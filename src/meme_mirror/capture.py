import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image

from meme_mirror.emotions import EmotionDetector
from meme_mirror.features import (
    FaceDetector, PoseDetector, FPS,
    draw_faces, draw_pose, draw_hands,
)
from meme_mirror.gesture_recognizer import GestureRecognizerWrapper
from meme_mirror.gestures import detect_gesture
from meme_mirror.retriever import MemeRetriever

WINDOW = "meme-mirror"
MEME_WINDOW = "closest meme"
HINTS = [
    "q / ESC  - quit",
    "m        - mirror on/off",
    "s        - save snapshot",
    "p        - toggle pose",
    "h        - toggle hands",
    "r        - toggle retrieval",
    "d        - toggle debug",
]

ANALYZE_EVERY = 3
MEME_EVERY = 10
MP_CONFIDENCE = 0.7   # порог уверенности MediaPipe, ниже — rule-based


def open_camera(index=0, width=1280, height=720, fps=30):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


class ClipEmbedder:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.eval()

    @torch.no_grad()
    def embed_bgr(self, frame_bgr):
        rgb = frame_bgr[:, :, ::-1]
        img = Image.fromarray(rgb)
        x = self.preprocess(img).unsqueeze(0)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].astype(np.float32)


def draw_hud(frame, mirror, fps, n_faces, n_hands, has_pose,
             meme_info, retrieval_on, resolution):
    for i, line in enumerate(HINTS):
        y = 25 + i * 22
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    info = (f"fps: {fps:5.1f}  faces: {n_faces}  hands: {n_hands}  "
            f"pose: {'Y' if has_pose else 'N'}  {resolution}")
    cv2.putText(frame, info, (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)

    if retrieval_on and meme_info:
        final, clip_s, label = meme_info
        txt = (f"match {final:.2f}  clip {clip_s:.2f}  "
               f"| {label.get('emotion','?')} / {label.get('gesture','?')}")
        cv2.putText(frame, txt, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)


def draw_emotion(frame, box, emotion, score):
    x, y, _, _ = box
    text = f"{emotion}  {score:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = max(y - 10, th + 6)
    cv2.rectangle(frame, (x - 2, ty - th - 6),
                  (x + tw + 6, ty + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)


def crop_face(frame, box, pad=60):
    x, y, w, h = box
    y1 = max(0, y - pad)
    y2 = min(frame.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(frame.shape[1], x + w + pad)
    return frame[y1:y2, x1:x2]


def stream(cap, window=WINDOW):
    mirror = True
    show_pose = True
    show_hands = True
    retrieval_on = True
    debug = False
    snap_count = 0

    detector = FaceDetector()
    pose_detector = PoseDetector(model_complexity=1)
    gesture_recognizer = GestureRecognizerWrapper(
        "models/gesture_recognizer.task", min_confidence=0.5)
    emotion_detector = EmotionDetector()
    retriever = MemeRetriever(
        "data/memes.index",
        "data/memes_meta.pkl",
        "data/memes/labels.json",
    )
    embedder = ClipEmbedder()
    fps = FPS()

    last_emotion = (None, 0.0)
    last_meme = None
    last_gesture = "none"
    last_mp_score = 0.0
    frame_idx = 0

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = f"{real_w}x{real_h}"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, real_w, real_h)
    cv2.namedWindow(MEME_WINDOW, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            boxes = detector.detect(frame)
            draw_faces(frame, boxes)

            pose_kp = None
            if show_pose:
                pose_kp = pose_detector.detect(frame)
                draw_pose(frame, pose_kp)

            # MediaPipe Gesture Recognizer (и руки, и жест)
            hands = []
            mp_tag, mp_score = "none", 0.0
            if show_hands:
                mp_tag, mp_score, hands = gesture_recognizer.recognize(frame)
                draw_hands(frame, hands)

            frame_idx += 1

            # emotion
            if boxes and frame_idx % ANALYZE_EVERY == 0:
                last_emotion = emotion_detector.predict(frame, boxes[0])
            if boxes and last_emotion[0]:
                draw_emotion(frame, boxes[0], *last_emotion)

            # gesture: MediaPipe если уверен, иначе rule-based
            if mp_score >= MP_CONFIDENCE and mp_tag != "none":
                last_gesture = mp_tag
                last_mp_score = mp_score
            else:
                last_gesture = detect_gesture(hands, pose_kp, frame.shape)
                last_mp_score = mp_score

            # retrieval
            if retrieval_on and boxes and frame_idx % MEME_EVERY == 0:
                crop = crop_face(frame, boxes[0], pad=60)
                if crop.size > 0:
                    vec = embedder.embed_bgr(crop)
                    emo = last_emotion[0]
                    ges = last_gesture
                    results = retriever.search(
                        vec, emotion=emo, gesture=ges, top_k=1,
                    )
                    if results:
                        final, clip_s, meta, label = results[0]
                        last_meme = (final, clip_s, meta, label)
                        if debug:
                            print(f"[retr] emo={emo} ges={ges} "
                                  f"(mp={mp_tag}/{mp_score:.2f}) "
                                  f"-> {meta['file']} final={final:.3f} "
                                  f"clip={clip_s:.3f}")

            if retrieval_on and last_meme:
                final, clip_s, meta, label = last_meme
                meme_img = cv2.imread(meta["path"])
                if meme_img is not None:
                    cv2.putText(meme_img,
                                f"{label.get('emotion','?')} | "
                                f"{label.get('gesture','?')}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(MEME_WINDOW, meme_img)

            meme_info = None
            if last_meme:
                meme_info = (last_meme[0], last_meme[1], last_meme[3])

            draw_hud(frame, mirror, fps.tick(), len(boxes), len(hands),
                     pose_kp is not None, meme_info, retrieval_on, resolution)

            if debug:
                cv2.putText(frame,
                            f"gesture: {last_gesture}  "
                            f"(mp: {mp_tag} {mp_score:.2f})",
                            (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("p"):
                show_pose = not show_pose
            elif key == ord("h"):
                show_hands = not show_hands
            elif key == ord("d"):
                debug = not debug
            elif key == ord("r"):
                retrieval_on = not retrieval_on
                if not retrieval_on:
                    cv2.destroyWindow(MEME_WINDOW)
            elif key == ord("s"):
                snap_count += 1
                path = f"snapshot_{snap_count:03d}.png"
                cv2.imwrite(path, frame)
                print(f"Saved {path}")
    finally:
        detector.close()
        pose_detector.close()
        gesture_recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = open_camera(0)
    stream(cam)