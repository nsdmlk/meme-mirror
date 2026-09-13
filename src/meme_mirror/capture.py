import cv2

from meme_mirror.emotions import EmotionDetector
from meme_mirror.features import FaceDetector, FPS, draw_faces

WINDOW = "meme-mirror"
HINTS = [
    "q / ESC  - quit",
    "m        - mirror on/off",
    "s        - save snapshot",
]

ANALYZE_EVERY = 3   # эмоции раз в N кадров (~10 Гц при 30fps)


def open_camera(index: int = 0,
                width: int = 1280,
                height: int = 720,
                fps: int = 30) -> cv2.VideoCapture:
    # AVFoundation даёт лучшую картинку на macOS
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        # fallback на дефолтный backend
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {real_w}x{real_h} @ {real_fps:.0f} fps")

    return cap


def draw_hud(frame, mirror: bool, fps: float, n_faces: int, resolution: str):
    for i, line in enumerate(HINTS):
        y = 25 + i * 22
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    info = f"fps: {fps:5.1f}   faces: {n_faces}   {resolution}"
    cv2.putText(frame, info, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)


def draw_emotion(frame, box, emotion: str, score: float):
    x, y, _, _ = box
    text = f"{emotion}  {score:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = max(y - 10, th + 6)
    cv2.rectangle(frame,
                  (x - 2, ty - th - 6),
                  (x + tw + 6, ty + 4),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)


def stream(cap: cv2.VideoCapture, window: str = WINDOW):
    mirror = True
    snap_count = 0

    detector = FaceDetector()
    emotion_detector = EmotionDetector()
    fps = FPS()

    last_emotion = (None, 0.0)
    frame_idx = 0

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = f"{real_w}x{real_h}"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, real_w, real_h)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            boxes = detector.detect(frame)
            draw_faces(frame, boxes)

            # эмоции — throttled
            frame_idx += 1
            if boxes and frame_idx % ANALYZE_EVERY == 0:
                last_emotion = emotion_detector.predict(frame, boxes[0])

            if boxes and last_emotion[0]:
                emo, score = last_emotion
                draw_emotion(frame, boxes[0], emo, score)

            draw_hud(frame, mirror, fps.tick(), len(boxes), resolution)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("s"):
                snap_count += 1
                path = f"snapshot_{snap_count:03d}.png"
                cv2.imwrite(path, frame)
                print(f"Saved {path}")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = open_camera(0)
    stream(cam)