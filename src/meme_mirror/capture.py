import cv2

from meme_mirror.features import FaceDetector, FPS, draw_faces

WINDOW = "meme-mirror"
HINTS = [
    "q / ESC  - quit",
    "m        - mirror on/off",
    "s        - save snapshot",
]


def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    return cap


def draw_hud(frame, mirror: bool, fps: float, n_faces: int):
    for i, line in enumerate(HINTS):
        y = 25 + i * 22
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    info = f"fps: {fps:5.1f}   faces: {n_faces}"
    cv2.putText(frame, info, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)


def stream(cap: cv2.VideoCapture, window: str = WINDOW):
    mirror = True
    snap_count = 0

    detector = FaceDetector()
    fps = FPS()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            boxes = detector.detect(frame)
            draw_faces(frame, boxes)

            draw_hud(frame, mirror, fps.tick(), len(boxes))
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