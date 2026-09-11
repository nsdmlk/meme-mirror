import time
import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, model_selection: int = 0, min_confidence: float = 0.5):
        self._mp_face = mp.solutions.face_detection
        self._face = self._mp_face.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame):
        """Return list of (x, y, w, h) boxes in pixel coords."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face.process(rgb)

        boxes = []
        if result.detections:
            for det in result.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                # clamp to frame
                x, y = max(0, x), max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)
                boxes.append((x, y, bw, bh))
        return boxes

    def close(self):
        self._face.close()


class FPS:
    """Simple exponential moving average FPS counter."""
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.fps = 0.0
        self._last = time.time()

    def tick(self) -> float:
        now = time.time()
        dt = now - self._last
        self._last = now
        if dt > 0:
            inst = 1.0 / dt
            self.fps = inst if self.fps == 0 else self.alpha * self.fps + (1 - self.alpha) * inst
        return self.fps


def draw_faces(frame, boxes, color=(0, 255, 0), thickness=2, corner_ratio=0.2):
    """Draw corner brackets instead of a full rectangle."""
    for (x, y, w, h) in boxes:
        cx = int(w * corner_ratio)
        cy = int(h * corner_ratio)

        # top-left
        cv2.line(frame, (x, y), (x + cx, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + cy), color, thickness)

        # top-right
        cv2.line(frame, (x + w, y), (x + w - cx, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + cy), color, thickness)

        # bottom-left
        cv2.line(frame, (x, y + h), (x + cx, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - cy), color, thickness)

        # bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - cx, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - cy), color, thickness)