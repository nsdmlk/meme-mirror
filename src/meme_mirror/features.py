import time
import cv2
import mediapipe as mp


# ---------- FACE ----------

class FaceDetector:
    def __init__(self, model_selection: int = 0, min_confidence: float = 0.5):
        self._mp_face = mp.solutions.face_detection
        self._face = self._mp_face.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame):
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
                x, y = max(0, x), max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)
                boxes.append((x, y, bw, bh))
        return boxes

    def close(self):
        self._face.close()


# ---------- POSE ----------

class PoseDetector:
    def __init__(self, min_confidence: float = 0.5, model_complexity: int = 1):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return None

        kp = {}
        for i, lm in enumerate(result.pose_landmarks.landmark):
            kp[i] = (int(lm.x * w), int(lm.y * h), lm.visibility)
        return kp

    def close(self):
        self._pose.close()


POSE_KEYPOINTS = {
    0: "nose",
    11: "l_shoulder", 12: "r_shoulder",
    13: "l_elbow",    14: "r_elbow",
    15: "l_wrist",    16: "r_wrist",
    23: "l_hip",      24: "r_hip",
}

POSE_EDGES = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
]


def draw_pose(frame, kp, color=(0, 255, 255), thickness=2):
    if not kp:
        return
    for a, b in POSE_EDGES:
        if a in kp and b in kp:
            pa, pb = kp[a], kp[b]
            if pa[2] > 0.5 and pb[2] > 0.5:
                cv2.line(frame, pa[:2], pb[:2], color, thickness)
    for i in POSE_KEYPOINTS:
        if i in kp and kp[i][2] > 0.5:
            cv2.circle(frame, kp[i][:2], 4, color, -1)


# ---------- HANDS ----------

class HandDetector:
    def __init__(self, max_hands: int = 2, min_confidence: float = 0.5):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        hands = []
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                hands.append(pts)
        return hands

    def close(self):
        self._hands.close()


def draw_hands(frame, hands, color=(255, 100, 255), radius=2):
    for pts in hands:
        for (x, y) in pts:
            cv2.circle(frame, (x, y), radius, color, -1)


# ---------- UTILS ----------

class FPS:
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
    """Corner brackets around each face."""
    for (x, y, w, h) in boxes:
        cx = int(w * corner_ratio)
        cy = int(h * corner_ratio)

        cv2.line(frame, (x, y), (x + cx, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + cy), color, thickness)
        cv2.line(frame, (x + w, y), (x + w - cx, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + cy), color, thickness)
        cv2.line(frame, (x, y + h), (x + cx, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - cy), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w - cx, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - cy), color, thickness)