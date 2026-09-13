import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# MediaPipe Gesture Recognizer -> наши теги
MP_TO_TAG = {
    "Thumb_Up":    "thumbs_up",
    "Thumb_Down":  "thumbs_down",
    "Pointing_Up": "point_up",
    "Victory":     "peace",
    "ILoveYou":    "none",
    "Open_Palm":   "hands_up",
    "Closed_Fist": "none",
    "None":        "none",
}


class GestureRecognizerWrapper:
    def __init__(self, model_path: str = "models/gesture_recognizer.task",
                 min_confidence: float = 0.5):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def recognize(self, frame_bgr):
        """Return (tag, score, hands_landmarks).

        hands_landmarks: list of [(x, y), ...21] in pixel coords.
        """
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))

        result = self._recognizer.recognize(mp_image)
        h, w = frame_bgr.shape[:2]

        hands = []
        for lms in result.hand_landmarks:
            pts = [(int(p.x * w), int(p.y * h)) for p in lms]
            hands.append(pts)

        if result.gestures:
            top = result.gestures[0][0]   # (category_name, score)
            name = top.category_name
            score = float(top.score)
            return MP_TO_TAG.get(name, "none"), score, hands

        return "none", 0.0, hands

    def close(self):
        self._recognizer.close()
