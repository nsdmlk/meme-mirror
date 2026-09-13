from hsemotion_onnx.facial_emotions import HSEmotionRecognizer


class EmotionDetector:
    def __init__(self, model_name: str = "enet_b0_8_best_vgaf"):
        self._recognizer = HSEmotionRecognizer(model_name=model_name)

    def predict(self, frame_bgr, box):
        """Return (emotion: str, score: float) for the face box."""
        x, y, w, h = box
        pad = 20
        y1 = max(0, y - pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)

        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None, 0.0

        try:
            emotion, scores = self._recognizer.predict_emotions(face, logits=False)
            return emotion, float(scores.max())
        except Exception:
            return None, 0.0