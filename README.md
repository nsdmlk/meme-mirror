
# Meme-mirror 🐱

**Real-time computer vision app that reads your emotion and gestures, then finds the meme that matches your vibe.**

![meme](cat.jpg)

<!-- if no GIF yet: ![meme](docs/meme_example.jpg) -->

---

## What it does

Point your webcam at yourself. The app:

1. Detects your **face**, **pose** (shoulders, elbows, wrists, hips), and **hands** (21 keypoints each)
2. Classifies your **emotion** (8 classes: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt)
3. Recognizes your **gesture** (thumbs up, point up, peace, pray, facepalm, middle finger, pointing fingers)
4. Retrieves the **most similar meme** from a local collection

Everything runs locally on CPU. No cloud, no API keys.

---

**Download models** (≈25 MB total):

```bash
mkdir -p models ~/.hsemotion

# gesture recognizer
curl -L -o models/gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task

# emotion model
curl -L -o ~/.hsemotion/enet_b0_8_best_vgaf.onnx \
  https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx
```

**Build the meme index:**

```bash
# put your memes into data/memes/*.jpg
# tag them in data/memes/labels.json

export HF_ENDPOINT=https://hf-mirror.com   # if outside US/EU
python scripts/build_index.py
```

**Run:**

```bash
python -m meme_mirror.capture
```

---

## Controls

| Key             | Action               |
| --------------- | -------------------- |
| `q` / `ESC` | quit                 |
| `m`           | mirror on/off        |
| `s`           | save snapshot        |
| `p`           | toggle pose skeleton |
| `h`           | toggle hands         |
| `r`           | toggle retrieval     |
| `d`           | toggle debug output  |

---

## Meme dataset format

Place images in `data/memes/` and tag them in `data/memes/labels.json`:

```json
{
  "cat_001.jpg": {"emotion": "neutral",   "gesture": "none",       "energy": "calm"},
  "cat_002.jpg": {"emotion": "happiness", "gesture": "thumbs_up",  "energy": "playful"},
  "cat_003.jpg": {"emotion": "neutral",   "gesture": "point_up",   "energy": "confident"}
}
```

**Allowed `emotion` tags:** `neutral`, `happiness`, `surprise`, `sadness`, `anger`, `disgust`, `fear`, `contempt`

**Allowed `gesture` tags:** `none`, `thumbs_up`, `thumbs_down`, `point_up`, `peace`, `pray`, `facepalm`, `middle_finger`, `pointing_fingers`, `hands_up`

---

## Roadmap

- [X] Camera capture + face detection
- [X] Emotion classification (HSEmotion, 8 classes)
- [X] Pose + hand tracking
- [X] Gesture recognition (MediaPipe + rules)
- [X] CLIP embeddings + FAISS index
- [X] Hybrid retrieval (CLIP + emotion + gesture)
- [ ] Temporal smoothing (gesture stable over N frames)
- [ ] Fine-tuned gesture classifier on custom dataset
- [ ] OBS / browser plugin wrapper
- [ ] Benchmark: retrieval accuracy vs. annotation

---

## Project structure

```
meme-mirror/
├── src/meme_mirror/
│   ├── capture.py            # main loop
│   ├── features.py           # face + pose detectors
│   ├── emotions.py           # HSEmotion ONNX wrapper
│   ├── gesture_recognizer.py # MediaPipe Tasks API wrapper
│   ├── gestures.py           # rule-based gesture fallback
│   └── retriever.py          # hybrid FAISS retrieval
├── scripts/
│   └── build_index.py        # CLIP embedding builder
├── data/memes/               # your memes (gitignored)
└── models/                   # downloaded ONNX models (gitignored)
```

---

## Why this project

Built as a portfolio piece to explore:

- **Multimodal CV pipelines** — combining multiple MediaPipe models with deep embedding retrieval in real time
- **Practical ML engineering** — ONNX inference, dependency conflicts (MediaPipe vs PyTorch), throttled pipelines
- **Product thinking** — not just "run CLIP on a webcam", but a coherent retrieval scoring function with explicit semantic tags

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [Ilya Emelyanov](https://github.com/nsdmlk) · Beijing Institute of Technology*
