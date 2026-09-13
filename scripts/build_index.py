import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

print("1. env set", flush=True)

import sys
print("2. sys ok", flush=True)

import torch
print("3. torch ok", flush=True)

import open_clip
print("4. open_clip ok", flush=True)

import faiss
print("5. faiss ok", flush=True)

from PIL import Image
print("6. PIL ok", flush=True)

import numpy as np
print("7. numpy ok", flush=True)

print("8. loading model...", flush=True)
m, _, pre = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
print("9. model loaded", flush=True)

print("10. testing encode...", flush=True)
img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
x = pre(img).unsqueeze(0)
print("11. preprocess ok", flush=True)
with torch.no_grad():
    feat = m.encode_image(x)
print("12. encode ok", feat.shape, flush=True)

print("13. listing memes...", flush=True)
files = sorted(f for f in os.listdir("data/memes") if f.lower().endswith((".jpg",".jpeg",".png",".webp")))
print(f"14. found {len(files)} files", flush=True)

for i, f in enumerate(files[:2]):
    print(f"15.{i} opening {f}", flush=True)
    im = Image.open(os.path.join("data/memes", f)).convert("RGB")
    print(f"16.{i} loaded, size={im.size}", flush=True)
    t = pre(im).unsqueeze(0)
    print(f"17.{i} preprocessed", flush=True)
    with torch.no_grad():
        v = m.encode_image(t)
    print(f"18.{i} encoded", v.shape, flush=True)

print("=== processing all 14 ===", flush=True)
for i, f in enumerate(files):
    print(f"[{i}] {f}", flush=True)
    try:
        im = Image.open(os.path.join("data/memes", f)).convert("RGB")
        t = pre(im).unsqueeze(0)
        with torch.no_grad():
            v = m.encode_image(t)
        print(f"  OK {v.shape}", flush=True)
    except Exception as e:
        print(f"  FAIL: {e}", flush=True)

print("=== building index ===", flush=True)
vectors = []
for f in files:
    im = Image.open(os.path.join("data/memes", f)).convert("RGB")
    t = pre(im).unsqueeze(0)
    with torch.no_grad():
        v = m.encode_image(t)
    v = v / v.norm(dim=-1, keepdim=True)
    vectors.append(v.cpu().numpy()[0].astype(np.float32))

print("vectors collected:", len(vectors), flush=True)
mat = np.stack(vectors).astype(np.float32)
print("mat:", mat.shape, flush=True)

print("creating faiss index...", flush=True)
index = faiss.IndexFlatIP(mat.shape[1])
print("faiss index created", flush=True)

print("adding to index...", flush=True)
index.add(mat)
print("added, ntotal =", index.ntotal, flush=True)

faiss.write_index(index, "data/memes.index")
print("index saved", flush=True)

import pickle
meta = [{"file": f, "path": os.path.join("data/memes", f)} for f in files]
with open("data/memes_meta.pkl", "wb") as fh:
    pickle.dump(meta, fh)
print("meta saved", flush=True)

print("DONE", flush=True)
