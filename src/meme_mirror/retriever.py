import json
import pickle
import numpy as np
import faiss


class MemeRetriever:
    def __init__(self, index_path, meta_path, labels_path):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

    def search(self, query_vec, emotion=None, gesture=None, top_k=3,
               w_clip=0.5, w_emotion=0.3, w_gesture=0.2):
        q = query_vec.astype(np.float32)[None, :]
        # берём больше кандидатов чтобы переранжировать
        scores, idxs = self.index.search(q, min(len(self.meta), 20))

        results = []
        for clip_score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            meta = self.meta[i]
            label = self.labels.get(meta["file"], {})

            emo_match = 1.0 if emotion and label.get("emotion") == emotion else 0.0
            ges_match = 1.0 if gesture and label.get("gesture") == gesture else 0.0

            final = (w_clip * float(clip_score)
                     + w_emotion * emo_match
                     + w_gesture * ges_match)

            results.append((final, float(clip_score), meta, label))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
