import numpy as np
import torch
import open_clip
from PIL import Image


class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 device: str = "cpu"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_pil(self, image: Image.Image) -> np.ndarray:
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].astype(np.float32)

    @torch.no_grad()
    def embed_bgr(self, frame_bgr) -> np.ndarray:
        rgb = frame_bgr[:, :, ::-1]
        img = Image.fromarray(rgb)
        return self.embed_pil(img)
