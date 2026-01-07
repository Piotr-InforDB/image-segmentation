import io
import os
from typing import Any

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from cog import BasePredictor, Input, Path

from classes.model_deeplabv3 import DeepLabV3Plus

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DeepLabV3Plus(n_classes=2, groupnorm=False)
        self.model.to(self.device)
        self.model.eval()

        ckpt_path = "models/checkpoint.pth"
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Checkpoint not found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        img = np.array(image).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))  # Fixed: removed the duplicate line

        tensor = torch.from_numpy(img).unsqueeze(0)  # [1, 3, H, W]
        return tensor.float()

    def postprocess_mask(self, logits: torch.Tensor) -> Image.Image:
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)  # [1, H, W]
        mask = preds[0].cpu().numpy().astype(np.uint8)  # [H, W]

        mask = (mask * 255).astype(np.uint8)
        return Image.fromarray(mask)

    def predict(
        self,
        image: Path = Input(description="Input RGB image"),
    ) -> Path:
        pil_img = Image.open(str(image))

        x = self.preprocess_image(pil_img).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        mask_img = self.postprocess_mask(logits)

        out_path = "/tmp/output.png"
        mask_img.save(out_path)
        return Path(out_path)
