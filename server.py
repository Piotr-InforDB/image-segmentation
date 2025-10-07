import io
import base64
import torch
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from albumentations.pytorch import ToTensorV2
import albumentations as A
from classes.model import UNet
from PIL import Image


IMAGE_SIZE = (768, 768)
MODEL_PATH = "models/unet_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(n_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = A.Compose([
    A.Resize(*IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImagePayload(BaseModel):
    image: str

def decode_base64_image(data: str):
    if data.startswith("data:image"):
        data = data.split(",")[1]
    missing_padding = len(data) % 4
    if missing_padding:
        data += "=" * (4 - missing_padding)
    image_bytes = base64.b64decode(data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

def encode_base64_image(image: np.ndarray):
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")

@app.post("/predict")
def predict(payload: ImagePayload):
    image = decode_base64_image(payload.image)

    transformed = transform(image=image)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(transformed)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8) * 255

    pred_b64 = encode_base64_image(pred)
    return {"prediction": pred_b64}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)