import os
import cv2
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from classes.model_deeplabv3 import DeepLabV3Plus

IMAGE_DIR = "C:/Users/piotr/Documents/DroneMissions/Gennep/DJI_202508201135_010_blok6"
BEST_MODEL_PATH = "models/checkpoint_best.pth"
OUTPUT_VIDEO = "predictions/output.mp4"

IMAGE_SIZE = (1280, 1280)
FPS = 8
THRESHOLD = 0.5

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = tensor.clone().cpu().numpy()
    img = img.transpose(1,2,0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.filenames = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        self.filenames.sort()
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.image_dir, fname)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_t = self.transform(image=img)["image"]
        else:
            img_t = img

        return img_t, fname, img  # also return raw image for display



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepLabV3Plus(n_classes=2).to(device)

checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()


transform = A.Compose([
    A.Resize(*IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])

dataset = InferenceDataset(IMAGE_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

split_width = IMAGE_SIZE[1] * 2
split_height = IMAGE_SIZE[0]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (split_width, split_height))

with torch.no_grad():
    for img_tensor, filename, raw_img in loader:

        img_tensor = img_tensor.to(device)

        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)[0].cpu().numpy()

        denorm = denormalize(img_tensor[0])

        pred_color = (preds * 255).astype(np.uint8)
        pred_color = cv2.cvtColor(pred_color, cv2.COLOR_GRAY2BGR)

        left = cv2.resize(pred_color, IMAGE_SIZE[::-1])
        right = cv2.resize(denorm, IMAGE_SIZE[::-1])

        combined = np.hstack((left, right))

        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        writer.write(combined_bgr)
        print(f"Added frame: {filename}")

writer.release()
print(f"\nVideo saved to: {OUTPUT_VIDEO}")
