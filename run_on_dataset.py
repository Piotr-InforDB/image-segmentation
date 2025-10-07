import torch

from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np


from classes.dataset import Data
from classes.model import UNet

DATASET_DIR = "dataset-768"
BATCH_SIZE = 1
best_model_path = "models/unet_best.pth"
IMAGE_SIZE = (768, 768)

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = tensor.clone().cpu().numpy()
    img = img.transpose(1,2,0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

# Validation transform (no augmentations)
val_transform = A.Compose([
    A.Resize(*IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# Dataset + DataLoader
val_dataset = Data(
    image_dir=f"{DATASET_DIR}/images/val",
    mask_dir=f"{DATASET_DIR}/masks/val",
    transform=val_transform
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=2).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# Run inference
with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for i in range(images.size(0)):
            img = denormalize(images[i])
            mask = masks[i].cpu().numpy()
            pred = preds[i].cpu().numpy()

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.imshow(img); plt.title("Image")
            plt.subplot(1, 3, 2); plt.imshow(mask, cmap="gray"); plt.title("Mask")
            plt.subplot(1, 3, 3); plt.imshow(pred, cmap="gray"); plt.title("Prediction")
            plt.suptitle(f"File: {i}")
            plt.show()
