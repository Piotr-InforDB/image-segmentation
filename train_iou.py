import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

from classes.dataset import Data
from classes.model import UNet


EPOCHS = 100
BATCH_SIZE = 2
DATASET_DIR = "dataset-768"
BEST_MODEL_PATH = "models/unet_best_iou.pth"


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(0.1, 0.1),
        rotate=(-15, 15),
        shear=(-10, 10),
        p=0.5
    ),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])


train_dataset = Data(
    image_dir=f"{DATASET_DIR}/images/train",
    mask_dir=f"{DATASET_DIR}/masks/train",
    transform=train_transform,
)
val_dataset = Data(
    image_dir=f"{DATASET_DIR}/images/val",
    mask_dir=f"{DATASET_DIR}/masks/val",
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=2).to(device)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

        dice_scores = []
        for i in range(pred.size(1)):
            pred_flat = pred[:, i].contiguous().view(-1)
            target_flat = target_one_hot[:, i].contiguous().view(-1)
            intersection = (pred_flat * target_flat).sum()

            dice = (2 * intersection + self.smooth) / \
                   (pred_flat.sum() + target_flat.sum() + self.smooth)

            dice_scores.append(dice)

        return 1 - torch.stack(dice_scores).mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.alpha * self.ce(pred, target) + (1 - self.alpha) * self.dice(pred, target)


criterion = CombinedLoss()


def compute_iou(preds, masks, num_classes=2):
    preds = preds.view(-1)
    masks = masks.view(-1)

    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = masks == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# scheduler tries to MAXIMIZE IoU → we pass negative IoU
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.75, patience=10
)


print("Training...")

best_val_iou = 0.0

train_losses = []
val_losses = []
train_ious = []
val_ious = []

for epoch in range(EPOCHS):
    start = time.time()

    model.train()
    train_loss = 0
    train_iou = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        # IoU metric
        preds = torch.argmax(outputs, dim=1)
        train_iou += compute_iou(preds, masks) * images.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_iou = train_iou / len(train_loader.dataset)


    model.eval()
    val_loss = 0
    val_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)

            val_loss += loss.item() * images.size(0)
            val_iou += compute_iou(preds, masks) * images.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_iou = val_iou / len(val_loader.dataset)


    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_ious.append(avg_train_iou)
    val_ious.append(avg_val_iou)

    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"New best model saved at epoch {epoch+1} | IoU={best_val_iou:.4f}")

    scheduler.step(1 - avg_val_iou)

    duration = time.time() - start
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
        f"Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f} | "
        f"lr: {optimizer.param_groups[0]['lr']:.6f} | {duration:.1f}s"
    )


model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.to(device)
model.eval()


plt.figure(figsize=(8, 5))
plt.plot(val_ious, label="Validation IoU")
plt.plot(train_ious, label="Training IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("IoU over epochs")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.grid(True)
plt.show()
