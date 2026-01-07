import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from classes.dataset import Data
import torch.nn.functional as F
from classes.model_deeplabv3 import DeepLabV3Plus

RESUME_FROM = None
# RESUME_FROM = 'models/checkpoint-1280-8s.pth'
EPOCHS = 100
BATCH_SIZE = 2
DATASET_DIR = "dataset-768-2S"
SAVE_DIR = "models"

os.makedirs(SAVE_DIR, exist_ok=True)

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

model = DeepLabV3Plus(n_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


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
            dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
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


def compute_iou(preds, masks, num_classes=2):
    preds = preds.view(-1)
    masks = masks.view(-1)

    ious = []

    # Only for classes > 0
    for cls in range(1, num_classes):
        pred_inds = preds == cls
        target_inds = masks == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union > 0:
            ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0


criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.75, patience=10
)

scaler = GradScaler('cuda')

start_epoch = 0
best_val_iou = 0
train_losses = []
val_losses = []
val_ious = []
best_model_path = os.path.join(SAVE_DIR, "checkpoint.pth")

if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"Resuming training from: {RESUME_FROM}")
    checkpoint = torch.load(RESUME_FROM, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_iou = checkpoint["best_val_iou"]

    # Load scaler state if available
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    # Load training history if available
    if "train_losses" in checkpoint:
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        val_ious = checkpoint["val_ious"]

    print(f"Resuming from epoch {start_epoch} with best IoU: {best_val_iou:.4f}")
else:
    print("Starting training from scratch...")

print("Training...")
for epoch in range(start_epoch, EPOCHS):
    start = time.time()

    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]", leave=False):
            images, masks = images.to(device), masks.to(device)

            # Use mixed precision for validation too
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)
            batch_iou = compute_iou(preds, masks, num_classes=2)

            val_iou += batch_iou * images.size(0)
            val_loss += loss.item() * images.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_iou = val_iou / len(val_loader.dataset)
    val_ious.append(avg_val_iou)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_iou": best_val_iou,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_ious": val_ious
        }, best_model_path)
        print(f"✓ Saved new best model at epoch {epoch + 1} with val IoU {best_val_iou:.4f}")

    duration = time.time() - start
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f} | duration: {duration:.2f}s")

# Save training history
np.savez(os.path.join(SAVE_DIR, 'training_history.npz'),
         train_losses=train_losses,
         val_losses=val_losses,
         val_ious=val_ious)

# Load best model for visualization
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

with torch.no_grad():
    images, masks = next(iter(val_loader))
    images, masks = images.to(device), masks.to(device)

    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

img = images[0].permute(1, 2, 0).cpu().numpy()
# Denormalize for visualization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img = std * img + mean
img = np.clip(img, 0, 1)

mask = masks[0].cpu().numpy()
pred = preds[0].cpu().numpy()

# IoU plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_ious) + 1), val_ious, label="Validation IoU", color="green")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Validation IoU over epochs")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, 'iou_plot.png'), dpi=150, bbox_inches='tight')
plt.show()

# Loss plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.ylim(0, .66)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'loss_plot.png'), dpi=150, bbox_inches='tight')
plt.show()

# Images plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(mask, cmap="gray")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(pred, cmap="gray")
plt.axis('off')
plt.savefig(os.path.join(SAVE_DIR, 'prediction_sample.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Training complete! Best validation IoU: {best_val_iou:.4f}")
print(f"✓ Model saved to: {best_model_path}")