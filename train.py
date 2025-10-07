import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

from classes.dataset import Data
from classes.model import UNet
import torch.nn.functional as F


EPOCHS = 200
BATCH_SIZE = 4
DATASET_DIR = "dataset-768"

#Augmentation
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


# Define Data
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

# Define Model
model = UNet(n_classes=2)
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


criterion = CombinedLoss()

# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.75, patience=5
)

best_val_loss = float("inf")
best_model_path = "models/unet_best.pth"

train_losses = []
val_losses = []

# Train
print("Training...")
for epoch in range(EPOCHS):
    start = time.time()

    # Training
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=False):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    # Evaluate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    duration = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f} | duration: {duration:.2f}s")


model.load_state_dict(torch.load(best_model_path))
model.to(device)
model.eval()

with torch.no_grad():
    images, masks = next(iter(val_loader))
    images, masks = images.to(device), masks.to(device)

    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

img = images[0].permute(1, 2, 0).cpu().numpy()
mask = masks[0].cpu().numpy()
pred = preds[0].cpu().numpy()

# Loss plot
plt.figure(figsize=(8,6))
plt.plot(range(1, EPOCHS+1), train_losses, label="Training Loss")
plt.plot(range(1, EPOCHS+1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.ylim(0, .66)
plt.legend()
plt.grid(True)
plt.show()

# Images plot
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(img)

plt.subplot(1,3,2)
plt.title("Mask")
plt.imshow(mask, cmap="gray")

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(pred, cmap="gray")
plt.show()
