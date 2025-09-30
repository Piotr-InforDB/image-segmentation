import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

EPOCHS = 200
BATCH_SIZE = 1

# Classes
class Data(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Apply Image transforms
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = T.ToTensor()(image)

        # Apply Mask transforms
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = np.array(mask)
        red_threshold = (mask[:, :, 0] > 150) & (mask[:, :, 1] < 100) & (mask[:, :, 2] < 100)
        mask_class = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        mask_class[red_threshold] = 1
        mask = torch.from_numpy(mask_class)

        return image, mask

    def __len__(self):
        return len(self.images)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        # Bottleneck
        b = self.bottleneck(self.pool2(e2))

        # Decoder
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.out_conv(d2)


#Augmentation
image_transform = T.Compose([
    T.Resize((512, 512)),
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    # T.RandomRotation(30),
    T.ToTensor()
])
mask_transform = T.Compose([
    T.Resize((512, 512), interpolation=Image.NEAREST),
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    # T.RandomRotation(30),
])


# Define Data
train_dataset = Data(
    image_dir="dataset/images/train",
    mask_dir="dataset/masks/train",
    image_transform=image_transform,
    mask_transform=mask_transform
)
val_dataset = Data(
    image_dir="dataset/images/val",
    mask_dir="dataset/masks/val",
    image_transform=T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ]),
    mask_transform=T.Compose([
        T.Resize((512, 512),interpolation=Image.NEAREST)
    ])
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Define Model
model = UNet(n_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = float("inf")
best_model_path = "models/unet_best.pth"

# Train
for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
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
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved new best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


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

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(img)

plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(mask, cmap="gray")

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(pred, cmap="gray")
plt.show()
