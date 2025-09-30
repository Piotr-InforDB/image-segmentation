import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

EPOCHS = 100
BATCH_SIZE = 1
IMAGE_SIZE = 512

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
        red_threshold = (mask[:, :, 0] > 100) & (mask[:, :, 1] < 100) & (mask[:, :, 2] < 100)
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

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(3, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


#Augmentation
image_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    # T.RandomRotation(30),
    T.ToTensor()
])
mask_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.NEAREST),
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

train_losses = []
val_losses = []

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
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

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
