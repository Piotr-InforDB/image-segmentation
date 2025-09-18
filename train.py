import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np


transform_img = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)
        self.transform = transform

        self.colormap = {
            (0, 0, 0): 0,
            (255, 0, 0): 1,
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.img_dir, img_filename)

        mask_filename = os.path.splitext(img_filename)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mask = mask.resize((512, 512), Image.NEAREST)

        mask_np = np.array(mask)
        mask_class = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
        for color, class_id in self.colormap.items():
            matches = np.all(mask_np == color, axis=-1)
            mask_class[matches] = class_id

        mask = torch.tensor(mask_class, dtype=torch.long)
        return image, mask




transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

train_data = SegDataset("dataset/images/train", "dataset/masks/train", transform)
val_data   = SegDataset("dataset/images/val", "dataset/masks/val", transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=10, shuffle=False)

model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=2)
model = model.to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(25):
    model.train()
    total_train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to("cuda"), masks.to("cuda")

        optimizer.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to("cuda"), masks.to("cuda")
            outputs = model(imgs)['out']
            loss = criterion(outputs, masks)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

torch.save(model.state_dict(), "building_segmentation.pth")
