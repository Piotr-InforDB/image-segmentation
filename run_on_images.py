import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from classes.model import UNet

IMAGE_DIR = "C:/Users/piotr/Downloads/test"
BATCH_SIZE = 1
best_model_path = "models/latest-768.pth"
IMAGE_SIZE = (768, 768)

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = tensor.clone().cpu().numpy()
    img = img.transpose(1,2,0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, self.filenames[idx]

transform = A.Compose([
    A.Resize(*IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])

dataset = InferenceDataset(IMAGE_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=2).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

os.makedirs("predictions", exist_ok=True)

with torch.no_grad():
    for images, filenames in loader:
        # images: [batch_size, 3, h, w]

        images = images.to(device)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        probs[probs < 0.9] = 0

        preds = torch.argmax(probs, dim=1)

        for i in range(images.size(0)):

            img = denormalize(images[i])
            pred = preds[i].cpu().numpy()
            prob_map = probs[i, 1].cpu().numpy()


            plt.figure(figsize=(15, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Image")

            plt.subplot(1, 3, 2)
            plt.imshow(pred, cmap="gray")
            plt.title("Prediction")

            plt.subplot(1, 3, 3)
            plt.imshow(prob_map, cmap="jet")
            plt.title(f"Probability map")

            plt.suptitle(filenames[i])
            plt.tight_layout()
            plt.show()