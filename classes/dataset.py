import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))

        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        red_threshold = (mask_rgb[:, :, 0] > 100) & \
                        (mask_rgb[:, :, 1] == 0) & \
                        (mask_rgb[:, :, 2] == 0)
        mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
        mask[red_threshold] = 1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()

        return image, mask

    def __len__(self):
        return len(self.images)