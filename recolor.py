import os
from PIL import Image
import numpy as np

# Paths
input_root = "dataset/masks"
output_root = "dataset/masks_red"

color_from = np.array([128, 0, 0])
color_to   = np.array([255, 0, 0])

os.makedirs(output_root, exist_ok=True)

os.makedirs(output_root, exist_ok=True)

for file in os.listdir(input_root):
    if file.endswith(".png") or file.endswith(".jpg"):
        path = os.path.join(input_root, file)
        mask = Image.open(path).convert("RGB")
        mask_arr = np.array(mask)

        match = np.all(mask_arr == color_from, axis=-1)
        mask_arr[match] = color_to

        out = Image.fromarray(mask_arr)
        out.save(os.path.join(output_root, file))

print("✅ Conversion complete. Red masks saved in dataset_rooftop/masks_red/")
