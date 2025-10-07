import os
from PIL import Image
import numpy as np

# Paths
input_root = "dataset_rooftops/masks"
output_root = "dataset_rooftops/masks_red"

os.makedirs(output_root, exist_ok=True)

for split in ["train", "val"]:
    input_dir = os.path.join(input_root, split)
    output_dir = os.path.join(output_root, split)
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".png") or file.endswith(".jpg"):
            path = os.path.join(input_dir, file)
            mask = Image.open(path).convert("L")  # grayscale

            mask_arr = np.array(mask)

            rgb = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype=np.uint8)
            rgb[mask_arr > 127] = [255, 0, 0]

            out = Image.fromarray(rgb)
            out.save(os.path.join(output_dir, file))

print("✅ Conversion complete. Red masks saved in dataset_rooftop/masks_red/")
