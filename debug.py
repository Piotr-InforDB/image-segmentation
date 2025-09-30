import numpy as np
from PIL import Image

mask = Image.open("dataset/masks/train/1.png").convert("RGBA")
mask = np.array(mask)

# What unique RGB values exist?
unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
print(unique_colors)