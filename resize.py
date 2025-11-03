import os
import random
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

INPUT_IMAGES = "dataset/images"
INPUT_MASKS = "dataset/masks"
OUTPUT_DIR = "dataset-512"
SIZE = (512, 512)
VAL_RATIO = 0.2
MAX_IMAGES = 100

def resize_and_save(input_path, output_path, size, is_mask=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with Image.open(input_path) as img:
        img = ImageOps.exif_transpose(img)
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        img = img.resize(size, resample)
        img.save(output_path)

def main():
    images = sorted([f for f in os.listdir(INPUT_IMAGES) if f.lower().endswith((".jpg", ".png"))])
    masks = sorted([f for f in os.listdir(INPUT_MASKS) if f.lower().endswith(".png")])

    paired = [(os.path.join(INPUT_IMAGES, img), os.path.join(INPUT_MASKS, mask))
              for img, mask in zip(images, masks)]

    if MAX_IMAGES is not None and MAX_IMAGES < len(paired):
        paired = random.sample(paired, MAX_IMAGES)

    train, val = train_test_split(paired, test_size=VAL_RATIO, random_state=42)

    for split_name, dataset in [("train", train), ("val", val)]:
        for img_path, mask_path in dataset:
            out_img = os.path.join(OUTPUT_DIR, "images", split_name, os.path.basename(img_path))
            out_mask = os.path.join(OUTPUT_DIR, "masks", split_name, os.path.basename(mask_path))

            resize_and_save(img_path, out_img, SIZE, is_mask=False)
            resize_and_save(mask_path, out_mask, SIZE, is_mask=True)

    print(f"✅ Done! Used {len(paired)} image-mask pairs, saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
