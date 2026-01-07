import os
import random
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIG
# ==========================================
INPUT_IMAGES = "dataset/images"
INPUT_MASKS = "dataset/masks"
OUTPUT_DIR = "dataset-768-2S"

TRAIN_SIZE = 0.8
FINAL_SIZE = (768, 768)

SLICE_LEVELS = [1, 2]

def load_and_resize(path, size, is_mask=False):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize(size, resample)


def slice_image(img, slices):
    w, h = img.size
    sw, sh = w // slices, h // slices

    patches = []

    for r in range(slices):
        for c in range(slices):
            left = c * sw
            top = r * sh
            patch = img.crop((left, top, left + sw, top + sh))
            patches.append((r, c, False, patch))  # False = not overlap

    return patches


def save_output(img, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, filename))


def main():
    # Collect paired images & masks
    imgs = sorted([f for f in os.listdir(INPUT_IMAGES) if f.lower().endswith((".jpg", ".png"))])
    masks = sorted([f for f in os.listdir(INPUT_MASKS) if f.lower().endswith(".png")])

    pairs = [(os.path.join(INPUT_IMAGES, img), os.path.join(INPUT_MASKS, mask))
             for img, mask in zip(imgs, masks)]

    all_patches = []

    index = 1
    for img_path, mask_path in pairs:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f'{index}: {name}')
        index = index + 1

        img_resized = load_and_resize(img_path, (5000, 5000), is_mask=False)
        mask_resized = load_and_resize(mask_path, (5000, 5000), is_mask=True)

        for slices in SLICE_LEVELS:
            img_patches = slice_image(img_resized, slices)
            mask_patches = slice_image(mask_resized, slices)

            for (r, c, overlapping, img_patch), (_, _, _, mask_patch) in zip(img_patches, mask_patches):

                img_final = img_patch.resize(FINAL_SIZE, Image.BILINEAR)
                mask_final = mask_patch.resize(FINAL_SIZE, Image.NEAREST)

                suffix = f"_s{slices}_r{r}_c{c}"
                filename = f"{name}{suffix}.png"

                min_val, max_val = mask_final.getextrema()
                if max_val == 0:
                    continue

                all_patches.append((img_final, mask_final, filename))

    print(f"\nTotal patches (after filtering): {len(all_patches)}")

    train_patches, val_patches = train_test_split(
        all_patches,
        train_size=TRAIN_SIZE,
        random_state=42
    )

    print(f"Train patches: {len(train_patches)}")
    print(f"Val patches: {len(val_patches)}")
    print(f"Actual train ratio: {len(train_patches) / len(all_patches):.2%}")

    for split_name, patches in [("train", train_patches), ("val", val_patches)]:
        for img_final, mask_final, filename in patches:
            save_output(img_final,
                        os.path.join(OUTPUT_DIR, "images", split_name),
                        filename)
            save_output(mask_final,
                        os.path.join(OUTPUT_DIR, "masks", split_name),
                        filename)

    print("\n✔ Dataset preparation complete (with overlaps and mask filtering)!")
    print(f"Output inside: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()