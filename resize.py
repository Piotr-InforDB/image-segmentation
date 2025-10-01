import os
from PIL import Image

INPUT_DIR = "dataset"
OUTPUT_DIR = "dataset-768"
SIZE = (768, 768)

def resize_and_save(input_path, output_path, size):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with Image.open(input_path) as img:
        if "masks" in input_path:
            img = img.resize(size, Image.NEAREST)
        else:
            img = img.resize(size, Image.BILINEAR)
        img.save(output_path)

def process_folder(subdir):
    input_subdir = os.path.join(INPUT_DIR, subdir)
    output_subdir = os.path.join(OUTPUT_DIR, subdir)

    for root, _, files in os.walk(input_subdir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".png")):
                input_path = os.path.join(root, fname)
                relative = os.path.relpath(input_path, INPUT_DIR)
                output_path = os.path.join(OUTPUT_DIR, relative)
                resize_and_save(input_path, output_path, SIZE)

if __name__ == "__main__":
    process_folder("images/train")
    process_folder("images/val")
    process_folder("masks/train")
    process_folder("masks/val")

    print("✅ Resizing complete! Saved in:", OUTPUT_DIR)
