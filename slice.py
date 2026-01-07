import os
from PIL import Image


INPUT_DIR = "dataset-2048"
OUTPUT_DIR = "dataset-2048-sliced"
SLICES = 2


def slice_image(img_path, out_path, slices):
    """Slice a single image into N x N patches."""
    img = Image.open(img_path)
    w, h = img.size

    slice_w = w // slices
    slice_h = h // slices

    basename = os.path.splitext(os.path.basename(img_path))[0]

    for row in range(slices):
        for col in range(slices):
            left = col * slice_w
            upper = row * slice_h
            right = left + slice_w
            lower = upper + slice_h

            patch = img.crop((left, upper, right, lower))
            out_name = f"{basename}_r{row}_c{col}.png"
            patch.save(os.path.join(out_path, out_name))


def process_folder(input_dir, output_dir, slices):
    """Recursively process all images in input_dir and preserve folder structure."""
    valid_ext = {".png", ".jpg", ".jpeg"}

    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        out_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(out_subdir, exist_ok=True)

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_ext:
                continue

            in_file = os.path.join(root, file)
            print("Slicing:", in_file)
            slice_image(in_file, out_subdir, slices)


if __name__ == "__main__":
    print("Starting slicing...")
    process_folder(INPUT_DIR, OUTPUT_DIR, SLICES)
    print("Done.")
