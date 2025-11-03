# Image Segmentation (U-Net) in PyTorch

Binary image segmentation pipeline using a U-Net model trained with PyTorch. Includes data prep utilities, training with augmentations, offline inference on datasets or image folders, and a FastAPI server for online predictions.

## Features
- U-Net architecture with skip connections (`classes/model.py`)
- Albumentations-based augmentations and normalization
- Combined loss: Cross-Entropy + Dice, IoU metric tracking (`train.py`)
- Inference on a validation split or arbitrary image folders
- FastAPI endpoint that accepts base64-encoded images and returns a mask

## Repository Layout
- `classes/model.py` – U-Net model
- `classes/dataset.py` – dataset that reads RGB images and red-coded masks
- `train.py` – training loop, validation, IoU and loss plots, checkpointing
- `run_on_dataset.py` – visualize predictions on a validation split
- `run_on_images.py` – visualize predictions for images in a folder
- `server.py` – FastAPI app exposing `/predict`
- `resize.py` – build `images/train|val` and `masks/train|val` splits with resizing
- `recolor.py` – convert grayscale masks to red RGB masks (thresholded)
- `test.jpg` – sample image

## Requirements
- Python 3.10+
- Recommended: NVIDIA GPU + CUDA (optional, CPU also works)

Install packages (create and activate a virtualenv/conda env first):

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA/CPU
pip install albumentations opencv-python pillow numpy matplotlib scikit-learn tqdm fastapi uvicorn pydantic
```

If you prefer a single command:

```bash
pip install torch albumentations opencv-python pillow numpy matplotlib scikit-learn tqdm fastapi uvicorn pydantic
```

Note: Install the correct Torch build for your CUDA/CPU setup; see pytorch.org.

## Data Preparation
This project expects paired images and masks. Masks should encode the positive class in red (R>100, G<100, B<100). The dataset class binarizes masks based on this red threshold (`classes/dataset.py:1`).

1) Put your raw data here:

```
dataset/
  images/
    img_001.jpg
    img_002.jpg
  masks/
    img_001.png  # same basename as image
    img_002.png
```

2) If your masks are grayscale (0/255), convert to red RGB masks. Adjust paths at the top of `recolor.py:1` and run:

```bash
python recolor.py
```

3) Build a resized train/val split. Adjust constants in `resize.py:1` (inputs, `OUTPUT_DIR`, `SIZE`, `VAL_RATIO`, `MAX_IMAGES`) and run:

```bash
python resize.py
```

This produces a structure like:

```
dataset-512/
  images/
    train/...
    val/...
  masks/
    train/...
    val/...
```

## Training
Edit the constants at the top of `train.py:1` as needed:
- `EPOCHS`, `BATCH_SIZE`
- `DATASET_DIR` (e.g., `dataset-512`)

Then run training:

```bash
python train.py
```

- Uses CombinedLoss (Cross-Entropy + Dice) and tracks IoU on validation.
- Saves the best checkpoint to `models/unet_best.pth`.
- Displays IoU and loss curves, and a qualitative example at the end.

Tip: Ensure your `IMAGE_SIZE` used in augmentation/inference matches the dataset produced by `resize.py`.

## Inference
Two options are provided.

- Validation split visualizer (`run_on_dataset.py:1`):
  - Set `DATASET_DIR`, `IMAGE_SIZE`, and `best_model_path`
  - Run: `python run_on_dataset.py`

- Folder of images (`run_on_images.py:1`):
  - Set `IMAGE_DIR`, `IMAGE_SIZE`, and `best_model_path`
  - Run: `python run_on_images.py`
  - The script visualizes image, predicted mask, and probability map.

Note: Keep `IMAGE_SIZE` and the model checkpoint consistent with how you trained. If you trained on 512×512, use 512×512 at inference unless your model and transforms explicitly support arbitrary sizes.

## REST API Server
Serve the model via FastAPI (`server.py:1`). Update at the top:
- `IMAGE_SIZE` (e.g., `(768, 768)`)
- `MODEL_PATH` (e.g., `models/unet_best.pth` or your own weights)

Start the server:

```bash
python server.py
# or
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Endpoint: `POST /predict`
- Body (JSON): `{ "image": "<base64 or data URL>" }`
- Returns: `{ "prediction": "<base64-encoded PNG mask>" }`

Minimal Python client example:

```python
import base64, requests
from PIL import Image
from io import BytesIO

# encode image as base64 (PNG)
img = Image.open("test.jpg").convert("RGB")
buf = BytesIO(); img.save(buf, format="PNG")
img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

resp = requests.post("http://localhost:8000/predict", json={"image": img_b64})
mask_b64 = resp.json()["prediction"]
open("prediction.png", "wb").write(base64.b64decode(mask_b64))
```

CORS is enabled for all origins for convenience; restrict in production.

## Notes & Troubleshooting
- Mask encoding: The loader expects red pixels for the positive class. Adjust the thresholding in `classes/dataset.py:1` if your masks use a different encoding.
- Model/image size: Keep `IMAGE_SIZE` consistent across training, inference, and the server for best results.
- Checkpoints: Training saves to `models/unet_best.pth`. The inference scripts and server reference different default filenames (`latest-768.pth` vs `unet_best.pth`). Point them to the file you actually have.
- GPU vs CPU: The scripts automatically use CUDA if available. CPU works but is slower.
- Dependencies: If OpenCV or Torch installs fail, verify your Python version and platform wheels.

---
