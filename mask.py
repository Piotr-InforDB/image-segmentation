import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
MODEL_PATH = "building_segmentation.pth"
IMAGE_PATH = "image.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

colormap = {
    0: (0, 0, 0),
    1: (255, 0, 0),
}

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")
orig_size = image.size  # (W,H) for resizing back later
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(input_tensor)['out']
    pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

mask_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
for class_id, color in colormap.items():
    mask_color[pred == class_id] = color

mask_pil = Image.fromarray(mask_color).resize(orig_size, Image.NEAREST)

overlay = Image.blend(image, mask_pil, alpha=0.5)

mask_pil.save("segmentation_mask.png")
overlay.save("segmentation_overlay.png")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)

plt.subplot(1,2,2)
plt.title("Segmentation Overlay")
plt.imshow(overlay)
plt.show()
