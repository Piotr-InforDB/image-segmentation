import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

def convert_to_groupnorm(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(module, name,
                    nn.GroupNorm(num_groups=min(num_groups, num_channels),
                                 num_channels=num_channels))
        else:
            convert_to_groupnorm(child, num_groups=num_groups)

class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes=2, groupnorm=True, num_groups=32):
        super().__init__()

        # use default pretrained weights
        self.model = deeplabv3_resnet50(weights="DEFAULT")

        # change final classifier head
        self.model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=1)

        # Replace BN → GN
        if groupnorm:
            convert_to_groupnorm(self.model, num_groups=num_groups)

    def forward(self, x):
        return self.model(x)["out"]
