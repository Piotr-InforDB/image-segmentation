import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    def __init__(self, n_classes=2, base_c=64):
        super().__init__()

        nb = base_c

        # Encoder
        self.c00 = ConvBlock(3, nb)
        self.p0 = nn.MaxPool2d(2)

        self.c10 = ConvBlock(nb, nb*2)
        self.p1 = nn.MaxPool2d(2)

        self.c20 = ConvBlock(nb*2, nb*4)
        self.p2 = nn.MaxPool2d(2)

        self.c30 = ConvBlock(nb*4, nb*8)
        self.p3 = nn.MaxPool2d(2)

        self.c40 = ConvBlock(nb*8, nb*16)

        # Decoder (nested)
        self.c01 = ConvBlock(nb + nb*2, nb)
        self.c02 = ConvBlock(nb*2 + nb, nb)
        self.c03 = ConvBlock(nb*2 + nb, nb)
        self.c04 = ConvBlock(nb*2 + nb, nb)

        self.c11 = ConvBlock(nb*2 + nb*4, nb*2)
        self.c12 = ConvBlock(nb*4 + nb*2, nb*2)
        self.c13 = ConvBlock(nb*4 + nb*2, nb*2)

        self.c21 = ConvBlock(nb*4 + nb*8, nb*4)
        self.c22 = ConvBlock(nb*8 + nb*4, nb*4)

        self.c31 = ConvBlock(nb*8 + nb*16, nb*8)

        self.final = nn.Conv2d(nb, n_classes, kernel_size=1)

    def forward(self, x):

        x00 = self.c00(x)
        x10 = self.c10(self.p0(x00))
        x20 = self.c20(self.p1(x10))
        x30 = self.c30(self.p2(x20))
        x40 = self.c40(self.p3(x30))

        # Up
        x01 = self.c01(torch.cat([x00, F.interpolate(x10, scale_factor=2, mode="bilinear", align_corners=True)], dim=1))
        x11 = self.c11(torch.cat([x10, F.interpolate(x20, scale_factor=2, mode="bilinear", align_corners=True)], dim=1))
        x21 = self.c21(torch.cat([x20, F.interpolate(x30, scale_factor=2, mode="bilinear", align_corners=True)], dim=1))
        x31 = self.c31(torch.cat([x30, F.interpolate(x40, scale_factor=2, mode="bilinear", align_corners=True)], dim=1))

        x02 = self.c02(torch.cat([x01, F.interpolate(x11, scale_factor=2, mode="bilinear", align_corners=True)], dim=1))
        x12 = self.c12(torch.cat([x11, F.interpolate(x21, scale_factor=2, mode="bilinear", align_corners=True)], dim=1))
        x22 = self.c22(torch.cat([x21, F.interpolate(x31, scale_factor=2, mode="bilinear", align_corners=True)], dim=1))

        x03 = self.c03(torch.cat([x02, F.interpolate(x12, scale_factor=2, mode="bilinear", align_corners=True)], dim=1))
        x13 = self.c13(torch.cat([x12, F.interpolate(x22, scale_factor=2, mode="bilinear", align_corners=True)], dim=1))

        x04 = self.c04(torch.cat([x03, F.interpolate(x13, scale_factor=2, mode="bilinear", align_corners=True)], dim=1))

        return self.final(x04)
