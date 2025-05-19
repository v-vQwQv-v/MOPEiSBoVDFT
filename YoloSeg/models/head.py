import copy
import math

import torch
from torch import nn

from models.block import Proto
from models.conv import Conv
import loss_1

class Segment(nn.Module):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__()
        self.nm = nm  # number of masks 
        self.npr = npr  # number of protos
        self.proto = Proto(ch, self.npr, self.nm)  # protos

        # c4 = max(ch[0] // 4, self.nm)
        c4 = max(min([ch]) // 4, nm)
        # self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        self.cv4 = nn.Sequential(Conv(ch, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1))

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x)  # mask protos
        mc = self.cv4(x)

        # x = Detect.forward(self, x)
        return mc, p
