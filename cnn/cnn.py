import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess_depth(images):
    """Normalize and resize a batch of single-channel depth images in meters."""
    images = torch.nan_to_num(images, nan=4.0, posinf=4.0, neginf=0.0)
    return F.interpolate(images.clamp(0.0, 4.0) / 4.0, size=(90, 160), mode="area")

depth_encoder = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
    nn.ReLU(),
    nn.Conv2d(16,32, kernel_size = 3, stride = 2, padding = 1),
    nn.ReLU(),
    nn.Conv2d(32,32, kernel_size = 3, stride = 2, padding = 1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((2,4)),
    nn.Flatten(),
    nn.Linear(32*2*4, 32)
)
