import torch.nn as nn

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
