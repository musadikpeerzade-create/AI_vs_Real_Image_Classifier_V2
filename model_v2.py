"""Model definition for the V2 CNN used in the image classification project.

This file exposes `BetterCNN`, a small convolutional network suitable for
classification on upsampled small images.
"""

import torch
import torch.nn as nn


class BetterCNN(nn.Module):
    """A simple, reliable convolutional classifier.

    The architecture stacks several conv blocks with BatchNorm and ReLU,
    followed by global average pooling and a small fully-connected head.
    """

    def __init__(self, num_classes=2):
        super(BetterCNN, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

        self.block1 = conv_block(3, 32)    # 128 -> 64
        self.block2 = conv_block(32, 64)   # 64 -> 32
        self.block3 = conv_block(64, 128)  # 32 -> 16
        self.block4 = conv_block(128, 256)  # 16 -> 8

        # Global Average Pooling to 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
