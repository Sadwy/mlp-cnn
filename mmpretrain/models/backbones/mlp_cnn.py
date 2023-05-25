# -----------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# -----------------------------------------------------------
# Modified from mmpretrain/models/backbones/lenet.py by Sadwy
# -----------------------------------------------------------
import torch.nn as nn

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class CNN(BaseBackbone):
    """CNN
    The input for CNN is a 28×28 grayscale image.
    [bn, 1, 28, 28] for FashionMNIST

    Args:
        num_classes (int): number of classes for classification.
    """

    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1),  # 26x26
            # nn.Tanh(),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, stride=1),  # 24x24
            # nn.Tanh(),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=3),  # 8x8
            # nn.Tanh(),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),  # 4x4
            # nn.Tanh(),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),  # 1x1
            # nn.Tanh(),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64, num_classes)

        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 10, kernel_size=5, stride=1),  # 24x24
        #     nn.AdaptiveAvgPool2d((12, 12)),
        #     nn.Conv2d(10, 20, kernel_size=5, stride=1),  # 8x8
        #     nn.AdaptiveAvgPool2d((4, 4)),
        # )
        # self.linear = nn.Sequential(
        #     nn.Linear(320, 50),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(50, num_classes),
        # )
        # self.linear = nn.Linear(320, num_classes)

    def forward(self, x):
        # breakpoint()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # x = self.linear(x.squeeze())

        return (x, )