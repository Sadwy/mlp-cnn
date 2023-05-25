# -----------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# -----------------------------------------------------------
# Modified from mmpretrain/models/backbones/lenet.py by Sadwy
# -----------------------------------------------------------
import torch.nn as nn

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class MLP(BaseBackbone):
    """MLP
    The input for MLP is a 28×28 grayscale image.
    [bn, 1, 28, 28] for FashionMNIST

    Args:
        num_classes (int): number of classes for classification.
    """

    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        # self.linear = nn.Sequential(
        #     nn.Linear(28*28, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, self.num_classes),
        #     # nn.ReLU(inplace=True),
        # )
        # self.linear = nn.Sequential(
        #     nn.Linear(28*28, 14*14),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(14*14, 7*7),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(7*7, num_classes),
        # )
        # self.linear = nn.Sequential(
        #     nn.Linear(28*28, 14*7),
        #     nn.ReLU(inplace=True),
        #     # nn.Tanh(),
        #     nn.Linear(14*7, num_classes),
        # )
        # self.linear = nn.Sequential(
        #    nn.Linear(28*28, 100),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(100, num_classes),
        # )
        self.linear = nn.Linear(28*28, self.num_classes)
        # self.l1 = nn.Linear(28*28, 100)
        # self.a1 = nn.ReLU(inplace=True)
        # self.l2 = nn.Linear(100, self.num_classes)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # x = self.l1(x)
        # x = self.a1(x)
        # x = self.l2(x)

        return (x, )
