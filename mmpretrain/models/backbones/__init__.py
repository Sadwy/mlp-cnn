# Copyright (c) OpenMMLab. All rights reserved.
from .mlp import MLP
from .mlp_cnn import CNN
from .lenet import LeNet5
from .resnet import ResNet, ResNetV1c, ResNetV1d

__all__ = [
    'MLP',
    'CNN',
    'LeNet5',
    'ResNet',
    'ResNetV1d',
    'ResNetV1c',
]
