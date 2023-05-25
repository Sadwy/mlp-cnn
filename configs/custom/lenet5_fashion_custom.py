# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
# Modified by Sadwy
# ---------------------------------------------
_base_ = [
    '../lenet/lenet5_mnist.py'
]

dataset_type = 'FashionMNIST'
data_preprocessor = dict(_delete_=True)

common_data_cfg = dict(
    type=dataset_type,
    data_prefix='data/fashionmnist')

train_dataloader = dict(
    dataset=dict(**common_data_cfg, test_mode=False),
)

val_dataloader = dict(
    dataset=dict(**common_data_cfg, test_mode=True),
)
test_dataloader = val_dataloader



# schedule settings
train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)