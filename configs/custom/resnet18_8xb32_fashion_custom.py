# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
# Modified by Sadwy
# ---------------------------------------------
_base_ = [
    '../_base_/models/resnet18.py',
    # '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        in_channels=1,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# schedule settings
auto_scale_lr = dict(base_batch_size=32)
train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)  # train 5 epochs


# dataset settings
dataset_type = 'FashionMNIST'
# data_preprocessor = dict(
#     num_classes=10,
#     # RGB format normalization parameters
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
# )

train_pipeline = [dict(type='Resize', scale=7), dict(type='PackInputs')]

test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='',
        data_root='data/fashionmnist',
        pipeline=train_pipeline,
        test_mode=False,
        download=True),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='',
        data_root='data/fashionmnist',
        pipeline=test_pipeline,
        test_mode=True,
        download=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
