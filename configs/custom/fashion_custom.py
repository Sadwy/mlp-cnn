# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
# Modified by Sadwy
# ---------------------------------------------
_base_ = [
    '../lenet/lenet5_mnist.py'
]

# model settings
model = dict(
    backbone=dict(
        type='CNN',  # CNN or MLP
        num_classes=10
    ),
)


# dataset settings
dataset_type = 'FashionMNIST'
data_preprocessor = dict(_delete_=True)
pipeline = [dict(type='PackInputs')]

common_data_cfg = dict(
    type=dataset_type,
    data_prefix='data/fashionmnist',
    pipeline=pipeline,
)

train_dataloader = dict(
    batch_size=128,
    dataset=dict(**common_data_cfg, test_mode=False),
)

val_dataloader = dict(
    batch_size=128,
    dataset=dict(**common_data_cfg, test_mode=True),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator


# schedule settings
train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer = dict(
#         type='Adam',
#         lr=0.001,
#         betas=(0.9, 0.999),
#         eps=1e-08,
#         weight_decay=0,
#         amsgrad=False),
# )
param_scheduler = dict(
    type='MultiStepLR',  # learning policy, decay on several milestones.
    by_epoch=True,  # update based on epoch.
    milestones=[40],  # decay at the 15th epochs.
    gamma=0.1,  # decay to 0.1 times.
)



# runtime settings
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    # 可视化验证结果，将 `enable` 设为 True 来启用这一功能。
    visualization=dict(type='VisualizationHook', enable=False),
)
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

# 随机种子 make the experiment as reproducible as possible.
randomness = dict(seed=None, deterministic=False)