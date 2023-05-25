# 这是使用GPT-3.5生成的配置文件, 无法使用
# 因为 OpenMMLab 2.0 之后, 许多配置名称发生了变化
# 比如 `log_config` 被整合到 `default_hook.logger` 中
# 而GPT生成的配置文件中, 仍然使用了 `log_config` 这个配置项 (即按照的是2.0之前的标准)
# 简言之, GPT生成的配置文件版本太低

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='LeNet',
        num_classes=10),
    neck=None,
    head=dict(
        type='ClsHead',
        in_channels=6,
        num_classes=10,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
    ))

# dataset settings
dataset_type = 'FashionMNIST'
img_norm_cfg = dict(
    mean=[0.5],
    std=[0.5],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/fashionmnist/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/fashionmnist/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/fashionmnist/test',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 6])
runner = dict(type='EpochBasedRunner', max_epochs=9)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

device_ids = range(2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/fashion_mnist'
load_from = None
resume_from = None
workflow = [('train', 1)]
