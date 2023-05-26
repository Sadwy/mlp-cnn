# Install
```shell
# 系统: Ubuntu 21.04
# 创建conda虚拟环境
conda create -n mmpretrain python=3.8 -y
conda activate mmpretrain

# 配置环境包
conda install pytorch=2.0.1 torchvision=0.15.2 -c pytorch -y
pip install openmim
python -m mim install mmcv==2.0.0
python -m mim install mmengine==0.7.3
pip install future tensorboard  # 可视化, 使用tensorboard可视化准确率和loss曲线

# 安装mmpretrain==1.0.0rc7
git clone https://github.com/Sadwy/mlp-cnn
cd mlp-cnn
python setup.py develop
```

# Train
```shell
python tools/train.py configs/custom/fashion_custom.py
```
## 修改配置 config
configs/custom/fashion_custom.py
- 第46行, `max_epochs`设置迭代次数
- 第13行, 选择使用MLP或CNN网络结构
- 第48行, `lr` 设置学习率
- 第31行和第36行, 修改batch_size (建议两行数值相同)
- 第47-48行, 默认使用的是SGD优化器. 如果使用Adam优化器, 则注释47-48行, 并取消49-59的注释.

# Visualization
训练时输出的信息中有显示 `Exp name`, 比如 `Exp name: fashion_custom_20230525_131400`. 则使用以下指令可视化:
```shell
tensorboard --host localhost --load_fast=true --logdir work_dirs/fashion_custom/20230525_131400
```
- 指令中的路径是训练时自动创建的. 其中`fashion_custom` 和 `20230525_131400` 分别是配置文件名称和训练时间, 视情况修改.
- 如果报错, 尝试修改指令中参数为 `--load_fast=false` (Details: [issue](https://github.com/tensorflow/tensorboard/issues/4784)). 这可能导致部分曲线显示异常.

# Appendix
另外实现了使用FashionMNIST数据集训练其他网络.
```shell
# lenet网络
python tools/train.py configs/custom/lenet5_fashion_custom.py

# resnet18网络
python tools/train.py configs/custom/resnet18_8xb32_fashion_custom.py 
```

# Citation
```
@misc{2023mmpretrain,
    title={OpenMMLab's Pre-training Toolbox and Benchmark},
    author={MMPreTrain Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpretrain}},
    year={2023}
}
```
