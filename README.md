## 一、实验任务

本实验旨在设计并实现一个基于卷积神经网络（CNN）的图像分类器，以理解CNN的基本结构、代码实现和训练过程。同时，通过应用dropout和多种normalization方法，分析它们对模型泛化能力的影响，并利用交叉验证找到神经网络的最佳超参数。

## 二、数据集

本实验使用MNIST手写数字数据集进行训练和测试。MNIST数据集包含60,000个训练图像和10,000个测试图像，每个图像为28x28像素的灰度手写数字。

## 三、模型结构

模型采用了以下结构：

- **Convolutional Layer 1**: 第一个卷积层使用32个过滤器，每个大小为3x3，步长为1。
- **Convolutional Layer 2**: 第二个卷积层使用64个过滤器，每个大小为3x3，步长为1。
- **Dropout Layer 1**: 在第一个全连接层之前使用dropout层，dropout率为0.25。
- **Fully Connected Layer 1**: 第一个全连接层将特征图转换为128维的向量。
- **Dropout Layer 2**: 在第二个全连接层之前使用dropout层，dropout率为0.5。
- **Fully Connected Layer 2**: 第二个全连接层输出10个类别的概率分布。

![ckpt29.png](resources%2Fckpt29.png)

## 四、超参数搜索
![result.png](resources%2Fresult.png)
使用Tune工具完成超参数搜索，
使用Hyperopt算法，最大迭代次数设置为10，每次将训练10个epoch作为测试
找到最佳的超参数组合为（以验证集准确率为标准）：
 ```
{'lr': 0.0014489483362154264, 'batch_size': 32, 'momentum': 0.9035754520022181, 'weight_decay': 0.0001114030365842323}
```
在验证集上的 最佳准确率为99.13%
搜索过程可视化如下：（横轴代表迭代次数）
- 准确率：
![tune-acc.png](resources%2Ftune-acc.png)
- 损失：
![tune-loss.png](resources%2Ftune-loss.png)

## 五、训练过程
在搜索到的最优参数下，训练过程如图：
![Pasted image 20231108195420.png](resources%2FPasted%20image%2020231108195420.png)
可以看出在第20个epoch处，模型达到最佳的泛化性能。

## 六、验证Normalization的作用

控制其余变量均相同，搭建三个不同的网络，一个不使用Normalization，一个使用BatchNorm，一个使用LayerNorm，分别进行10个epoch训练，得到的在验证集上的准确率如下图所示。
![normalization.png](resources%2Fnormalization.png)
- 可见使用Normalization可以使模型更快地达到更佳的泛化性能。
实验并没有看出两种规范化方法在效果上有明显的区别。

## 七、模型集成

为了进一步提高模型的准确率和鲁棒性，实验中尝试了**模型集成**的方法。dropout在训练过程中不仅可以防止过拟合，还可以被视作模型集成的一种形式。
此外，通过训练多个网络并在预测时取它们输出概率的平均或者投票，可以显著提升模型性能。

## 八、总结

本实验通过实现和训练卷积神经网络，深入理解了CNN的结构和功能。超参数搜索证明了选择合适的学习率、批量大小、动量和权重衰减对提高模型性能至关重要。dropout和模型集成等策略对于增强模型的泛化能力和准确率也起到了积极作用。