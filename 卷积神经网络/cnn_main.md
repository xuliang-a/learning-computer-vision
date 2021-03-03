# 目录

* [Q1-卷积神经网络是什么和多层感知机有什么区别？](#Q1-卷积神经网络是什么和多层感知机有什么区别)

* [Q2-卷积层输出尺寸和感受野怎么计算？](#Q2-卷积层输出尺寸和感受野怎么计算)

* [Q3-卷积层参数数量和计算量怎么计算？](#Q3-卷积层参数数量和计算量怎么计算)

* [Q4-什么是池化层，有哪些池化类型？](#Q4-什么是池化层-有哪些池化类型)

* [Q5-卷积层和池化层有什么区别？](#Q5-卷积层和池化层有什么区别)

* [Q6-怎么组成一个用于图像分类的基本完整的卷积神经网络？](#Q6-怎么组成一个用于图像分类的基本完整的卷积神经网络)

* [Q7-一个1乘1的卷积层有什么作用？](#Q7-一个1乘1的卷积层有什么作用)

* [Q8-用于图像分类的卷积神经网络发展过程是怎样的？](#Q8-用于图像分类的卷积神经网络发展过程是怎样的)

  - [LeNet(1998)](#LeNet-1998)

  - [AlexNet(2012)](#AlexNet-2012)

  - [ZFNet(2013)](#ZFNet-2013)

  - [NiN(2013)](#NiN-2013)

  - [VGG(2014)](#VGG-2014)

  - [GoogLeNet(2014)](#GoogLeNet-2014)

  - [SPPNet(2015)](#SPPNet-2015)

  - [ResNet(2015)](#ResNet-2015)

  - [DenseNet(2017)](#DenseNet-2017)

* [Q9-用于分类的卷积神经网络最后几层一般是什么层？](#Q9-用于分类的卷积神经网络最后几层一般是什么层)

* [Q10-有哪些变种卷积？](#Q10-有哪些变种卷积)

* [中英词汇对照](#中英词汇对照)

# 卷积神经网络

## Q1-卷积神经网络是什么和多层感知机有什么区别
在卷积神经网络（Convolutional Neural Network，CNN）出现之前，多层感知机（Multi-Layer Perceptron，MLP）是比较常见神经网络。多层感知机相邻层节点通常是全连接的，也就是输入层（Input Layer）的每个节点会与输出层（Output Layer）每个节点相连接。与多层感知机不同，卷积神经网络的主要组成部分是卷积层（Convolution Layer），每个卷积层通过特定数目的卷积核（Convolution Kernel）与输入图像进行扫描计算，由于卷积核的尺寸一般是要小于输入图像的尺寸，所以卷积层输出的每个节点只与输入层的部分节点相连接，称为局部连接，该特点与多层感知机的全连接不同。另外卷积神经网络还有另一个特征，权值共享，具体来说卷积层输出的每个节点，与输入层所连接的权值是一样的，都是卷积核的内部的参数。

参考《百面深度学习》P4-7

一个多输入通道（Channel）的卷积计算例子

![一个多输入通道的卷积计算例子](https://zh.d2l.ai/_images/conv_multi_in.svg)

摘自《动手学深度学习》

## Q2-卷积层输出尺寸和感受野怎么计算

卷积层输出的尺寸由卷积核尺寸（$ k\times k $）、卷积核滑动步长（Stride）和对原图边缘所填充（Padding）的尺寸所决定。

- 若没有步长s和填充p，有 $ \mbox{输出边长} = \mbox{输入边长}  - k + 1 $

- 若宽或高的两侧一共填充p(注意是一共填充还是分别填充)，没有步长s，有 $ \mbox{输出边长}  = \mbox{输入边长} - k + p + 1 $

- 若宽或高的两侧分别填充p，步长s，有 $ \mbox{输出边长} = (\mbox{输入边长}- k + 2p + s) / s = (\mbox{输入边长} - k + 2p)/s + 1 $，若是小数则取下限

一个计算卷积尺寸的例子，若输入图像为$3\times 3$, 核为$2\times 2$,宽或高的两侧分别填充1，在高和宽上步幅分别为3和2，有

$ [(3 - 2 + 2)/3 + 1] \times [{(3 - 2 + 2)/2 + 1}] = 2\times 2 $

![一个计算卷积尺寸的例子](https://zh.d2l.ai/_images/conv_stride.svg)

参考《动手学深度学习》

感受野，也就是卷积结果对应输入的区域尺寸，其实就是反过来求解卷积层的输入尺寸

$ \mbox{卷积层的输入尺寸（感受野）} = (\mbox{卷积层输出尺寸}-1)\times s + k - 2p  $

根据这个公式可以从后向前计算感受野，向前一层一层计算就可以计算到在原始图片上对应的感受野了。

## Q3-卷积层参数数量和计算量怎么计算

卷积层的参数量，取决于卷积核的个数和该卷积核的参数量，若卷积核大小为$k_w\times k_h$, 输入特征图通道数为$c^i$ ,输出特征图通道数（卷积核个数）为$c^o$，则
参数量 = $c^o \times c^i \times k_w\times k_h$

卷积层的计算量，取决于卷积核在每个滑到的窗口的计算量和滑动次数，在每个滑动窗内计算量约为$c^i \times k_w\times k_h$，卷积核滑动次数就是输出特征图的数据个数，即$c^o \times o_w\times o_h$, $o_w$和$o_h$分别是输出的宽度和长度，总计算量 = $c^o \times o_w\times o_h \times c^i \times k_w\times k_h$

参考《百面深度学习》P10

## Q4-什么是池化层 有哪些池化类型

池化（Pooling）层又称为降采样层(Downsampling Layer)，它可以缓解卷积层对位置的过度敏感性、降低网络参数和防止过拟合。

池化操作可以降低图像维度的原因，本质上是因为图像具有一种“静态性”的属性，这个意思是说在一个图像区域有用的特征极有可能在另一个区域同样有用。

池化类型如下：

|                  池化类型                   |                      示意图                       | 作用                                                         |
| :-----------------------------------------: | :-----------------------------------------------: | :----------------------------------------------------------- |
|          一般池化(General Pooling)          |   ![max_pooling](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch05_%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(CNN)/img/ch5/general_pooling.png)   | 通常包括最大池化(Max Pooling)和平均池化(Mean Pooling)。以最大池化为例，池化范围$(2\times2)$和滑窗步长$(stride=2)$ 相同，仅提取一次相同区域的范化特征。 |
|        重叠池化(Overlapping Pooling)        | ![overlap_pooling](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch05_%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(CNN)/img/ch5/overlap_pooling.png) | 与一般池化操作相同，但是池化范围$P_{size}$与滑窗步长$stride$关系为$P_{size}>stride$，同一区域内的像素特征可以参与多次滑窗提取，得到的特征表达能力更强，但计算量更大。 |
| 空间金字塔池化$^*$(Spatial Pyramid Pooling) | ![spatial_pooling](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch05_%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(CNN)/img/ch5/spatial_pooling.png) | 在进行多尺度目标的训练时，卷积层允许输入的图像特征尺度是可变的，紧接的池化层若采用一般的池化方法会使得不同的输入特征输出相应变化尺度的特征，而卷积神经网络中最后的全连接层则无法对可变尺度进行运算，因此需要对不同尺度的输出特征采样到相同输出尺度。 |

参考《深度学习500问》、《动手学深度学习》

## Q5-卷积层和池化层有什么区别

卷积层和池化层在结构上具有一定的相似性，都对感受域内的特征进行提取，并根据填充步长获取到不同维度的输出。

需要注意的是在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加。这意味着池化层的输出通道数与输入通道数相等。

其它区别：

|            |                 卷积层                 |              池化层              |
| :--------: | :------------------------------------: | :------------------------------: |
| **稳定性** | 输入特征发生细微改变时，输出结果会改变 | 感受域内的细微变化不影响输出结果 |
|  **作用**  |        感受域内提取局部关联特征        |  感受域内提取泛化特征，降低维度  |
| **参数量** |      与卷积核尺寸、卷积核个数相关      |          不引入额外参数          |


摘自《深度学习500问》


## Q6-怎么组成一个用于图像分类的基本完整的卷积神经网络

以图像分类任务为例，在表示卷积神经网络中，一般包含5种类型的网络层次结构：输入层、卷积层、激活层（Activation Layer）、池化层和全连接层（Full Connected Layer）。

| CNN层次结构 |             输出尺寸              | 作用                                                         |
| :---------: | :-------------------------------: | :----------------------------------------------------------- |
|   输入层    |      $W_1\times H_1\times 3$      | 卷积网络的原始输入，可以是原始或预处理后的像素矩阵           |
|   卷积层    |      $W_1\times H_1\times K$      | 参数共享、局部连接，利用平移不变性从全局特征图提取局部特征   |
|   激活层    |      $W_1\times H_1\times K$      | 将卷积层的输出结果进行非线性映射                             |
|   池化层    |      $W_2\times H_2\times K$      | 进一步筛选特征，可以有效减少后续网络层次所需的参数量         |
|  全连接层   | $(W_2 \cdot H_2 \cdot K)\times C$ | 将多维特征展平为2维特征，通常低维度特征对应任务的学习目标（类别或回归值） |

> $W_1\times H_1\times 3$对应原始图像或经过预处理的像素值矩阵，3对应RGB图像的通道;$K$表示卷积层中卷积核（滤波器）的个数;$W_2\times H_2$ 为池化后特征图的尺度，在全局池化中尺度对应$1\times 1$;$(W_2 \cdot H_2 \cdot K)$是将多维特征压缩到1维之后的大小，$C$对应的则是图像类别个数。

摘自《深度学习500问》

## Q7-一个1乘1的卷积层有什么作用

1乘1卷积层首先在NiN网络中提出，后来的GoogLeNet也借鉴了该卷积。

$1\times 1 $卷积的作用主要为以下两点：
- 实现信息的跨通道交互和整合，具体来说就是实现了不同通道同一位置的信息融合。
- 对卷积核通道数进行降维和升维，减小参数量，控制模型复杂度。

摘自《深度学习500问》

使用输入通道数为3、输出通道数为2的 1×1 卷积核的计算图例

![一个1乘1卷积的计算例子](https://zh.d2l.ai/_images/conv_1x1.svg)

假设将通道维当作特征维，将高和宽维度上的元素当成数据样本，那么 1×1 卷积层的作用与全连接层等价。

为了便于理解1乘1卷积和全连接的关系，画了一个图

![1乘1卷积和全连接的关系](ch01_img/1乘1卷积和全连接的关系.png)

摘自《动手学深度学习》

## Q8-用于图像分类的卷积神经网络发展过程是怎样的

### LeNet 1998

LeNet-5被广泛用于银行手写体数字识别，是现代卷积神经网络的原型, 这个名字来源于LeNet论文的第一作者Yann LeCun, 5是研究成果的代号。

![LeNet-5结构](https://img-blog.csdn.net/20180606092950828)

#### 1. 特征图尺寸计算

LeNet模型若不考虑输入层，一共包含7个层。即卷积、池化、卷积、池化、卷积和2个全连接层，再简单点说就是三个卷积层和两个全连接层。

- C1是卷积层，有6个尺寸为5×5的卷积核，输入图像的尺寸是32×32，经过卷积计算后得到了6幅28×28的特征图；

- S2是下采样层，S2对6幅特征图进行平均池化操作，池化窗口大小为2×2，步长为2，S2层的输出为6幅14×14的特征图；

- C3是卷积层，由于该卷积层与S2层是局部连接的关系，所以该层分别有6个尺寸为5×5×3的卷积核、9个尺寸为5×5×4的卷积核和1个5×5×6的卷积核，经过卷积计算后得到16幅10×10的特征图，局部连接关系见表；

![S2和C3的连接关系](https://img-blog.csdn.net/20180606094255999)

对应的行就是6幅C3输入的特征图, 对应的列就是C3输出结果对应的特征图, 具体来说就是生成的16个通道的特征图分别按照相邻的3个特征图,相邻的4个特征图,非相邻的4个特征图和全部6个特征图进行映射。

- S4是下采样层，S4对这16幅特征图进行平均池化操作，池化窗口大小为2×2，步长为2，S4层的输出为16幅5×5的特征图；

- C5是卷积层，有120个尺寸为16×5×5的卷积核，得到了120幅1×1的特征图，其实就是得到了一个120维的向量，这层相当于一个输出个数为120的全连接层；

- 接下来的F6是全连接层，有84个输出；

- 最后一层是输出层，有10个输出，也就是要识别的数字0到9的类别个数。

#### 2. 参数量的计算

参数量包括权重和偏置。

卷积层参数计算以C1层为例, 有$ (5\times 5 + 1)\times 6$，1是偏置

![卷积参数量计算](https://img-blog.csdn.net/20180606093453532)

该模型C3卷积层的参数量，采用了稀疏连接进行限制

$ (5\times 5\times 3 + 1)\times 6 + (5\times 5\times 4 + 1)\times 6 + (5\times 5\times 4 + 1)\times 3 + (5\times 5\times 6 + 1)\times 1 = 1516 $

![C3卷积层示意](https://img-blog.csdn.net/20180606094210133)

池化层参数计算以S2层为例，有$ (1 + 1)\times 6$，两个1一个是权重，一个是偏置，在没有权重和偏置的情况下可认为参数量是0

![池化参数量计算](https://img-blog.csdn.net/20180606094138206)

参考博主https://blog.csdn.net/saw009/article/details/80590245


LeNet的卷积用法:

- $5\times 5 $ 的卷积核，没有填充和步长，正常计算
- $5\times 5 $ 的卷积核，没有填充和步长，输入特征图尺寸也是 $5\times 5 $的,正常计算后变为 $1\times 1 $, 选取的核尺寸若和输入尺寸已知,即可将卷积后尺寸压缩到 $1\times 1 $

LeNet的池化用法:

- 输入的特征图的尺寸是偶数，$2\times 2$，步长为2，没有填充，使输出特征图的尺寸变为输入的一半

LeNet论文:

      LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

---

### AlexNet 2012

AlexNet取得了2012年的ImageNet大规模视觉识别挑战赛（ImageNet Large Scale Visual Recognition Challenge，ILSVRC）的竞赛冠军，由Hinton和他的学生Alex Krizhevsky设计的。也是在那年之后，更多的更深的神经网路被提出。

#### 1. 总体结构

1. 包含八个学习层：5个卷积层和3个全连接层

![](https://img-blog.csdn.net/20180829094734541?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZW55dXBpbmczMzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

2. 网络太大因此将网络分布在两个GPU上，GPU间可以直接互相读写内存，而不需要通过主机内存。

3. Alex采用的并行方案基本上每个GPU放置一半的核（或神经元），还有一个额外的技巧：只在某些特定的层上进行GPU通信。这意味着，例如，第3层的核会将第2层的所有核映射作为输入。然而，第4层的核只将位于相同GPU上的第3层的核映射作为输入。

![](https://img-blog.csdn.net/20180829094603154?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZW55dXBpbmczMzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 2. 详解AlexNet网络结构

##### 1 先不看成多个GPU

1. 输入图像$227\times 227\times 3$

2. 卷积层1（96个 $11\times 11\times 3$的卷积核，步长是4）

   - 计算输出卷积形状，

     $\lfloor(227-11+0+4)/4 \rfloor * \lfloor(227-11+0+4)/4 \rfloor = 55 \times 55$

     得出输出特征图形状为$55\times 55 \times 96$

   - 计算输出池化形状（$3\times 3, s = 2$）

     $\lfloor(55-3+0+2)/2 \rfloor * \lfloor(55-3+0+2)/2 \rfloor = 27 \times 27$

     得出输出特征图形状为 $27\times 27\times 96$

3. 卷积层2（256个 $5 \times 5 \times 96$的卷积核，分别填充2（上下左右分别填充），步长1）

   - 计算输出卷积形状，

     $\lfloor(27-5+4+1)/1 \rfloor * \lfloor(27-5+4+1)/1 \rfloor = 27 \times 27$

     得出输出特征图形状为 $27\times 27 \times 256$

   - 计算输出池化形状（$3\times 3, s = 2$）

     $\lfloor(27-3+0+2)/2 \rfloor * \lfloor(27-3+0+2)/2 \rfloor = 13 \times 13$

     得出输出特征图形状为$13\times 13\times 256$

4. 卷积层3（384个 $3\times 3\times 256$的卷积核，填充1，步长1）

   - 计算输出卷积形状，

     $\lfloor(13-3+2+1)/1 \rfloor * \lfloor(13-3+2+1)/4 \rfloor = 13 \times 13$

     得出输出特征图形状为 $13\times 13 \times 384$

5. 卷积层4（384个 $3\times 3\times 384$的卷积核，填充1，步长1）

   - 计算输出卷积形状，

     $\lfloor(13-3+2+1)/1 \rfloor * \lfloor(13-3+2+1)/4 \rfloor = 13 \times 13$

     得出输出特征图形状为 $13\times 13 \times 384$

6. 卷积层5(256个 $3 \times 3 \times 384$的卷积核，分别填充1（上下左右分别填充），步长1)

   - 计算输出卷积形状，

     $\lfloor(13-3+2+1)/1 \rfloor * \lfloor(13-3+2+1)/4 \rfloor = 13 \times 13$

     得出输出特征图形状为$13\times 13 \times 256$

   - 计算输出池化形状（$3\times 3, s = 2$）

     $\lfloor(13-3+0+2)/2 \rfloor * \lfloor(13-3+0+2)/2 \rfloor = 6 \times 6$

     得出输出特征图形状为$6\times 6\times 256$

7. 全连接层，有4096个神经元，输出$4096\times 1$向量

8. 全连接层，有4096个神经元，输出$4096\times 1$向量

9. 全连接层，有1000个神经元，输出$1000\times 1$向量

总的来说就是卷积池化、卷积池化、卷积、卷积、卷积池化和3个全连接层


AlexNet的卷积用法:

- $11\times 11 $ 的卷积核，步长为4，没有填充，正常计算

- $5\times 5 $ 的卷积核，分别填充为2和步长为1，使输出特征图的尺寸和输入一致

- $3\times 3 $ 的卷积核，分别填充为1和步长为1，使输出特征图的尺寸和输入一致

AlexNet的池化用法:

- 输入特征图的尺寸为奇数，$3\times 3$，步长为2，没有填充，使输出特征图的尺寸变为输入的一半取下限

  该池化的池化范围3>步长，属于重叠池化。

##### 2 多个GPU

可参考博主https://blog.csdn.net/chenyuping333的图

![多GPU的AlexNet计算](https://img-blog.csdn.net/20180829094658984?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZW55dXBpbmczMzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

AlexNet论文：

      Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
    
---

### ZFNet 2013

ZFNet是2013年ILSVRC的冠军。ZFNet实际上是微调了AlexNet，然后通过转置卷积（Deconvolution）的方式可视化各层的输出特征图，提高了可解释性。

![ZFNet结构](https://img-blog.csdn.net/20180829101607795?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZW55dXBpbmczMzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

注意，

layer1和layer2的padding是valid类型，valid类型的计算公式 $ \lfloor (输入尺寸-k)/s \rfloor + 1 $, 就是不进行填充

layer3、layer4和layer5的padding都是same类型，same类型的计算公式 $ \lfloor (输入尺寸- 1)/s\rfloor + 1$, 就是一共填充k-1

ZFNet的卷积用法:

- ZFNet用 $7\times 7$ 步长为2的卷积核代替AlexNet中$11\times 11 $ 步长为4的卷积核，第一层的卷积核和步长不宜过大，否则会导致后续学习的特征不够细致。

- ZFNet用 $5\times 5 $ 的卷积核，步长为2

- ZFNet用 $3\times 3 $ 的卷积核，步长为1

ZFNet的池化用法:

- 和AlexNet采用相同的池化方式，$3\times 3$，步长为2，没有填充，使输出特征图的尺寸变为输入的一半

  该池化的池化范围3>步长2属于重叠池化。

---

### NiN 2013

Network In Network由Minlin等人提出，在CIFAR-10和CIFAR-100分类任务中达到了当时最好的水平。

为了提高特征的抽象表达能力，作者提出了使用多层感知机（多层全连接层和非线性函数的组合）来替代传统卷积层，新卷积层称为mlpconv层。

传统线性卷积层示意图

![传统卷积层示意图](http://img.blog.csdn.net/20160623184637569)

单通道mlpconv层示意图

![单通道mlpconv层示意图](http://img.blog.csdn.net/20160623184654929)

由于多层感知机可以用 $1\times 1$卷积层替代，所以新层称为cccp层(cascaded cross channel parametric pooling)

跨通道cccp层示意图

![跨通道cccp层示意图](http://img.blog.csdn.net/20160623191820505)

NIN网络由3个cccp层和一个全局平均池化层（Global Average Pooling，GAP）组成。

全连接层与全局平均池化对比图, 节省参数。

![全连接层与全局平均池化对比图](https://www.freesion.com/images/187/422561fc0b3638305c8bf47cb7dece83.JPEG)

参考《深度学习500问》P145的网络参数配置表，可以计算网络各个阶段的尺寸

1. 输入图像 $32 \times 32$

2. cccp层1

   - 卷积层：核 $ 3\times 3 \times 16$, stride步长为1
   
     计算特征图尺寸 $ (32-3+1)/1 \times (32-3+1)/1 \times 16 = 30 \times 30 \times 16$
     
   - 全连接层（$1 \times 1 $卷积层，输出通道数16）：
     
     计算特征图尺寸 $ (30-1+1)\times (30-1+1) \times 16 = 30\times 30\times 16$
     
3. cccp层2
   
   - 卷积层：核 $ 3\times 3 \times 64$, stride步长为1
   
     计算特征图尺寸 $ (30-3+1)/1 \times (30-3+1)/1 \times 64 = 28 \times 28 \times 64$
     
   - 全连接层（$1 \times 1 $卷积层，输出通道数64）：
     
     计算特征图尺寸 $ (28-1+1)\times (28-1+1) \times 64 = 28\times 28\times 64$
     
4. cccp层3

   - 卷积层：核 $ 3\times 3 \times 100$, stride步长为1
   
     计算特征图尺寸 $ (28-3+1)/1 \times (28-3+1)/1 \times 100 = 26 \times 26 \times 100$
     
   - 全连接层（$1 \times 1 $卷积层，输出通道数100）：
     
     计算特征图尺寸 $ (26-1+1)\times (26-1+1) \times 64 = 26\times 26\times 100$
     
5. 全局平均采样层GAP

   池化窗口$ 26 * 26 * 100$ , 步长为1
   
   输出特征图 $ (26 - 26 + 1)/1 \times (26-26+1)/1 \times 100 = 1 \times 1 \times 100$
   

NiN论文：

      Lin, M., Chen, Q., & Yan, S. (2013). Network in network. arXiv preprint arXiv:1312.4400.
---
### VGG 2014

VGG是Oxford牛津大学的Visual Geometry Group的组提出，并在2014年ILSVRC取得亚军。VGG研究的初衷是想搞清楚卷积网络深度是如何影响大规模图像分类与识别的精度和准确率的，最初是VGG-16号称非常深的卷积网络全称为（GG-Very-Deep-16 CNN），VGG在加深网络层数同时为了避免参数过多，在所有层都采用3x3的小卷积核。

#### 1. VGG-16 的总体结构

VGG的输入被设置为224x224大小的RGB图像，在训练集图像上对所有图像计算RGB均值，然后把图像作为输入传入VGG卷积网络。

VGG全连接层都为3层，根据卷积层+全连接层总数目的不同可以从VGG11 ～ VGG19，最少的VGG11有8个卷积层与3个全连接层，最多的VGG19有16个卷积层+3个全连接层。

VGG池化层都为5层，VGG网络并不是在每个卷积层后面跟上一个池化层，而是分布在不同的卷积层之下。

下图是VGG-11、13、16和19的结构图：

![](https://img-blog.csdn.net/20180831091052723?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZW55dXBpbmczMzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

下图是VGG16的结构图：

![](https://img-blog.csdn.net/20180831091117872?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZW55dXBpbmczMzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 2. 以VGG-11的结构为例

1. 输入图像 $224\times 224\times 1$

2. VGG块1

   - 卷积层1（64个 $3\times 3\times 1$ 的卷积核，填充1（上下左右分别填充），步长1）
   
     计算卷积输出形状 $\lfloor(224-3+2+1)/1 \rfloor * \lfloor(224-3+2+1)/1 \rfloor = 224 \times 224$
     
     特征图输出形状为 $ 224 \times 224 \times 64$
     
   - 池化（$2\times 2$，步长为2）
     
     计算池化输出形状 $\lfloor(224-2+0+2)/2 \rfloor * \lfloor(224-2+0+2)/2 \rfloor = 112 \times 112$
     
     特征图输出形状为 $ 112 \times 112 \times 64$

3. VGG块2

   - 卷积层2（128个 $3\times 3\times$ 64的卷积核，填充1（上下左右分别填充），步长1）

     计算卷积输出形状 $\lfloor(112-3+2+1)/1 \rfloor * \lfloor(112-3+2+1)/1 \rfloor = 112 \times 112$
     
     特征图输出形状为 $ 112 \times 112 \times 128$
     
   - 池化（$2\times 2$，步长为2）
     
     计算池化输出形状 $\lfloor(112-2+0+2)/2 \rfloor * \lfloor(112-2+0+2)/2 \rfloor = 56 \times 56$
     
     特征图输出形状为 $ 56 \times 56 \times 128$
     
4. VGG块3

   - 卷积层3（核通道256）
   
     特征图输出形状为 $ 56 \times 56 \times 256$
     
   - 卷积层4（核通道256）
   
     特征图输出形状为 $ 56 \times 56 \times 256$
     
   - 池化
   
     特征图输出形状为 $ 28 \times 28 \times 256$

5. VGG块4

   - 卷积层5（核通道512）
   
     特征图输出形状为 $ 28 \times 28 \times 512$
     
   - 卷积层6（核通道512）
   
     特征图输出形状为 $ 28 \times 28 \times 512$
     
   - 池化
   
     特征图输出形状为 $ 14 \times 14 \times 512$
     
5. VGG块5

   - 卷积层7（核通道512）
   
     特征图输出形状为 $ 14 \times 14 \times 512$
     
   - 卷积层8（核通道512）
   
     特征图输出形状为 $ 14 \times 14 \times 512$
     
   - 池化
   
     特征图输出形状为 $ 7 \times 7 \times 512$
     
 6. 全连接层，有4096个神经元，输出 $4096\times 1$ 向量
 7. 全连接层，有4096个神经元，输出 $4096\times 1$ 向量
 8. 全连接层，有1000个神经元，输出 $1000\times 1$ 向量


VGG-11的卷积用法:

- $3\times 3$ 的卷积核，填充1（上下左右分别填充），步长1，使输出特征图尺寸与输入一致

VGG-11的池化用法:

- $2\times 2$，步长为2，没有填充，使输出特征图的尺寸变为输入的一半

#### 3. VGG 块

VGG块的组成规律是：

1.	连续使用数个相同的填充为1、窗口形状为 $3\times 3$ 的卷积层后接上一个步幅为2、窗口形状为2×2的最大池化层。

2.	卷积层保持输入的高和宽不变，而池化层则对其减半。


        Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

---

### GoogLeNet 2014

GoogLeNet是2014年ILSVRC在分类任务上的冠军。

该网络设计了Inception块来代替人工选择卷积的类型，然后堆叠Inception块，去掉了全连接层，使用了NIN中全局平均池化的思想。

作者首先提出了下图的Inception的基本结构

![Inception module基本结构](https://img-blog.csdn.net/20160225155336279)

该模块有如下特点：

- 与传统的卷积不同，该卷积拓宽了网络的宽度

- 采用了1、3、5这样不同尺度的卷积核，在步长为1时，可以设置分别填充的padding分别为0、1、2，这样卷积之后特征图的维度就一致了，方便进行拼接

- 由于pooling在很多地方都很有用，所以Inception中也嵌入了

为了进一步减少参数量，GoogLeNet借鉴了NiN网络的 $1 \times 1$ 卷积。

在没改进时Inception的参数量为（不考虑偏置）,若输入和各支路输出的通道数都为$ C1 = 16$

$ (1\times 1 + 3\times 3 + 5\times 5 + 0) \times 16 \times 16 = 8960$

在1、3、5前加上 $1 \times 1$ 卷积层可以有效地减少特征图厚度，从而减少参数量。

下图为Inception模块

![Inception改进结构](https://img-blog.csdn.net/20160225155351172)

以单个的 $5 \times 5 $的卷积为例计算参数量，若输入上一层的输出通道数为128，卷积核个数为256。

有该卷积的参数量为，$ 128 \times 5 \times 5 \times 256 = 819200$

若该 $ 5 \times 5 $卷积前添加一个 32个$ 1\times 1$的卷积，那么

该卷积层的参数量为，$ 128 \times 1 \times 1 \times 32 +  32 \times 5 \times 5 \times 256 = 204800$

可以看出加了 $1\times 1$ 卷积之后，原参数量是新参数量的$ 819200 / 204800 = 4$倍。

再来看整个结构，若Inception右边网络设置 $1\times 1$卷积核的个数为$C2 = 8$，满足$C1>C2$

那么网络的的参数量为$ 2\times C1 \times 1 \times 1 \times C1 + 2\times (C1 \times 1 \times 1 \times C2) + C2 \times 3 \times 3 \times C1 + C2 \times 5 \times 5 \times C1   $
即$ 2\times 16 \times 1 \times 1 \times 16 + 2\times (16 \times 1 \times 1 \times 8) + 8 \times 3 \times 3 \times 16 + 8 \times 5 \times 5 \times 16   = 5120$

可以看出在保证输出尺寸的情况下，减少了很多参数。

这种结构也称为卷积神经网络的瓶颈结构，即在计算比较大的卷积层之前先用一个 $1 \times 1$ 卷积来压缩大卷积层输入特征图的通道数，以减小计算量，在大卷积层完成计算后，根据实际需要，有时候会再次使用一个 $1 \times 1$ 卷积对输出层特征图的通道数复原。

GoogLeNet完整结构如下：

![GoogLeNet完整结构图](https://img-blog.csdn.net/20160225155414702)

1. 输入 $224\times 224\times 3$

2. 卷积层 $7\times 7$卷积核，stride=2, 分别padding=3, 输出通道64
   
   $(224-7+6+2)/2 = 112.5$，输出特征图尺寸 $112\times 112 \times 64$

3. 池化层 $3\times 3$窗口，stride=2,分别padding=1

   $(112-3+2+2)/2 = 56.5$，输出特征图尺寸 $56 \times 56 \times 64$

4. 卷积层 $3\times 3$卷积核，stride=1, 分别padding=1, 输出通道192

   输出特征图尺寸 $56\times 56\times 192$
   
5. 池化层 $3\times 3$窗口，stride=2,分别padding=1

   $(56-3+2+2)/2 = 28.5$，输出特征图尺寸 $28 \times 28 \times 192$

6. Inception层

   - 支路1：$ 1\times 1$卷积层，核个数64，输出尺寸$28\times 28 \times 64$;
    
   - 支路2：
   
      + $1 \times 1$卷积层，核个数96，输出尺寸 $28 \times 28 \times 96$;
       
      + $3\times 3$卷积层，核个数128，分别padding = 1, 输出尺寸 $ 28 \times 28 \times 128$;

   - 支路3：
   
      + $1 \times 1$卷积层，核个数32，输出尺寸 $28 \times 28 \times 32$;
       
      + $5\times 5$卷积层，核个数32，分别padding = 2, 输出尺寸 $ 28 \times 28 \times 32$;

   - 支路4：
   
      + $3\times 3$的pooling层，分别padding为1， 输出尺寸 $ 28\times 28 \times 192$

      + $1\times 1$的卷积层，核个数32，输出尺寸 $28 \times 28 \times 32$
    
   - 拼接结果得到本层输出特征图尺寸 $ 28 \times 28\times (64+128+32+32) = 28\times 28 \times 256$

     ![拼接尺寸](https://pic4.zhimg.com/v2-57e6660e8c7f9c061f42d6f24ccbd957_r.jpg)

其它类推

       Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., & Anguelov, D. & Rabinovich, A.(2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).

后续还有Inception-v2、Inception-v3、Inception-v4

---

### SPPNet 2015

SPPNet的基础是ZFNet，通过将ZFNet的第一个全连接层替换为SPP层，即可得到SPPNet。

传统CNN和SPPNet的对比

![传统CNN和SPPNet的对比](ch01_img/传统CNN和SPPNet.jpeg)

从上面的架构中可以看出，SPPNet与经典CNN最主要的区别在于两点：

第一点：不再需要对图像进行裁剪和放缩这样的预处理；

第二点：在卷积层和全连接层交接的地方添加所谓的空间金字塔池化层，即（spatial pyramid pooling），这也是SPP-Net网络的核心所在。

一个包含3级金字塔的SPPNet的结构图

![一个包含3级金字塔的SPPNet的结构图](https://img-blog.csdn.net/20180912185049651?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3Byb2dyYW1taW5nZm9vbDU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

SPP层的目的是保证经过该层后，不管多大尺寸的输入，在金字塔的每一级都产生一个固定尺寸的输出，最后该层输出一个由金字塔各级的固定长度向量拼接而成的向量。

金字塔中每一级池化具体的做法，可以理解将原来卷积神经网络中固定尺寸的池化窗口根据输入和输出修改成自适应的池化窗口。

假设任意尺寸输出为 $a\times a$, 想要一个 $n\times n$的输出结果，那么有公式窗口尺寸 $window=\lceil a/n \rceil $ ，步长stride= $window=\lfloor a/n \rfloor $

该计算公式本质上将特征图均分为 $n\times n$个子区域，然后对各个子区域max pooling

当 $a\times a$为 $13 \times 13$时，要得到 $4 \times 4$的输出，$win=4, stride=3$

当 $a\times a$为 $13 \times 13$时，要得到 $2 \times 2$的输出，$win=7, stride=6$

当 $a\times a$为 $13 \times 13$时，要得到 $1 \times 1$的输出，$win=13, stride=13$

这种多级池化的机制下会对目标的形变问题有很好的健壮性.

若网络的输入要处理任意大小的图像时，要至少考虑两种不同的预定义大小，比如 $224 \times 224$和 $180 \times 180$，当输入是 $224 \times 224$时将图像缩放成 $180 \times 180$，从而可以实现一个固定尺寸的网络。

        He, K., Zhang, X., Ren, S., & Sun, J. (2015). Spatial pyramid pooling in deep convolutional networks for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 37(9), 1904-1916.
        
---

### ResNet 2015

ResNet是2015年的ILSVRC的冠军。引入了残差网络，解决了深层网络在训练过程中梯度弥散的问题，使深层模型更容易训练。

假设遇到一个简单的任务，只需要少层的conv即可拟合，过多的conv反而效果不好。这时100层中90层被去除，只留10层。假设又有一个复杂任务，需要很多层conv才行，这时100层中10层去除，留下90层。ok，resnet就是一种能同时适应这两种任务的网络结构，其可以“调整网络的层数以适应任务”

当网络在内部的其中一层已经达到了最佳的情况，这时剩下的特征层应该自动学习成恒等映射 $f(x) = x$的形式。

由于网络随着深度的增加会有梯度消失和梯度爆炸的问题，那么让深度网络实现和浅层网络一样的性能，即让深度网络后面的层实现恒等映射，根据这个想法，作者提出了残差块。

![ResNet结构](https://img2018.cnblogs.com/i-beta/1557203/202002/1557203-20200214172343785-2084773936.jpg)

若将输入设为x， 将该网络层设为H，那么就有此层的输出 $H(x)$, 网络普遍学习的都是 $x -> H(x)$ 的映射关系。

若去学习输出与输出之间的差值，即学习H(x) - x ，为了在H(x)和输入x相等时进行恒等映射，那么有最终学习变为(H(x) - x) + x

若输入x和H(x)相等那么就有了恒等映射 x - > x

令 $F(x)=H(x)-x$, 有最终学习 $F(x) + x$

34层残差网络（右），34层原始卷积网络（中），19层VGG网络（左）如下图

![34层残差网络（右），34层原始卷积网络（中），19层VGG网络（左）如下图](https://pic2.zhimg.com/80/v2-03f393009c383ce8ec8b956399a105a8_720w.jpg?source=1940ef5c)

对于短路连接，当输入输出维度一致时可直接相加，需要注意的是图中的虚线代表残差块的前后维度不一致的情况。

可以有两种策略处理通道维度不一致

- 使用0来对通道数进行填充以使像素点对齐

- 使用 $1\times 1$卷积来对通道进行调整

对于尺寸不一致，（？？？这是一个问题，没看懂具体做法）

- 将x做一个线性映射： $ H(x) = F(x, W_i) + x => H(x) = F(x, W_i) + W_sx$ 网址连接：https://zhuanlan.zhihu.com/p/106764370


        He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

---      

### DenseNet 2017

DenseNet主要借鉴了ResNet和Inception的思想。卷积神经网络提高效果的方向，要么深（比如ResNet，解决了网络深时候的梯度消失问题）要么宽（比如GoogleNet的Inception）。

如下图使一个5层的dense块

![如下图使一个5层的dense块](https://upload-images.jianshu.io/upload_images/5067993-6c3dbdc5ece30919?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

从图中可以看出稠密块将所有层连接了起来，也就是每一层的输入来自前面所有层的输出。

与ResNet的主要区别在于，DenseNet里模块 B 的输出不是像ResNet那样和模块 A 的输出相加，而是在通道维上连结。

![ResNet和DenseNet](https://zh.d2l.ai/_images/densenet.svg)

这样模块 A 的输出可以直接传入模块 B 后面的层。在这个设计里，模块 A 直接跟模块 B 后面的所有层连接在了一起。这也是它被称为“稠密连接”的原因。

DenseNet网络结构图

![](https://upload-images.jianshu.io/upload_images/5067993-bbd7b1856ac7cb1f?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

DenseNet由若干个密集连接块组成，密集连接块之间通过转换层（Transition Layer）连接，每一个密集连接块又由若干个卷积层组成，同一块中的每一个卷积层与前面的卷积层都存在直接连接。

和ResNet问题一样，若块之间的特征图尺寸不匹配，DenseNet在两个块之间使用转换层，转换层就是用 $1\times1$卷积来调整通道

下面是一种用于ImageNet的DenseNet结构

1. 输入图像 $224\times 224 \times 3$

2. Feature Block

   - 卷积层1，64个 $7\times 7$, 填充3， 步长2的卷积核
  
      * $(224-7+6+2)/2 = 112.5$
   
      * 输出尺寸 $112\times 112\times 64$
   
   - BN层，输入输出维度不变 $112\times 112\times 64$

   - RELU激活层，输入输出维度不变 $112\times 112\times 64$

   - 池化层，$3\times 3$, 填充1， 步长2，输出尺寸$56\times 56\times 64$

3. 稠密块1

   - 稠密层0
   
     * BN层，输入输出维度不变 $56\times 56\times 64$
     
     * RELU激活层，输入输出维度不变 $56\times 56\times 64$
     
     * 可选的Bottlenneck用来调整特征图通道
       
       - $1\times 1 \times 128$卷积， 输出特征图 $56\times 56\times 128$

       - BN层，输出特征图 $56\times 56\times 128$

       - RELU层，输出特征图 $56\times 56\times 128$
       
     * 卷积层 $3\times 3 \times 32$, 填充1, 步长1，输出特征图 $ 56\times 56\times 32$

     * 可选Dropout层，防止过拟合输出特征图 $ 56\times 56\times 32$


   - 稠密层1，输入 $ 56\times 56\times 32$，输出 $ 56\times 56\times 32$

   - 稠密层2，输入 $ 56\times 56\times (32+32)$，输出 $ 56\times 56\times 32$

   - 稠密层3，输入 $ 56\times 56\times (32+32+32)$，输出 $ 56\times 56\times 32$
   
   - 若稠密层共L+1层,从0到L
   
   - 稠密层L，输入 $ 56\times 56\times (32\times L)$，输出 $ 56\times 56\times 32$

4. Transition Block

   - BN层，输出特征图 $ 56\times 56\times 32$

   - RELU层，输出特征图 $ 56\times 56\times 32$
   
   - $1\times 1 \times 128$卷积， 若稠密块包含m个特征图（这里是32），这里可以设置一个压缩因子compression(大于0并小于1)来对通道进行控制，输出特征图 $56\times 56\times (32\times compression)$
   
   - Average Pooling $2 \times 2$, 步长2， 输出特征图 $28\times 28\times (32\times compression)$
   
   设压缩因子compression为1

5. Dense Block 2 输入特征图 $28\times 28\times 32$，输出特征图 $28\times 28\times 32$

6. Transition 2 输入特征图 $28\times 28\times 32$，输出特征图 $14\times 14\times 32$
   
7. Dense Block 3 输入特征图 $14\times 14\times 32$，输出特征图 $14\times 14\times 32$

8. Transition 3 输入特征图 $14\times 14\times 32$，输出特征图 $7\times 7\times 32$

9. Classification Block

   - BN 输出特征图 $7\times 7\times 32$

   - RELU 输出特征图 $7\times 7\times 32$

   - Poolling $7\times 7$ ,stride=1 , 输出特征图 $1\times 1\times 32$

   - flatten 将 $1\times 1\times 32$ 铺平成 $1\times 32$

   - Linear全连接，输出 $1\times classes_num$

DenseNet论文：

        Huang, G., Liu, Z., Weinberger, K. Q., & van der Maaten, L. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (Vol. 1, No. 2).

网络可以参考博主https://blog.csdn.net/chenyuping333

Caffe和Tensorflow的实现可以参考《深度学习卷积神经网络从入门到精通》

## Q9-用于分类的卷积神经网络最后几层一般是什么层

用于分类任务的卷积神经网络，其前面若干层一般是卷积层、池化层等，但网络末端一般是几层全连接层。

因为多个全连接层组合在一起就是经典的多层感知机分类模型，卷积神经网络中前面的卷积层为多层感知机提取深层的、非线性的特征。

最近，分类网络在卷积层之后、最后一层之前通常采用全局平均池化，并有如下优点：

（1）大大降低了参数量和计算量。假设输入特征图的尺寸为$w\times h$, 输出通道数为c，则全局平均池化的参数为0，计算量仅为$c \times w \times h$, 而对于k个输出单元的全连接层来说，则参数量和计算量均为$k \times c \times w \times h$

（2）具备良好的可解释性，知道特征图上哪些点对最后的分类贡献最大。

摘自《百面学深度学习》

![](https://img-blog.csdn.net/20180201141956028?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveWltaW5nc2lsZW5jZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## Q10-有哪些变种卷积

转置卷积、空洞卷积、可变形卷积、分组卷积

参考《百面深度学习》P11-19

# 中英词汇对照

卷积神经网络 Convolutional Neural Network，CNN

多层感知机 Multi-Layer Perceptron，MLP

输入层 Input Layer

输出层 Output Layer

卷积层 Convolution Layer

卷积核 Convolution Kernel

通道 Channel

步长 Stride

填充 Padding

池化 Pooling

降采样层 Downsampling Layer

最大池化 Max Pooling

平均池化 Mean Pooling

激活层 Activation Layer

全连接层 Full Connected Layer

ImageNet大规模视觉识别挑战赛 ImageNet Large Scale Visual Recognition Competition，ILSVRC

转置卷积 Deconvolution

全局平均池化层 GAP Global Average Pooling
