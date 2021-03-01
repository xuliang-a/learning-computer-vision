# 目录

* [Q1-目标检测的单步模型和两步模型是什么意思？](#Q1-目标检测的单步模型和两步模型是什么意思)

* [Q2-两步模型的发展过程是什么样的？](#Q2-两步模型的发展过程是什么样的)

  - [RCNN](#RCNN)

  - [Fast RCNN](#Fast-RCNN)

  - [Faster RCNN](#Faster-RCNN)

* [Q3-Fast RCNN的ROI是如何映射到特征图上的？](#Q3-Fast-RCNN的ROI是如何映射到特征图上的)

* [Q4-Fast RCNN的ROI-Pooling是什么？](#Q4-Fast-RCNN的ROI-Pooling是什么)

* [Q5-Faster RCNN的RPN网络是什么？](#Q5-Faster-RCNN的RPN网络是什么)

* [Q6-非极大值抑制处理的流程是什么？](#Q6-非极大值抑制处理的流程是什么)

* [Q7-单阶段检测算法的发展过程是什么样的？](#Q7-单阶段检测算法的发展过程是什么样的)

  - [YOLO](#YOLO)

  - [YOLOv2](#YOLOv2)

  - [YOLO9000](#YOLO9000)

  - [YOLOv3](#YOLOv3)

* [Q8-增强模型对小目标的检测效果有哪些方法？](#Q8-增强模型对小目标的检测效果有哪些方法)

* [Q9-常用的图像增广方法有哪些？](#Q9-常用的图像增广方法有哪些)

* [Q10-目标检测模型如何进行评估？](#Q10-目标检测如何进行评估)

* [中英文词汇](#中英文词汇)

# 目标检测

## Q1-目标检测的单步模型和两步模型是什么意思

单步模型（One stage）没有独立地、显示地提取提议区域（候选区域）（Region Proposal，RP）的过程，直接由输入图像得到其中存在的目标的类别和位置信息的模型。

两步模型（Two stage）有独立的、显示的候选区域的提取过程，即先在输入图像上筛选出一些可能存在物体的候选区域，然后针对每个候选区域，判断是否存在物体，若存在就给出物体的类别和修正信息。

**优缺点：** 一般来说，单步模型在计算效率上有优势，两步模型在检测精度上有优势。

对于单步模型和两步模型在**速度和精度上的差异**，学术界一般认为有如下原因：

（1）多数的单步模型是利用预设的锚框来捕捉可能存在于图像中各个位置的物体，单步模型会对数量庞大的锚框进行是否有物体及物体所述类别的密集分类，由于图像实际物体数目远小于锚框数目，所以正负样本极其不均衡，导致分类器效果不好。由于两步模型中，含有独立的候选区域提取步骤，在第一步中会筛选掉大部分不含有待检测物体的区域，即负样本，再传递给第二步进行分类和候选区域位置修正时，正负样本比例均衡。

（2）两步模型在候选区域提取过程中会对候选框的位置和大小进行修正，在第二阶段区域特征已被对齐，然后在第二阶段候选框会再次修正，所以带来了更高的精准度。单步模型输入特征未被对齐，质量较差，所以在分类和定位的精度较差。

（3）由于两步模型在第二步需要对每一个候选区域进行分类和位置回归，所以在候选区域数目非常大时，计算量与之成正比，所以存在计算量大，速度慢的问题。
	
摘自《百面深度学习》P234-236

- **Two stage目标检测算法**

    - 先进行区域生成（一个有可能包含待检物体的预选框），再通过卷积神经网络进行样本分类。

    - 任务：特征提取—>生成RP—>分类/定位回归。
    
    - 常见的有R-CNN、Fast R-CNN、Faster R-CNN。

- **One stage目标检测算法**

    - 不用RP，直接在网络中提取特征来预测物体分类和位置。

    - 任务：特征提取—>分类/定位回归
    
    - 常见的有OverFeat、SSD和YOLO系列模型

摘自《深度学习500问》

## Q2: 两步模型的发展过程是什么样的？

### RCNN	

RCNN是目标检测的奠基之作，RCNN是第一个将卷积神经网络用于目标检测的深度学习模型，具体来说我总结为三个步骤：

第一步，输入图像并选取提议区域，对提议区域的选择可以使用选择性搜索（Selective Search）方法将输入图像中具有相似颜色直方图特征的区域进行递归合并，来提取1千到2千个提议区域；

第二步，提取特征，在选取提议区域之后，首先将每个区域首先进行缩放和裁剪，使区域能匹配卷积神经网络的输入维度，在缩放之后将区域送入卷积神经网络生成固定长度的特征，来提取特征；

第三步，分类和回归计算，对每一个提议区域的特征送入SVM分类器进行目标的分类，然后进一步将该类别的特征送入多层感知机进行目标坐标位置的回归修正；

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/8.2.1-1.png)

### Fast RCNN

由于RCNN需要对每一个提议区域进行特征提取计算即卷积计算，而这些提议区域很多都是高度重叠的，所以为了减少训练时间，Fast RCNN提出了改进方法。

第一步，输入图像并选取提议区域；

第二步，提取特征，将整幅输入图像作为卷积神经网络的输入，得到一幅特征图，和RCNN不同的是，RCNN将每个提议区域都送入一次卷积神经网络，而Fast RCNN直接将提议区域位置映射到了卷积神经网络的最后一层的特征图上，该方法节约了时间。由于接下来需要对每个不同尺寸提议区域对应的特征图送入多层感知机，而多层感知机的输入是固定的，所以提出了ROI Pooling层来将不同尺寸的特征下采样到相同尺寸，得到最后固定长度的特征。

第三步，对特征进行分类和回归。

其实Fast RCNN与SPPNet设计思路类似，SPPNet是空间金字塔池化而非ROI Pooling。

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/8.2.2-1.png)

### Faster RCNN

该网络从提议区域的生成速度，来对网络的性能进行提高，该网络提出了一个新颖的RPN网络来改进耗时的选择性搜索。

第一步，输入图像提取特征图，和Fast RCNN相同，Faster RCNN对输入图像做一次卷积计算得到一幅特征图。

第二步，将特征图作为新颖的RPN网络（Region Proposal Networks）的输入来提取提议区域。接着和Fast RCNN一样将提议区域和特征图进行映射然后送入ROI Pooling，得到固定尺寸的特征。

第三步，对特征进行分类和回归。

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/8.2.3-1.png)
![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/8.2.3-2.png)

## Q3-Fast RCNN的ROI是如何映射到特征图上的

感兴趣区域（Region of Interest，ROI），指的是由选择性搜索提取的候选区域RP。

ROI映射的目标是原图ROI区域的中心点尽量接近特征图对应区域的中心点。

关于ROI映射我看到了如下映射方法：

- 把在输入图像上的ROI各个坐标除以“输入图片与Feature Map的大小的比值”，得到了feature map上的box坐标。（摘自《深度学习500问》）

  例子,输入图像 $600 \times 800$, 特征图 $38 \times 50$，原图的ROI左上角坐标(30,40)、左下角坐标(200,400)，
  
  那么有在特征图上的左上角坐标 $( 30 \times (38/600), 40 \times (50/800) )$和左下角坐标 $( 200 \times (38/600), 400 \times (50/800) )$，四舍五入后得到(2,3)和(13, 25)

- SPPNet的ROI映射

在SPPNet中，假设(x’,y’)表示特征图上坐标点，(x,y)表示该坐标点在原始输入图像上的对应点。则有结论 (x,y)=(S * x’,S * y’) 其中S代表所有卷积层和池化层的stride乘积

为了处理有小数的情况，同时左上角点和右下角点都向图像内侧近似(左上角要向右下偏移，右下角要想要向左上偏移)，所以左上角加一 右下角减一 同时为了减小这种近似产生的误差 所以左上角向下取整 右下角向上取整。

最后，左上角点 x’= ⌊x/S⌋+1 右下角点 x’=⌈x/S⌉-1

该推导用到的计算公式：

若卷积核边长为k，填充是p，步长是s，则有如下坐标计算，

- 对于卷积和池化层，$ p_i = s \times p_{i+1} + ((k-1)/2-p)$

- 对于激活层，$ p_i = p_{i+1}$

一个计算卷积后图像中坐标的例子，

![一个计算卷积后图像中坐标的例子](https://pic2.zhimg.com/v2-c1ce5a16dbd75553be1a9ea8921f3c35_r.jpg)

为了简化计算，可以将$$((k-1)/2-p)$$化简，将每一个卷积层和池化层的填充设置为小于等于当前层滤波器尺寸一半的最大整数（就是取下限），即 $p=\lfloor k_i / 2 \rfloor$。

那么，就有 $ p_i = S_i \times p_{i+1} + ((k_i-1)/2 - \lfloor k_i / 2 \rfloor)$

- 当 $k_i$为奇数时, $((k_i-1)/2 - \lfloor k_i / 2 \rfloor) = 0$，有 $p_i = S_i \times p_{i+1} $

- 当 $k_i$为偶数时，$((k_i-1)/2 - \lfloor k_i / 2 \rfloor) = -1/2$，有 $p_i = S_i \times p_{i+1} - 1/2$

因为 $p_i$是坐标值，不可能取到小数，所以可以得到 $p_i = S_i \times p_{i+1} $，公式这样就得到了化简， $p_i$ 只跟 $p_{i+1}$和步长有关。

将公式一层一层进行级联，得到 $p_0 = S \times p_{i+1}$, 其中 $S=\prod_0^i s_i$;

对于特征图上的(x’,y’)，则有结论该坐标点在原始输入图像上的对应点(x,y)=(S * x’,S * y’) 其中S代表所有卷积层和池化层的stride乘积。

然后根据前面说的左上角向右下角偏移，右下角向左上角偏移再调整一下得到左上角点 $(x' = \lfloor x/S \rfloor + 1, y' = \lfloor y/S \rfloor + 1)$，右下角点$(x' = \lceil x/S \rceil - 1, y' = \lceil y/S \rceil - 1)$


## Q4-Fast RCNN的ROI Pooling是什么

ROI pooling层是pooling层的一种，由于是针对ROI进行的池化操作，所以称为ROI Pooling

第一步，根据输入的图像将提议区域ROI映射到特征图上对应的位置；

第二步，将映射后的ROI区域划分为与输出维度相同的切片；

第三步，对每个切片进行最大池化操作。

这样就可以从不同尺寸的ROI提议区域得到固定大小的特征图，该ROI Pooling的结果特征图不依赖ROI的尺寸。

例如：输入图像经过一系列卷积和池化操作缩小32倍后的输出特征图的尺寸为：$8 \times 8$，提议区域ROI对应于输入图像的左上角和右下角坐标分别为（0,100）和（198,224），规定输出大小为：$2 \times 2$。

第一步，进行ROI映射，可能会产生量化误差。

映射到特征图后ROI左上角坐标:（0,100/32），将坐标向下取整，变为（0,3）

映射到特征图后ROI右下角坐标（198/32,224/32），将坐标向下取整，变为（6,7）

第二步，将ROI划分成$2 \times 2$的区域切片，可能会产生量化误差。

映射到特征图后ROI的宽为7，划分2份后，每个区域的宽为：7/2 = 3.5，左半部分的宽取3，右半部分的宽取4

映射到特征图后ROI的长为5，划分2份后，每个区域的长为：5/2 = 2.5，上半部分的长取2，下半部分的长取3

第三步，对划分后的区域进行最大池化操作。

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/8.1.11.gif)

## Q5-Faster RCNN的RPN网络是什么

RPN网络是候选区域网络，用来替代选择性搜索来生成ROI，这个新颖的RPN网络实质上是一种基于神经网络的的二分类和边界框回归模型；

![RPN](https://img-blog.csdn.net/20180120181848383?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFucmFuMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

1. 以特征图每个单元为中心，生成k个不同大小和宽高比的锚框并标注它们，一共会产生 $k\times w\times h $个锚框。

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/8.2.3-4.png)

2. 对于一个$ w\times h$ 的特征图，使用一共填充为2，步长1的 $3\times 3$ 卷积层变换卷积神经网络的输出，并将输出通道数记为 c=256 。这样，卷积神经网络为图像抽取的特征图中的每个单元均得到一个长度为 256 的新特征,即得到一个$ w\times h \times 256$的特征。

3. 用锚框中心单元长度为 c= 256 的特征分别在分类层得到2k个得分，用于预测该锚框的二元类别（含目标还是背景的概率），在回归层得到4k个坐标偏移量，用于边界框的回归，该分类层和回归层都可以使用 $1 \times 1$卷积来实现全连接层的所需输出的功能。

4. 最后使用非极大值抑制，从预测类别为目标的预测边界框中移除相似的结果。最终输出的预测边界框即兴趣区域池化层所需要的提议区域。

参考https://blog.csdn.net/lanran2/article/details/54376126

## Q6-非极大值抑制处理的流程是什么

目的：忽略相互间高度重叠的锚框（即：将Iou重叠率大于0.7框，比较分类的得分，保留分类的得分大的框）

![](https://images2017.cnblogs.com/blog/606386/201708/606386-20170826153025589-977347485.png)

**非极大值抑制的方法：**

非极大值抑制（Non-Maximum Suppression，NMS）步骤如下：

1.设置一个Score的阈值，一个IOU的阈值；

2.对于每类对象，遍历属于该类的所有候选框，

①过滤掉Score低于Score阈值的候选框；

②找到剩下的候选框中最大Score对应的候选框，添加到输出列表；

③进一步计算剩下的候选框与②中输出列表中每个候选框的IOU，若该IOU大于设置的IOU阈值，将该候选框过滤掉，否则加入输出列表中；

④最后输出列表中的候选框即为图片中该类对象预测的所有边界框

Top-N排序
经过NMS后的框还是过多，所以采用top-N排序的方式选出128个正例（若正例少于128，则选择所有的正例）、128个负例用于目标检测。
用于目标检测的总个数为256。

3.返回步骤2继续处理下一类对象。


例如：先假设有6个矩形框，根据分类器的类别分类概率(即：为背景或目标的得分)做排序，假设从小到大属于车辆的概率 分别为A、B、C、D、E、F。

1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;

2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。

3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。

就这样一直重复，找到所有被保留下来的矩形框。

**交并比**（Intersection-over-Union，IoU）

它是目标检测中使用的一个概念，是产生的候选框（candidate bound）与原标记框（ground truth bound）的交叠率，即它们的交集与并集的比值。

![](https://img-blog.csdnimg.cn/20181102130324332.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwNjM4ODMx,size_16,color_FFFFFF,t_70)

对于非极大值抑制算法来说，IoU指的是两个候选框之间的交叠率。

## Q7-单阶段检测算法的发展过程是什么样的

### YOLO

YOLO（You Only Look Once: Unified, Real-Time Object Detection）是one-stage detection的开山之作。

基于先产生候选区再检测的方法虽然有相对较高的检测准确率，但运行速度较慢。

YOLO创造性的将物体检测任务直接当作回归问题（regression problem）来处理，将候选区和检测两个阶段合二为一。只需一眼就能知道每张图像中有哪些物体以及物体的位置。

从结构上来说，YOLO借鉴了GoogLeNet结构，不同的是YOLO使用 $1 \times 1$卷积层和 $3 \times 3$卷积层来代替Inception模块，这个检测网络包括24个卷积层和2个全连接层。

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/YOLOv1-02.png)

事实上，YOLO也并没有真正的去掉候选区，而是直接将输入图片划分成7x7=49个网格，每个网格预测两个边界框，一共预测49x2=98个边界框。可以近似理解为在输入图片上粗略的选取98个候选区，这98个候选区覆盖了图片的整个区域，进而用回归预测这98个候选框对应的边界框。


从输出可以看出YOLO网络最后的输出的尺寸为 $7\times 7 \times 30$，30是指对 $7 \times 7$中的每一个网格都预测了20个类别、置信度c（边界框与标注框的IOU）和边界框(x,y,w,h)，边界框的中心是(x,y),边界框的宽高w,h是相对于原始输入图像的宽高比例，然后在每个网格上预测两个边界框。

![YOLO输入与输出的映射关系](https://upload-images.jianshu.io/upload_images/2709767-100b9cca5ff41ab7.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

![YOLO的输出](https://upload-images.jianshu.io/upload_images/2709767-6bd185455eff98cd.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

当图像经过YOLO网络后生成了$7\times 7 \times 2$个候选框，最后采用非极大值抑制算法得到筛选出的候选框。

YOLO算法将目标检测看成回归问题，采用的是均方差损失函数将样本标签中不同部分的均方差损失相加在一起，作为整体误差，具体来说有中心点、边框高度和宽度、边框内有物体的置信度、边框内无物体的置信度和对象分类。

![YOLO样本标签与网络实际输出](https://upload-images.jianshu.io/upload_images/2709767-2424b2881c518390.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

![YOLO给出的损失函数](https://upload-images.jianshu.io/upload_images/2709767-73e19de371eceedf.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

YOLO的最后一层采用线性激活函数，其它层都是Leaky ReLU。训练中采用了drop out和数据增强（data augmentation）来防止过拟合。

参考https://www.jianshu.com/p/cad68ca85e27

---

### YOLOv2

YOLOv2采用了新的网络结构Darknet-19。

![Darknet-19](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/YOLOv2-02.png)

Darknet-19包括19个卷积层和5个最大池化层，主要采用了 $1 \times 1$卷积层和 $3 \times 3$卷积层。

YOLOv2网络在Darknet-19的基础上形成

![YOLOv2网络](https://img-blog.csdnimg.cn/20191118110947444.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3podzg2NDY4MDM1NQ==,size_16,color_FFFFFF,t_70)

YOLO2网络中第0-22层是Darknet-19网络，后面第23层开始，是添加的检测网络。

若Darknet-19表示为

![](https://img-blog.csdn.net/20180630232927217?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlbnh1ZWxpdQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

有对应的YOLOv2网络

![](https://img-blog.csdn.net/20180630233004461?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlbnh1ZWxpdQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

YOLOv2的输出是 $13 \times 13 \times 125$，也就是划分成 $13\times 13$个格子，每个格子预测5个长宽比不同的边框,每个边框有20个类和置信度还有坐标，$5 \times (4 + 1 + 20) = 125$

YOLOv2的改进策略如下：

- 关于输入

1. 首先从输入图像上来说，大部分网络的输入都是以小于 $256\times 256$分辨率的图像作为输入，YOLOv2将输入图像分辨率提升至 $448 \times 448$，从而提高了检测精度。

2. YOLOv2希望在不同尺寸的输入图像上都可以稳健地检测，所以采用了多尺度训练的方法，也就是训练过程中每迭代10次，就会随机选择输入图像的尺寸。

3. 由于YOLOv2可以适应多种尺寸的输入图像，若采用更高分辨率比如 $608 \times 608$则会更能提升检测精度。

- 关于网络

4. 使用了新的网络结构Darknet-19，该网络降低了模型计算量。

5. 在每个卷积层后加批归一化层（Batch Normalization,BN），BN层可以起到一定正则化效果，能提升模型收敛速度，防止过拟合。

6. YOLOv2使用了多尺度的特征图做检测，提出转移层将高分辨率特征图和低分辨率特征图结合在一起，YOLOv2提取Darknet-19的最后一个最大池化层的输入 $26 \times 26 \times 512$，然后一个分支继续正常计算得到 $13 \times 13 \times 1024$的特征图，另一个分支先经过 $1\times 1 \times 64$降低维度得到 $26\times 26 \times 64$的特征图，然后经过转移层的处理变成 $13 \times 13 \times 256$的特征图，最后两个分支相连接变为$13\times 13 \times 1280$的特征图，最后在该特征图上做预测，也就是使用了细粒度特征提高了准确率。

- 关于锚框

7. YOLOv2去掉了YOLO中的全连接层，使用 $1\times 1 \times 125$的卷积直接预测结果，也就是使用卷积来预测锚框。

8. 关于锚框个数的选择使用了维度聚类，作者使用了k-means聚类的方法，在训练集上得到了不同数量的边框，经过实验，作者确定在他的训练数据集上聚类数量是5的时候是一个准确率和召回率的平衡值。所以在自己训练的时候，最好也根据自己的训练集的特点，生成自己的锚框个数。

9. 和Faster RCNN类似，YOLOv2也是将预测锚框的偏移量作为预测结果，Faster R-CNN 在预测锚框的坐标偏移量时，由于没有对偏移量进行约束，每个位置预测的边界框可以落在图片任何位置，会导致模型不稳定，加长训练时间。YOLOv2，对其范围进行了限制，坐标通过sigmoid来将坐标的偏移量映射到0到1之间，取不到0和1。

--- 

### YOLO9000

YOLO9000是在YOLOv2上提出的一种混合目标检测数据集和分类数据集的联合训练方法，可以检测超过9000个类别的模型。

1. YOLO9000使用一种树形结构WordTree，将数据集组织起来，组织之后的数据集有9418个类别，使用one-hot编码的形式，将数据集中每个物体的类别标签组织成一个长度为9418的向量，在WordTree中，从该物体对应的名词到根节点上的路径出现的次对应的类别标号置为1，其余置为0；

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/YOLOv2-04.png)

2. YOLO9000使用的是YOLOv2的结构，锚框个数由5调整为3，那么每个网格就要预测 $3 \times (4+1+9418) = 28269$个值。

3. WordTree中每个节点的子节点都属于同一个子类，分层次的对每个子类中的节点进行一次softmax处理，以得到同义词集合中的每个词的下义词的概率。当需要预测属于某个类别的概率时，需要预测该类别节点的条件概率。即在WordTree上找到该类别名词到根节点的路径，计算路径上每个节点的概率之积。预测时，YOLOv2得到置信度，同时会给出边界框位置以及一个树状概率图，沿着根节点向下，沿着置信度最高的分支向下，直到达到某个阈值，最后到达的节点类别即为预测物体的类别。

---

### YOLOv3

YOLOv3是在YOLOv2的基础上做的一些尝试性改进。

1. 在Darknet-19的基础上引入了残差块，并进一步加深了网络，改进后的网络有53个卷积层，取名为Darknet-53，网络结构如下图所示

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/YOLOv3-01.png)

2. YOLOv3从不同尺度提取特征，相比YOLOv2，YOLOv3提取最后3层特征图，不仅在每个特征图上分别独立做预测，同时通过将小特征图上采样到与大的特征图相同大小，然后与大的特征图拼接做进一步预测。用维度聚类的思想聚类出9种尺度的anchor box，将9种尺度的anchor box均匀的分配给3种尺度的特征图.如下图是在网络结构图的基础上加上多尺度特征提取部分的示意图（以在COCO数据集(80类)上256x256的输入为例）：

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/YOLOv3-03.png?raw=true)

另一个图

![](https://img-blog.csdnimg.cn/20190329210004674.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xpdHQxZQ==,size_16,color_FFFFFF,t_70)

DBL代表卷积+BN+Leaky ReLU


## Q8-增强模型对小目标的检测效果有哪些方法

针对于小目标检测，可以从一下几个角度入手：

- 在模型的设计方面，可以采用特征金字塔来增强网络对多尺度尤其是小尺度特征的感知和处理能力；尽可能提升网络的感受野，使得网络能够更多地利用上下文信息来增强检测效果；同时减少网络总的下采样比例，使最后用于检测的特征分辨率更高。

- 在训练方面，可以提高小物体样本在总体样本中的比例；也可以利用数据增强手段，将图像缩小以生成小物体样本。

- 在计算量允许的范围内，可以尝试使用更大的输入图像尺寸。

摘自《百面深度学习》P240

## Q9-常用的图像增广方法有哪些

- 翻转和裁剪

  比如可以左右翻转和上下翻转，通过随机裁剪让物体以不同的比例出现在图像的不同位置，来降低模型对目标位置的敏感性。

- 变化颜色

  可以从4个方面改变图像的颜色：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
 
摘自《动手学深度学习》P240

## Q10-目标检测如何进行评估

参考https://www.zhihu.com/question/53405779

# 中英文词汇

单步模型（One stage）

候选区域 提议区域（Region Proposal，RP）

两步模型（Two stage）

选择性搜索（Selective Search）

感兴趣区域（Region of Interest，ROI）

非极大值抑制（Non-Maximum Suppression，NMS）
