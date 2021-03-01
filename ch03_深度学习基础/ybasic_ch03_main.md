# 深度学习基础理论
## Q1：深度学习与机器学习的区别

## Q2：加深网络层数的意义

## Q3：欠拟合，过拟合是什么，如何解决

**欠拟合**指的是模型在训练和预测时的表现都不好的情况。

**过拟合**是指模型在训练集上的表现很好，但在测试集和新数据上表现较差的情况。

（1）降低欠拟合的方法：

   - **添加新特征**。  

   当特征不足或者现有特征与样本标签的相关性不强时，模型容易出现欠拟合。

   - **增加模型复杂度**。  

   简单模型的学习能力较差，通过增加模型的复杂度可以使模型拥有更强的拟合能力。例如，在线性模型中增加高    次项，在神经网络模型中增加网络层数或神经元个数。

   - **减小正则化系数**。    

   正则化用来防止过拟合，但当模型出现欠拟合现象时，需要有针对性地减少正则化系数。

（2）降低过拟合的方法：

   - **获取更多的训练数据**。 

   让模型学习到更多更有效的特征，减少噪声的影响。

   - **降低模型复杂度**。  

   适当的降低模型复杂度可以避免拟合过多的采样噪声。例如，在神经网络模型中减少网络层数、神经元个数等；    在决策树模型中降低树的深度，进行剪枝。

   - **正则化方法**。  

   正则化可以用来防止过拟合，在损失函数后面加上一个正则化项，避免权重过大带来的过拟合。

   - **集成学习方法**。  

   把多个模型集成在一起，来降低单一模型的过拟合风险。

## Q4：如何解决梯度消失和梯度爆炸

**梯度消失**
   
   根据链式法则，如果每一层神经元对上一层的输出的偏导乘上权重结果都小于1的话，那么即使这个结果是      0.99，在经过足够多层传播之后，误差对输入层的偏导会趋于0。这种情况会导致靠近输入层的隐含层神经元    调整极小。

**梯度爆炸**

   根据链式法则，如果每一层神经元对上一层的输出的偏导乘上权重结果都大于1的话，在经过足够多层传播之    后，误差对输入层的偏导会趋于无穷大。这种情况又会导致靠近输入层的隐含层神经元调整变动极大。

**梯度消失的解决方案：**

   - 使用relu、leak relu和elu等激活函数。这些激活函数设计思想是将激活函数的导数限制在一定范围内；

   - 使用LSTM、GRU。通过控制门来解决相隔较远的数据之间的数据传递问题；

   - 残差结构。残差的捷径（shortcut）部分可以很轻松的构建几百层，一千多层的网络，而不用担心梯度消失过快的问题；

   - 预训练加微调。预训练（pre-training）的基本思想是每次训练一层隐节点，训练时将上一层隐节点的输      出作为输入，而本层隐节点的输出作为下一层隐节点的输入；在预训练完成后，再对整个网络进行“微调”（fine-tunning）

**梯度爆炸的解决方案：**
  
   - 梯度剪切（裁剪）。设计思想是设置一个梯度剪切阈值，然后更新梯度的时候，如果梯度超过这个阈值，那      么就将其强制限制在这个范围之内；

   - 权重正则化。常见的是l1正则化和l2正则化，正则化是通过对网络权重做正则限制过拟合；

   - 使用relu、leak relu和elu等激活函数。这些激活函数设计思想是将激活函数的导数限制在一定范围内；

   - 批量归一化。网络的反向传播式子中有权重的存在，所以权重的大小会导致梯度的消失和爆炸，batchnorm就是通过对每一层的输出规范为均值和方差一致的方法，消除了权重带来的放大缩小的影响，进而解决梯度消失和爆炸的问题。



## Q4：如何解决网络退化

## Q4：为什么需要非线性激活函数

## Q5：如何选择激活函数

## Q6：常见的激活函数

**1. sigmoid激活函数**  

函数表达式：$f(x) = \frac {1} {1 + e^{-x}}$，其值域为(0,1)。

函数图像为：
![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9PrS2jqcgp04sYOZNhbMVWe5nFPYqgmwEMyFYMqhWsHUjkwrJLPpeTvVRTGOF54Q7sgCInu1ME0w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

导数为：$f^{'}(x) = \frac {1}{1+e^{-x}}\left( 1- \frac{1}{1+e^{-x}} \right)=f(x)(1-f(x))$

1.1 什么情况下适合使用 Sigmoid 激活函数

   - 对每个神经元的输出进行了归一化。由于Sigmoid 函数的输出值限定在 0 到 1 之间，相当于对输出进行了归一化处理；

   - 用于将预测概率作为输出的模型。由于概率的取值范围是 0 到 1，因此 Sigmoid 函数非常合适；

   - 梯度平滑。避免「跳跃」的输出值；

   - 函数是可微的。这意味着可以找到任意两个点的 sigmoid 曲线的斜率；

1.2 Sigmoid 激活函数有哪些缺点

   - Sigmoid 函数执行指数运算，计算机运行得较慢

**2. tanh激活函数**

函数表达式：$f(x) = \frac {e^x - e^{-x}} {e^x + e^{-x}}$，其值域为(-1,1)。

函数图像为：

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9PrS2jqcgp04sYOZNhbMVWtz9Dn7SzuKsicEnDnGEegkH3Wlt5FE2ybkyXdW6m363azzMA0ibbraPA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

导数为：$f(x) = - (tanh(x))^2$

导数的图像为：

![](https://img-blog.csdnimg.cn/20200407212806764.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2tlZXBwcmFjdGljZQ==,size_16,color_FFFFFF,t_70)

2.1 sigmoid和tanh激活函数的区别：

   - 当输入较大或较小时，输出几乎是平滑的并且梯度较小，这不利于权重更新。二者的区别在于输出间隔，      tanh 的输出间隔为 1，并且整个函数以 0 为中心，比 sigmoid 函数更好；
  
   - 在 tanh 图中，负输入将被强映射为负，而零输入被映射为接近零；
   
   - 在一般的二元分类问题中，tanh 函数常用于隐藏层，而 sigmoid 函数常用于输出层

**3. Relu激活函数**

函数表达式：$ f(x) = max(0, x)$，其值域为$ [0,+∞) $。

函数图像为：

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch03_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/img/ch3/3-32.png)

导数图像为：

![](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjIwMTE1NzE5MzMy?x-oss-process=image/format,png)

3.1 relu激活函数的优点

   - 解决了梯度消失、爆炸的问题
   
   - 计算方便，计算速度快
   
   - 加速了网络的训练

3.2 relu激活函数的缺点

   - 由于负数部分恒为0，会导致一些神经元无法激活（可通过设置小学习率部分解决）
   
   - 输出不是以0为中心的

**4. Leak Relu 激活函数**

函数表达式：$ f(x) = max(kx, x) $，其中k是leak系数，一般选择0.01或0.02，其值域为$ (-∞,+∞) $。

函数图像为：

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9PrS2jqcgp04sYOZNhbMVWL4kB5fRTec1zZk4saEztrGYnvCAgm8cZG4AoWbriaD4GRGtnMgY0DTg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

4.1 为什么Leaky ReLU 比 ReLU 好：

   - Leaky ReLU 通过把 x 的非常小的线性分量给予负输入（0.01x）来调整负值的零梯度（zero        gradients）问题；

   - leak 有助于扩大 ReLU 函数的范围，通常 a 的值为 0.01 左右；

Leaky ReLU 的函数范围是（负无穷到正无穷）


**5. elu 激活函数**

函数表达式：![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9PrS2jqcgp04sYOZNhbMVWfguia0LXZjVRfoFS3PuViapJt9gygKnJVgfHDROzB0a8kcO8xQURyAcw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

函数和导数的图像为：

![](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjIwMTM0NjE0MTIx?x-oss-process=image/format,png)

5.1 ELU 具有 ReLU 的所有优点：

   - 没有 Dead ReLU 问题，输出的平均值接近 0，以 0 为中心；

   - ELU 通过减少偏置偏移的影响，使正常梯度更接近于单位自然梯度，从而使均值向零加速学习；

   - ELU 在较小的输入下会饱和至负值，从而减少前向传播的变异和信息。

**6. SoftPlus 激活函数**

函数表达式：$f(x) = ln( 1 + e^x) $，其值域为$ (0,+∞) $。

函数图像为：

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch03_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/img/ch3/3-30.png)

**7. softmax 函数**

函数表达式：![](https://pic3.zhimg.com/50/v2-39eca1f41fe487983f5111f5e5073396_hd.jpg)，其值域为：$ (0,+∞) $。

函数图像为：

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9PrS2jqcgp04sYOZNhbMVWmk9OHeNrtt74bsmaDV8l2kXic1Xlxxcv1LvFwuQILPKfm1e3jtDsibNw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Softmax 多用于多分类神经网络输出。

7.1 softmax激活函数的缺点：

   - 在零点不可微；

   - 负输入的梯度为零，这意味着对于该区域的激活，权重不会在反向传播期间更新，因此会产生永不激活的死      亡神经元

## Q7：为什么要归一化

## Q8：归一化的类型

## Q9：dropout系列问题

## Q9：深度学习中常用的数据增强方法


