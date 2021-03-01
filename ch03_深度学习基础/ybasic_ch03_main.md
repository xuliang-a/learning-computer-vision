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

sigmoid和tanh激活函数的区别：

   - 当输入较大或较小时，输出几乎是平滑的并且梯度较小，这不利于权重更新。二者的区别在于输出间隔，tanh 的输出间隔为 1，并且整个函数以 0 为中心，比 sigmoid 函数更好；
   - 在 tanh 图中，负输入将被强映射为负，而零输入被映射为接近零；
   - 在一般的二元分类问题中，tanh 函数常用于隐藏层，而 sigmoid 函数常用于输出层

**3. Relu激活函数**

函数表达式：$ f(x) = max(0, x)$，其值域为$ [0,+∞) $。

函数图像为：

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch03_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/img/ch3/3-32.png)

**4. Leak Relu 激活函数**

函数表达式：$ f(x) =  \left\{
   \begin{aligned}
   ax, \quad x<0 \\
   x, \quad x>0
   \end{aligned}
   \right. $，其值域为$ (-∞,+∞) $。

函数图像为：

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9PrS2jqcgp04sYOZNhbMVWL4kB5fRTec1zZk4saEztrGYnvCAgm8cZG4AoWbriaD4GRGtnMgY0DTg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**5. SoftPlus 激活函数**

函数表达式：$f(x) = ln( 1 + e^x) $，其值域为$ (0,+∞) $。

函数图像为：

![](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch03_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/img/ch3/3-30.png)

**6. softmax 函数**

函数表达式：$ \sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}} $，其值域为$ (0,+∞) $。

函数图像为：

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9PrS2jqcgp04sYOZNhbMVWmk9OHeNrtt74bsmaDV8l2kXic1Xlxxcv1LvFwuQILPKfm1e3jtDsibNw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Softmax 多用于多分类神经网络输出。

## Q7：为什么要归一化

## Q8：归一化的类型

## Q9：dropout系列问题

## Q9：深度学习中常用的数据增强方法


