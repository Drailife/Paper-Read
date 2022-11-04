<center><h1>PointNext</h1></center>
## 一些知识
1. 编码器结构：
编码器部分主要由普通卷积层和下采样层将feature map尺寸缩小，使其成为更低维度的表征。目的时尽可能多的提取低级特征和高级特征，从而利用提取到的空间信息和全局信息精确分割。

2. 解码器结构：
解码器部分主要由普通卷积、删除改样层和融合层组成。利用上采样操作逐步恢复空间维度，融合编码过程中提取到的特征，在尽可能减少信息损失的前提下完成同尺寸输入输出。


**sota**实际上就是State of the arts 的缩写，指的是在某一个领域做的Performance最好的model，一般就是指在一些benchmark的数据集上跑分非常高的那些模型

PointNet++  +  PointMlp

## 出发点

论文指出：性能提高的很大一部分是由于改进了培训策略，即**数据增强和优化技术**，以及增加了模型大小，而非架构创新。因此，PointNet++的全部潜力还有待挖掘

实验提出了一组改进的训练策略，显著提高了 PointNet++ 的性能： 在不改变架构的情况下，PointNet++ 在 ScanObjectNN 对象分类上的整体准确率（OA）可以从 77.9% 提高到 86.1%

## 工作

1. 对点云领域的训练策略进行系统研究 
2. PointNeXt is scalable， faster than SOTA

```ad-note
title:PointNet++ SA改进
\\collapse:true
发现半径是特定于数据集的，可以对性能产生重大影响
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210041520986.png)
**SA块包括一个用于对输入点进行下采样的子采样层、一个用于查询每个点的邻居的分组层、一组用于提取特征的共享多层感知器（MLP）以及一个用于聚合邻居中的特征的简化层**
$$x_i^{l+1}=R_{j:(i,j)\in N}\{H_{\Phi([x_j^l;\ p_j^l - p_j^l])}\}$$

**解释**：
1. R 是**reduction layer**（e.g. max-pooling） 该还原层从点i的邻域（表示为{${j:(i,j)\in N}$}）
2. $p_i^j \ x_i^l \ x_j^l$ 第$l^th$层的输入坐标，输入特征，以及点 i的第 j个邻居的特征
3. $H_{\Phi}$表示共享的MLPs，它的输入是$x_j^l$和相对坐标在特征纬度的级联


我们发现等式中的相对坐标 Δp = $p_j^l -p_i^l$使网络优化更难，导致性能下降。因此，我们建议通过半径对 ∆p 进行归一化:
$$x_i^{l+1}=R_{j:(i,j)\in N}\{H_{\Phi([x_j^l;\ (p_j^l - p_j^l)/r^l])}\}$$
如果没有归一化，**相对坐标的值会非常小（小于半径）**，需要网络学习更大的权重以应用于 ∆p。这使得优化变得不简单，所提出的归一化还减少了不同阶段之间的 Δp 方差
```


## 结构改进
1. 包括在模型输入处添加了一层额外的MLP，
2. 用于缩放模型架构的Inverted Residual MLP (InvResMLP)模块
3. **decoder**部分使用与encoder对称的channel size。


**我们发现无论是使用更多的SA模块还是使用更大的channel size都不会显着提高准确性，却反而导致thoughput显著下降**。这主要是梯度消失和过度拟合导致的
论文提出**Inverted Residual MLP (InvResMLP）模块以实现高效实用的模型缩放**

0.  添加了一个stem MLP，即在体系结构开始处插入的附加MLP层，以将输入点云映射到更高的维度。
1. 用于缩放模型架构的Inverted Residual MLP (InvResMLP)模块，SA由2个加到了4个。
2. 发现无论是使用更多的SA模块还是使用更大的channel size都不会显著提高准确性，为了缓解梯度消失问题（尤其是当网络更深时），在模块的输入和输出之间添加了**残差连接**。
3. 引入**inverted bottleneck**的设计将第二个 MLP 的输出通道扩展了 4 倍，以提高特征提取的能力

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210041143960.png)
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210041540070.png)


 我们将 stem MLP 的channel大小表示为 C，C越大，网络的宽度越大。
 将 InvResMLP 模块的数量表示为 B。 B越大，网络的深度越大。
 **通过改变C和B的值，从而实现PointNeXt在宽度和深度层面的缩放** 
 
+ PointNeXt-S: C = 32, B = 0
+ PointNeXt-B: C = 32, B = (1, 2, 1, 1)
+ PointNeXt-L: C = 32, B = (2, 4, 2, 2)
+ PointNeXt-XL: C = 64, B = (3, 6, 3, 3)

## 效果
**在所有研究的数据集上，PointNeXt超越了现有的SOTA方法，并且在性能和效率上都有很好的表现。**
```ad-note
title:ScanObjectNN
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210041531278.png)

```
```ad-note
title:ModelNet40
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210041534085.png)
```
```ad-note
title:优化策略
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210042026926.png)
 On  ScanObjectNN
```
