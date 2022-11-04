*Time: [[2022-10-03]]*
*Author: DIridescent*

# RETHINKING NETWORK DESIGN AND LOCAL GEOM-ETRY IN POINT CLOUD

Our PointMLP also follows the design philosophy of PointNet++ but explores a simpler yet much deeper network architecture —— 和PointNet/PointNet++具有一定的相似性,但是更加的简单却深入

近些年来的工作
**DeepGCNs: Making GCNs Go as Deep as CNNs.**
PointNet/PointNet++
RSCNN
RPNet
PointConv
Point Transformer
受到ResNet网络的启发

 ## 0. 前置知识
### 感受野
在卷积神经网络中，感受野（Receptive Field）的定义是卷积神经网络每一层输出的特征图（feature map）上的像素点在输入图片上映射的区域大小。再通俗点的解释是，特征图上的一个点对应输入图上的区域
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210041037848.png)
举例说明： 两层3\*3的的卷积核的感受野为5\*5

## 1. 出发点
作者认为详细的局部几何信息可能不是点云分析的关键，
很多网络对点云的处理依赖于**复杂的局部提取器**，作者认为复杂的提取程序存在一些缺点：
1. 由于繁琐的计算和内存访问开销，这阻碍了应用程序在自然场景中的效率。
2. 对于这方面的研究使得已经能够很好地描述局部几何性质。
引入了一个纯粹的残差MLP网络，称为PointMLP，它没有集成复杂的局部几何提取器，但仍然具有很强的竞争力。

## 2. 论文目标
we aim at the ambitious goal of building a deep network for point cloud analysis using **only residual feed-forward MLPs**, **without** any delicate local feature explorations

## 3. PointMLP 框架
$$g_i=\Phi_{pos}(A(\Phi_{Pre}(f_{i,j})|j=1,...,K))$$
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210032133815.png)

和**ResNet**网络挺像
其中 两个$\Phi$ 是 **residual point MLP blocks**, 共享权重的多层感知机
$\Phi_{pre}$：learn shared weights from a **local region** 
$\Phi_{pos}$ : leveraged to extract deep aggregated features(从前者聚合之后的特诊里面提取深度聚合特征)
MLP由FC、归一化和激活层组合而成（重复两次）
$A(.)$ 表示聚合函数  -> MaxPooling
上式描述了PointMLP的一个阶段。对于层次结构和深度网络，递归地重复 s 个阶段的操作

PointMLP 首先使用几何仿射模块对局部点进行仿射变换，然后通过几个残差 MLP 模块 (Residual Point Block) 来提取深层的特征。注意此时的局部区域中仍包含多个点，作者通过一个简单的聚合器 (使用的是 max-pooling) 来将局部多个点聚合成一个点以描述局部信息， 并且再次使用残差 MLP 模块来提取特征。
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210032354283.png)

PointMLP 通过重复多个阶段 (每个阶段中通道数翻倍) 逐步扩大感受野，并模拟完整的点云几何信息。

在实验中用了四个阶段 两个$\Phi$ 操作中都有两个残差块。采用**K近邻算法**(kNN)来选择近邻点，并设置K为24.

## 4. PointMLP-elite
更加轻量级

为了进一步提高推理速度、减轻模型大小，该研究减少了每个阶段的通道数以及残差 MLP 模块的个数，并在残差 MLP 模块中引入了瓶颈 (bottleneck) 结构。

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210032355381.png)

速度更快，正确率也很高


## 5. 几何仿射模块

在 $\Phi$ 中可以简单地通过考虑更多的阶段或堆积更多的块来增加深度，但我们注意到，**简单的深度MLP结构会降低精度和稳定性**，使模型的鲁棒性降低。这可能是由于**局部区域几何结构稀疏、不规则造成的**。不同局部区域之间不同的几何结构可能需要不同的提取器，但共享剩余mlp很难实现这一点。我们具体化了这种直觉，并开发了一个轻量级的几何仿射模块来解决这个问题。

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210032209580.png)
论文指出 局部几何仿射模块，将局部点转换为正态分布，可以自适应地变换局部区域中的点特征，保持原来的几何性质(不是太懂)


![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210032322818.png)
在ScanObjectNN测试集上使用/不使用几何仿射模块的PointMLP的四个运行结果

研究者希望这个新颖的想法能够激发大家重新思考点云中的网络设计和局部几何操作

## 6. 分类准确性

1. 94.5%, on the ModelNet40
2. we outperform related works by3.3% accuracy on the real-world ScanObjectNN dataset, with a significantly higher inference speed.

```ad-note
title:ModelNet40
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210040009725.png)
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210040011701.png)
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210040011678.png)


```
```ad-note
title:ScanobjectNN
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210032359143.png)

```
```ad-note
title:ShapeNet
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210040005323.png)
ShapeNetPart数据集上的部分分割结果
```

## 7. 优势
  从结果来看：
	  1. 提高点云分类准确率
	  2. 提高了推理速度
  
  从网络的结构来看
	 1.  只利用mlp进行处理，满足点云的序列不变性特征
	 2.  网络引入残差结构，使得PointMLP可以很容易地扩展到几十层，得到深度特征表示
	 3.  由于没有包含复杂的提取器，主要操作只是高度优化的前馈mlp，即使我们引入更多的层，我们的PointMLP仍然可以高效地执行

<img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210032058348.png" height=300>
<center><font size =2>on ModelNet40</font></center>

## 想法
0. Point MLP的成功说明了其作者给我们提供的新想法是可行的，我们可以在数据增强方面（旋转等）下功夫，让网络学习更多的特征
1. 可以利用PointMLP与卷积进行叠加，但加上卷积之后时候还需要单独考虑排列不变性？

 