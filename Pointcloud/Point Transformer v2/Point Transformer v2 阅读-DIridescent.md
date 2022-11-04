## 工作
1. We propose an effective **grouped vector attention (GVA)** with a novel weight encoding layer that enables efficient information exchange within and among attention groups.

2. We introduce an **improved position encoding schem**e to utilize point cloud coordinates better and further enhance the spatial reasoning ability of the model. 

3. We design the **partition-based(基于分区) pooling strategy** to enable more efficient and spatially betteraligned information aggregation compared to previous methods.

```ad-important
title:point transformer v2
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210181707957.png)
```

+ 效果
	+ Achieves state-of-the-art on several challenging 3D point cloud understanding benchmarks, including 3D point cloud segmentation on ScanNet v2 and S3DIS and 3D point cloud classification on ModelNet40.
+ code
	+ Our code will be available at https://github.com/Gofinge/PointTransformerV2.

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210181707957.png)


## Grouped vector attention

**Scalar attention**
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182144355.png)

**Vector attention**
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182146597.png)

+ Hadamard product. 
+ $\gamma$  is a relation function
+ ω : $R^c → R^c$ is a learnable weight encoding (e.g., MLP)

<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191125749.png" height  =20></div>

**PTv1 attention层的缺点**
PTv1中的向量注意层利用MLPs作为权重编码，将$q,k$的减法关系映射为一个可以调制值向量的各个通道的注意权重向量。然而，随着模型深度和通道数量的增加，权重编码参数的数量也急剧增加，导致严重的过拟合，限制了模型深度。

**论文改进方法**
Our proposed **grouped vector attention** inherits the merits of both **vector attention and multi-head attention** while being more powerful and efficient

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191311857.png)

我们将值向量  $v ∈ R^c$  的通道均匀地划分为 g 个组 (1 ≤ g ≤ c)。权重编码层输出具有 g 个通道的**grouped attention vector**。同一 attention group 的 $v$  的通道共享来自attention group的相同标量注意权重。

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182136024.png)
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191125749.png" height  =20></div>

**Grouped linear**
受 MSA 权重编码功能的启发，我们设计了分组线性层 ζ(r) : $R^c  → R^g$，其中不同组的输入向量独立地用不同的参数投影。分组线性进一步减少了权重编码函数中的参数数量。我们最终采用的分组权重编码函数由分组线性层、归一化层、激活层和一个全连接层组成，以允许组间信息交换。

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182302343.png)

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182253666.png)
<font color='blue'>蓝线</font>表示作用于输入关系标量的可学习参数
<font color='red'>红线</font>表示乘以输入关系标量。
<font color='orange'>橙色</font>线条标识哪个值要素受输入标量权重的影响。

## Position Encoding Multipler

```ad-important
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210181707957.png)
```

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182309699.png)
$δ_{mul}, δ_{bias}$ : $R^d → R^d$ are two MLP position encoding functions, which take relative positions as input. 
Position encoding multiplier compliments group vector attention to achieve a good balance of network capacity.(位置编码倍增器补充组向量注意，实现网络容量的良好平衡)

```ad-note
title: difference
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210181707957.png)
```

## Partition-based Pooling

基于点的方法所采用的传统基于采样的池化过程使用了采样和查询方法的组合。在采样阶段，使用**最远点采样**或**网格采样**对保留给下一个编码阶段的点进行采样。对于每个采样点，将执行一个**邻居查询**来聚合来自相邻点的信息，在这些基于采样的池化过程中，点的查询集**不是空间对齐**的，因为每个查询集之间的**信息密度和重叠**是不可控制的。为了解决这个问题，我们提出了一种更高效、更有效的基于分区的池化方法。

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182330419.png)



![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182326921.png)

**Unpooling**。
+ 插值解池的常用做法也适用于基于分区的池。
+ 这里我们介绍一种更直接和有效的解池方法。为了将融合点集M’解池回M，在池化过程中记录M中的点位置，我们只需要得到M中每个点的特征。在池化阶段，借助基于网格的划分[M1，M2，…，Mn‘]，我们可以将点特征映射到同一子集的所有点。
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182333826.png)
要将融合点集 M' 解池回 M，M 中的点位置是池化过程中记录的，我们只需要获取 M 中每个点的特征

## Network Architecture

**Backbone structure**
我们采用了具有跳跃连接的 U-Net 架构。编码器和解码器有四个阶段，块深度分别为 [2, 2, 6, 2] 和 [1, 1, 1, 1]。四个阶段的网格大小乘数为 [x3.0, x2.5, x2.5, x2.5]，表示在前一个池化阶段的扩展率
初始特征维度是 48，我们首先将输入通道嵌入到这个数字中，其中包含 6 个注意力组的基本块。然后，每次进入下一个编码阶段时，我们将这个特征维度和注意力组加倍。对于四个编码阶段，特征维度为[96,192,384,384]，对应的注意力组为[12,24,48,48]。 

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182342740.png)

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182352686.png)


## 实验

### ScanNet v2 & S3DIS
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182343188.png)

### ModelNet40
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182345998.png)
### Pooling method
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182346186.png)
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210182349681.png)

对于我们通过网格实现的基于分区的池化，基本网格大小为 0.02 米，这与数据预处理期间的体素化网格大小相同。网格大小乘数是前一个池化阶段的网格大小扩展比。例如[×4.0,×2.0,×2.0,×2.0]表示网格大小分别为：[0.08,0.16,0.32,0.64]米。