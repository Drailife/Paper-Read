自注意力因为它本质上是一个集合运算符：位置信息作为作为集合处理的元素的属性提供 ， 3D 点云本质上是具有位置属性的点集，因此自注意力机制似乎特别适合此类数据。

**架构**

```ad-important
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210162109242.png)
```

## 1. Point Transformer Layer

### Based on **vector self-attention**

**vector self-attention:**
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210162107183.png" height=80></div>
β is a relation function (e.g., subtraction) and $\gamma$ is a mapping function (e.g., an MLP)

We use the **subtraction relation** and add a **position encoding δ** to both the attention vector $\gamma$ and the transformed features α
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210162010026.png" height=80></div>

参数说明：
+  **φ, ψ, and α** are pointwise feature transformations(逐点特征变换), such as linear projections(线性投影) or MLPs
+  **ρ** is a normalization function such as softmax
+  The mapping function **γ** is **an MLP with two linear layers and one ReLU nonlinearity**
psai
φ(xi)：把中心点的特征向量xi送到Query里面。
ψ(xj)：把每个邻居点的特征向量xj送到Key里面。
φ(xi)−ψ(xj)：计算中心点和邻居点的关系，类比于原始自注意力中的QK的点乘，只不过这里换成了减法。
γ(φ(xi)−ψ(xj)+δ)：γ()是一个MLP, 经过一个MLP之后，得到中心点和邻居的权重。
ρ(γ(φ(xi)−ψ(xj)+δ))：将γ(φ(xi)−ψ(xj)+δ)权重分数图进行Softmax归一化。
α(xj)：把每个邻居点的特征向量xj送到Value里面。
最后输出这个点的包含邻居关系的特征向量yi。

对点云做这么一套操作，即每个点都有特征向量yi，包含和邻居的关系的向量表示。

（注意xi和xj是点的特征向量，而不是坐标点xyz）

Here the subset $\mathscr{X} (i) ⊆ \mathscr{X}$ is a set of points in a local neighborhood (specifically, k nearest -neighbors) of $x_i$  *论文指出k 设置为 16*

在每个数据点周围的本地邻域内应用自我注意力机制
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210162026585.png)


### Position Encoding

**Position encoding plays an important role in selfattention, allowing the operator to adapt to local structure in the data**.

序列和图像网格的标准位置编码方案是手工制作的，例如基于正弦和余弦函数或归一化范围值。在3D点云处理中，3D点坐标本身是位置编码的自然候选对象。我们通过引入**可训练的参数化位置编码**来超越这一点(**We go beyond this by introducing trainable, parameterized position encoding**)

**position encoding function δ**
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210162031913.png)
参数说明：
+ $p_i$ and $p_j$ are the 3D point coordinates for points i and j
+ The encoding function θ is an MLP with **two linear layers and one ReLU nonlinearity**

### Point Transformer Block

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210162048668.png)
linear projections that can reduce dimensionality and accelerate processing(可以降低维度和加速处理的线性投影)

The point transformer block **facilitates information exchange between these localized feature vectors**(局部特征向量), producing **new feature vectors** for all data points as its output. The information aggregation adapts both to the **content of the feature vectors and their layout in 3D**(既适应特征向量的内容，又适应其三维布局)


## 3. Network Architecture

用于语义分割和分类的点转换网络中的特征编码器有五个阶段，分别对逐步下采样的点集进行操作。各阶段的下采样率为[1,4,4,4,4]，因此各阶段产生的点集的基数为[N, N/4, N/16, N/64, N/256]，其中N为输入点的个数

+ Consecutive stages are connected by transition modules: **transition down for feature encoding** and **transition up for feature decoding**.

### Transition Down
+ We perform farthest **point sampling** in P1 to identify a well-spread subset P2 ⊂ P1 with the requisite cardinality.(确定一个分布良好的(具有必要的基数的)子集P2⊂P1)

+ Each input feature goes through a **linear transformation**, followed by **batch normalization and ReLU**, followed by **max pooling** onto each point in P2 from its k neighbors in P1(每个输入特征都经过线性变换，然后是批量归一化和 ReLU，然后是最大池化)
确定局部区域 --> mlp --> local max pooling
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210162109242.png)

### Transition UP

对于语义分割等密集预测任务，我们采用 U-net 设计，其中上述编码器与对称解码器耦合。解码器中的连续级由向上转换模块连接。它们的主要功能是将下采样输入点集 P2 的特征映射到其超集 P1 ⊃ P2。为此，每个输入点特征都经过一个线性层处理，然后进行批量归一化和 ReLU，然后通过**三线性插值**将特征映射到更高分辨率的点集 P1 上。这些来自前面解码器阶段的插值特征与来自相应编码器阶段的特征进行汇总，通过跳跃连接提供。


## 实验

**ModelNet40 & ShapeNetPart**
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191411336.png)

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191413957.png)

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191412306.png)
