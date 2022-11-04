# A Transformer-Based Capsule Network for 3D Part–Whole Relationship Learning

# 灵感来源
在本文中，我们特别设计了一种新的神经网络来探索3D模型的局部到全局的认知以及3D空间中结构上下文特征的聚集，灵感来自于Transformer最近在自然语言处理(NLP)方面的成功以及在图像分析任务(如图像分类和目标检测)方面的巨大进步。

# 方法
我们提出局部形状标记来编码局部几何信息。On this basis, we design a shape-Transformer-based capsule routing algorithm。
# Dataset
1. SHREC10
2. SHREC15
3. ModelNet40

it is more important to be able to learn the relationship between its local shape and the whole for the model

# Transformer 的优势
These Transformer architectures do not have the structural induction bias provided by convolution to the local spatial structure. Instead, they are completely based on flexible attention distribution. This mechanism enables us to quickly establish the relationship between each local patch

# 贡献
+ 3D shape Transformer.  我们提出了一种基于局部形状表示的新型自注意力计算方法。它允许一种类似于标准 1D selfattention 的机制，以网格模型表面的局部形状作为标记，并为其设计匹配相似性度量。因此，适用于 NLP 的著名 1D Transformer 可以适应 3D 网格任务
+ Multi-head shape attention layer. 我们提出了一种多头形状注意力机制来形成多个子空间，使模型能够关注信息的不同方面。这扩大了底层局部形状之间组合的可能性，使模型学习到的局部组合信息更加准确。
+ Vector representation。基于 3D 网格数据，我们提出了一种新的初级胶囊构造方法来提高胶囊网络的性能
+ 3D vector-type network 我们构造了一种新的矢量型网格跨囊神经网络，并将其应用于三维变形模型的识别。实验表明，与其他方法相比，该网络能够尊重模型本身的几何特征，具有更好的分类效果和学习能力

