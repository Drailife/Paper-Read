## 工作
1. 基于坐标的输入嵌入模块 —— 解决点云的无序性
2. 优化 offer-attention 模块 —— 优化之前Transformer的self-attention层，使用offset。把点云看作图、把浮点数矩阵。看作attention map
		A. 相对坐标更为可靠
		B. 图卷积学习中，Laplacian 矩阵更为有效
3. 邻近嵌入模块 —— 加强局部的效果，而不只是考虑单点无关联情况
4. 
## PCT archiecture
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210281840351.png)

**LBR** combines **Linear, BatchNorm and ReLU layers.**
**LBRD** means LBR followed by a Dropout layer

## Offset-Attention

## Neighbor Embedding for Augmented Local Feature Representation
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210291314699.png)

PCT with point embedding is an effective network for extracting global features. However, it ignore the local neighborhood information which is also essential in point cloud learning.

## Experiments

We now evaluate the performance of 

**na ̈ıve PCT** (NPCT, with point embedding and self-attention), 
**simple PCT**:  (SPCT, with point embedding and offset-attention)
**full PCT**:  (with neighbor embedding and offset-attention) on two public datasets, ModelNet40 [32] and ShapeNet [37], giving a comprehensive comparison with other methods.

# Code
## Input Embedding
```python
# LBR  
xyz = x.permute(0, 2, 1) # [B 3 npoints]  
batch_size, _, _ = x.size()  
x = F.relu(self.bn1(self.conv1(x)))  # [B 64 npoints]  
# B, D, N  
x = F.relu(self.bn2(self.conv2(x)))  # [B 64 npoints]  
x = x.permute(0, 2, 1)           # [B npoints 64]  
# SG -> 最远点采样(512) + KNN(32)  
# 先从原始的点云数据中采取512个点 knn32  
new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)           
feature_0 = self.gather_local_0(new_feature)  
# print(feature_0.shape)  
feature = feature_0.permute(0, 2, 1)  
# 从上述采取的512个点中采取256个点 knn 32new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)   
feature_1 = self.gather_local_1(new_feature)
```
## Sample_and_group
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210311446223.png)

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210292127015.png)
