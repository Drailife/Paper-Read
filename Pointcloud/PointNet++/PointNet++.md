## PointNet++ 和 PointNet的联系：
Our work can be viewed as an extension of PointNet with added hierarchical structure.
PointNet++ applies PointNet recursively on a nested partitioning of the input set.

## The design of PointNet++ has to address two issues
1. how to generate the partitioning of the point set
2. how to abstract sets of points or local features through a local feature learner.

## PointNet++ Architecture
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202209212107254.png)

The set abstraction level is made of three key layers: *Sampling layer, Grouping layer and PointNet layer*.
1. The Sampling layer selects a set of points from input points, which defines the centroids of local regions.
	+ farthest point sampling (FPS) 
	+ To evenly cover the whole set, the centroids are selected among input point set by a **farthest point sampling (FPS)** algorithm
	+ **FPS实现方式如下**：随机选择一个点作为初始点作为已选择采样点，计算未选择采样点集中每个点与已选择采样点集之间的距离distance，将距离最大的那个点加入已选择采样点集，然后更新distance，一直循环迭代下去，直至获得了目标数量的采样点
	+ <https://zhuanlan.zhihu.com/p/266324173>

3. Grouping layer then constructs local region sets by finding "neighboring" points around the centroids. 
	+  input :point set of size $N × (d + C)$
	+  groups of point sets of size $N^` × K × (d + C)$
	
3. PointNet layer uses a mini-PointNet to encode local region patterns into feature vectors
	+ $N ′$ local regions of points with data size $N ′ ×K ×(d+C)$
	+ Output data size is $N ′ × (d + C′)$   Each local region in the output is abstracted by its centroid and local feature that encodes the centroid's neighborhood

## Non-uniform Sampling Density in point cloud
```ad-note
collapse:true
title:Non-uniform Sampling Density in point cloud
1. Multi-scale grouping (MSG)
+ 应用具有不同尺度的分组层，然后根据PointNet提取每个尺度的特征。将不同尺度的特征串联起来，形成多尺度特征
2. Multi-resolution grouping (MRG)
	+ The MSG approach above is computationally expensive
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202209221043720.png)
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202209221046400.png)
SSG: single scale grouping
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202209221059932.png)
When the density of a local region is low, the first vector may be less reliable than the second vector
Compared with MSG, this method is computationally more efficient since we avoids the feature extraction in large scale neighborhoods at lowest levels
```



```ad-summary
collapse:true
1.  point-wise MLP，仅仅是对每个点表征，对局部结构信息整合能力太弱 --> **PointNet++的改进：sampling和grouping整合局部邻域**
2.  global feature直接由max pooling获得，无论是对分类还是对分割任务，都会造成巨大的信息损失 --> **PointNet++的改进：hierarchical feature learning framework，通过多个set abstraction逐级降采样，获得不同规模不同层次的local-global feature**
3.  分割任务的全局特征global feature是直接复制与local feature拼接，生成discriminative feature能力有限 --> **PointNet++的改进：分割任务设计了encoder-decoder结构，先降采样再上采样，使用skip connection将对应层的local-global feature拼接**

```

```ad-info
title: 感受野的概念？
collapse:true
待会再学
```

数据归一化
```python
def pc_normalize(pc):  
    l = pc.shape[0]  # [B, N, C]
    centroid = np.mean(pc, axis=0)  
    pc = pc - centroid  
    m = np.max(np.sqrt(np.sum(pc**2, axis=1))) # 球半径  
    pc = pc / m  # 得到球半径为一的点云  
    return pc
```
最远点采样寻找质心
球查询寻找每个区域内的采样点
MSG
```ad-todo
title:Tasks
- [x] PointNet结构的理解
- [x] 复现PoinitNet的分类部分
- [x] PointNet++结构的理解
- [ ] PointNet++复现（看代码理解过程中）
```

```ad-question
1. 归一化方程，以centroid为中心，半径为1
2. 最远点采样的具体算法
3. Point++ 是否需要有旋转不变性的考虑？
4. MRG和MSG在空间上如何理解
```


