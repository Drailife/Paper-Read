<center><h1>PointNet++学习与详解</h1></center>
## PointNet回顾
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202209161537948.png)
<center><font size = 2> Figure PointNet 结构</font></center>

1. 通过MLP将每一个点映射到高维的特征
2. 通过一个对称函数(Maxpooling)得到特征
3. 通过全连接层完成k分类
<img src = 'https://drailife.oss-cn-beijing.aliyuncs.com/img/202209221803019.png' width="300">
## PointNet的不足
1. 在后续论文中证明了T-Net部分并没有太大作用，即没能解决PointCloud旋转不变性的问题
2. 在PointNet中，只获得了每个点的特征和全局特征，忽略了PointCloud的局部特征

## PointNet++简介

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202209212107254.png)
<center><font size = 2>PointNet++ Architecture</font></center>
## SetAbstraction
主要包括三个部分$Sampling \ Layer  \ \ Grouping \ Layer \ \ PointNet \ Layer$



