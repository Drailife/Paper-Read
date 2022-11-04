## 一、函数
1. nn.Parameter：
可以看作是一个类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
```python
self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))  
self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

grouped_points = self.affine_alpha*grouped_points + self.affine_beta
```

2. torch.std
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210121623564.png)
3. torch.topk()
返回列表中最大的n个值
```python
m=torch.arange(0,10)  
             print(m.topk(3))

torch.return_types.topk(
values=tensor([9, 8, 7]),
indices=tensor([9, 8, 7]))
```

## 二、仿射函数
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210121831689.png)

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210121637886.png)
embedding： 
		使得从三维度到64个维度，之后经过仿射函数，-->后续残差网络

简单的深度mlp结构会降低精度和稳定性，从而使模型的健壮性。
原因是局部区域的几何结构稀疏和不规则。不同局部区域之间的不同几何结构可能需要不同的提取器
$\alpha \  \beta$ 是**可学习的**参数，σ是一个标量，它描述了所有本地组和通道之间的特征偏差。
通过这样做，我们将局部点变换为正态分布，同时保持原始的几何属性
{$f_{i,j}$} : 所有的 通过Grouping 操作后的 points数据
$f_i$ 最远点采样得到的质心数据
$f_{i,j}$ 由质心数据 $f_i$ 通过**knn**(注：选取**距离最远**的一组点，也可以使用球查询的方式) 
k 为上述knn查找的点数


可以把中间那一部分看成是 **数据输入**  $\alpha$看成是需要学习的权重，$\beta$ 看成是偏置
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210121821420.png)

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210121548375.png)


使用上述仿射函数，pointmlp在ScanObjectNN上的正确率

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210121549262.png)


将仿射函数改为使用简单的归一化函数
$$\{f_{i,j}\} = \frac{f_{i,j}-f_i}{\sigma}$$
测试条件：
batch_size:	 32
model:	 pointMLP
num_classes:	 15
epoch:	 200
num_points:	 1024
learning_rate:	 0.01
weight_decay:	 0.0001
++++++++++++++++Final results++++++++++++++++++
++  Last Train time: 152 | Last Test time: 13  ++++++++++
++  Best Train loss: 1.068 | Best Test loss: 1.343  +++++++
++  Best Train acc_B: 99.985 | Best Test acc_B: **83.905**  ++     ACC
++  Best Train acc: 99.991 | Best Test acc: **85.531**  ++++++  
++++++++++++++++++++++++++++++++++++++++++++
仿射函数在总正确率上可以增强左右的正确率