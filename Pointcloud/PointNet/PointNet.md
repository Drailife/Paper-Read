<center><h1>PointNet</h1></center>
```ad-note
title:Key
Key to our approach is the use of a single symmetric function, m-ax pooling

```

```ad-note
title:Architecture
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202209161537948.png)

![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202209161547067.png)

```
$$f(x_1,x_2,……,x_n) = \gamma(Max_{i=1,.,n}\{h(x_i)\})$$
h():实现升维操作
Max():对称函数
r(): 分类/分割
```ad-question
1. T-Net结构被后续证明是无用的，那么PointNet没有解决旋转不变性的问题，也得到了比较好的正确率
2. Point++ 是否需要有旋转不变性的考虑？
```

