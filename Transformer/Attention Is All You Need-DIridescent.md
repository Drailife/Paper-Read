**自回归**：输入又是  输出
**点积(dot-product)** 矩阵乘法
**哈达玛积(Hadamard product)**  $A_{m*n}\ hadamard \ product \ B_{m*n}=C_{m*n}$
资料来源：[Self-Attention自注意力机制 讲解](http://t.csdn.cn/njtqI)

## 0.最原始的self-attention

Input Embedding: 将输入中的每个词表示成一个向量

---

self-attention 最原始的形式：$$Softmax(XX^T)X$$
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210190951906.png)

向量内积的意义：**表征两个向量的夹角，表征一个向量在另一个向量上的投影**
投影值大和小又有什么意义呢？
**投影值大，意味两个向量相关度高**，**如果两个向量夹角是90°，那么这两个向量线性无关，完全没有相关性**，词向量之间相关度高表示什么？是不是在一定程度上（不是完全）表示，**在关注词A的时候，应给予词B更多的关注？**
**经过softmax** 当我们关注“早”这个字的时候，我们应该分配0.4的注意力（attention）给它本身，剩下0.4的注意力给“上”，最后的0.2的注意力给“好”

接下来利用相似度来提取sequence的信息

 得到一个新的value,然后再与相似度进行加权求和


## 1.架构

```ad-important
title: transformer
collapse:true
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210151838970.png" height=600></div>
```

## 2.Self-Attention

<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210151954951.png" height=400></div>
输出b考虑到了**所有的输入**

如何产生输出 **b** ?
我们应用线性投影或 MLP 将点特征 fi 投影到 query q、key k 和 value v 的特征向量
1. 首先找到序列中相关的向量, 设两个向量之间的关联为  $\alpha$  求法如下: 1.将输入的两个向量分别乘以两个矩阵 $W^q \ 和 \ W^k$  得到 **q** 和 **k** 两个向量，然后将他们做**dot-product**运算得到 $\alpha$

```ad-note
title: Figure 1
collapse:true
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210152004858.png" height =300></div>
```

2. 求序列中所有向量相关性(**包括自己**)的方法如下 ，计算出$a_1$ 和所有向量的相关性之后然后输入到 **Normalization** (softmax、relu)  

```ad-note
title:Figure 2
collapse:true
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210152013542.png" height =320></div>
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210152044704.png" height =160></div>
```

3. 根据得到的 $\alpha$ 来提取序列中的重要的信息 将所有的向量乘以 $W^v$ 得到 $v$  然后用乘以对应的$\alpha$ 并相加得到对应的 $b$  

```ad-note
title: Figure 3
collapse:true
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210152024869.png" height =320></div>
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210152047249.png" height =180></div>
```

**$W^q \ W^k  \ W^v$ 需要通过数据学习到**

## 3.Multi-head Self-attention
两个头示例
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210152053160.png" height =320></div>

```ad-note
collapse:true
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210161131431.png)
```


## 4.Transformer 中的Attention
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191021761.png)

$$Attention(Q, K, V ) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$


($d_k$ 是键向量的维度，论文中为64):
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191025374.png)
当d变得很大时，$q\cdot k$ 中的元素的方差也会变得很大，如果 $q \cdot k$ 中的元素方差很大，那么$Softmax(A)$的分布会趋于陡峭（分布方差大，分布集中在绝对值大的区域）总结一下就是$Softmax(A)$的分布会和$d_k$有关。因此$A$每个元素除以 $d_k$ 后，方差又变为了1。

## 5.Positional Encoding

问题：上述的 *self-attention* 没有考虑到位置信息 
使用不同频率的正弦和余弦函数：

<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210152141744.png" height =100></div>
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210152101994.png" height =220></div>


## 6. 编码器
each layer has two sub-layers. 
+ **multi-head self-attention mechanism**,
- **positionwise fully connected feed-forward network**
	+ *This consists of two linear transformations with a ReLU activation in between*

The output of each sub-layer is **LayerNorm(x + Sublayer(x))**

残差 注意力机制 FC Norm

```ad-important
title: transformer
collapse:true
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210151838970.png" height=600></div>
```
<div align='center'><img src='https://drailife.oss-cn-beijing.aliyuncs.com/img/202210152205500.png' height=300></div>
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191029795.png)

为了促进这些剩余连接，模型中的所有子层以及嵌入层产生维度$d_{Model}$=512的输出。
内核大小为 1 的两个卷积。输入和输出的维数为 $d_model = 512$，内层的维数为 f = 2048
### Layer Normalization
LN是在同一个样本中不同神经元之间进行归一化，而BN是在同一个batch中不同样本之间的同一位置的神经元之间进行归一化。
BN是对于相同的维度进行归一化，NLP中输入的都是词向量，一个N维的词向量，单独去分析它的每一维是没有意义地，在每一维上进行归一化也是适合地，因此这里选用的是LN。

```ad-important
title: Batch Norm
collapse:true
<div align="center"><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210161234382.png" height =200></div>```
```
BatchNorm把一个batch中同一通道的所有特征视为一个分布（有几个通道就有几个分布），并将其标准化。这意味着:
-   不同图片的的同一通道的相对关系是保留的，即不同图片的同一通达的特征是可以比较的
```ad-important
title: Layer Norm
collapse:true
<div align="center"><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210161218253.png" height =800></div>
```
LayerNorm把一个样本的所有词义向量视为一个分布（有几个句子就有几个分布），并将其标准化：
+ 同一句子中词义向量（上图中的V1, V2, …, VL）的相对大小是保留的，或者也可以说LayerNorm不改变词义向量的方向，只改变它的模
## 7.解码器

编码器通过处理输入序列开启工作，顶端编码器的输出之后会变转化为一个包含向量 K 和 V的注意力向量集
在解码器中，自注意力层只被允许处理输出序列中更靠前的那些位置。在softmax步骤前，它会把后面的位置给隐去（把它们设为-inf）

### Mask self-attention

```ad-important
title: transformer
collapse:true
<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210151838970.png" height=600></div>
```

<div align='center'><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210161150611.png" height =270><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191037097.png" height=270><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191043784.png" height =290><img src="https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191046681.png
" height =280></div>



![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202210191039854.png)







