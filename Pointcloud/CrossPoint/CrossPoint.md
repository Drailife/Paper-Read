<center><h1>CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding</h1></center>
 ```ad-summary
title:Approach

&emsp;&emsp;Our approach in particular, follows a joint objective of embedding the augmented versions of same point cloud close together in the feature space, while preserving the 3D-2D correspondence between them and the rendered 2D image of the original 3D point cloud
![](https://drailife.oss-cn-beijing.aliyuncs.com/img/202209142110666.png)
1. We aim to train a point cloud feature extractor $f_{θP (.)}$ in a selfsupervised manner to be effectively transferable to downstream tasks. To this end, we use an image feature extractor $f_{θI (.)}$, multi-layer perceptron (MLP) projection heads $g_{\phi P (.)}$ and $g_{\phi I (.)}$ for point cloud and image respectively

3. Given an input 3D point cloud $P_i$, we construct augmented versions $P_i^{t1}$ and $P_i^{t1}$ of it. We compose $t_1$ and $t_2$ by randomly combining transformations from $T$ in a sequen-tial manner
```

```ad-info
title: Evaluate
1. object classification
2. few-shot learning and 
3. part segmentation on a diverse range of synthetic and real-world datasets

Result: CrossPoint outperforms previous unsupervised learning methods
   
```

 ```ad-question
title:Questions

intra-modal
 crossmodal
 self-supervised learning
```

```ad-note
title:Dataset
synthetic:
1. ModelNet40
real:
2. ScanObjectNN
3. ShapeNetPart
```
```ad-note
PointNet is an MLP based feature extractor
DGCNN is built on graph convolutional networks

```
