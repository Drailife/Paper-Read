## 求解欧式距离

```python
import torch  
  
a = torch.arange(12).reshape(2,2,3)  
b = torch.ones([2,2,3])  
print(a,b,sep='\n')  
dist = torch.sum((a-b)**2, -1)  
print(dist.shape,dist)

output:
D:\Anaconda3\envs\torch\python.exe E:/GitHub_Drailife/PointCloud/草稿.py 
tensor([[[ 0,  1,  2],
         [ 3,  4,  5]],

        [[ 6,  7,  8],
         [ 9, 10, 11]]])
tensor([[[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]]])
torch.Size([2, 2]) 
tensor([[  2.,  29.],
        [110., 245.]])

Process finished with exit code 0

```
# 按照输入的点云数据和索引返回索引的点云数据
```python
import torch  
  
def index_points_1(points, idx):  
    device = points.device  
    B = idx.shape[0]  
    batch_indices = torch.arange(B, 
         dtype=torch.long).to(device).view(B,1).repeat(1, idx.shape[1])  
    print(batch_indices)  
    new_points = points[batch_indices, idx, :]  # 此处运用了广播机制  
    return new_points
  
point = torch.tensor([[[1, 2, 3],  
                       [4, 5, 6]],  
  
                      [[7, 8, 9],  
                       [10, 11, 12]]])  
print(point.shape)  
idxs = torch.tensor([[0,1], [1,0]])  
ans = index_points(point, idx=idxs)  
print(ans)

output:
torch.Size([2, 2, 3])
tensor([[0, 0],
        [1, 1]])
tensor([[[ 1,  2,  3],
         [ 4,  5,  6]],

        [[10, 11, 12],
         [ 7,  8,  9]]])
```

