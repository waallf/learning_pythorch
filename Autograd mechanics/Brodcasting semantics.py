'''
pytorch中的广播遵循的两个规则
1.必须至少有一维
2.维度必须相等，或者一个tensor的维度为1，或者该维度不存在
'''
import torch
# 维度相等的就不说了

#tensor为空，导致不能广播
x1 = torch.empty((0,))
y1 = torch.empty(2,2)
z1 = x1+y1
#维度不相同，但是维度存在且不相同，导致不能广播

x2 = torch.empty(2,2)
y2= torch.empty(3,2)
z2 = x2 + y2

#可以广播的情况
x3 = torch.empty(5,2,2,2)
y3= torch.empty( 1,2,2)
z3 = x2 + y2 #广播后的size是两个tensor中最大的值，z3的size是（5,2,2,2）

#在本地操作是，例如add_ 操作，不能用广播改变其tensor大小
x4 = torch.empty(5,2,2,2)
y4= torch.empty( 1,2,2)
x4.add_(y4) #可以执行，并没有改变x4的维度

y4.add_(x4) #就会报错，因为改变的是y4的维度

#BP中的广播
torch.add(torch.ones(4,1),torch(4)) #会先产生一个（4,1） ，最后产生的是（4,4）

#这种情况会导致BP出现错误，可以设置torch.utils.backcompat.broadcast_warning.enabled=True，会产生python警告
torch.utils.backcompat.broadcast_warning.enabled=True
torch.add(torch.ones(4,1),torch(4))
