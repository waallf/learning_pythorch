import torch
print(torch.__version__)
x = torch.randn(5,5) #默认的requires_grad = False
y = torch.randn(5,5)
a = x+y
'''
通过设置requires_grad来设置参数用不用更新，例如在finetune时候，
通过设置为False来固定参数
'''
z = torch.randn((5,5),requires_grad=True)
b = a + z
print(a.requires_grad)
print(b.requires_grad)


################################
'''
导入resnet模型，固定前面来改变最后的全连接层的参数
'''
import torchvision
import torch.nn as nn
import torch.optim as optim
model = torchvision.models.resnet18(pretrained= True)
for param in model.parameters():
    param.requires_grad = False
#替换掉全连接层
model.fc = nn.Linear(512,100)
optimizer = optim.SGD(model.parameters(),lr=1e-2,momentum=0.9)





