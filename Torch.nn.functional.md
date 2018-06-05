## Convolution函数  
`torch.nn.functional.conv1d(input,weight,bias = None,stride=1,padding=0,dilation=1,groups=1)`  
```
import torch.nn.functional as F
filters = autograd.Variable(torch.randn(33,16,3))
inputs = autograd.Variable(torch.randn(20,16,50))
F.conv1d(inputs,filters)
```  
##Pooling函数  
`torch.nn.functional.avg_pool1d(input,kernek_size,stride=None,padding=0,ceil_mode=False,count_include_pad = True)`  
##非线性激活函数  
`torch.nn.functional.relu(input,inplace = False)`  
##Normalization函数  
`torch.nn.functional.batch_norm(input,running_mean,running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)`  
##Dropout函数  
`torch.nn.functional.dropout(input,p=0.5,training=False,inplace=False)`  
##距离函数  
`torch.nn.functional.pairwise_distance(x1,x2,p=2,eps=1e-06)`  

*在torch.nn中介绍的所有类，在torch.nn.functional中都有涉及。
