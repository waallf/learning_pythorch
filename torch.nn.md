# torch.nn  
## Parameters  
### class torch.nn.Parameter()  
variable的一种，当和Modules一起使用时会有一些特殊属性，即：当Paramenters赋值给Moudle的属性时  
会自动的被加到Moudle的参数列表中。
* 参数 data - Tensor  
* requires_grad(bool) ,默认为True，在BP过程中会对其求微分  
# Conteiners(容器)  
## class torch.nn.Module  
* 所有网络的基类  
```
import torch.nn as nn  
impirt torch.nn.function as F
class Model(nn.Module)
  def __init__(self):
    super(Model,self).__init__()
    self.conv1 = nn.Conv2d(1,20,5)
    self.conv2 = nn.Conv2d(20,20,5)
  def forward(self,x):
    x = F.relu(self.conv1(x))
    return F.relu(self.convx(x))
    
```  
## add_moudle(name,module)  
* 将一个child module添加当前的modle，被添加的model可以通过name属性来获取  
```
import torch.nn as nn  
class Model(nn.Module):  
  def __init__(self):
    super(Model,self).__init__()  
    self.add_module("conv",nn.Conv2d(10,20,4)) 
model = Model()  
print(model.conv)
输出：
Conv2d(10,20,kernel_size=(4,4),stride =(1,1))
```  
## children()  
* 返回当前模型子模块的迭代器  
```
import torch.nn as nn
class Model(nn.Module):
  def __init__(self):
    super(Model,self).__init__()
      self.add_module("conv",nn.Conv2d(10,20,4))
      self.add_modulde("conv1",nn.Conv2d(20,10,4))
model = Model()
for sub_module in model.children():
  print(sub_module)
```
## forward(*input)
* 定义了每次执行的计算步骤，在所有子类中都需要重写这个函数  
## load_state_dict(state_dict)  
* 将state_dict中的parameters和buffers复制到此module和它的后代中。  
## modeles()  
* 返回当前模型所有模块的迭代器  
```
import torch.nn as nn  
class Model(nn.Module):
  def __init__(self):
    super(Model.self).__init__()
    self.add_module("conv",nn.Conv2d(10,20,4))
    self.add_module("conv1",nn.Conv2d(20,10,4))
model = Model()
for model in model.modules():
  print(module)
```
## named_children()  
* 返回包含模型当前子模块的迭代器，迭代返回模块名字和模块本身  
```
for name,module in model.named_children():
  if name in ['conv4','conv5']:
    print(module)
```  
## parameters  
* 返回一个包含模型所有参数的迭代器  
```
for param in model.parameters():
  print(type(param.data),param.size())
```  
## register_buffer(name,tensor)  
* 通常被用在这么一种情况：我们需要保存一个状态，但是这个状态不能看作为模型的参数。
## state_dict(destination=None,prefix="")  
* 返回一个字典，保存着module的所有状态  
```
import torch
from torch.autograd import Variable
import torch.nn as nn
class Model(nn.Module):
  def __init__(self):
    super(Model,self).__init__()
    self.conv2 = nn.Liner(1,2)
    self.vari = Variable(torch.rand([1]))
    self.par = nn.Parameter(torch.rand([1]))
    self.register_buffer("buffer",torch.randn([2,3]))
model = Model()
print(model.state_dict().keys())
输出： odict_keys(["par","buffer","con2.weight","conv2.bias"])
```  
## train(mode = True)  
* 将module设置为training mode  
## zero_grad()  
* 将module中的所有模型参数的梯度设置为0  
## class torch.nn.Sequential()  
* 一个时序容器，Modules会以他们传入的顺序被添加到容器中  
```
model = nn.Sequential(
         nn.Conv2d(1,20,5),
         nn.Relu(),
         nn.Conv2d(20,64,5),
         nn.Relu())  
 model = nn.Sequential(OrderedDict([
            ("conv1",nn.Conv2d(1,20,5)),
            ("relu",nn.Relu(1,20,5)),
            ("conv2"nn.Conv2d(20,64,5)),
            ])
 ```  
 ## class torch.nn.ModuleList(modules=None)  
 * 将一个submodules保存在一个list中  
 ```
 class MyModule(nn.Module):
  def __init__(self):
    super(MyModule,self).__init()
    self.linears = nn.ModuleList([nn.Linear(10,10) for i in range(10) ])
   def forward(self.x):
    for i,l in enumerate(self.linears):
      x = self.linears[i //2](x) +l(x)
      return x
 ```  
 ## class torch.nn.ParameterList()  
 * 将参数放在一个列表中  
 # 卷积层  
 ## class torch.nn.Conv1d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,biad=True)  
 * Parameters:  
 * in_channels -输入信号的通道  
 * out_channels -卷积产生的通道  
 * kerner_size -卷积核的尺寸  
 * stride -卷积步长  
 * padding -输入的每一条边补充0的层数  
 * dilation -卷积核元素之间的距离  
 * groups -从输入通道到输出通道的阻塞连接数  
 * bias  -设置为True 添加偏置  
 ## class torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True)  
 ## class torch.nn.Conv3d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1.bias=True)  
 ## class torch.nn.ConvTranspose(in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0,groups=1,bias=True)  
 ## class torch.nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0,groups=1,bias=True)  
 # 池化层  
 ## class torch.nn.MaxPool1d(kernel_size,stride=None,padding=0,dilation=1,return_indices=False,ceil_mode=False)  
 * return_indices -如果等于True，会返回输出最大值的序号  
 * ceil_mode 如果等于true，输出时会向上取整  
 ## class torch.nn.MaxPool2d(kernel_size,stride=None,padding =0,dilation=1,return_indices=False,ceil_mode=False)  
 ## class troch.nn.MaxPool3d(kernel_size,stride=None,padding=0,dilation=1,return_indices=False,ceil_mode=False)  
 ## class torch.nn.MaxUnpool1d(kernel_size,stride=None,padding=0)  
 * MaxUnpool1d输入MaxPool1d的输出，包括最大值的索引，不是最大值的地方补0  
 ## class torch.nn.MaxUnpool2d(kernel_size,stride=None,padding=0)  
 ## class torch.nn.AvgPool1d(kernel_size,stride=None,padding=0,ceil_mode=False,count_include_pad=True)  
 # Non-Linear Activations  
 ## class torch.nn.Relu()    
 ## class torch.nn.Relu6()  
 * return min(max(0,x),6)  
 ## class troch.nn.ELU(alpha=1.0,inplace=False)  
 * return max(0,x) +min(0,alpha*(e^x-1))  
 ## class torch.nn.PRelu(num_parameters=1,init=0.25)  
 * return max(0,x)+a*min(0,x)  
 * a是一个可学习的参数，当没有声明时所有的输入中只有一个参数a，  
 * 如果是nn.PRelu(nChaeenls),a将应用到每个输入  
 * num_parameters:需要学习的a的个数  
 * init ：a的初始值  
 ## class torch.nn.LeakyRelu(negative_slope=0.01,inplace=False)  
 * return max(0,x) +{negative_slope}*min(0,x)  
 * negative_slope：控制负斜率的角度  
 ## class torch.nn.Threshold(thershold,value,inplace=False)  
 * return y=x,if x>=threshold y=value,if x< threshold  
 * threshold 阈值  
 * value 输入值小于阈值会被value代替  
 ## class torch.nn.LogSigmoid  
 * return log(1/(1 +e^{-x}))  
 # Normalization layers  
 ## class torch.nn.BatchNorm1d(num_features,eps=1e-0.5,momentum=0.1,affine-True)  
 * momentum:动态均值和动态方差所使用的动量  
 * affine :设置为True,给该层添加课学习的仿射变换  
 # Recurrent layers  
 ## class torch.nn.RNN  
 * input_size :输入x特征数量  
 * hidden_szie:隐层的特征数量 
 * num_layers :RNN的特征数量  
 * nonlinearity 指定非线性函数使用tanh还是relu，默认tanh  
 * bias :如果是False，那么RNN层就不适用偏置  
 * batch_first:为True时，那么输入Tensor的shaoe应该是[batch_size,time_step,frature]  
 * dropout:如果值为0，那么除了最后一层外，其它层的输出都会套上一个dropout层  
 * bidirectional :如果为True,将变成一个双向RNN  
 * 输入：(input,h_0) :input(seq_len,batch,input_size)  
 *       h_0(num_layers*num_directions,batch,hidden_size)保存着初始隐状态  
 * 输出：(out_put,h_n) : out_put(seq_len,batch,hidden_size*num_directions)  
        h_n(num_kayers*num_directions,batch,hidden_size) 保存着最后一个时刻隐状态  
 ## class torch.nn.LSTM()  
 * input_size :输入的特征维度  
 * hidde_size :隐状态的特征维度  
 * num_layers :层数
 
 
 
          )
