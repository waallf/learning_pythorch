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
 ## class torch.nn.RNNCell(inpu_size,hidden_size,bias=True,nonlinearity="tanh")  
 * 与RNN的区别：
              1.RNNCell只接受输入序列中的单步的输入，且必须传入隐藏层状态  
              2.RNN可以接受一个序列的输入，默认会传入全为0的隐藏层  
 ## class torch.nn.Linear(in_features,out_features,bias=True)  
 * in_feature -输入样本的大小  
 * out_features 输出样本的大小  
 # Dropout layers  
 ## class torch.nn.Dropout(p=0.5,inplace=False)  
 ## class torch.nn.Dropout2d(p=0.5,inplace=False)  
 * 随机将输入张量中整个通道设置为0.  
 # Sparse layers  
 ## class torch.nn.Embedding(num_embeddings,embedding_dim,padding_idx=None,max_norm=None,norm_type=2,scale_grad_by_freq=False,sparse=False)  
* 保存了固定字典和大小的简单查询表  
* num_embeddings 嵌入字典的大小  
* embedding_dim  每个嵌入向量的大小  
* padding_idx   输出遇到此下标时用零填充  
* max_norm      重新归一化词嵌入，使他们的范数的值小于max_norm  
* norm_type     p范数时的p  
* scale_grad_by_freq (boolean, optional) - 如果提供的话，会根据字典中单词频率缩放梯度  
* 输入LongTensor(N,W),N=mini-batch,w=每个mini-batch中提取的下标数  
* 输出：（N,W,embedding_dim）  
# Distance functions  
## class torch.nn.PairwiseDistance(p=2,eps=1e-06)  
* 按批计算向量V1，V2之间的距离  
# loss function  
## class torch.nn.L1Loss(size_average=True)  
* size_average = False时，输出结果将不会除以n  
## class torch.nn.MSELoss(size_average=Tre)  
## class torch.nn.CrossEntropyLoss(weight=None,size_average=True)  
* weight:一维向量，n个元素，分别代表n类的权重，如果样本很不均衡的话非常有用  
## class torch.nn.NLLLoss(weight=None,size_averae=True)  
* 可以通过在最后一层添加LogSoftmax来获得类别的log-probabilities  
* 如果不想加一个额外层的话，可以使用CrossEntropyLoss  
## class torch.nn.KLDivLoss(weight=None,size_average=True)  
* 计算KL散度  
## class torch.nn.BCELoss(weight=None,size_average=True)  
* 计算targrt与output之间的二进制交叉熵  
# Vision layers  
## class torch.nn.PixelShuffle(upscale_factor)  
* 将shape为[N,C*R^2,H,W]重新排列为[N,C,H*R,W*R]  
## class torch.nn.UpsamplingNearest2d(size=None,scale_factor=None)  
* 对多channel输入进行2D最近邻上采样  
* size 一个包含两个整数的元组。制定了输出的长宽  
* scale_factor 长和宽的一个乘子  
# Multi-GPU layers  
## class torch.nn.DataParallel(module,device_ids=None,output_device=None,dim=0)  
* 通过mimi-batch划分到不同设备上来实现module的并行。在forward过程中  
module会在设备上都复制一遍，每个副本都会处理部分输入，在backward过程中，副本上的梯度会累加到
原始moudle上  
* moudle 要被并行的moudle  
* device_ids -CUDA设备，默认为所有设备  
* output_device 输出设备，默认为device_ids[0]  
# Utilities  
## torch.nn.utils.clip_grad_norm(parameters,max_norm,norm_type=2)  
* parameters -可迭代的variables，它们的梯度即被标准化  
* max_norm - 计算范数后的不可超过的值  
* norm—type - 标准化的类型  
## torch.nn.utils.rnn.PackedSequence(_cls,data,batch_sizes)  
* data 包含打包后序列的Varible  
* batch_size 包含mini-batch中每个序列长度的列表  
## torch.nn.utils.rnn.pack_padded_sequence(input,lengths,batch_first=False)  
* 将一个填充过的序列压紧  
* 输入shape是（TxBx*），T是最长序列长度，B是batch_size，如果batch_first=True的话  
那么相应的input_size就是（BxTx*）  
* 只要是维度大于等于2的input都可以作为这个函数的参数，你可以用它来打包labels，然后用  
RNN的输出和打包后的labels来计算loss，通过PackedSequence对象的.data属性可以获取Variable  
## torch.nn.utils.rnn.pad_packed_sequence(sequence,batch_first=False)
* 和上面的函数相反，把压紧的序列再填充回来  
```
import torch  
import torch.nn as nn 
from torch.autograd import Variable
from torch.nn import utils as nn_utils
batch_size = 2 
max_length =3
hidden_size = 2
n_layers =1
tensor_in = torch.FloatTensor([[1,2,3],[1,0,0]]).resize_(2,3,1)
tensor_in = Variable(tensor_in)
seq_lengths = [3,1]
pack = nn_utils.rnn.pack_padded_sequence(tensor_in,seq_lengths,batch_first=True)
rnn = bb.RNN(1,hidden_size,n_layers,batch_size=True)
h0 = Variable(torch.randn(n_layers,batch_size,hidden_size))
out,_ = rnn(pack,h0)
unpacked = nn_utils.rnn.pad_packed_sequence(out)
print(unpacked)

```

