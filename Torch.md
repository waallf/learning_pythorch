# torch.numel(input)
* 返回input中张量的个数  
`torch.numel(input) -int`
* 例子
```
a = torch.randn(1,2,3,4,5)
torch.numel(a)
120
```
# torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None)  
* precision -浮点数输出的精度位数  
* threshold -显示数组元素的总数（默认1000）  
* edgeitems -显示中每维两端显示元素的个数（默认值为3） 
* linewidth -用于插入行间隔的每行字符数（默认800） 
* profile - pretty 打印的完全默认值，可以覆盖上述所有选项  
# 创建操作
## torch.eye(n,m=None,out = None)  
* 返回对角线位置全为1，其他位置全为0
* n -行数  
* m -列数，默认为n  
* out - Tensor  
## from_numpy  
`torch.from_numpy(ndarray) -> Tensor`  
* 将numpy.ndrray转化为ptorch中的tensor，两者占据共同的存储空间，改变一个另一个也会改变  
## torch.linspace  
`torch.linspace(start，end，steps = 100,out=None)`  
*返回一维张量，从start到end平均生成五个样本  
## torch.logspace
`torch.logspace(start,end,steps = 100,oyt = None)`  
* 返回10^start 到10^end 均匀的steps个点  
## torch.rand
* 返回（0,1）之间的随机数  
## torch.randn  
* 返回均值为0，方差为1的随机数  
## torch.randperm  
`torch.randperm(n)`  
* 返回0，n-1的随机整数序列  
## torch.arange  
`torch.arange(start,end,step=1,out = None)  
*返回从start-end，以step为步长的一组序列  
# 索引，切片，连接，换位  
## torch.cat
`torch.cat(inputs,dimension = 0)`  
* 进行连接操作  
* inputs -Tensor的序列  
* dimensi-沿着哪个维度进行拼接  
## torch.chunk  
`troch.chunk(tensor,chunks,dim=0)`
* 在给定维度上进行分块  
* chunks 分块的个数  
## torch.gather  
`totch.gather(input,dim,index,out = None)`  
* 沿着给定轴，将输入索引Index指定位置进行聚合
**对于二维tensor,dim =0,那列搜索，dim =1按行搜索，聚合不知道是啥意思，但是运行代码看，就是换了个位置**  
```
t = torch.Tensor([[1,2],[3,4]])

torch.gather(t,1,torch.LongTensor([[0,0],[1,0]]))
1 1
4 3
#按行搜索,:
第一行的第0个 第一行的第0个
第二行的第1个 第二行的第0个

torch.gather(t,0,torch.LongTensor([[0,0],[1,0]]))
1 2
4 3
#按列搜索
第一列的第0个，第二列的第0个
第一列的第1个，第二列的第0个 
```
## torch.index_slect  
`torch.index_select(input,dim,index,out = None)`
* 按照index指定的维度进行提取  
## torch.masked_select  
`torch.masked_select(input,mask,out = None)`
* maks 与input一样维度的二元值，且为ByteTensor
* 返回一个一维的张量
## torch.nonzero  
* 返回的是input中的非零元素索引的张量，输出张量中的每行包含输出中非零元素的索引
* 返回z*n 的Tensor ，z是全部非零元素的个数，n是输入的维度  
## torch.split  
`torch.split(tensor,split_size,dim=0)`  
* 将张量分割成单块为split_size大小，如果不能整分，最后一块将小于其他块  
## torch.squeeze  
`torch.squeeze(input,dim = None,out =None)`  
* 将张量中维度为1的去掉  
* 如果给定dim，则只在指定维度上进行操作  
## torch.stack 
`torch.stack(sequeue,dom=0)`  
* 沿着一个新的维度对输入张量进行连接，序列中所有张量都应为相同形状  
* sequeue 带连接的张量序列  
## torch.t 
`torch.t(input,None)`  
* 输入一个矩阵，并转置0,1维
## torch.transpose  
`torch.transpose(input,dim0,dim1,out=None)`  
* 输入矩阵，交换维度dim0,与dim1  
## torch.unbind  
`torch.unbind(tensor,dim =0)`  
* 返回沿着指定维度切片后的一个元组  
## torch.unsqueeze  
`torch.unsqueeze(input,dim,out = None)  
* 返回一个新的张量，对输入的维度插入一个维度1  
# 随机抽样  
## torch.manual_seed  
* 设定生成随机数的种子  
## torch.initial_seed  
* 返回生成随机数的原始种子  
## torch.multinomial  
`torch.multinomial(input,num_samples,replacement = False,out = None)`  
* 返回一个张量，每行从input相应行中定义的多项分布中抽取num_samoles个样本  
* input每行的值不需要总和为1，但是必须非负切总和不能为0  
* replacement为true，则样本抽取可以重复，否则一个样本不能被重复抽取  
## torch.normal()  
`torch.normal(means,std,out = None)`  
* 返回一个张量，means是一个张量，包含每个输出元素的正太分布均值，std是一个张量，代表标准差  
# 序列化  
##torch.save  
`torch.save(obj,f,pickle_module = (module = 'pickle' from '/home/' jenkins/miniconda/lib/python3.5/pickle.py'>, pickle_protocol=2))`  
* 保存一个对象到一个硬盘文件   
* obj-保存对象  
* f -文件对象或一个文件名的字符串  
## torch.load()  
`torch.load(f,map_location = None,pickle_module <module 'pickle' from '/home/jenkins/miniconda/lib/python3.5/pickle.py'>)`  
* 从磁盘中读取一个通过torch.save()保存的对象，可以通过map_location动态的进行内存从映射，
* f - 类文件对象  
* map_location 一个函数或者字典规定如何remap存储位置  
```
torch.load('tensor.pt')
torch.load("tensor.pt",map_location = {'cuda:1':'cuda:0'})  

```  
# 并行化  
## torch.get_num_threads()  
* 返回int值,得到并行化cpu的线程数  
## torch.set_num（int）  
* 设定用于并行化cpu操作的线程数  
# 数学操作  
## torch.addcdiv
`torch.addcdiv(tensor,value = 1,tensor1,tensor2,out=None)`  
* 用tensor2对tensor1逐元素相除，然后乘以标量值value，并加到tensor  
## torch.ceil  
* 对输入input张量每个元素向上取整  
## torch.floor  
* 返回不小于元素的最大整数  
## torch.fmod  
* 计算余数，余数的正负与被除数相同  
## torch.lerp  
`torch.lerp(start,end,weight)`  
* 对两个张量以start,end做线性插值，out = start + weight(end - start)  
## torch.round  
* 将每个元素舍入到最近的整数  
## torch.sign  
* 返回每个元素的正负  





