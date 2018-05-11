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
# 沿着给定轴，将输入索引Index指定位置进行聚合
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
* 返回z*n 的Tensor ，z是全部非零元素的个数，n是输入的维度  
## torch.split  
`torch.split(tensor,split_size,dim=0)`  
* 将张量分割成单块为split_size大小，如果不能整分，最后一块将小于其他块  
## torch.squeeze  
`torch.squeeze(input,dim = None,out =None)`  
* 将张量中维度为1的去掉  
* 如果给定dim，则只在指定维度上进行操作  

