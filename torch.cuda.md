# torch.cuda  
使用is_available()来确定系统是否支持CUDA  
class torch.cuda.device(idx)  
上下文管理器，更改所选设备  
* 参数 idx 设备索引选择  
torch.cuda.device_count()  
返回可得到的GPU数量  
class torch.cuda.device_of(obj)  
将当前设备更改为给定对象的上下文管理器  
`torch.cuda.is_available()`  
返回一个Bool值，指示CUDA当前是否可用  
`torch.cuda.set_device(device)`  
设置设备  
* device(int),所选设备，如果参数为负，则此函数无效操作  
`torch.cuda.stream(stream)`  
选择给定流的上下文管理器  
`torch.cuda.synchronize()`  
等待当前设备上所有流中的所有核心完成  
`torch.cuda.comm.broadcast(tensor,devices)`  
向一些GPU广播张量  
`torch.cuda.comm.reduce_add(inputs,destination=None)`  
来自多个GPU的张量相加  
`toech.cuda.comm.scatter(tensor,devices.chunk_sizes=None,dim=0,streams=None)`  
打散横跨多个GPU的张量  
* 参数： tensor 要分散的张量  
* devices 指定设备  
* chunk_sizes 要放置在设备上的块大小，默认分成相等的块  
* dim 沿着这个维度来切分  
`torch.cuda.comm.gather(tensor,dim=0,destination=None)`  
从多个GPU收集张量  
##流和事件  
## class torch.cuda.Stream  
* 参数  
* device 分配流的设备  
* priority(int,optional)流的优先级较低的设备有较高的优先级  
* query() 检查所有提交的用过是否已经完成  
* record_event(event=None)记录一个事件  
* synchronize()等待此流中所有核心完成  
* wait_event(event) 将所有未完成的工作提交到流等待事件  
* wait_stream(stream)与另一个流同步  
## cuda 事件的包装  
`class torch.cuda.Event(enable_timing=False,blocking=False,interprocess=False,_handle=None)`  
* 参数  
* enable_timing 指示事件是否应该被测量时间  
* blocking 如果为True,wait()将被阻塞  
* interprocess 如果为true，则可以进程之间共享事件  
* elapsed_time(end_event)  返回事件记录之前经过的时间  
* query() 检查事件是否被记录  
* record  记录给定流的事件  
* synchronize()与事件同步  
* wait(stream=None)使给定的流等待事件  
