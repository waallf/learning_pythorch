封装了multiprocessing模块，用于在相同数据的不同进程中共享视图  

一旦张量或者存储被移动到共享单元，它可以不需要任何其他复制操作，发送到其他进程中  
`toch.multiprocessing.get_all_sharing_strategies()`  
返回一组由当前系统所支持的共享策略  
`torch.multiprocessing.get_sharing_strategy()`  
返回当前策略共享cpu中的张量  
`torch.multiprocessing.set_sharing_strategy(new_strategy)`  
设置共享CPU张量的策略  
* 参数：new_strategy() 被选中策略的名字，应当是get_all_sharing_strategies()中的一个  

