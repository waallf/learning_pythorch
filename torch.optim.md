## 构建  
为了构建一个optimizer，需要给他一个包含了需要优化的参数（必须都是variable对象）  
```
optimzer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)  
optimzer = optim.Adama([var1,var2],lr=0.0001)
```  
* 为每个参数设置单独的参数  
```
optim.SGD([
            {'params':model.base.parapeters()},
            {'params':model.classifier.parameters(),'lr':1e-3}
          ],lr=1e-2,momentum=0.9)
```  
## 进行优化  
step()方法，两种方式来使用  
1. optimzer.step()  
```
for input,target in dataset:
  optimizer.zero_grad()
  output = model(input)
  loss = loss_fn(output,target)
  loss.backward()
  optimzer.step()
```  
2. optimizer.step(closure)  
```
for input,target in dataset:
  def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output,target)
    loss.backward()
    return loss 
   optimizer.step(closure)
```  
## class torch.optim.Optimizer(params,defaults)  
* 参数  
* params  指定需要被优化的参数  
* defaults 包含了优化选项默认值得字典  
## load_state_dict(state_dict)  
加载optimizer状态  
* 参数  
* state_dict()-- optimizer的状态，应当调用state_dict()所返回的对象  
## state_dict()  
以dict返回optimizer的状态  
包含两项：
  state 一个保存了当前优化状态的dict，optimizer的类别状态不同，state也不同  
  param_groups 包含了全部参数的dict  
## zero_grad()  
清空所有被优化过的Variable的梯度  
## class torch.optim.Adadelta(params,lr=1.0,rho=0.9,eps=1e-6,weight_decay=0)  
* 参数  
* rho 用于计算平方梯度的运行平均值的系数  
* eps 在分母上加一个很小的数（默认：0.9）  
## class torch.optim.Adagrad(params,lr=0.01,lr_decay=0,weight_decay=0)  
* 参数  
* lr_decay 学习衰减率  




