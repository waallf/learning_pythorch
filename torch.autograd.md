#torch.autograd  
* 使用自动求导需要将所有的tensor包含进Variable对象中即可  

## torch.autograd.backward(variables,grad_variables,retain_variables = False)  
*variables 被求微分的叶子节点  
*grad_variable 对应variable的梯度，仅当variable不是标量切需要梯度的时候使用  
*retain_variables(bool) 计算梯度是所需要的buffer在计算完梯度后不会被释放，如果想对一个子图多次求微分，需要设置为true  
*给定图的叶子节点，计算图中变量的梯度和，如果，variables中任何一个variable是非标量的，且require_grad=True  
那么此函数需要需要指定grad_variables，它的长度应该和variables的长度匹配，里面保存了相关variable的梯度，对于不需要  
gradient_tensor的variable,可以写成None  
## class torch.autograd.Variable  
variable是Tensor对象的一个thin wrapper，它同时保存着Variable的梯度和创建这个variable的function的引用，这个引用可以用来  
追溯创建这个variable的整条链，如果variable是被用户创建，那么它的creator是None,这种对象称为leaf variables  
** 参数  
* data 包含的Tensor  
* grad 保存着Variable的梯度  
* requires_grad 指示这个variable是否被一个包含variable的子图创建  
* volatile 指示这个Variable是否被用于推断模式，即不保存历史信息  
* creator 创建这个variable的Function，对于leaf variable，这个属性为None  
## backward(gradient=None,retain=variable=False)  
当前variable对leaf variable求偏导  
计算图通过链式法则求导，如果variable是非标量的，且requires_grad=Ture,那么此函数需要指定gradinet,他的形状应该和variable长度  
相匹配，里面保存了variable的梯度  
此函数积累leaf variable的梯度，在调用此函数之前将variable的梯度置0  
* 参数  
* gradient 其他函数对于此variable的导数，仅当variable不是标量的时候使用，类型和形状应该和self.data一致  
* retain_variable =True 计算梯度经历过一次backward过程后不会被释放  
## detach()  
返回一个新的Variable，从当前图中分离下来的  
返回的variable得require_grad=False,如果输出volatile=True，那么返回的Variable volatile=True  
## death_()  
将一个variable从创建它的图中分离，并把它置成leaf variable  

## register_hook  
注册一个bachward钩子  
每次gradients被计算的时候，hook都会被调用，返回一个替代当前梯度的新梯度  
他有一个方法，handle.remove，可以用这个方法将hook从module移除  
```
v = Variable(torch.Tensor([0,0,0]),requires_grad=True)  
h = v.register_hook(lambda grad:grad * 2)
v.backward(torch.Tensor([1,1,1]))
#先计算原始梯度，再进hook，获得一个新梯度
print(v.grad.data)
h.remove()
```  
## reinforce(reward)  
注册一个奖励，这个奖励由一个随机过程得到的  
* reward 每一个元素的reward，必须和variable形状相同  
# class torch.autograd.Function  
记录操作历史，每一个执行在variables上的operation都会创建一个Function  
这个Function对象执行计算工作，同时记录下来，当backward被调用时，计算图  
通过调用每个Function对象的backward(),同时将返回的梯度传递给下一个function  
* 参数  
* saved_tensors 调用forward()时需要被保存的Tensor得tuple  
* need_input_grad 长度为输入数量的布尔值组成的tuple，指示给定的input是否需要  
  梯度  
* num_inputs forward的输入参数数量  
* num_outputs forward返回的Tenosr数量  
* requires_grad 布尔值，指示backwar以后会不会被调用  
* previous_functions 长度为num_inputs的Tuple (int,Function),Tuple中每个单元中保存  
 着创建input的Function得引用和索引  
## backward()  
必须接收和forward的输出相同个数的参数，而且它需要返回和forward的输入参数相同个数的Tensor  
即backward的输入参数是此操作的输出的值得梯度，backward的返回值是此操作输入值得梯度  
## mark_dirty()  
将输入的tensor标记为in-place operation 被修改过  
这个方法至多被调用一次，仅仅用在forward方法里，每个在forward中被in-place操作修改的tensor都应该  
传递给这个方法，这样可以保持检查的正确性。  
## mark_non_differebtiable()  
将输出标记为不可微，这个方法将输出标记为不可微，在backward中，依旧需要接收forward输出值得梯度，但  
这些梯度一直是None  
## mark_shared_storage()  
将给定的tensor pairs标记为共享存储空间  
## save_for_backward()  
将传入的tensor保存起来，留着backward时候使用  
被保存的tensor可以通过saved_tensors属性获取。



