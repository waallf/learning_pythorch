import torch

#选择GPU设备torch.cuda.device

cuda = torch.device('cuda') #默认的GPU
cuda0 = torch.device('cuda:0')
cuda2 = torch.device("cuda:2")

x = torch.tensor([1.,2.],device = cuda0)

y = torch.tensor([1.,2.]).cuda()
#x.device, y.device is device(type='cuda', index=0)

with torch.cuda.device(1):
    a = torch.tensor([1.,2.],device = cuda)
    b = torch.tensor([1.,2.]).cuda()

    b2 = torch.tensor([1.,2.]).to(device = cuda)
    # a.device and b.device,b2.device are device(type='cuda', index=1)
    c = a + b
    # c.device is device(type='cuda', index=1)

    z = x + y
    # z.device is device(type='cuda', index=0)
    d = torch.randn(2, device=cuda2)
    e = torch.randn(2).to(cuda2)
    f = torch.randn(2).cuda(cuda2)
    # d.device, e.device, and f.device are all device(type='cuda', index=2)
#利用参数来选择是GPU还是CPU

import argparse
import torch
parser = argparse.ArgumentParser(description="pytorch Example")
parser.add_argument('--disable-cuda',action = 'store_true')

args = parser.parse_args()
args.device = None
if not args.disable-cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device("cpu")
x = torch.empty((8,42),device = args.device)

#如果想在相同设备上创建tensor
x = torch.empty((2,2))
y = x.new_full([3,2],fill_value=0.3)
#输出是一个3*2 ，值全部是 0.3
