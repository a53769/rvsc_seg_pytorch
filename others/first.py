import torch as t
t.cuda.set_device(0)#第0个gpu
a = t.Tensor([2]).cuda()
b = t.Tensor([3]).cuda()

print(a+b)

matrix = t.randn(3,3).cuda()
print(matrix)

print(matrix.t())