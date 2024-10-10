# cha_grad

# Project Overview
This is the fun implementation of deep learning framework,similar to PyTorch,but built from scratch It's base is numpy array wrapped around Tensor class.The purpose is to enhance understanding and readability of deep learning algorithms.


# Example
Implementing in cha_grad and in pytorch
```bash
from cha_grad.tensor import Tensor,layer__init

a=Tensor(layer__init(3,3))
b=Tensor(layer__init(1,3))
c=Tensor(np.array(3.0))
x=a.add(b)
z=x.mul(c)
z.mean().backward()
##
print(a.grad)
print(b .grad)

```

Implementing in pytorch
```bash
import torch
a=torch.randn(3,3,requires_grad=True)
b=torch.randn(1,3,requires_grad=True)
c=torch.tensor(3.0,requires_grad=True)
x=a+b
z=x*c
z.mean().backward()
##
print(a.grad)
print(b .grad)
```

