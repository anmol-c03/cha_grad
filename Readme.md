# cha_grad

# Project Overview
This is the fun implementation of deep learning framework,similar to PyTorch,but built from scratch It's base is numpy array wrapped around Tensor class.The purpose is to enhance understanding and readability of deep learning algorithms.

# Overview
Fun Grad aims to provide a better understanding of how deep learning frameworks like PyTorch work under the hood. It is built around the concept of Tensors, allowing for operations such asbackpropagation , all implemented in an easy-to-understand manner.

The framework is educational and simple, with an emphasis on clarity and readability of both code and algorithm.
Features :
    * Custom Tensor class to handle data and gradients.
    * Autograd system(cha_grad) for backpropagation.
    * Operations like matrix multiplication, addition, etc.
    * Basic optimization algorithms (like SGD, Adam) implemented from scratch.
    * Example usage with a custom MNIST classifier.

# Installation Steps
1.Clone the repository:
```python
git clone https://github.com/anmol-c03/cha_grad.git
```

2.Navigate to the project directory:
```bash
cd src
```
3. Install the required dependencies:
```bash
pip install numpy
```

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
# Example Usage
1. Create a new python script as my_script.py 
```bash 
  src/my_script.py
```
2.Inside `my_script.py`, import and use the `Tensor` class and other components like this:

```bash
from cha_grad.tensor import Tensor
import numpy as np

# Create some tensors
x = Tensor(np.random.randn(3, 3))
w = Tensor(np.random.randn(3, 1))

# Perform matrix multiplication
out = x.matmul(w)

# Calculate gradients
out.mean().backward()

print("Output:", out)
print("Gradients for x:", x.grad)
```

3. Run your script
```bash
python3 src/my_script.py
```

Project Structure
The folder structure of the project is as follows:
```bash
cha_grad/
├── notebook/
│   └── NN_from_scratch.ipynb    # Jupyter notebook for neural network example
├── src/
│   ├── cha_grad/
│   │   ├── tensor.py            # Core tensor operations and autograd system
│   │   ├── optim.py             # Optimizers like SGD, Adam
│   │   ├── utils.py             # Custom operations like dot, mul, etc.
│   ├── my_script.py             # Custom script using cha_grad
│   ├── mnist.py                 # Example usage on MNIST dataset
│   └── scratch.py               # Example or test script
└── README.md                    # Project description and instructions
```