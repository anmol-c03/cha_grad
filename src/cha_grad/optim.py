from torch import Tensor
import numpy as np
class optimizer:
    def __init__(self,params,lr=0.001):
        self.params=params
        if lr<=0:
            raise ValueError(f'Learning rate should be non-zero,{lr}')
        if isinstance(lr,Tensor):
            raise ValueError('Learning rate and tensors shouldnot be tensors')
        
        self.lr=lr

    def zero_grad(self):
        for t in self.params:
            if t.grad is not None:
                t.grad=None

class SGD(optimizer):
    def __init__(self):
        super().__init__()
        pass
    def step(self):
        for t in self.tensors:
            t.data-=self.lr * t.grad  

class Adam(optimizer):
    def __init__(self,params,
                 lr=0.001,
                 eps=1e-7,
                 betas=(0.9, 0.999),
                 weight_decay=0):
        super().__init__(params,lr)
        if not 0.0<=eps<=1e-5:
            raise ValueError(f'epsilon is invalid value{eps}')
        if not all(0.0<=b<1.0 for b in betas):
            raise ValueError(f'either {betas[0]} or {betas[1]} is outside the range of 0 and 1')
        if weight_decay<0.0:
            raise ValueError(f'Weight decay should be non-negative, {weight_decay}')
        self.eps=eps
        self.b1=betas[0]
        self.b2=betas[1]
        self.weight_decay=weight_decay
        self.first_moment=[np.zeros_like(t.grad) for t in self.params]
        self.second_moment=[np.zeros_like(t.grad) for t in self.params]
        self.t=0
    
    def step(self):
        for i,t in enumerate(self.params):
            self.t+=1
            self.first_moment[i] = self.b1*self.first_moment[i] + (1.-self.b1)*t.grad
            self.second_moment[i] = self.b1*self.second_moment[i] + (1.-self.b2)*(t.grad**2)
            first_unbias=self.first_moment[i] / (1. - self.b1**(self.t))
            second_unbias=self.second_moment[i]  / (1. - self.b2**(self.t))
            t.data-=self.lr*first_unbias/(np.sqrt(second_unbias)+self.eps)
            

        
