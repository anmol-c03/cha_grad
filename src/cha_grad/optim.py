import torch
from torch import Tensor
import numpy as np
class Optimizer:
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

class SGD(Optimizer):
    def __init__(self,params,lr):
        super().__init__(params,lr)
        pass
    def step(self):
        for t in self.params:
            # print(t.grad)
            t.data-=self.lr * t.grad  

class Adam(Optimizer):
    def __init__(self,params,
                 lr=0.001,
                 eps=1e-8,
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
        self.first_moment=[torch.zeros_like(t.data) for t in self.params]
        self.second_moment=[torch.zeros_like(t.data) for t in self.params]
        self.t=0
        print(self.lr)
        print(self.b1,self.b2)
    
    def step(self):
        for i,p in enumerate(self.params):
            self.t+=1
            grad=p.grad
            if self.weight_decay!=0:
                grad+=self.weight_decay*p.data
            self.first_moment[i] = self.b1*self.first_moment[i] + (1 -self.b1)*grad
            self.second_moment[i] = self.b2*self.second_moment[i] + (1 -self.b2)*(torch.square(grad))
            first_unbias=self.first_moment[i] / (1. - self.b1**(self.t))
            second_unbias=self.second_moment[i]  / (1. - self.b2**(self.t))
            p.data-=self.lr*first_unbias/(torch.sqrt(second_unbias)+self.eps)
            

class AdamW(Optimizer):
    def __init__(self,params,
                 lr=0.001,
                 eps=1e-8,
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
        self.first_moment=[torch.zeros_like(t.data) for t in self.params]
        self.second_moment=[torch.zeros_like(t.data) for t in self.params]
        self.t=0
        
    # still ha to define and look upto the scheduler
    def scheduler(self,i):
        return 1
    
    def step(self):
        for i,p in enumerate(self.params):
            self.t+=1
            grad=p.grad
            if self.weight_decay!=0:
                grad+=self.weight_decay*p.data
            self.first_moment[i] = self.b1*self.first_moment[i] + (1 -self.b1)*grad
            self.second_moment[i] = self.b2*self.second_moment[i] + (1 -self.b2)*(torch.square(grad))
            first_unbias=self.first_moment[i] / (1. - self.b1**(self.t))
            second_unbias=self.second_moment[i]  / (1. - self.b2**(self.t))
            sch=self.scheduler(self.t)
            p.data-=sch*(self.lr*first_unbias/(torch.sqrt(second_unbias)+self.eps)+self.weight_decay*p.data)
            
