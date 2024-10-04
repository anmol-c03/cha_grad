import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


g=torch.Generator().manual_seed(2147483647)
# class Module:
#     def __init__(self):
#         pass

#     def parameters(self,params):
#         return params

class Linear:
    '''Though i have defined the weights as (fan_in,fan_out)
     torch.nn.linear defines weight as  (fan_out,fan_in)
     and thats the huge diff and one should always consider that while initializing linear layer'''
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight=torch.randn((fan_in,fan_out),generator=g)/fan_in**0.5
        self.bias=torch.zeros(fan_out) if bias else None

    # def round_clip(self,xs):
    #     x=xs/torch.mean(xs)
    #     x=x.view(-1)
    #     out=torch.zeros(x.shape)
    #     out=torch.clamp(torch.round(x),-1,1)
    #     return out 
    
    def __call__(self,x):
        y=x @ self.weight
        if self.bias is not None:
            y+=self.bias
        # self.out=self.round_clip(y)
        self.out=y
        return self.out
        
    def parameters(self):
        return [self.weight]+([] if self.bias is None else [self.bias])
    
#actual implementation of linear layer in pytorch looks similar to this
# class Linear:
#     def __init__(self,fan_in,fan_out,bias=True):
#         self.weight=torch.randn((fan_out,fan_in),generator=g)/fan_in**0.5
#         self.bias=torch.zeros(fan_out) if bias else None  

#     def __call__(self,x):
#         y=x @ self.weight.T
#         if self.bias is not None:
#             y+=self.bias
#         # self.out=self.round_clip(y)
#         self.out=y
#         return self.out
#--------------------------------------------------------------------------------------------------------------------------------------------
class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
      
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):

    if self.training:
      if x.ndim==2:
          dim=0
      elif x.ndim==3:
          dim=(0,1)
      xmean = x.mean(dim, keepdim=True) 
      xvar = x.var(dim, keepdim=True) 
    else:
      xmean = self.running_mean
      xvar = self.running_var
    
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) 
      
    self.out = self.gamma * xhat + self.beta

    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

#--------------------------------------------------------------------------------------------------------------------------------------------
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
      
  def parameters(self):
    return []
#--------------------------------------------------------------------------------------------------------------------------------------------      
class Embedding:
    def __init__(self,num_emb,emb_dim):
        self.weight=torch.randn(num_emb,emb_dim,generator=g)

    def __call__(self,index):
        self.out=self.weight[index]
        return self.out

    def parameters(self):
        return [self.weight]

#--------------------------------------------------------------------------------------------------------------------------------------------
class FlattenC:
    def __init__(self,n):
        self.n=n
        
    def __call__(self,x):
        B,T,C=x.shape
        x=x.view(B,T//self.n,C*self.n)
        if x.shape[1]==1:
            x=x.squeeze(1)
        self.out=x
        return self.out

    def parameters(self):
        return []
#----------------------------------------------------------------------------------------------------------------------------------------------
class Sequential:
    def __init__(self,layers):
        self.layers=layers

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        self.out=x
        return self.out

    def eval(self):
        for layer in self.layers:
            layer.training=False

    def train(self):
        for layer in self.layers:
            layer.training=True

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
#----------------------------------------------------------------------------------------------------------------------------------------------      
class ReLU:
   def __init__(self):
      pass
   
   def __call__(self, x):
      pass
   
   def parameters(self):
      return []
   
#----------------------------------------------------------------------------------------------------------------------------------------------      
class CrossEntropyLoss:
   def __init__(self):
      pass
   
   def __call__(self,logits,y):
        prob=F.log_softmax(logits,dim=1)
        loss=-prob[torch.arange(y.shape[0]),y].mean()
        return loss

   def parameters(self):
      return []