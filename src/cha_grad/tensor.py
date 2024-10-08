# inspired from https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
import functools
import numpy as np

np.random.seed(1337)
def layer__init(fan_in,fan_out):
   return np.random.uniform(-1,1,(fan_in,fan_out))/np.sqrt(fan_out).astype(np.float32)

class Tensor:
    def __init__(self, data):

        self.data = data
        self.grad = None

        self._prev=None
        if type(self.data)!=np.ndarray:
            print(f'error constructing Tensor from {self.data} as data is not numpy array')
            assert(False)

    def __repr__(self):
        return f'Tensor with data {self.data} and grad {self.grad}'
    
    def backward(self,allow_fill=True):
        if self._prev is  None:
           return

        if self.grad is None and allow_fill==True:
            '''since gradient w.r.t  single value is always calculated 
            This line of code is executed only for last node in graph i.e output of NN because
            the gradient calculated is gradient  with itself which is always 1.'''

            assert self.data.size==1
            self.grad = np.ones_like(self.data) 
        
        assert(self.grad is not None)

        grads=self._prev.backward(self._prev,self.grad)
        # print(grads)
        if len(self._prev.parents)==1:
            grads=[grads]
        for prev_node,grad in zip(self._prev.parents,grads):
            if prev_node.data.shape!=grad.shape:
                print(f'Error: dimensions of gradients {grad.shape} and parameters {prev_node.data.shape} do not match')
                assert(False)
            prev_node.grad=grad
            prev_node.backward(False)
    
    def mean(self):
        p=Tensor(np.array([1/self.data.size]))
        return self.sum().mul(p)

class Function:
    def __init__(self,*tensors):
        self.parents = tensors 
        self.saved_tensors = []
    
    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)
    
    def apply(self, arg, *x):
        if isinstance(arg,Tensor):
            op=self
            x=[arg]+list(x)
        else:
            op=arg
            x=[self]+list(x)
        ctx = op(*x) # op is fxn(subclass/derived class)  of Function and op(*x) is obj insantiation 
        ret = Tensor(op.forward(ctx, *[t.data for t in x]))
        ret._prev=ctx # Tensor class _prev contains the obj of function class whose parents are actual prev node
        return ret
    
def register(name,fxn):
    setattr(Tensor,name,functools.partialmethod(fxn.apply,fxn))

class ReLU(Function):
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return np.maximum(0, x)
  @staticmethod
  def backward(ctx, grad):
    x, = ctx.saved_tensors
    out=np.where(x>0,1,0)
    return grad * out
register('relu', ReLU)

class Add(Function):
  @staticmethod
  def forward(ctx,x,y):
    ctx.save_for_backward(x, y)
    return x + y
  @staticmethod
  def backward(ctx, grad):
    x, y = ctx.saved_tensors
    return grad, grad
register('add', Add)

class MUL(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x * y
  @staticmethod
  def backward(ctx, grad):
    x, y = ctx.saved_tensors
    return grad*y,grad*x
register('mul', MUL)

class pow(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return np.power(x, y)
  
  @staticmethod
  def backward(ctx, grad):
    x, y = ctx.saved_tensors

    return grad * np.power(x, y-1) * y, grad * np.power(x, y) * np.log(x)
register('pow', pow)

class log(Function):
   @staticmethod
   def forward(ctx, x):
     ctx.save_for_backward(x)
     a=np.where(x>0,x,-x)
     b=np.where(x>0,1,-1)
     return b * np.log(a) 
   @staticmethod
   def backward(ctx, grad):
     x = ctx.saved_tensors[0]
     return grad * (1/x)
register('log',log)

class Sum(Function):
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return np.array([np.sum(x)]) # if used np.sum returns float64 instead of ndrarry throwing error at init tensor init

  @staticmethod
  def backward(ctx, grad):
    x = ctx.saved_tensors[0]
    return grad * np.ones_like(x)
register('sum', Sum)

class Div(Function):
   @staticmethod
   def forward(ctx, x, y):
     return MUL.forward(ctx,x, 1/y)
   @staticmethod
   def backward(ctx, grad):
      return MUL.backward(ctx,grad)
register('div', Div)

class Sub(Function):
  @staticmethod
  def forward(ctx, x,y):
    ctx.save_for_backward(x,y)
    return x-y
  @staticmethod
  def backward(ctx, grad):
    x, y = ctx.saved_tensors
    return grad, -grad
register('sub', Sub)
  
class Matmul(Function):
   @staticmethod
   def forward(ctx, x, y):
     ctx.save_for_backward(x,y)
     return np.matmul(x,y)
   
   @staticmethod
   def backward(ctx, grad):
     x, y = ctx.saved_tensors
     return np.matmul(grad, y.T), np.matmul(x.T, grad)
register('matmul',Matmul)

class sigmmoid(Function):
   @staticmethod
   def forward(ctx, x):
    out=1.0/(1.0+np.exp(-x))
    ctx.save_for_backward(np.array([out]))
    return out
   @staticmethod
   def backward(ctx,grad):
      x=ctx.saved_tensors
      return x*(1-x)*grad
register('sigmoid', sigmmoid)

x=layer__init(3,3)
a=Tensor(x)
b=Tensor(np.array([3]))
z=a.log()
print('z',z)
z.mean().backward()
print(a.grad)
import torch
y=torch.tensor(x)
p=torch.log(y)
print(p)
print((z.data==p.values).all())



# class Sigmoid(Function)