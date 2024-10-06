# inspired from https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
import functools
import numpy as np

np.random.seed(1337)
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
   
a=Tensor(np.random.uniform(-1,1,(3,3)))
b=np.random.uniform(-2,1,(3,3))
c=Tensor(b)
b=-b
d=Tensor(b)
z=a.sub(c)
y=a.add(d)
z.mean().backward()
e=c.grad
y.mean().backward()
f=d.grad
print(e,'\n',f)





