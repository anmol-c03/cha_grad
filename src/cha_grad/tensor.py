# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
# inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
from functools import partialmethod
import numpy as np

# **** start with two base classes ****

class Tensor:
  def __init__(self, data):
    # print(type(data), data,'\n')
    if type(data) != np.ndarray:
      print("error constructing tensor with %r" % data)
      assert(False)
    self.data = data
    self.grad = None

    # internal variables used for autograd graph construction
    self._ctx = None

  def __repr__(self):
    return "Tensor %r with grad %r" % (self.data, self.grad)

  def backward(self, allow_fill=True):
    #print("running backward on", self)
    if self._ctx is None:
      return

    if self.grad is None and allow_fill:
      # fill in the first grad with one
      # this is "implicit gradient creation"
      assert self.data.size == 1
      self.grad = np.ones_like(self.data)

    # print(f'grad{self.grad}')
    assert(self.grad is not None)
    print(self._ctx)
    grads = self._ctx.backward(self._ctx, self.grad)
    print(grads)
    if len(self._ctx.parents) == 1:
      grads = [grads]
    # print(grads)
    # print(self._ctx.parents)
    for t,g in zip(self._ctx.parents, grads):
      if g.shape != t.data.shape:
        print("grad shape must match tensor shape in %r, %r != %r" %
          (self._ctx, g.shape, t.data.shape))
        assert(False)
      t.grad = g
      t.backward(False)

  def mean(self):
    div = Tensor(np.array([1/self.data.size]))
    return self.sum().mul(div)

# An instantiation of the Function is the Context
class Function:
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  # note that due to how partialmethod works, self and arg are switched
  def apply(self, arg, *x):
    # support the args in both orders
    if type(arg) == Tensor:
      op = self
      x = [arg]+list(x)
    else:
      op = arg
      x = [self]+list(x)
    ctx = op(*x)
    ret = Tensor(op.forward(ctx, *[t.data for t in x]))
    ret._ctx = ctx
    return ret

def register(name, fxn):
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))

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

class Sub(Function):
  @staticmethod
  def forward(ctx, x,y):
    ctx.save_for_backward(x,y)
    return x-y
  @staticmethod
  def backward(ctx, grad):
    x, y = ctx.saved_tensors
    return grad, -grad if x>y else -grad,grad
register('sub', Sub)

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

a=Tensor(np.random.uniform(-1, 1,(2,3)))
b=Tensor(np.random.uniform(-1, 1,(2,3)))
z=a.add(b).mean()
print(f'z is {z}')
z.backward(True)
# print(z)
