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
        if len(self._prev.parents)==1:
            grads=[grads]
        for prev_node,grad in zip(self._prev.parents,grads):
            if prev_node.data.shape!=grad.shape:
                print(grad,grad.shape)
                print(f'Error: dimensions of gradients {grad.shape} and parameters {prev_node.data.shape} do not match')
                assert(False)
            print(grad,grad.shape)
            prev_node.grad=grad
            prev_node.backward(False)
    
    def mean(self):
        p=Tensor(np.array([1/self.data.size]))
        return self.sum().mul(p)
    
    def cross_entropy(self,y):
       return  self.mul(y).mean()


    
class Function:
    def __init__(self,*tensors):
        self.parents = tensors 
        self.saved_tensors = []
      
    def __repr__(self) -> str:
        return f'{self.__class__.__name__} with parents {self.parents}'
    
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



def equal_size(x,y,res_grad):
    count=0
    while len(x.shape)!=len(y.shape): 
      x=np.expand_dims(x,axis=0)
      count+=1
    for i in range(len(x.shape)):
      if x.shape[i]==1:
          res_grad=res_grad.sum(i,keepdims=True)
    
    for _ in range(count):
        res_grad=res_grad.squeeze(0)
    return res_grad
          # return np.array([res_grad])

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
    if x.size==y.size:
      return grad, grad
    else:  
      max=x.size>y.size    
      if max:
        grad_=equal_size(y,x,grad)
        return grad,grad_
      else:
        grad_=equal_size(x,y,grad)
        return grad_,grad
register('add', Add)

class MUL(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x * y
  @staticmethod
  def backward(ctx, grad):
    x, y = ctx.saved_tensors
    grad_x=grad*y
    grad_y=grad*x
    if x.size==y.size:
      return grad_x, grad_y
    else:
      max=x.size>y.size
      if max:
        grad_=equal_size(y,x,grad_y)
        return grad_x,grad_
      else:
        grad_=equal_size(y,x,grad_x)
        return grad_,grad_y
register('mul', MUL)

class pow(Function):
  @staticmethod
  def forward(ctx, x, y):
    out= np.power(x, y)
    ctx.save_for_backward(x, y,out)
    return out
  
  @staticmethod
  def backward(ctx, grad):
    x, y,out = ctx.saved_tensors
    if not (x>0).all():
       raise Exception ('Domain error of log results error in backprop ')

    else:
      return grad * np.power(x, y-1) * y, (grad * out * np.log(x)).sum()

register('pow', pow)

class log(Function):
  ''' This is special implementation of log function i prepared for special purposes
      How it differs?
      Normal log when encounters negative number  give domain error
      This log is used in special case when user want to take log value abs value of all element and then assign sign to the log value
       
        
         Example:
          x=[-1,2,3,-4]
          b=[-1,1,1,-1]
           
          output:
           np.log(abs(x)) doesnot throw domain error as all values are positive  
           b*nlog(a) assign sign correspondingly'''
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    b=np.where(x>0,1,-1)
    return b * np.log(abs(x)) 
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
    x,_ = ctx.saved_tensors
    return grad * np.ones_like(x)
register('sum', Sum)

class Div(Function):
  @staticmethod
  def forward(ctx, x, y):
    if y.any()==0:
      raise ValueError ('Divide by zero error')
    ctx.save_for_backward(x, y)
    return x/y 
  
  @staticmethod
  def backward(ctx, grad):
    x, y = ctx.saved_tensors
    grad_x=grad/y
    grad_y=-grad*x*np.power(y,-2)
    if x.size==y.size:
      return grad_x, grad_y
    else:
      max=x.size>y.size
      if max:
        grad_=equal_size(y,x,grad_y)
        return grad*y,grad_
      else:
        grad_=equal_size(y,x,grad_x)
        return grad_,grad*x
register('div', Div)

class Logsoftmax(Function):
  @staticmethod
  def forward(ctx, logits):
     max_=logits.max(1,keepdims=True)
     out=logits-(max_+np.log(np.exp(logits-max_).sum(1,keepdims=True)))
     ctx.save_for_backward(out)
     return out
  
  @staticmethod
  def backward(ctx, grad):
    output, = ctx.saved_tensors
    return grad - np.exp(output)*grad.sum(axis=1).reshape((-1, 1))
register('log_softmax', Logsoftmax)
  
class Sub(Function):
  @staticmethod
  def forward(ctx, x,y):
    ctx.save_for_backward(x,y)
    return x-y
  
  @staticmethod
  def backward(ctx, grad):
    x, y = ctx.saved_tensors
    if x.size==y.size:
       return grad, grad
    else:  
      max=x.size>y.size    
      if max:
        grad_=equal_size(y,x,grad)
        return grad,-grad_
      else:
        grad_=equal_size(x,y,grad)
        return grad_,-grad
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
    ctx.save_for_backward(out)
    return out
   @staticmethod
   def backward(ctx,grad):
      x=ctx.saved_tensors[0]
      return x*(1-x)*grad
register('sigmoid', sigmmoid)

class exp(Function):
   @staticmethod
   def forward(ctx, x):
    out=np.exp(x)
    ctx.save_for_backward(out)
    return out
   @staticmethod
   def backward(ctx, grad):
      out=ctx.saved_tensors[0]
      return out*grad
register('exp',exp)


# class Conv2D(Function):
#    #img (in_c,h,w)
#    @staticmethod
#    def forward(ctx, input,k_size, bias=None, stride=`1, padding=0):
#       in_c,out_c,=ctx.saved_tensors
#       for _ in range(out_c):
#         temp=np.random.uniform(-1,1,(in_c,k_size[0],k_size[1]))     
