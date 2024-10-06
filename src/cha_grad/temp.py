# inspired from https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
import functools
import numpy as np

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
            assert self._prev is not None

            if self.grad is None and allow_fill==True:
                '''since gradient w.r.t  single value is always calculated 
                This line of code is executed only for last node in graph i.e output of NN because
                the gradient calculated is gradient  with itself which is always 1.'''

                assert self.data.size==1
                self.grad = np.ones_like(self.data) 
            
            assert(self.grad is not None)

            grads=self._prev.backward(self._prev,self.grad)
            if len(self._prev.parents)==1:
                grads=[grad]
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




