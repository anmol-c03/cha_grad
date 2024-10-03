class optimizer:
    def __init__(self,tensors,lr=0.001):
        self.tensors=tensors
        self.lr=lr
        
    def zero_grad(self):
        for t in self.tensors:
            if t.grad is not None:
                t.grad=None
class SGD(optimizer):
    
    def step(self):
        for t in self.tensors:
            t.data-=self.lr * t.grad  

