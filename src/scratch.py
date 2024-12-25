import numpy as np
from tqdm import tqdm,trange

import matplotlib.pyplot as plt


from cha_grad.utils import Dataloader,fetch_data
from cha_grad.optim import SGD,Adam,AdamW
from cha_grad.tensor import Tensor,layer__init



np.set_printoptions(suppress=True)

class scratch_net():
    def __init__(self):
        super().__init__()
        self.l1=Tensor(layer__init(28*28,128))
        self.l2=Tensor(layer__init(128,10))
    
    def __call__(self,x):
        x=x.matmul(self.l1)
        x=x.relu()
        x=x.matmul(self.l2)
        x=x.log_softmax()
        return x
    
batch_size=32
train,test=fetch_data()
x_train=Dataloader(train,batch_size)
x_test=Dataloader(test,batch_size)

model=scratch_net()

# import sys;sys.exit(0)
lr=0.001
weight_decay=0.001
optim=AdamW([model.l1,model.l2],lr,weight_decay=weight_decay)
# optim=SGD([model.l1.weight,model.l2.weight],lr)
losses,accs=[],[]
steps=1000
def train():
    for i in (t := trange(steps)):
        optim.zero_grad()
        samp=np.random.randint(0,len(x_train),(batch_size,))
        x,y=x_train[samp]
        x=np.array(x)
        y=np.array(y)
        Y=np.zeros((len(samp),10),np.float32)
        Y[range(len(samp)),y]=-1
        logits=model(Tensor(x))
        loss=logits.cross_entropy(Tensor(Y))
        loss.backward()
        optim.step()
        loss=loss.data
        losses.append(loss)
        pred=np.argmax(logits.data,-1)
        acc=(pred==y).astype(float).mean()
        accs.append(acc)
        t.set_description(f'{loss.item():.2f} {acc.item():.2f}')
    
train()
# plt.plot(accs)
# plt.plot(losses)
# plt.ylim(0,1.5)
# plt.savefig('train.jpg')