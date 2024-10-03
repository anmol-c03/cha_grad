import numpy as np
from tqdm import tqdm,trange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from cha_grad.utils import Dataloader,fetch_data
from cha_grad.nn import SGD


np.set_printoptions(suppress=True)

class scratch_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(28*28,128)
        self.l2=nn.Linear(128,10)
    
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=self.l2(x)
        return x
    
batch_size=32
train,test=fetch_data()
x_train=Dataloader(train,batch_size)
x_test=Dataloader(test,batch_size)

model=scratch_net()
lr=0.001
optim=SGD([model.l1.weight,model.l2.weight],lr)
losses,accs=[],[]
steps=2
for i in (t := trange(steps)):
    optim.zero_grad()
    samp=torch.randint(0,len(x_train),(batch_size,))
    x,y=x_train[samp]
    logits=model(x)
    loss=F.cross_entropy(logits,y)
    loss.backward()
    optim.step()
    losses.append(loss.item())
    pred=torch.argmax(logits,dim=-1)
    acc=(pred==y).float().mean()
    accs.append(acc)
    t.set_description(f'{loss:.2f},{acc:.2f}')
    
plt.plot(accs)
plt.plot(losses)
plt.ylim(0,1.5)

