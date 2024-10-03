import numpy as np
from tqdm import tqdm,trange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.datasets as datasets


np.set_printoptions(suppress=True)


train = datasets.MNIST(root='./data', train=True, download=True, transform=None)
test = datasets.MNIST(root='./data', train=False, download=True, transform=None)
class scratch_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(28*28,128)
        self.l2=nn.Linear(128,10)
    
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=self.l2(x)
        return x

class Dataloader:
    def __init__(self,data,bs):
        self.data=data
        self.bs=bs

    def __getitem__(self,idx):
        img=torch.zeros(self.bs,28*28)
        label=torch.zeros(self.bs,1)

        for i in range(len(idx)):
            img[i]=torch.tensor(np.array(self.data[idx[i]][0]).reshape(-1,28*28)).float()
            label[i]=torch.tensor((self.data[idx[i]][1]))
        return (img,label.view(-1).long())

    def __len__(self):
        return len(self.data)
    def __repr__(self):
        return f'Dataset with batch_size {self.bs}'
        
    def __str__(self):
        return f'Dataset with batch_size {self.bs}'
        
batch_size=32
x_train=Dataloader(train,batch_size)
x_test=Dataloader(test,batch_size)

model=scratch_net()
losses,accs=[],[]
optim=torch.optim.Adam(model.parameters())
for i in (t := trange(1000)):
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
    t.set_description(f'{loss},{acc}')
    
plt.plot(accs)
plt.plot(losses)
plt.ylim(0,1.5)

samp=torch.randint(0,len(x_test),(batch_size,))
x,y=x_test[samp]
logits=model(x)
pred=torch.argmax(logits,dim=-1)
loss=F.cross_entropy(logits,y)
print(y)
print(pred)
acc=(pred==y).float().mean()
print(acc)
print(loss)