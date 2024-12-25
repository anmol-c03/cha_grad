import numpy as np
import ssl
import torchvision.datasets as datasets


def fetch_data():

    ssl._create_default_https_context = ssl._create_unverified_context


    train = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    test = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    return train, test

class Dataloader:
    def __init__(self,data,bs):
        self.data=data
        self.bs=bs

    def __getitem__(self,idx):
        img=np.zeros((self.bs,784))
        label=np.zeros((self.bs,1))

        for i in range(len(idx)):
            img[i]=np.array(self.data[idx[i]][0]).reshape(-1,28*28).astype(np.float32)
            label[i]=(self.data[idx[i]][1])
        return (img,label.reshape(-1).astype(np.int32))

    def __len__(self):
        return len(self.data)
    def __repr__(self):
        return f'Dataset with batch_size {self.bs}'
        
    def __str__(self):
        return f'Dataset with batch_size {self.bs}'
        