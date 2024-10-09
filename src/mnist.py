from scratch import x_test,model,batch_size
import numpy as np
from cha_grad.tensor import Tensor
def eval(x_test,batch_size):

    samp=np.random.randint(0,len(x_test),(batch_size,))
    x,y=x_test[samp]
    x,y=np.array(x),np.array(y)
    Y=np.zeros((len(samp),10),np.float32)
    Y[range(len(samp)),y]=-1
    logits=model(Tensor(x))
    pred=np.argmax(logits.data,1)
    loss=logits.cross_entropy(Tensor(Y))
    print('y is',y)
    print('pred is ',pred)
    acc=(pred==y).astype(float).mean()
    print(f'acc is {acc:.2f}')
    print(f'Total_loss {loss.data.item():.2f}')

eval(x_test,batch_size)