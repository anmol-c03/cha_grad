from scratch import x_test,model,batch_size
import torch
import torch.nn.functional as F

def eval(x_test,batch_size):

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

eval(x_test,batch_size)