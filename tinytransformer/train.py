import torch
from .model import TinyTransformer
def train(text, steps=300, block_size=64, seed=7):
    torch.manual_seed(seed); chars=sorted(set(text)); enc={c:i for i,c in enumerate(chars)}; data=torch.tensor([enc[c] for c in text])
    model=TinyTransformer(len(chars), min(block_size,max(4,len(text)-1))); opt=torch.optim.AdamW(model.parameters(),lr=3e-3); losses=[]
    for _ in range(steps):
        starts=torch.randint(0,max(1,len(data)-model.block_size-1),(16,)); x=torch.stack([data[i:i+model.block_size] for i in starts]); y=torch.stack([data[i+1:i+model.block_size+1] for i in starts]); _,loss=model(x,y); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    return model, enc, losses
def generate(model, enc, prompt='AI ', length=80):
    inv={v:k for k,v in enc.items()}; x=torch.tensor([[enc.get(c,0) for c in prompt]])
    for _ in range(length):
        logits,_=model(x[:,-model.block_size:]); x=torch.cat([x,torch.multinomial(torch.softmax(logits[:,-1,:],-1),1)],1)
    return ''.join(inv[i] for i in x[0].tolist())
