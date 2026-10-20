import torch
from torch import nn
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, block_size=64, emb=64, heads=4, layers=2):
        super().__init__(); self.block_size=block_size
        self.tok=nn.Embedding(vocab_size,emb); self.pos=nn.Embedding(block_size,emb)
        layer=nn.TransformerEncoderLayer(emb, heads, dim_feedforward=4*emb, batch_first=True, dropout=0.0)
        self.encoder=nn.TransformerEncoder(layer,layers); self.head=nn.Linear(emb,vocab_size)
    def forward(self, x, targets=None):
        t=x.size(1); idx=torch.arange(t,device=x.device); h=self.tok(x)+self.pos(idx)[None,:,:]
        mask=torch.triu(torch.ones(t,t,device=x.device), diagonal=1).bool(); logits=self.head(self.encoder(h,mask=mask))
        loss=None
        if targets is not None: loss=nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss
