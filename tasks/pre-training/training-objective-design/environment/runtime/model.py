"""Fixed causal model used by the small-model training tasks."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab: int = 50257
    width: int = 768
    layers: int = 12
    heads: int = 12
    context: int = 2048


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.ln1 = nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, 3*c.width)
        self.proj = nn.Linear(c.width, c.width)
        self.ln2 = nn.LayerNorm(c.width)
        self.mlp = nn.Sequential(nn.Linear(c.width, 4*c.width), nn.GELU(), nn.Linear(4*c.width, c.width))

    def forward(self, x, mask=None):
        b,t,d = x.shape
        q,k,v = self.qkv(self.ln1(x)).chunk(3, -1)
        q,k,v = [z.view(b,t,self.heads,d//self.heads).transpose(1,2) for z in (q,k,v)]
        # The fused implementation preserves causal dependencies without
        # materializing a quadratic FP32 attention matrix for every layer.
        if mask is None:
            h = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        else:
            causal = torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
            h = F.scaled_dot_product_attention(q,k,v,attn_mask=causal & mask)
        h = h.transpose(1,2).contiguous().view(b,t,d)
        x = x + self.proj(h)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c=Config()):
        super().__init__()
        self.config = c
        self.wte = nn.Embedding(c.vocab,c.width)
        self.wpe = nn.Embedding(c.context,c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.ln = nn.LayerNorm(c.width)
        self.head = nn.Linear(c.width,c.vocab,bias=False)
        self.head.weight = self.wte.weight

    def forward(self, x, mask=None, position_ids=None):
        pos = torch.arange(x.shape[1],device=x.device) if position_ids is None else position_ids
        h = self.wte(x)+self.wpe(pos)
        for block in self.blocks:
            h = block(h,mask)
        return self.head(self.ln(h))


def token_loss(logits, targets):
    # Recompute each small FP32 softmax during backward rather than retaining
    # several full [tokens,vocabulary] FP32 buffers. The token objective and its
    # normalization are identical, including ignored targets.
    from torch.utils.checkpoint import checkpoint
    flat=logits.reshape(-1,logits.shape[-1]);labels=targets.reshape(-1)
    count=(labels!=-100).sum()
    def summed_loss(x,y):
        return F.cross_entropy(x.float(),y,ignore_index=-100,reduction='sum')
    total=logits.new_zeros((),dtype=torch.float32)
    for start in range(0,len(labels),256):
        x=flat[start:start+256];y=labels[start:start+256]
        value=checkpoint(summed_loss,x,y,use_reentrant=False) if torch.is_grad_enabled() and x.requires_grad else summed_loss(x,y)
        total=total+value
    return total/count
