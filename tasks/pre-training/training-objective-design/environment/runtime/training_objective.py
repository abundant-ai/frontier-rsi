"""Trusted training model and bounded auxiliary tensors for objective research."""
import math
import torch
from adapter import load as bounded_load

CAP = 250_000_000

def load(path,required=('configure','training_loss')):
    return bounded_load(path,required,allow_attention=True)


def prepare(model,candidate,cap=CAP):
    base=tuple(model.parameters())
    count=sum(p.numel() for p in base)
    shapes={name:tuple(p.shape) for name,p in model.named_parameters()}
    specs=candidate.configure(shapes)
    if type(specs) not in (tuple,list) or len(specs)>64:
        raise ValueError('configure returns up to 64 auxiliary tensor declarations')
    for spec in specs:
        if type(spec) is not dict or set(spec)!={'shape','std'}:
            raise ValueError('auxiliary declaration requires shape and std')
        shape,std=spec['shape'],spec['std']
        if type(shape) not in (tuple,list) or not 1<=len(shape)<=4 or any(type(n) is not int or n<1 for n in shape):
            raise ValueError('invalid auxiliary tensor shape')
        if type(std) not in (float,int) or not math.isfinite(std) or not 0<=std<=1:
            raise ValueError('invalid auxiliary initialization')
        count+=math.prod(shape)
        if count>cap:raise ValueError('total training-parameter cap')
    if count>cap:raise ValueError('total training-parameter cap')
    exemplar=base[0]
    auxiliary=tuple(torch.nn.Parameter(torch.randn(tuple(s['shape']),device=exemplar.device,dtype=exemplar.dtype)*s['std']) for s in specs)
    return base+auxiliary,auxiliary,count


def forward(model,ids):
    """The callback gets a callable, never the mutable nn.Module or optimizer."""
    if ids.ndim != 2 or not 1 <= ids.shape[1] <= 1024:
        raise ValueError('training objective requires 1–1024 input tokens')
    pos=torch.arange(ids.shape[1],device=ids.device)
    hidden=model.wte(ids)+model.wpe(pos)
    for block in model.blocks:hidden=block(hidden)
    hidden=model.ln(hidden)
    return model.head(hidden),hidden


def export(model):
    return {name:t.detach().cpu().contiguous().clone() for name,t in model.state_dict().items()}
