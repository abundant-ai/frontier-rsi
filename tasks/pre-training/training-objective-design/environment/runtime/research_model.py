"""Model scaffolding for editable sequence computation.

Candidate hooks are loaded through the numerical edit language. The runner
manages model weights, data splits, and resource measurements.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARCH_ROWS=(7,8,10,11,12,13,14,15,21)

def defaults(row,smoke=False):
 return dict(width=32 if smoke else 768,depth=2 if smoke else 12,heads=4 if smoke else 12,
             vocab=64 if smoke else 50257,inner=128 if smoke else 3072,
             channels=8 if smoke else 64,units=8 if row==10 else 1,
             hidden=32 if smoke else (16276 if row==10 else 3072),extra=0)

def checked_config(row,values,smoke=False):
 base=defaults(row,smoke)
 if not isinstance(values,dict) or set(values)!=set(base):raise ValueError('configuration keys differ')
 if any(type(v)is not int or (v<0 if k=='extra' else v<=0) for k,v in values.items()):raise ValueError('positive integer dimensions required')
 if values['vocab']!=base['vocab'] or values['width']%values['heads'] or values['channels']%2:raise ValueError('vocabulary/head dimensions differ')
 editable={7:{'channels','extra'},8:{'width','depth','heads','inner','channels','hidden','extra'},11:{'width','depth','heads','inner','channels','hidden'},12:{'channels','inner','extra'},14:{'extra'},21:{'channels','inner','extra'}}.get(row,set())
 if any(values[k]!=v for k,v in base.items() if k not in editable):raise ValueError('fixed architecture dimensions differ')
 if max(values.values())>200000 or values['depth']>64 or values['units']>32:raise ValueError('configuration bounds')
 return values


def storage_bytes(value):
 seen=set()
 def visit(x):
  if isinstance(x,torch.Tensor):
   key=(x.device,x.untyped_storage().data_ptr());n=x.untyped_storage().nbytes()
   if key not in seen:seen.add(key);return n
   return 0
  if isinstance(x,(tuple,list)):return sum(visit(v) for v in x)
  if isinstance(x,dict):return sum(len(str(k).encode())+visit(v) for k,v in x.items())
  if x is None:return 0
  if type(x)is bool:return 1
  if type(x) in (int,float):return 8
  raise ValueError('state must contain only tensors and scalar metadata')
 return visit(value)


class SequenceModel(nn.Module):
 def __init__(self,c,hooks,row):
  super().__init__();self.c=c;self.hooks=hooks;self.row=row;self.benchmark_checks_on_device=False
  d=c['width'];h=c['heads'];q=h*c['channels']
  self.embedding=nn.Embedding(c['vocab'],d)
  self.layers=nn.ModuleList()
  for i in range(c['depth']):
   block=nn.ModuleDict({'norm':nn.LayerNorm(d),'qkv':nn.Linear(d,3*q),'output':nn.Linear(q,d),'norm2':nn.LayerNorm(d),
                        'up':nn.Linear(d,c['inner']),'down':nn.Linear(c['inner'],d)})
   if row==10 and i==c['depth']-1:
    block['router']=nn.Linear(d,c['units'],bias=False)
    block['experts']=nn.ModuleList([nn.Sequential(nn.Linear(d,c['hidden'],bias=False),nn.GELU(),nn.Linear(c['hidden'],d,bias=False)) for _ in range(c['units'])])
   if c['extra']:
    block['extra_a']=nn.Linear(d,c['extra'],bias=False);block['extra_b']=nn.Linear(c['extra'],d,bias=False)
   self.layers.append(block)
  self.norm=nn.LayerNorm(d);self.loads=[]
 def forward(self,ids,state=None,position=0,project_all=True):
  b,t=ids.shape;x=self.embedding(ids);new=[];self.loads=[]
  if state is None:state=[None]*len(self.layers)
  finite_outputs=[]
  for index,block in enumerate(self.layers):
   z=block['norm'](x);q,k,v=block['qkv'](z).chunk(3,-1)
   q,k,v=[a.view(b,t,self.c['heads'],self.c['channels']).transpose(1,2) for a in (q,k,v)]
   q,k=self.hooks.position(q,k,position)
   extra=(block['extra_a'].weight.transpose(0,1),block['extra_b'].weight.transpose(0,1)) if 'extra_a' in block else None
   y,s=self.hooks.mix(q,k,v,state[index],position,index,extra)
   if y.shape!=q.shape:raise ValueError('sequence hook output')
   if self.benchmark_checks_on_device or (self.row==7 and t==1):
    # Preserve the nonfinite-output check while synchronizing once for the
    # complete decode step rather than once per transformer layer.
    finite_outputs.append(torch.isfinite(y).all())
   elif not torch.isfinite(y).all():raise ValueError('sequence hook output')
   storage_bytes(s);new.append(s)
   x=x+block['output'](y.transpose(1,2).reshape(b,t,-1));z=block['norm2'](x)
   if 'experts' in block:
    scores=block['router'](z);flat=z.reshape(-1,z.shape[-1]);scores=scores.reshape(-1,scores.shape[-1])
    assignment=self.hooks.route(scores,self.loads,index)
    if assignment.shape!=scores.shape or not torch.isfinite(assignment).all() or (assignment<0).any() or not torch.allclose(assignment.sum(-1),torch.ones_like(assignment.sum(-1)),atol=1e-5):raise ValueError('routing weights must sum to one for every token')
    self.loads.append((assignment.detach()>0).sum(0));result=torch.zeros_like(flat)
    for unit,expert in enumerate(block['experts']):
     ix=torch.nonzero(assignment[:,unit]>0).flatten()
     if len(ix):result=result.index_add(0,ix,(expert(flat[ix])*assignment[ix,unit,None]).to(result.dtype))
    x=x+result.view_as(z)
   else:x=x+self.hooks.feedforward(z,(block['up'].weight,block['up'].bias,block['down'].weight,block['down'].bias,extra))
  if finite_outputs:
   finite=torch.stack(finite_outputs).all()
   if self.benchmark_checks_on_device:torch._assert_async(finite,'sequence hook output')
   elif not finite:raise ValueError('sequence hook output')
  return F.linear(self.norm(x if project_all else x[:,-1:]),self.embedding.weight),new


def validate_capacity(model,row,smoke=False):
 n=sum(p.numel() for p in model.parameters());limit=500_000_000 if row==8 else 500_000_000 if row==10 else 150_000_000
 if n>(2_000_000 if smoke else limit):raise ValueError('parameter cap exceeded')
 return n


def measured_forward(model,x,state=None,position=0,profile=False):
 if profile:
  from arithmetic import Arithmetic
  with Arithmetic() as accounting:result=model(x,state,position)
  if accounting.flops<=0:raise ValueError('arithmetic accounting unavailable')
  return result,accounting.flops/x.numel()
 return model(x,state,position),None
