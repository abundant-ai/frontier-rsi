"""Executed tensor arithmetic accounting, including intermediate computation."""
import math
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class Arithmetic(TorchDispatchMode):
 def __init__(self):super().__init__();self.flops=0;self.operations={}
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  result=func(*args,**(kwargs or {}));name=str(func).split('.')[1]
  def count(x):
   if isinstance(x,torch.Tensor):return x.numel()
   if isinstance(x,(list,tuple)):return sum(count(v) for v in x)
   return 0
  n=count(result);cost=0
  if name in ('mm','bmm','addmm','baddbmm'):
   left,right=(args[1],args[2]) if name in ('addmm','baddbmm') else (args[0],args[1])
   cost=2*math.prod(left.shape[:-2])*left.shape[-2]*left.shape[-1]*right.shape[-1]
   if name in ('addmm','baddbmm'):cost+=n
  elif name in ('_softmax','softmax','_log_softmax','log_softmax'):cost=5*n
  elif 'layer_norm' in name or 'native_layer_norm' in name:cost=8*args[0].numel()
  elif name in ('gelu','silu','sigmoid','tanh','exp','log','sin','cos','sqrt','rsqrt','pow'):cost=8*n
  elif name in ('sum','mean','amax','amin','max','min','cumsum'):cost=args[0].numel()
  elif 'norm' in name or name in ('std','var'):cost=8*args[0].numel()
  elif name in ('add','sub','mul','div','clamp','clamp_min','clamp_max','square','reciprocal','add_','sub_','mul_','div_'):cost=n
  elif 'scaled_dot_product' in name:
   q,k,v=args[:3]
   pairs=math.prod(q.shape[:-2])*q.shape[-2]*k.shape[-2]
   cost=2*pairs*(q.shape[-1]+v.shape[-1])+5*pairs
  self.flops+=cost;self.operations[name]=self.operations.get(name,0)+cost
  return result
