"""Additional operation contracts and deterministic tensor workloads.

Straightforward implementations intentionally materialize intermediate results.
"""
import math
import torch
from torch.nn import functional as F


def reference(kind,a):
 if kind in ('op11','op12'):return a['a'] @ a['b']
 if kind=='op13':
  k=a['state'].float() @ a['key_projection'].float();v=a['state'].float() @ a['value_projection'].float()
  scores=a['q'].float() @ k.float().transpose(-1,-2)/math.sqrt(k.shape[-1])
  return (scores.softmax(-1) @ v.float()).to(a['q'].dtype)
 if kind=='op14':
  # Each token is assigned to two supplied matrices, with prescribed weights.
  results=[]
  for slot in range(a['assignments'].shape[1]):
   assigned=[]
   for row in range(a['x'].shape[0]):
    unit=int(a['assignments'][row,slot])
    assigned.append(a['x'][row] @ a['weights'][unit])
   results.append(torch.stack(assigned)*a['coefficients'][:,slot,None])
  return sum(results)
 if kind=='op15':
  # Return the reindexed payload and the sum restored to original token order.
  moved=a['x'].index_select(0,a['indices']);restored=torch.zeros_like(a['x'])
  restored.index_add_(0,a['indices'],moved)
  return moved,restored
 if kind=='op16':
  logits=a['logits'].float()/a['temperature'];values,order=logits.sort(dim=-1,descending=True,stable=True)
  rank=torch.arange(logits.shape[-1],device=logits.device)[None,:]
  values=values.masked_fill(rank>=a['k'],float('-inf'));prob=values.softmax(-1)
  remove=(prob.cumsum(-1)-prob)>=a['p'];prob=prob.masked_fill(remove,0)
  prob=prob/prob.sum(-1,keepdim=True)
  full=torch.zeros_like(prob).scatter(-1,order,prob)
  # Uniforms include zero: use the first strictly greater CDF value.
  # FP64 accumulation also prevents tail rounding into zero-mass tokens.
  cumulative=full.double().cumsum(-1);cumulative=(cumulative/cumulative[:,-1:]).contiguous();u=a['uniforms'].double().contiguous()
  sampled=torch.searchsorted(cumulative,u,right=True).clamp_max(logits.shape[-1]-1)
  return full,sampled
 raise ValueError(kind)


def inputs(kind,seed,index,device='cpu',smoke=False):
 g=torch.Generator(device=device).manual_seed(seed*1009+index)
 dtype=torch.float32 if device=='cpu' else torch.bfloat16
 def rand(*shape):return torch.randn(shape,generator=g,device=device,dtype=dtype)*.3
 if kind in ('op11','op12'):
  sizes=(16,24,32) if smoke else (64,127,256,1024,2048,4096,8192)
  m,k,n=[sizes[(index+j)%len(sizes)] for j in range(3)]
  a,b=rand(m,k),rand(k,n)
  if kind=='op12' and index%4==0:a[0,:]*=32;b[:,0]*=32
  return {'a':a,'b':b}
 if kind=='op13':
  b=1 if smoke else (1,2,4,8,16,32)[index%6]
  t=12 if smoke else (1024,2048,4096,8192,16384,32768)[index%6]
  latent=8 if smoke else (256,512)[index%2];d=8 if smoke else 128
  while b*t*latent*2>2*1024**3:b//=2
  return {'q':rand(b,1,d),'state':rand(b,t,latent),'key_projection':rand(latent,d),'value_projection':rand(latent,d)}
 if kind=='op14':
  n=8 if smoke else (128,256,512,1024,2048,4096,8192)[index%7]
  width=16 if smoke else (512,1024,2048)[index%3];units=4 if smoke else (8,16,32)[index%3]
  return {'x':rand(n,width),'weights':rand(units,width,width),
          'assignments':torch.randint(units,(n,2),generator=g,device=device),
          'coefficients':torch.rand(n,2,generator=g,device=device,dtype=dtype)}
 if kind=='op15':
  n=16 if smoke else (128,512,2048,8192,32768,65536)[index%6];width=8 if smoke else (128,512,1024,2048)[index%4]
  # Half of workloads include duplicated indices, exercising inverse accumulation.
  ids=torch.randperm(n,generator=g,device=device) if index%2==0 else torch.randint(n,(n*2,),generator=g,device=device)
  return {'x':rand(n,width),'indices':ids}
 if kind=='op16':
  b=2 if smoke else 1 if index<5 else (1,8,32,128,256)[index%5];v=16 if smoke or index<5 else (8192,32768,65536,131072)[index%4]
  samples=4096 if smoke else 1_000_000 if index<5 else 1
  return {'logits':rand(b,v),'uniforms':torch.rand(b,samples,generator=g,device=device),
          'k':min(v,(1,5,20,50,100)[index%5]),'p':(.8,.9,.95,1.)[index%4], 'temperature':(.5,1.,2.)[index%3]}
 raise ValueError(kind)
