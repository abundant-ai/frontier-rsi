"""Unoptimized operation contracts and reproducible workload generation.

This module defines the reference operations and workload generation.
"""
import math
import torch
from torch.nn import functional as F


def attention(q,k,v,allowed=None):
    logits = q.float() @ k.float().transpose(-1,-2) / math.sqrt(q.shape[-1])
    if allowed is not None:
        logits = logits.masked_fill(~allowed, float('-inf'))
    return (logits.softmax(-1) @ v.float()).to(q.dtype)


def reference(kind, a):
    if kind in ('op11','op12','op13','op14','op15','op16'):
        from extra_kernel_ops import reference as extra_reference
        return extra_reference(kind,a)
    if kind in ('op01','op04'):
        return attention(a['q'],a['k'],a['v'],a.get('mask'))
    if kind=='op02':
        return a['a'] @ a['b']
    if kind=='op03':
        return tuple(a[f'a{i}'] @ a[f'b{i}'] for i in range(a['count']))
    if kind=='op05':
        pages=a['pages']; table=a['table']
        k=torch.cat([pages[table[i],0] for i in range(table.numel())],dim=-2)
        v=torch.cat([pages[table[i],1] for i in range(table.numel())],dim=-2)
        return attention(a['q'],k.unsqueeze(0),v.unsqueeze(0))
    if kind=='op06':
        return attention(a['q'],a['k'],a['v'],a['mask'])
    if kind=='op07':
        x=a['x']; norm=(x.float().square().mean(-1,keepdim=True)+1e-5).rsqrt()
        return (x.float()*norm*a['weight'].float()).to(x.dtype)
    if kind=='op08':
        return (a['x'] * torch.sigmoid(a['x'])) * a['gate']
    if kind=='op09':
        logits=a['logits'].float()
        logprobs=logits-logits.exp().sum(-1,keepdim=True).log() if a.get('unstable',False) else logits-logits.logsumexp(-1,keepdim=True)
        return -logprobs.gather(-1,a['labels'][:,None]).mean()
    if kind=='op10':
        result=[]
        for i in range(a['count']):
            p,g,m,v=[a[f'{key}{i}'] for key in ('p','g','m','v')]
            m1=.9*m+.1*g; v1=.999*v+.001*g*g
            step=a['step']; update=(m1/(1-.9**step))/((v1/(1-.999**step)).sqrt()+1e-8)
            result.extend((p*(1-1e-3*.01)-1e-3*update,m1,v1))
        return tuple(result)
    raise ValueError(kind)


def inputs(kind,seed,index,device='cpu',smoke=False):
    if kind in ('op11','op12','op13','op14','op15','op16'):
        from extra_kernel_ops import inputs as extra_inputs
        return extra_inputs(kind,seed,index,device,smoke)
    g=torch.Generator(device=device).manual_seed(seed*1009+index)
    dtype=torch.float32 if device=='cpu' or kind in ('op07','op08','op10') and index%2 else torch.bfloat16
    def rand(*shape):return torch.randn(shape,generator=g,device=device,dtype=dtype)*.3
    if kind in ('op01','op04','op06'):
        t=(16,24,32)[index%3] if smoke else (256,512,1024,2048,4096,8192)[index%6]
        h=2 if smoke else (8,16)[index%2]; d=8 if smoke else (64,128)[index%2]
        b=1 if smoke else 1+index%4; qt=1 if kind=='op04' else t
        if not smoke and kind=='op04':
            t=(1024,2048,4096,8192,16384,32768,65536)[index%7];b=(1,2,4,8,16,32)[index%6]
            while b>1 and b*h*t*d*4>4*1024**3:b//=2
        if not smoke and kind!='op04':
            while b>1 and b*h*t*t*16>16*1024**3:b//=2
        out={'q':rand(b,h,qt,d),'k':rand(b,h,t,d),'v':rand(b,h,t,d)}
        if kind=='op01':out['mask']=torch.ones(t,t,device=device,dtype=torch.bool).tril()
        if kind=='op06':
            block=8 if smoke else (64,128)[index%2]
            n=(t+block-1)//block
            mask=torch.rand(n,n,generator=g,device=device)<(.1+.1*(index%4))
            mask.fill_diagonal_(True)
            out['mask']=mask.repeat_interleave(block,0).repeat_interleave(block,1)[:t,:t]
        return out
    if kind in ('op02','op03'):
        dims=(16,24,32) if smoke else (128,256,511,1024,2048,4096,8192)
        count=((2+index%3) if smoke else (2,4,8,16,32,64)[index%6]) if kind=='op03' else 1
        result={'count':count}
        for i in range(count):
            m,n,k=[dims[(index+i+j)%len(dims)] for j in range(3)]
            if kind=='op03' and not smoke:m,n,k=min(m,2048),min(n,2048),min(k,2048)
            result[f'a{i}']=rand(m,k);result[f'b{i}']=rand(k,n)
        if kind=='op02':return {'a':result['a0'],'b':result['b0']}
        return result
    if kind=='op05':
        n=4 if smoke else (32,64,128,256,512,1024)[index%6]
        page=4 if smoke else (16,32)[index%2]; h=2 if smoke else 8;d=8 if smoke else 64
        return {'q':rand(1,h,1,d),'pages':rand(n,2,h,page,d),'table':torch.randperm(n,generator=g,device=device)}
    if kind in ('op07','op08'):
        m=8 if smoke else (1,128,1024,4096)[index%4]
        n=16 if smoke else (128,512,1024,4096,8192)[index%5]
        return {'x':rand(m,n),**({'weight':rand(n)} if kind=='op07' else {'gate':rand(m,n)})}
    if kind=='op09':
        m=4 if smoke else (1,16,128,512)[index%4];v=32 if smoke else (8192,32768,65536,131072)[index%4]
        return {'logits':rand(m,v),'labels':torch.randint(v,(m,),generator=g,device=device)}
    if kind=='op10':
        out={'count':3 if smoke else 8+index%8,'step':1+index%100}
        size=32 if smoke else 8_000_000 if index==39 else (1024,16384,131072,1048576)[index%4]
        for i in range(out['count']):
            for key in ('p','g','m','v'):out[f'{key}{i}']=rand(size).abs() if key=='v' else rand(size)
        return out
    raise ValueError(kind)


def tensors(result):
    return list(result) if isinstance(result,(list,tuple)) else [result]


def output_and_gradients(fn,a,gradients=False):
    differentiable=[]
    a=dict(a)
    if gradients:
        for key,value in a.items():
            if isinstance(value,torch.Tensor) and value.is_floating_point():
                a[key]=value.detach().clone().requires_grad_(True);differentiable.append(a[key])
    out=tensors(fn(a))
    if gradients:
        loss=sum(x.float().square().mean() for x in out)
        grads=torch.autograd.grad(loss,differentiable,allow_unused=False)
        out+=list(grads)
    return [x.detach() for x in out]
