import math
import torch
from torch.nn import functional as F

def settings():
    # Schedule in optimizer steps; horizon is ~95% of the steps a 1800 s run
    # completes (measured 4.35 steps/s with packing and checkpointing).
    return {'horizon':7400,'boost':2.5,'boost_start':50,'boost_end':700,'decay_start':0.6,'eta_end':0.02,'scale_max':0.25,'temp_start':0.05,'temp_end':0.55}


def configure(shapes):
    # A zero-gradient clock: only decoupled weight decay touches it, so its
    # mean square shrinks by exactly (1-lr*wd)^2 per optimizer step.
    return [{'shape':(1048576,),'std':1.0}]


def pack(inputs,targets):
    mask=targets!=-100
    xs=inputs[mask];ys=targets[mask]
    n=xs.shape[0];width=inputs.shape[1]
    rows=(n+width-1)//width;pad=rows*width-n
    xs=torch.cat([xs,xs.new_zeros(pad)])
    ys=torch.cat([ys,ys.new_zeros(pad)-100])
    return xs.reshape(rows,width),ys.reshape(rows,width)


def eta(t):
    # Target multiplier on the Adam update ratio for steps t.
    c=settings();u=t/c['horizon']
    boost=torch.where((t<c['boost_end'])&(t>=c['boost_start']),torch.full_like(u,c['boost']),torch.ones_like(u))
    frac=((u-c['decay_start'])/(1-c['decay_start'])).clamp(0,1)
    decay=1-(1-c['eta_end'])*frac
    return boost*decay


def log_scale(steps):
    # Adam's bias-corrected update ratio is s_t/sqrt(vhat_t), vhat_t =
    # v_t/(1-b2^t). For a target ratio eta_t this gives exactly
    # log v_t = log v_{t-1} + log b2 - log(1-(1-b2)eta_t^2/(1-b2^t)),
    # s_t = eta_t*sqrt(vhat_t), starting from s_1 = 1.
    t=torch.arange(1,steps+1,dtype=torch.float64)
    c=1-torch.exp(t*math.log(0.999))
    e=torch.minimum(eta(t-1),(0.9*c/0.001).sqrt())
    step=math.log(0.999)-torch.log(1-0.001*e*e/c)
    step=torch.where(t<1.5,torch.full_like(t,math.log(0.001)),step)
    logv=step.cumsum(0)
    logs=torch.log(e)+0.5*(logv-torch.log(c))
    return torch.where(t<1.5,torch.zeros_like(t),logs)


def loss_scale(step):
    c=settings()
    peak=log_scale(c['horizon']+1).max()
    return math.exp(float(log_scale(step+1)[-1]-peak))*c['scale_max']


def temperature(step):
    # Logit multiplier for the training loss. The fixed initialization has
    # unit-variance tied embeddings (logit std ~28); training on scaled logits
    # keeps the softmax well conditioned, and raising the multiplier as
    # c0/(1-(1-c0)u) asks the final LayerNorm gain to shrink linearly, well
    # within Adam's per-step reach, until the deployed logits are trained at 1.
    c=settings();c0=c['temp_start']
    u=min(step/(c['temp_end']*c['horizon']),1.0)
    return c0/(1-(1-c0)*u)


def training_loss(forward,inputs,targets,auxiliary):
    clock=auxiliary[0]
    step=int(round(float(torch.log(clock.float().square().mean()))/(2*math.log(1-6e-5))))
    step=max(step,0)
    x,y=pack(inputs,targets)
    logits,hidden=forward(x)
    loss=F.cross_entropy(temperature(step)*logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100)
    return loss*loss_scale(step)+0.0*clock[0]
