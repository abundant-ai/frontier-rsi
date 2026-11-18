import torch
import math

def configure(defaults):
 return defaults

def position(q,k,offset):
 d=q.shape[-1]
 p=torch.arange(offset,offset+q.shape[-2],device=q.device).float()
 frequency=torch.exp(torch.arange(0,d,2,device=q.device).float()*(-math.log(10000.)/d))
 angle=p[:,None]*frequency[None,:]
 cosine=angle.cos()[None,None,:,:];sine=angle.sin()[None,None,:,:]
 qr=torch.stack((q[...,0::2]*cosine-q[...,1::2]*sine,q[...,0::2]*sine+q[...,1::2]*cosine),-1).flatten(-2)
 kr=torch.stack((k[...,0::2]*cosine-k[...,1::2]*sine,k[...,0::2]*sine+k[...,1::2]*cosine),-1).flatten(-2)
 return qr.to(q.dtype),kr.to(k.dtype)

def mix(q,k,v,state,offset,layer,extra=None):
 if state is not None:
  k=torch.cat((state[0],k),-2);v=torch.cat((state[1],v),-2)
 if state is None:
  output=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
 elif q.shape[-2]==1:
  output=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.)
 else:
  query=torch.arange(offset,offset+q.shape[-2],device=q.device)
  key=torch.arange(k.shape[-2],device=k.device)
  output=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=key[None,:]<=query[:,None],dropout_p=0.)
 return output,(k,v)


def route(scores,loads,layer):
 return scores.softmax(-1)

def objective(logits,targets,state):
 return torch.nn.functional.cross_entropy(logits.float().flatten(0,1),targets.flatten(),ignore_index=-100)

def feedforward(x,weights):
 up,bias,down,output_bias,extra=weights
 return torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x,up,bias)),down,output_bias)
