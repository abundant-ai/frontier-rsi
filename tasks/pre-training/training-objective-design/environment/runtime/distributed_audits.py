"""Distributed gradient consistency utilities.

Gradient fragments are temporary scratch tensors.
"""
import hashlib,hmac
import torch

CHUNK_ELEMENTS=262144
SAMPLED_TENSORS=8
FULL_AUDITED_CALLS=32


def number(key,*parts):
 return int.from_bytes(hmac.new(bytes.fromhex(key),'|'.join(map(str,parts)).encode(),hashlib.sha256).digest(),'big')


def schedule(key,iterations):
 return sorted(sorted(range(iterations),key=lambda step:number(key,'audit',step))[:min(FULL_AUDITED_CALLS,iterations)])


def input_seed(key,seed,step):
 return number(key,'input',seed,step)%(2**63-1)


def snapshot(tensors,key,step,rank):
 names=list(tensors)
 if not names:raise ValueError('timed execution produced no tensors')
 if any(not isinstance(tensors[name],torch.Tensor) or tensors[name].numel()==0 for name in names):
  raise ValueError('invalid timed output or gradient')
 chosen=[]
 for pick in range(min(SAMPLED_TENSORS,len(names))):
  ticket=number(key,'tensor',step,rank,pick)%sum(tensors[name].numel() for name in names)
  for name in names:
   ticket-=tensors[name].numel()
   if ticket<0:break
  chosen.append(name);names.remove(name)
 result={}
 for name in chosen:
  value=tensors[name].detach().reshape(-1)
  # Large outputs get several independently placed fragments. Training chooses
  # eight parameter tensors by size, then one fragment within each tensor.
  for chunk in range(SAMPLED_TENSORS if len(tensors)==1 else 1):
   count=min(CHUNK_ELEMENTS,value.numel());start=number(key,'slice',step,rank,name,chunk)%(value.numel()-count+1)
   result[f'{name}|{chunk}|{start}|{count}']=value[start:start+count].float().cpu().clone()
 return result
