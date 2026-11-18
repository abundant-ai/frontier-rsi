"""Tensor-only representation serialization; never loads Python objects."""
import json
from pathlib import Path
from safetensors.torch import save_file,load_file
import torch


def write(path,value):
 path=Path(path);path.mkdir(parents=True,exist_ok=True);tensors={};tensor_names={}
 def pack(x):
  if isinstance(x,torch.Tensor):
   if id(x) not in tensor_names:
    name=str(len(tensors));tensor_names[id(x)]=name;tensors[name]=x.detach().cpu().contiguous().clone()
   return {'tensor':tensor_names[id(x)]}
  if isinstance(x,tuple):return {'tuple':[pack(v) for v in x]}
  if isinstance(x,list):return {'list':[pack(v) for v in x]}
  if isinstance(x,dict):
   if not all(type(k)is str for k in x):raise ValueError('representation dictionary keys must be strings')
   return {'dict':{k:pack(v) for k,v in x.items()}}
  if x is None or type(x) in (int,float,bool):return {'scalar':x}
  raise ValueError('representation must contain tensors and scalar metadata only')
 tree=pack(value);save_file(tensors,str(path/'payload.safetensors'));(path/'layout.json').write_text(json.dumps(tree,allow_nan=False))

def read(path,device):
 path=Path(path);tensors=load_file(str(path/'payload.safetensors'),device=str(device))
 def unpack(x):
  if 'tensor' in x:return tensors[x['tensor']]
  if 'tuple' in x:return tuple(unpack(v) for v in x['tuple'])
  if 'list' in x:return [unpack(v) for v in x['list']]
  if 'dict' in x:return {k:unpack(v) for k,v in x['dict'].items()}
  return x['scalar']
 return unpack(json.loads((path/'layout.json').read_text()))
