import torch

def execute_layer(summary,index,total):
 return True

def propose(tokens,history,limit):
 return tokens[:,:0]

def encode(tensor):
 return tensor.clone()

def decode(payload):
 return payload

def retain(keys,values,capacity,position):
 n=keys.shape[-2]
 return torch.arange(max(0,n-capacity),n,device=keys.device)

def compress(tokens,question):
 return tokens[:,:2048]

def schedule(requests,clock):
 if len(requests)==0:
  return []
 return [(requests[0]['id'],256 if requests[0]['remaining']>0 else 1)]

def plan(question,history,remaining):
 if len(history)==0:
  return ('sample',512)
 return ('answer',history[0]['answer'])

def calibrate(tensor,statistics):
 return encode(tensor)
