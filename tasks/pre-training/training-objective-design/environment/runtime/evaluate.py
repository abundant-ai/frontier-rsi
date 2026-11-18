"""Trusted evaluation of safetensors weights, never imports candidate code."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from model import Config,GPT,token_loss
from train import answer,encode_pair,logprob


def checked_state(model,path):
 state=load_file(str(path),device='cpu');expected=model.state_dict()
 if set(state)!=set(expected):raise ValueError('checkpoint tensor keys differ from fixed model')
 for key,tensor in state.items():
  if tensor.shape!=expected[key].shape or not torch.isfinite(tensor).all():raise ValueError('invalid checkpoint tensor '+key)
 model.load_state_dict(state,strict=True)


@torch.no_grad()
def score(c,weights,data,assets,device='cuda',smoke=False,limit=None,max_new_tokens=2048,measurement=None):
 if c['family']=='text':
  cfg=Config(vocab=64,width=32,layers=2,heads=4,context=64) if smoke else Config()
  model=GPT(cfg);checked_state(model,weights);model.to(device).eval()
  tokens=np.load(data);length=16 if smoke else 1024
  total=0.;count=0
  token_budget=128 if smoke else 2_000_000
  if limit is not None:
   if not isinstance(limit,int) or limit<1:raise ValueError('evaluation limit must be positive')
   token_budget=min(token_budget,limit*length)
  for offset in range(0,min(len(tokens)-length-1,token_budget),length):
   z=torch.tensor(np.array(tokens[offset:offset+length+1]),device=device,dtype=torch.long)
   loss=token_loss(model(z[:-1][None]),z[1:][None]);total+=float(loss)*length;count+=length
  if not count:raise ValueError('empty evaluation split')
  if measurement is not None:measurement.update(evaluated_next_tokens=count,window_count=count//length,context_length=length,loss_sum=total)
  return total/count
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(assets/'model',local_files_only=True);tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(assets/'model',local_files_only=True,torch_dtype=torch.bfloat16 if device=='cuda' else torch.float32,attn_implementation='sdpa')
 if weights is not None:checked_state(model,weights)
 model.to(device).eval();rows=json.loads(Path(data).read_text())
 if limit is not None:rows=rows[:limit]
 if c['mode'] in ('policy','allocation'):
  correct=0
  for row in rows:
   prompt=tok.apply_chat_template([{'role':'user','content':row['question']}],tokenize=False,add_generation_prompt=True)
   x=tok(prompt,return_tensors='pt',truncation=True,max_length=1024).input_ids.to(device)
   seq=model.generate(x,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
   response=tok.decode(seq[0,x.shape[1]:],skip_special_tokens=True)
   correct+=answer(response) is not None and answer(response)==answer(row['answer'])
  return correct/len(rows)
 if c['mode']=='preferences':
  reference=AutoModelForCausalLM.from_pretrained(assets/'model',local_files_only=True,torch_dtype=model.dtype).to(device).eval()
  correct=0.
  for row in rows:
   x,y=encode_pair(tok,row['chosen'],device);r,ry=encode_pair(tok,row['rejected'],device)
   margin=float((logprob(model,x,y)-logprob(reference,x,y))-(logprob(model,r,ry)-logprob(reference,r,ry)))
   correct+=1 if margin>0 else .5 if margin==0 else 0
  return correct/len(rows)
 # Fixed candidate-set task accuracy for adaptation, with independently held-out preference labels.
 correct=0.
 for row in rows:
  x,y=encode_pair(tok,row['chosen'],device);r,ry=encode_pair(tok,row['rejected'],device)
  delta=float(logprob(model,x,y)-logprob(model,r,ry));correct+=1 if delta>0 else .5 if delta==0 else 0
 return correct/len(rows)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--weights');p.add_argument('--baseline',action='store_true');p.add_argument('--data',required=True);p.add_argument('--assets',default='/opt/assets/public');p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--max-new-tokens',type=int,default=2048);p.add_argument('--out');a=p.parse_args()
 if not a.weights and not a.baseline:p.error('--weights or --baseline is required')
 value=score(json.loads(Path(a.config).read_text()),Path(a.weights) if a.weights else None,Path(a.data),Path(a.assets),a.device,a.smoke,a.limit,a.max_new_tokens)
 if not math.isfinite(value):raise ValueError('nonfinite score')
 if a.out:Path(a.out).write_text(json.dumps({'score':value}))
 print(json.dumps({'score':value}))
if __name__=='__main__':main()
