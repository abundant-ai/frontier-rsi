"""Private full-workload packing validation, independent of candidate Python.

Timing remains BF16. FP32 gradients avoid mistaking shape-dependent BF16
rounding for a change in the mathematical training computation.
"""
from pathlib import Path
import torch
from packing import padded,check
from performance import documents_for
from model import GPT,Config,token_loss

LAYOUT_KEYS=('layout_x','layout_y','layout_mask','layout_pos')


def checked_layout(outputs,documents):
 if set(outputs)!=set(LAYOUT_KEYS):raise ValueError('packing output must contain exactly four layout tensors')
 layout=tuple(outputs[k] for k in LAYOUT_KEYS)
 check(documents,layout)
 return layout


def gradients(model,layout,device,huggingface=False):
 model.zero_grad(set_to_none=True)
 x,y,mask,pos=(t.to(device) for t in layout)
 # No autocast and no TF32: this is the independent FP32 numerical check.
 if huggingface:
  additive=torch.zeros(mask.shape,device=device,dtype=torch.float32).masked_fill(~mask,float('-inf'))
  logits=model(input_ids=x,attention_mask=additive,position_ids=pos).logits
 else:logits=model(x,mask=mask,position_ids=pos)
 loss=token_loss(logits,y);loss.backward()
 result={'loss':loss.detach().cpu(),**{n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}}
 if any(not torch.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite FP32 validation result')
 return result


def validate(outputs,seed,device,assets,smoke=False,tolerance=.02):
 documents=documents_for(seed,assets,smoke)
 layout=checked_layout(outputs,documents)
 torch.manual_seed(seed)
 if smoke:model=GPT(Config(vocab=64,width=32,layers=2,heads=4,context=64))
 else:
  from transformers import AutoModelForCausalLM
  model=AutoModelForCausalLM.from_pretrained(assets/'model',local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
  model.config.use_cache=False
  # Recompute activations only in the trusted numerical check to keep full
  # FP32 validation within one H100. Candidate timing is unaffected.
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 model.to(device).train()
 if device=='cuda':
  old_matmul=torch.backends.cuda.matmul.allow_tf32;old_cudnn=torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 try:
  reference=gradients(model,padded(documents,0),device,not smoke)
  candidate=gradients(model,layout,device,not smoke)
 finally:
  if device=='cuda':
   torch.backends.cuda.matmul.allow_tf32=old_matmul;torch.backends.cudnn.allow_tf32=old_cudnn
 if set(reference)!=set(candidate):raise ValueError('FP32 gradient keys differ')
 worst=0.;worst_name=''
 for name,expected in reference.items():
  actual=candidate[name]
  error=float((actual-expected).norm()/expected.norm().clamp_min(1e-8))
  if error>worst:worst=error;worst_name=name
 if worst>tolerance:raise ValueError(f'FP32 loss/gradient mismatch {worst_name}: NRMSE={worst:.6g}, limit={tolerance}')
 return {'precision':'float32','worst_nrmse':worst,'worst_tensor':worst_name,'tokens':sum(len(d)-1 for d in documents)}
