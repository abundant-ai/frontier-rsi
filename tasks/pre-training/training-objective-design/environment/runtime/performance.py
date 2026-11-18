"""Public training-execution workload runner for input preparation and memory policy."""
import argparse,json,time,contextlib,secrets
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from model import GPT,Config,token_loss
from packing import padded,check
from restricted import load


def documents_for(seed,assets,smoke):
 random=np.random.default_rng(seed)
 if smoke:return [torch.tensor(random.integers(0,64,size=n),dtype=torch.long) for n in (9,13,17)]
 tokens=np.load(assets/'train.npy',mmap_mode='r');docs=np.load(assets/'packing_documents.npy')
 return [torch.from_numpy(np.array(tokens[int(docs[i,0]):int(docs[i,0])+min(2048,int(docs[i,1]))])).long() for i in random.integers(0,len(docs),size=8)]


def compute(mode,hook,seed,device,assets,smoke,iterations,input_seeds=None,audit_callback=None):
 torch.manual_seed(seed)
 if smoke:model=GPT(Config(vocab=64,width=32,layers=2,heads=4,context=64))
 elif mode=='batches':
  from transformers import AutoModelForCausalLM
  model=AutoModelForCausalLM.from_pretrained(assets/'model',local_files_only=True,torch_dtype=torch.bfloat16)
  # Common execution policy keeps the legal padded starter within the 64 GiB cap.
  model.config.use_cache=False
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 else:model=GPT()
 model.to(device).train();random=np.random.default_rng(seed)
 documents=documents_for(seed,assets,smoke) if mode=='batches' else None
 if not smoke and mode!='batches':
  tokens=np.load(assets/'train.npy',mmap_mode='r');docs=np.load(assets/('packing_documents.npy' if mode=='batches' else 'documents.npy'))
  documents=[]
  for idx in random.integers(0,len(docs),size=8):
   offset,n=map(int,docs[idx]);n=min(2048,n)
   if n>=2:documents.append(torch.from_numpy(np.array(tokens[offset:offset+n])).long())
 if mode=='batches':
  batch=hook.batch(documents,0) if hook else padded(documents,0);check(documents,batch)
 else:
  n=16 if smoke else 2048;b=2 if smoke else 8;v=64 if smoke else 50257
  x=torch.randint(v,(b,n),device=device);batch=(x,torch.roll(x,-1,1),None,None)
 batch=tuple(x.to(device) if x is not None else None for x in batch);x,y,mask,pos=batch
 elapsed=[];last=None
 for step in range(iterations):
  if input_seeds is not None and mode!='batches':
   # Fresh operands are prepared outside timing. A saved answer from an earlier
   # iteration must not be a valid answer to the next timed invocation.
   generator=torch.Generator(device=device).manual_seed(input_seeds[step])
   x=torch.randint(v,(b,n),generator=generator,device=device);y=torch.roll(x,-1,1)
  model.zero_grad(set_to_none=True)
  if device=='cuda':torch.cuda.synchronize()
  begin=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else contextlib.nullcontext():
   if mode=='batches' and not smoke:
    # HF attention masks are additive; document positions are reset by the batch contract.
    additive=torch.zeros_like(mask,dtype=model.dtype).masked_fill(~mask,float('-inf'))
    logits=model(input_ids=x,attention_mask=additive,position_ids=pos).logits
   elif mode=='batches':logits=model(x,mask=mask,position_ids=pos)
   else:logits=hook.forward(model,x) if hook else model(x)
   loss=token_loss(logits,y)
  loss.backward()
  if device=='cuda':torch.cuda.synchronize()
  elapsed.append((time.perf_counter()-begin)*1000)
  if mode=='batches':
   # Return only the layout. Dependencies and FP32 gradients are recomputed
   # independently; BF16 layout rounding is not truth.
   last={name:value.detach().cpu().contiguous() for name,value in zip(('layout_x','layout_y','layout_mask','layout_pos'),batch)}
  else:last={'loss':loss.detach().float().cpu(),**{name:p.grad.detach().float().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}}
  if audit_callback is not None:audit_callback(step,last)
 return last,sorted(elapsed)[len(elapsed)//2],int((y!=-100).sum())


def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--config',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--assets',default='/opt/assets/public');p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 c=json.loads(Path(a.config).read_text());m=load(a.candidate,c['entrypoints']);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(min(1.,c['memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory))
 quick=c.get('validation_profile')=='quick-gpu-v1'
 iterations=2 if a.smoke or quick else 100
 audits=[];input_seeds=None;callback=None
 if c['mode']!='batches':
  input_seeds=[secrets.randbits(63) for _ in range(iterations)]
  # Selection stays in the trusted runner, outside the submitted numerical API.
  # Audit multiple timed calls, including the final full-gradient check.
  selected=set(secrets.SystemRandom().sample(range(iterations),min(32,iterations)))|{iterations-1}
  def callback(step,tensors):
   if step not in selected:return
   names=[name for name in tensors if name!='loss']
   if step!=iterations-1:
    # Weight by tensor size so large matrix gradients cannot hide behind many
    # tiny bias/normalization tensors. Sampling is without replacement.
    remaining=names;names=[]
    for _ in range(min(8,len(remaining))):
     total=sum(tensors[name].numel() for name in remaining);ticket=secrets.randbelow(total)
     for name in remaining:
      ticket-=tensors[name].numel()
      if ticket<0:break
     names.append(name);remaining=[other for other in remaining if other!=name]
   path=f'audit-{step}.safetensors'
   save_file({name:tensors[name].clone() for name in ['loss',*names]},str(out/path))
   audits.append({'step':step,'input_seed':input_seeds[step],'file':path,'gradient_names':names})
 result,ms,tokens=compute(c['mode'],m,a.seed,a.device,Path(a.assets),a.smoke,iterations,input_seeds,callback)
 save_file(result,str(out/'output.safetensors'))
 (out/'measurement.json').write_text(json.dumps({'latency_ms':ms,'tokens':tokens,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,'iterations':iterations,'timed_execution_audits':audits}))
if __name__=='__main__':main()
