"""Run numerical sequence hooks and write model outputs."""
import argparse,contextlib,json,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from research_hooks import load
from research_model import defaults,checked_config,SequenceModel,validate_capacity,storage_bytes,measured_forward
from gpu_timing import queued_cuda_call

HOOKS=['configure','position','mix','route','objective','feedforward']

def setup(candidate,row,seed,device,smoke):
 torch.manual_seed(seed);hooks=load(candidate,HOOKS,allow_attention=row==7)
 c=checked_config(row,hooks.configure(defaults(row,smoke)),smoke)
 # Reset the initialization seed after processing configuration.
 torch.manual_seed(seed)
 from types import SimpleNamespace
 baseline=load(Path(__file__).with_name('sequence_baseline_fused.py' if row==15 else 'sequence_baseline.py'),HOOKS,allow_attention=row==15)
 editable={7:{'mix','position','feedforward','objective'},8:{'mix','position','feedforward','objective'},10:{'route'},11:set(),12:{'mix','position','feedforward','objective'},13:{'mix'},14:{'mix','objective'},15:{'position'},21:{'mix','position','feedforward','objective'}}[row]
 selected=SimpleNamespace(**{name:getattr(hooks if name in editable else baseline,name) for name in HOOKS})
 model=SequenceModel(c,selected,row).to(device)
 validate_capacity(model,row,smoke)
 return model,c

def sync(device):
 if str(device).startswith('cuda'):torch.cuda.synchronize()

def retrieval(seed,length,vocab):
 g=torch.Generator().manual_seed(seed);x=torch.randint(4,vocab,(1,length),generator=g)
 key=int(torch.randint(4,vocab,(1,),generator=g));value=int(torch.randint(4,vocab,(1,),generator=g))
 index=int(torch.randint(0,max(1,length-8),(1,),generator=g));x[0,index:index+3]=torch.tensor([1,key,value]);x[0,-2:]=torch.tensor([2,key])
 return x,value

def train(a,c,model,out):
 tokens=np.random.default_rng(a.seed).integers(4,c['vocab'],8192,dtype=np.int32) if a.smoke else np.load(Path(a.assets)/'train.npy',mmap_mode='r')
 opt=torch.optim.AdamW(model.parameters(),lr=6e-4);begin=time.monotonic();steps=0;loads=[]
 length=16 if a.smoke else 1024;g=np.random.default_rng(a.seed)
 def checkpoint():
  save_file({k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()},str(out/'model.safetensors'))
  (out/'architecture.json').write_text(json.dumps(c))
 last=begin
 while steps<2 if a.smoke else time.monotonic()-begin<a.seconds-15:
  if a.row==15 and not a.smoke:length=(1024,2048,4096)[steps%3]
  offset=int(g.integers(0,len(tokens)-length-1));z=torch.tensor(np.array(tokens[offset:offset+length+1]),device=a.device).long()[None,:]
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16) if a.device=='cuda' else contextlib.nullcontext():
   logits,state=model(z[:,:-1]);loss=model.hooks.objective(logits,z[:,1:],state) if a.row in (8,12,14,21) else torch.nn.functional.cross_entropy(logits.float().flatten(0,1),z[:,1:].flatten())
   if a.row==14:
    memory,target=retrieval(a.seed+steps%10000,length,c['vocab']);r,_=model(memory.to(a.device));loss=loss+torch.nn.functional.cross_entropy(r[:,-1].float(),torch.tensor([target],device=a.device))
  if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1
  if model.loads:
   loads.append(torch.stack(model.loads).detach().cpu())
   if len(loads)==100:
    counts=torch.stack(loads).sum((0,1));ratio=float(counts.max()/counts.float().mean().clamp_min(1))
    if a.row==10 and ratio>1.5:raise ValueError('100-update assignment imbalance exceeds 1.5')
    loads=[]
  if time.monotonic()-last>60:checkpoint();last=time.monotonic()
 if not steps:raise ValueError('no completed update')
 checkpoint();return {'steps':steps}

def predict(a,c,model,out):
 model.load_state_dict(load_file(str(Path(a.weights)/'model.safetensors')),strict=True);model.eval()
 x=load_file(a.input)['input'].long().to(a.device)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16) if a.device=='cuda' else contextlib.nullcontext():
  sync(a.device);start=time.perf_counter();(_,state),flops=measured_forward(model,x[:,:min(1024,x.shape[1])],profile=True);sync(a.device)
  # Separate timing from arithmetic profiling.
  start=time.perf_counter();logits,state=model(x,project_all=False);sync(a.device);latency=time.perf_counter()-start
  state_size=storage_bytes(state)
  next_token=logits[:,-1].argmax(-1,keepdim=True)
  steps=2 if a.smoke else a.output_length;timing={}
  if a.row in (7,11) and a.device=='cuda' and not a.smoke:
   model.benchmark_checks_on_device=True
   elapsed=0.;delay=2_000_000;prefilled=0
   try:
    for i in range(steps):
     def step():
      result,new_state=model(next_token,state,x.shape[1]+i,project_all=False)
      return result[:,-1].argmax(-1,keepdim=True),new_state
     (next_token,state),ms,queue=queued_cuda_call(step,delay,max_attempts=1)
     elapsed+=ms;delay=queue['next_delay_cycles'];prefilled+=int(queue['queue_prefilled'])
   finally:model.benchmark_checks_on_device=False
   decode=elapsed/1000/steps
   timing={'timing_protocol':'queued-cuda-events-v1','queue_prefilled_steps':prefilled,'queue_retries':0,'queue_delay_cycles':delay}
  else:
   start=time.perf_counter()
   for i in range(steps):
    result,state=model(next_token,state,x.shape[1]+i,project_all=False);next_token=result[:,-1].argmax(-1,keepdim=True)
   sync(a.device);decode=(time.perf_counter()-start)/steps
  state_size=max(state_size,storage_bytes(state))
 save_file({'logits':logits.detach().float().cpu()},str(out/'predictions.safetensors'))
 return {'latency':latency,'decode_latency':decode,'state_bytes':state_size,'flops_per_token':flops,'parameters':validate_capacity(model,a.row,a.smoke),'batch':x.shape[0],'length':x.shape[1],**timing}

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--row',type=int,required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--seconds',type=int,default=120);p.add_argument('--device',default='cuda');p.add_argument('--assets',default='/opt/assets/public');p.add_argument('--out',required=True);p.add_argument('--output-length',type=int,default=16);p.add_argument('--weights');p.add_argument('--input');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 start=time.monotonic();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model,c=setup(a.candidate,a.row,a.seed,a.device,a.smoke)
 if a.row==15 and not a.smoke:
  model.load_state_dict(load_file(str(Path(a.assets)/'position_start.safetensors')),strict=True)
 if not a.smoke and not a.input:a.seconds=max(1,a.seconds-(time.monotonic()-start))
 result=predict(a,c,model,out) if a.input else train(a,c,model,out)
 result['elapsed']=time.monotonic()-start;(out/'measurement.json').write_text(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
