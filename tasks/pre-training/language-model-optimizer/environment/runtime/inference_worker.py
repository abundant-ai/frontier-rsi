"""Run frozen-model inference and write predictions."""
import argparse,json,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from research_hooks import load
from research_model import storage_bytes
from inference_core import HOOKS,frozen,sync,StateCache,cached_call,quantize_model,install_execution_policy,generate_verified,reason
from serving import run_trace
from gpu_timing import queued_cuda_call


def run(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(a.seed)
 hooks=load(Path(__file__).with_name('inference_baseline.py') if a.baseline else a.candidate,HOOKS)
 model,tok=frozen(a.assets,a.device,a.smoke)
 row=a.row;payload=0
 if row==16 and not a.baseline:install_execution_policy(model,hooks)
 if row==31 and not a.baseline:
  calibration=load_file(a.calibration)['input'].to(a.device)
  start=time.monotonic();payload=quantize_model(model,hooks,calibration,getattr(a,'prepared',None))
  if time.monotonic()-start>1800:raise ValueError('calibration/processing deadline')
 if row in (18,22):
  questions=json.loads(Path(a.input).read_text());answers=[];counts=[]
  for q in questions:
   value,count=reason(model,tok,hooks,q,a.device,a.smoke);answers.append(value);counts.append(count)
  result={'answers':answers,'generated_tokens':counts}
 elif row in (48,55):
  requests=json.loads(Path(a.input).read_text());plan_path=getattr(a,'replay_plan',None)
  if plan_path and not a.baseline:raise ValueError('only the independent reference may replay an execution plan')
  plan=json.loads(Path(plan_path).read_text()) if plan_path else None
  result=run_trace(model,hooks,requests,a.rate,a.device,row,a.smoke,replay_plan=plan)
 else:
  inputs=load_file(a.input);x=inputs['input'].long().to(a.device);original_length=x.shape[-1];preprocess=0.
  if row==33 and not a.baseline:
   sync(a.device);start=time.perf_counter();x=hooks.compress(x.clone(),inputs['question'].to(a.device));sync(a.device);preprocess=time.perf_counter()-start
   if x.dtype!=torch.int64 or x.ndim!=2 or x.shape[0]!=1 or not x.numel() or x.shape[1]>original_length:raise ValueError('invalid downstream input')
  state_bytes=0;timing={};sync(a.device);start=time.perf_counter()
  with torch.no_grad():
   if row in (32,34) and not a.baseline:
    cache=StateCache(model,hooks,row,x.shape[0]);parts=[]
    for offset in range(0,x.shape[1],256):
     last=cached_call(model,x[:,offset:offset+256],cache,offset);state_bytes=max(state_bytes,cache.bytes())
    logits=last[:,-1:]
   else:logits=model(x,use_cache=False,logits_to_keep=1).logits
   sync(a.device);prefill=time.perf_counter()-start
   count=a.output_length;start=time.perf_counter()
   if row==30:generated=generate_verified(model,hooks,x,count)
   else:
    generated=[];next_token=logits[:,-1].argmax(-1,keepdim=True)
    if row not in (32,34) or a.baseline:cache=None
    queued=row==32 and not a.baseline and a.device=='cuda' and not a.smoke
    gpu_elapsed=0.;delay=2_000_000
    if queued:cache.benchmark_checks_on_device=True
    for index in range(count):
     generated.append(next_token)
     if row in (32,34) and not a.baseline:
      if queued:
       def decode_step():
        logits=cached_call(model,next_token,cache,x.shape[1]+index)
        return logits[:,-1].argmax(-1,keepdim=True)
       next_token,ms,queue=queued_cuda_call(decode_step,delay,max_attempts=1)
       gpu_elapsed+=ms;delay=queue['next_delay_cycles']
       state_bytes=max(state_bytes,cache.bytes());continue
      step=cached_call(model,next_token,cache,x.shape[1]+index);state_bytes=max(state_bytes,cache.bytes())
     else:
      z=model(next_token,past_key_values=cache,use_cache=True) if cache is not None else model(torch.cat((x,*generated),-1),use_cache=True)
      step=z.logits;cache=z.past_key_values
     next_token=step[:,-1].argmax(-1,keepdim=True)
    if queued:
     cache.benchmark_checks_on_device=False
     timing={'timing_protocol':'queued-cuda-events-v1','queue_retries':0,'queue_delay_cycles':delay}
    generated=torch.cat(generated,-1)
   sync(a.device);decode=gpu_elapsed/1000 if row!=30 and queued else time.perf_counter()-start
  save_file({'logits':logits.float().detach().cpu(),'generated':generated.cpu()},str(out/'predictions.safetensors'))
  result={'prefill_latency':prefill,'decode_latency':decode/max(1,count),'elapsed':prefill+decode+preprocess,'state_bytes':state_bytes,'payload_bytes':payload,'input_tokens':x.numel(),'original_tokens':original_length,'preprocess_latency':preprocess,'batch':x.shape[0],**timing}
 (out/'measurement.json').write_text(json.dumps(result,allow_nan=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--row',type=int,required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--device',default='cuda');p.add_argument('--assets',default='/opt/assets/public');p.add_argument('--out',required=True);p.add_argument('--input',required=True);p.add_argument('--calibration');p.add_argument('--prepared');p.add_argument('--output-length',type=int,default=16);p.add_argument('--rate',type=float,default=1.);p.add_argument('--baseline',action='store_true');p.add_argument('--replay-plan');p.add_argument('--smoke',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
