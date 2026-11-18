"""Four-rank training operations and launch contracts."""
import argparse,contextlib,json,math,os,tempfile,time,hashlib
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from model import GPT,Config,token_loss
from restricted import load
from distributed_audits import schedule,input_seed,snapshot


class TrainingGraph(GPT):
    def forward(self,x):
        from torch.utils.checkpoint import checkpoint
        pos=torch.arange(x.shape[1],device=x.device)
        h=self.wte(x)+self.wpe(pos)
        for block in self.blocks:h=checkpoint(block,h,use_reentrant=False)
        return self.head(self.ln(h))


def backward_reference(model,x,y,scale):
 (token_loss(model(x),y)*scale).backward()
 for p in model.parameters():
  if p.grad is not None:dist.all_reduce(p.grad);p.grad.div_(dist.get_world_size())
 return []


def reduction_reference(a):
 x=a['x'].clone();dist.all_reduce(x)
 return torch.nn.functional.silu(x+a['bias'])


def workload(mode,seed,index,rank,device,smoke):
 g=torch.Generator(device=device).manual_seed(seed+rank*1009+index*100003)
 if mode=='training':
  length=16 if smoke else (1024,2048,4096)[index%3]
  # The global update contains 65536 tokens, divided across four ranks and
  # fixed microbatches. Full FP32 verification uses the identical token set.
  micro=1;accum=1 if smoke else 65536//(4*length)
  x=torch.randint(64 if smoke else 50257,(accum,micro,length),generator=g,device=device)
  y=torch.randint(64 if smoke else 50257,x.shape,generator=g,device=device)
  return x,y
 dtype=torch.float32 if smoke or index%2 else torch.bfloat16
 elements=64 if smoke else (262144,1048576,4194304,16777216,67108864,268435456)[index%6]
 return {'x':torch.randn(elements,generator=g,device=device,dtype=dtype)*.1,
         'bias':torch.randn(elements,generator=g,device=device,dtype=dtype)*.1}


def worker(rank,world,init_method,candidate,mode,seed,index,out,phase,smoke,quick=False,audit_key=None):
 device='cpu' if smoke else 'cuda:'+str(rank)
 if not smoke:torch.cuda.set_device(rank)
 torch.set_num_threads(1)
 options={} if smoke else {'device_id':torch.device(device)}
 dist.init_process_group('gloo' if smoke else 'nccl',init_method=init_method,rank=rank,world_size=world,**options)
 try:
  torch.manual_seed(seed)
  hook=None if candidate is None else load(candidate,['backward'] if mode=='training' else ['run'])
  torch.manual_seed(seed)
  timed=phase in ('timing','timing-reference')
  repeats=(2 if smoke else 1 if quick and mode=='training' else 2 if quick else 100) if timed else (2 if smoke else 1)
  audited=schedule(audit_key,repeats) if timed else []
  audit_files=[]
  def retain(step,tensors):
   if step not in audited:return
   name=f'{phase}-audit-{step}-rank{rank}.safetensors'
   save_file(snapshot(tensors,audit_key,step,rank),str(Path(out)/name))
   audit_files.append({'step':step,'file':name,'input_seed':input_seed(audit_key,seed,step)})
  if mode=='training':
   cfg=Config(vocab=64,width=16,layers=2,heads=2,context=64) if smoke else Config(width=1024,layers=24,heads=16,context=4096)
   model=TrainingGraph(cfg).to(device).train()
   x,y=workload(mode,seed,index,rank,device,smoke)
   # Forward recomputation is part of the fixed runner in both numerical and
   # timed phases; candidates may alter execution through their callback.
   callback=backward_reference if hook is None else hook.backward
   opt=torch.optim.SGD(model.parameters(),lr=1e-4)
   elapsed=[]
   for repeat in range(repeats):
    if timed:x,y=workload(mode,input_seed(audit_key,seed,repeat),index,rank,device,smoke)
    model.zero_grad(set_to_none=True);dist.barrier()
    if not smoke:torch.cuda.synchronize()
    begin=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16) if timed and not smoke else contextlib.nullcontext():
     for xx,yy in zip(x,y):
      work=callback(model,xx,yy,1/len(x))
      if work is not None:
       for request in work:request.wait()
    if timed:opt.step()
    dist.barrier()
    if not smoke:torch.cuda.synchronize()
    elapsed.append((time.perf_counter()-begin)*1000)
    if timed:retain(repeat,{name:p.grad for name,p in model.named_parameters() if p.grad is not None})
   if not timed:
    tensors={name:p.grad.detach().cpu().contiguous() for name,p in model.named_parameters() if p.grad is not None}
    save_file(tensors,str(Path(out)/f'{phase}-rank{rank}.safetensors'))
  else:
   a=workload(mode,seed,index,rank,device,smoke)
   if not timed:a={k:v.float() for k,v in a.items()}
   callback=reduction_reference if hook is None else hook.run
   elapsed=[]
   for repeat in range(repeats):
    if timed:a=workload(mode,input_seed(audit_key,seed,repeat),index,rank,device,smoke)
    values={k:v.clone() for k,v in a.items()};dist.barrier()
    if not smoke:torch.cuda.synchronize()
    begin=time.perf_counter();result=callback(values)
    dist.barrier()
    if not smoke:torch.cuda.synchronize()
    elapsed.append((time.perf_counter()-begin)*1000)
    if timed:retain(repeat,{'output':result})
   if not timed:save_file({'output':result.detach().cpu().contiguous()},str(Path(out)/f'{phase}-rank{rank}.safetensors'))
  if timed:
   (Path(out)/f'{phase}-audits-rank{rank}.json').write_text(json.dumps({'iterations':repeats,'audited_steps':audited,'files':audit_files,'audit_key_sha256':hashlib.sha256(bytes.fromhex(audit_key)).hexdigest(),'precision':'float32' if smoke or mode=='reduction' and index%2 else 'bfloat16'}))
  maximum=torch.tensor(sorted(elapsed)[len(elapsed)//2],device=device);dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
  if rank==0:
   (Path(out)/(phase+'-measurement.json')).write_text(json.dumps({'latency_ms':float(maximum),'ranks':world}))
 finally:dist.destroy_process_group()


def run(candidate,mode,seed,index,out,phase='timing',smoke=False,quick=False,audit_key=None):
 if phase in ('timing','timing-reference') and (not isinstance(audit_key,str) or len(audit_key)!=64):raise ValueError('private timed-execution challenge key required')
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 # All ranks share this container's filesystem. A unique FileStore avoids
 # releasing an ephemeral TCP port before spawned rank zero can bind it.
 with tempfile.TemporaryDirectory(prefix='rendezvous-',dir=out) as store:
  init_method=(Path(store).resolve()/'store').as_uri()
  mp.spawn(worker,args=(4,init_method,candidate,mode,seed,index,str(out),phase,smoke,quick,audit_key),nprocs=4,join=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate');p.add_argument('--mode',choices=['training','reduction'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--out',required=True);p.add_argument('--phase',choices=['timing','timing-reference','candidate','reference'],default='timing');p.add_argument('--smoke',action='store_true');p.add_argument('--quick',action='store_true');p.add_argument('--audit-key');a=p.parse_args()
 run(None if a.phase in ('reference','timing-reference') else a.candidate,a.mode,a.seed,a.index,a.out,a.phase,a.smoke,a.quick,a.audit_key)
if __name__=='__main__':main()
