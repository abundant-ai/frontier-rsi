"""Fixed programs, shape distribution, and compiler-configuration runner.

A submission chooses launch configuration, never the mathematical computation.
"""
import argparse,json,math,resource,time
from pathlib import Path
import torch
from restricted import load


def shape_for(program,index,smoke=False):
 if smoke:return (3,16+8*(index%2))
 return ((1,8,32,128,512,1024)[index%6],(127,256,511,1024,2048,4096,8192)[(index+program)%7])


def evaluate(program,x,y):
 if program<10:
  scale=1.+program/20.
  return ((x.float()*scale).sigmoid()*y.float()+x.float().square()*.01).to(x.dtype)
 mode=program%5;z=x.float()+y.float()*(1.+program/20.)
 # Cancellation can leave a very small scalar after thousands of FP32
 # additions. Both implementations accumulate sums in FP64, then return FP32;
 # otherwise reduction-tree rounding alone can exceed the fixed NRMSE gate.
 if mode==0:return z.double().sum(-1).float()
 if mode==1:return z.square().mean(-1)
 if mode==2:return z.amax(-1)
 if mode==3:return torch.logsumexp(z,-1)
 return (z*z.sigmoid()).sum(-1)


def validate_config(config,program,shape):
 if not isinstance(config,dict) or set(config)!={'block','warps','stages'}:raise ValueError('configure returns exactly block, warps, stages')
 if any(type(v) is not int for v in config.values()):raise ValueError('compiler configuration values must be integers')
 if config['warps'] not in (1,2,4,8,16) or config['stages'] not in (1,2,3,4,5):raise ValueError('compiler configuration range')
 block=config['block']
 if block not in (32,64,128,256,512,1024,2048,4096,8192,16384):raise ValueError('block range')
 if program>=10 and block<shape[-1]:raise ValueError('reduction block must cover the row')
 return config


_KERNEL=None

def get_kernel():
 global _KERNEL
 if _KERNEL is not None:return _KERNEL
 import triton
 import triton.language as tl
 @triton.jit
 def kernel(X,Y,O,N:tl.constexpr,W:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
  row=tl.program_id(0)
  offset=tl.arange(0,B)
  if P<10:
   ids=row*B+offset;xx=tl.load(X+ids,ids<N,other=0).to(tl.float32);yy=tl.load(Y+ids,ids<N,other=0).to(tl.float32)
   z=xx*(1.+P/20.);v=yy/(1.+tl.exp(-z))+.01*xx*xx;tl.store(O+ids,v,ids<N)
  else:
   ids=row*W+offset;xx=tl.load(X+ids,offset<W,other=0).to(tl.float32);yy=tl.load(Y+ids,offset<W,other=0).to(tl.float32)
   z=xx+yy*(1.+P/20.)
   if P%5==0:v=tl.sum(tl.where(offset<W,z,0).to(tl.float64),0).to(tl.float32)
   elif P%5==1:v=tl.sum(tl.where(offset<W,z*z,0),0)/W
   elif P%5==2:v=tl.max(tl.where(offset<W,z,float('-inf')),0)
   elif P%5==3:
    z=tl.where(offset<W,z,float('-inf'));maximum=tl.max(z,0);v=maximum+tl.log(tl.sum(tl.exp(z-maximum),0))
   else:v=tl.sum(tl.where(offset<W,z/(1.+tl.exp(-z)),0),0)
   tl.store(O+row,v)
 _KERNEL=kernel
 return kernel


def launch(program,x,y,config):
 import triton
 kernel=get_kernel()
 output=torch.empty_like(x) if program<10 else torch.empty(x.shape[0],device=x.device,dtype=torch.float32)
 grid=(triton.cdiv(x.numel(),config['block']),) if program<10 else (x.shape[0],)
 kernel[grid](x,y,output,x.numel(),x.shape[-1],program,config['block'],num_warps=config['warps'],num_stages=config['stages'],enable_fp_fusion=False)
 return output


def measure(fn,device,repetitions):
 for _ in range(2 if device=='cpu' else 25):fn()
 if device=='cuda':torch.cuda.synchronize()
 samples=[]
 for _ in range(repetitions):
  if device=='cuda':begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
  else:begin=time.perf_counter()
  result=fn()
  if device=='cuda':end.record();torch.cuda.synchronize();samples.append(begin.elapsed_time(end))
  else:samples.append((time.perf_counter()-begin)*1000)
 return result,sorted(samples)[len(samples)//2]


def run(candidate,seed,out,device='cuda',smoke=False,quick=False):
 resource.setrlimit(resource.RLIMIT_CPU,(7200,7200));torch.set_num_threads(2)
 hook=load(candidate,['configure']);torch.manual_seed(seed);began=time.monotonic();ratios=[];max_error=0.
 for program in range(20):
  for index in (range(2) if smoke else (0,39) if quick else range(40)):
   shape=shape_for(program,index,smoke);cfg=validate_config(hook.configure(program,shape),program,shape)
   x=torch.randn(shape,device=device);y=torch.randn(shape,device=device)
   default={'block':256 if program<10 else 2**math.ceil(math.log2(shape[-1])),'warps':4,'stages':2}
   if device=='cuda':candidate_fn=lambda:launch(program,x,y,cfg);baseline_fn=lambda:launch(program,x,y,default)
   else:candidate_fn=baseline_fn=lambda:evaluate(program,x,y)
   actual,ms=measure(candidate_fn,device,2 if smoke or quick else 100)
   expected=evaluate(program,x,y);error=float((actual-expected).norm()/expected.norm().clamp_min(1e-8));max_error=max(error,max_error)
   if actual.shape!=expected.shape or not torch.isfinite(actual).all() or error>1e-4:
    raise ValueError(f'compiler workload numerical mismatch: seed={seed}, program={program}, index={index}, shape={shape}, config={cfg}, nrmse={error}, reference_norm={float(expected.norm())}')
   _,baseline_ms=measure(baseline_fn,device,2 if smoke or quick else 100);ratios.append(baseline_ms/max(ms,1e-9))
   if time.monotonic()-began>3600:raise ValueError('one GPU-hour suite execution budget exceeded')
 output={'metric':math.exp(sum(math.log(x) for x in ratios)/len(ratios)),'cases':len(ratios),'max_nrmse':max_error,'elapsed':time.monotonic()-began,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
 Path(out).mkdir(parents=True,exist_ok=True);(Path(out)/'measurement.json').write_text(json.dumps(output));return output


def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');p.add_argument('--quick',action='store_true');a=p.parse_args();run(a.candidate,a.seed,a.out,a.device,a.smoke,a.quick)
if __name__=='__main__':main()
