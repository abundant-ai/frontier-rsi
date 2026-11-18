"""Run a requested kernel case and write its output tensors."""
import argparse,json,time
from pathlib import Path
import torch
from safetensors.torch import save_file
from kernel_ops import inputs,output_and_gradients,reference
from restricted import load
from gpu_timing import queued_cuda_call


def clone_operands(values):
 return {k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in values.items()}


def validate_update_sequence(fn,a):
  # Neither initial state nor freshly sampled gradients may alias the
  # reference's operands: candidate callbacks can update tensors in place.
  actual_state=clone_operands(a);expected_state=clone_operands(a)
  for step in range(1,101):
   actual_state['step']=expected_state['step']=step
   for i in range(a['count']):
    grad=torch.randn_like(a[f'g{i}'])*.3
    actual_state[f'g{i}']=grad;expected_state[f'g{i}']=grad.clone()
   actual=fn(actual_state);expected=reference('op10',expected_state)
   if len(actual)!=len(expected):raise ValueError('multi-step output count')
   for i,(x,y) in enumerate(zip(actual,expected)):
    tol=.0001 if y.dtype==torch.float32 else .02
    if x.shape!=y.shape or not torch.isfinite(x).all() or float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-8))>tol:raise ValueError('multi-step state mismatch')
   for i in range(a['count']):
    for j,key in enumerate(('p','m','v')):
     actual_state[f'{key}{i}']=actual[3*i+j];expected_state[f'{key}{i}']=expected[3*i+j]


def measure(fn,a,gradient,warmup,repetitions,device,kind):
 if kind=='op10':validate_update_sequence(fn,a)
 # Keep warm-up mutations out of the source for measured inputs.
 warmup_values=clone_operands(a)
 for _ in range(warmup):output_and_gradients(fn,warmup_values,gradient)
 # Validation state and warm-up copies must be released before timed work.
 # Keeping them live charges temporary tensors against the candidate cap.
 del warmup_values
 if device=='cuda':torch.cuda.synchronize()
 elapsed=[];result=None;delay=2_000_000
 for iteration in range(repetitions):
  values=clone_operands(a)
  # Independently new operands every invocation; validate every timed result.
  for key,value in list(values.items()):
   if isinstance(value,torch.Tensor) and value.is_floating_point():
    value=(torch.rand_like(value) if kind=='op16' and key=='uniforms' else torch.randn_like(value)*.3)
    values[key]=value.abs() if kind=='op10' and key.startswith('v') else value
  # References receive a separate pre-invocation snapshot, never tensors that
  # candidate code has been allowed to mutate.
  reference_values=clone_operands(values)
  if device=='cuda':
   result,milliseconds,queue=queued_cuda_call(lambda:output_and_gradients(fn,values,gradient),delay,max_attempts=1)
   delay=queue['next_delay_cycles'];elapsed.append(milliseconds)
  else:
   start=time.perf_counter();result=output_and_gradients(fn,values,gradient)
   elapsed.append((time.perf_counter()-start)*1000)
  expected=output_and_gradients(lambda v:reference(kind,v),reference_values,gradient)
  if len(result)!=len(expected):raise ValueError('output count')
  if kind=='op16':
   from kernel_checks import sampling
   sampling(result,expected,reference_values,repetitions<10)
   continue
  for i,(actual,target) in enumerate(zip(result,expected)):
   if actual.shape!=target.shape or not torch.isfinite(actual).all():raise ValueError('nonfinite or incorrect shape')
   if kind=='op15' and i==0 and not torch.equal(actual,target):raise ValueError('indexed copies must be bitwise exact')
   if not target.is_floating_point() and not torch.equal(actual,target):raise ValueError('discrete outputs differ')
   error=float((actual.float()-target.float()).norm()/target.float().norm().clamp_min(1e-8))
   tolerance=.0001 if kind in ('op07','op08','op10') and target.dtype==torch.float32 else .01 if kind in ('op02','op03') else .02
   if kind=='op12':tolerance=.05
   if kind=='op09' and i==0:
    if abs(float(actual)-float(target))>.001:raise ValueError('loss mismatch')
   elif error>tolerance:raise ValueError('output or gradient mismatch')
 return output_and_gradients(fn,a,gradient),sorted(elapsed)[len(elapsed)//2]


def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--config',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 c=json.loads(Path(a.config).read_text());m=load(a.candidate,['run'])
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(min(1.,c['memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory))
 values=inputs(c['operation'],a.seed,a.index,a.device,a.smoke)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 quick=c.get('validation_profile')=='quick-gpu-v1'
 warmup=1 if a.smoke or quick else 25;repetitions=2 if a.smoke or quick else 100
 result,ms=measure(m.run,values,c.get('gradients',False),warmup,repetitions,a.device,c['operation'])
 save_file({f'output_{i}':x.cpu().contiguous() for i,x in enumerate(result)},str(out/'output.safetensors'))
 meta={'latency_ms':ms,'timing_protocol':'queued-cuda-events-v1' if a.device=='cuda' else 'cpu-smoke','peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0}
 if c['operation']=='op02':
  from basic_gemm import matmul
  _,ref_ms=measure(lambda x:matmul(x['a'],x['b']),values,False,warmup,repetitions,a.device,c['operation'])
  meta['reference_latency_ms']=ref_ms
 (out/'measurement.json').write_text(json.dumps(meta))
if __name__=='__main__':main()
