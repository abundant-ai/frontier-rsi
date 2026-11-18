"""Five-minute full-size GPU diagnostic, never a benchmark grading entrypoint."""
import json,math,time,statistics,signal,os
from pathlib import Path
from grade import execute,Invalid,Infrastructure
from research_grade import grade_seed
import torch


def run(contract,tests,submission,log,device='cuda',executor=execute,clock=time.monotonic,workspace=None):
 if contract.get('validation_profile')!='quick-gpu-v1':raise Infrastructure('quick profile requires an explicitly diagnostic package')
 if contract.get('family') not in ('kernel','text','post','merge','research_arch','research_learning','research_inference','performance','schedule','distributed'):raise Infrastructure('unsupported quick GPU task')
 if device!='cuda' or not torch.cuda.is_available():raise Infrastructure('quick validation still requires a real GPU')
 if contract.get('gpu_type','H100') not in torch.cuda.get_device_name(0):raise Infrastructure('wrong GPU for quick validation')
 if contract.get('gpu_count',1)==4 and torch.cuda.device_count()!=4:raise Infrastructure('four GPUs required for distributed validation')
 seed=json.loads((tests/'seeds.json').read_text())[0]
 start=clock();deadline=start+300
 def limited_execute(command,out,timeout,**kwargs):
  remaining=deadline-clock()
  if remaining<=0:raise Invalid('quick validation exceeded 300-second execution budget')
  return executor(command,out,min(timeout,remaining),**kwargs)
 worker=(Path(workspace) if workspace is not None else Path('/run/rsi-workers'))/f'quick-{seed}'
 worker.parent.mkdir(parents=True,exist_ok=True);worker.parent.chmod(0o711)
 if contract.get('family') in ('performance','schedule','distributed'):
  value,measurement=systems_seed(contract,submission,worker,seed,device,limited_execute)
 elif contract.get('family')=='kernel':
  value,measurement=kernel_seed(contract,submission,worker,seed,device,limited_execute)
 elif contract.get('family') in ('text','post','merge'):
  value,measurement=ordinary_seed(contract,submission,worker,seed,device,limited_execute)
 else:value,measurement=grade_seed(contract,submission,worker,seed,False,device,limited_execute,quick=True)
 if not math.isfinite(value):raise Invalid('nonfinite quick diagnostic metric')
 if clock()>deadline:raise Invalid('quick validation exceeded 300-second execution budget')
 report={'quick_validation_only':True,'full_validation':False,'profile':'quick-gpu-v1',
         'seed':seed,'training_seconds':120 if contract.get('family') in ('text','post') else 90 if contract.get('family') in ('text','post','research_arch','research_learning') else 0,'execution_budget_seconds':300,'elapsed_seconds':clock()-start,
         'raw_metric':value,'research_validation':measurement}
 # This reward means that a diagnostic completed, not that task quality was validated.
 return {'valid':1,'reward':1,'quick_validation_only':1},report


def systems_seed(contract,submission,worker,seed,device,execute):
 import sys
 from grade import read_outputs,nrmse
 family=contract['family']
 if family=='distributed':
  from distributed_validation import compare,compare_timing
  import secrets
  audit_key=secrets.token_hex(32)
  for phase in ('reference','candidate','timing-reference','timing'):
   dest=worker/phase
   command=[sys.executable,'/opt/runtime/distributed_ops.py','--candidate',str(submission),'--mode',contract['mode'],'--seed',str(seed),'--index','0','--out',str(dest),'--phase',phase,'--quick']
   if phase.startswith('timing'):command+=['--audit-key',audit_key]
   execute(command,dest,180,memory_gib=contract['memory_gib'],gpus=4)
   if os.geteuid()==0:os.chown(dest,0,0);dest.chmod(0o700)
  validation=compare(worker/'candidate',reference_folder=worker/'reference',tolerance=.02 if contract['mode']=='training' else .0001)
  validation['timed_execution']=compare_timing(worker/'timing',worker/'timing-reference',audit_key,seed,contract['mode'],0,quick=True)
  measurement=json.loads((worker/'timing/timing-measurement.json').read_text())
  if measurement['ranks']!=4:raise Invalid('distributed rank count differs')
 elif family=='schedule':
  command=[sys.executable,'/opt/runtime/compiler_suite.py','--candidate',str(submission),'--out',str(worker),'--seed',str(seed),'--device',device,'--quick']
  execute(command,worker,300,memory_gib=contract['memory_gib'])
  measurement=json.loads((worker/'measurement.json').read_text())
  if measurement['cases']!=40 or not math.isfinite(measurement['max_nrmse']) or measurement['max_nrmse']>1e-4:raise Invalid('incomplete or incorrect compiler diagnostic')
  if measurement['peak_allocated_bytes']>contract['memory_gib']*1024**3:raise Invalid('compiler memory cap exceeded')
  value=measurement['metric']
  if not math.isfinite(value) or value<=0:raise Invalid('invalid compiler speedup')
  return value,measurement
 else:
  command=[sys.executable,'/opt/runtime/performance.py','--config','/app/contract.json','--candidate',str(submission),'--out',str(worker),'--seed',str(seed),'--device',device]
  execute(command,worker,180,memory_gib=contract['memory_gib'])
  actual=read_outputs(worker)
  if contract['mode']=='batches':
   from packing_validation import validate
   validation=validate(actual,seed,device,Path('/opt/assets/public'),False,contract['tolerance']);tokens=validation['tokens']
  else:
   from performance_validation import validate_execution
   try:tokens,validation=validate_execution(worker,seed,device,Path('/opt/assets/public'),False,contract['tolerance'],quick=True)
   except ValueError as e:raise Invalid(str(e)) from e
  measurement=json.loads((worker/'measurement.json').read_text())
  if measurement['peak_allocated_bytes']>contract['memory_gib']*1024**3:raise Invalid('performance memory cap exceeded')
 ms=measurement['latency_ms']
 if not math.isfinite(ms) or ms<=0:raise Invalid('invalid systems latency')
 value=tokens/(ms/1000) if family=='performance' and contract['mode']=='batches' else ms
 return value,{'measurement':measurement,'numerical_validation':validation,'real_dimensions':True}


def kernel_seed(contract,submission,worker,seed,device,execute):
 import sys
 from safetensors.torch import load_file
 from kernel_ops import inputs,output_and_gradients,reference
 from grade import nrmse
 cases=[];metrics=[]
 # Parameter-update case 39 is the largest real allocation and must be part
 # of its diagnostic: small cases cannot expose verifier-buffer cap failures.
 for index in ((0,4,39) if contract['operation']=='op10' else (0,4,5)):
  dest=worker/f'case-{index}'
  command=[sys.executable,'/opt/runtime/kernel_worker.py','--config','/app/contract.json','--candidate',str(submission),'--out',str(dest),'--seed',str(seed),'--index',str(index),'--device',device]
  execute(command,dest,120,memory_gib=contract['memory_gib'])
  path=dest/'output.safetensors'
  if path.is_symlink() or not path.is_file():raise Invalid('missing ordinary kernel output')
  actual=load_file(str(path));a=inputs(contract['operation'],seed,index,device,False)
  expected=output_and_gradients(lambda v:reference(contract['operation'],v),a,contract.get('gradients',False))
  if set(actual)!={f'output_{i}' for i in range(len(expected))}:raise Invalid('kernel tensor keys differ')
  if contract['operation']=='op16':
   from kernel_checks import sampling
   sampling([actual[f'output_{i}'] for i in range(len(expected))],[x.cpu() for x in expected],{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in a.items()},True)
  else:
   for i,target in enumerate(expected):
    value=actual[f'output_{i}'];target=target.cpu()
    if not target.is_floating_point() or contract['operation']=='op15' and i==0:
     if not torch.equal(value,target):raise Invalid('discrete kernel output differs')
    tolerance=contract['tolerance']
    if contract['operation'] in ('op07','op08','op10'):tolerance=.0001 if target.dtype==torch.float32 else .02
    error=abs(float(value)-float(target)) if contract['operation']=='op09' and i==0 else nrmse(value,target)
    if error>(.001 if contract['operation']=='op09' and i==0 else tolerance):raise Invalid('kernel output or gradient mismatch')
  measurement=json.loads((dest/'measurement.json').read_text());ms=measurement['latency_ms']
  if not math.isfinite(ms) or ms<=0:raise Invalid('invalid kernel latency')
  if measurement['peak_allocated_bytes']>contract['memory_gib']*1024**3:raise Invalid('kernel memory cap exceeded')
  if measurement.get('timing_protocol')!='queued-cuda-events-v1':raise Invalid('unexpected kernel timing protocol')
  metric=ms
  if contract['operation']=='op02':
   reference_ms=measurement['reference_latency_ms']
   if not math.isfinite(reference_ms) or reference_ms<=0:raise Invalid('invalid reference timing')
   metric=reference_ms/ms
  metrics.append(metric);cases.append({'index':index,**measurement})
 return math.exp(statistics.mean(math.log(x) for x in metrics)),{'cases':cases,'real_dimensions':True,'warmup':1,'repetitions':2}


def ordinary_seed(contract,submission,worker,seed,device,execute):
 import sys
 worker.mkdir(parents=True,exist_ok=True)
 trained=worker/'trained'
 if contract['family']=='merge':
  command=[sys.executable,'/opt/runtime/model_construction.py','--candidate',str(submission),'--out',str(trained)]
 else:
  command=[sys.executable,'/opt/runtime/train.py','--config','/app/contract.json','--candidate',str(submission),'--out',str(trained),'--seed',str(seed),'--device',device,'--seconds','120']
 execute(command,trained,160,memory_gib=contract['memory_gib'])
 weights=trained/'model.safetensors'
 if weights.is_symlink() or not weights.is_file() or weights.stat().st_size>8*1024**3:raise Invalid('invalid regular model checkpoint')
 if contract['family']=='text':
  from evaluate import score
  value=score(contract,weights,Path('/opt/assets/private/eval.npy'),Path('/opt/assets/public'),device,False,limit=3)
  return value,{'evaluation_contexts':3,'context_length':1024,'training_seconds':120}
 def evaluate(mode,data,label,baseline=False):
  # Private labels stay in this root-owned verifier process. The worker UID
  # intentionally cannot read /opt/assets/private. Only tensor weights cross
  # the boundary; no candidate module is imported during private evaluation.
  from evaluate import score
  value=score({**contract,'family':'post','mode':mode},None if baseline else weights,
              Path('/opt/assets/private')/data,Path('/opt/assets/public'),device,
              False,limit=3,max_new_tokens=512)
  if not math.isfinite(value):raise Invalid('nonfinite quick evaluation')
  return value
 if contract['family']=='merge':
  reasoning=evaluate('policy','math_eval.json','math')
  preference=evaluate('adapt','pref_eval.json','preference')
  return (reasoning+preference)/2,{'reasoning_accuracy':reasoning,'preference_accuracy':preference,'evaluation_items_per_split':3,'generation_cap':512}
 mode=contract['mode'];reasoning=mode in ('policy','allocation')
 value=evaluate(mode,'math_eval.json' if reasoning else 'pref_eval.json','task-objective')
 if mode!='preferences':return value,{'evaluation_items':3,'generation_cap':512,'training_seconds':120}
 baseline=evaluate('policy','math_eval.json','baseline-capability',True)
 candidate=evaluate('policy','math_eval.json','candidate-capability')
 warnings=[]
 if candidate<baseline-.02:warnings.append('sampled capability-regression accuracy drop exceeds 2 percentage points')
 return value,{'preference_accuracy':value,'baseline_capability':baseline,'candidate_capability':candidate,'evaluation_items_per_split':3,'generation_cap':512,'diagnostic_quality_warnings':warnings}


def main():
 tests=Path('/tests');log=Path('/logs/verifier');log.mkdir(parents=True,exist_ok=True)
 for name in ('reward.json','reward.txt'):(log/name).unlink(missing_ok=True)
 def expired(*_):raise TimeoutError('quick validation exceeded 300-second execution budget')
 signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,300)
 try:
  rewards,report=run(json.loads((tests/'contract.json').read_text()),tests,Path('/opt/submission/candidate.py'),log)
 except Infrastructure as error:
  (log/'infrastructure_error.json').write_text(json.dumps({'error':str(error),'quick_validation_only':True}));raise
 except (Invalid,ValueError,RuntimeError,OSError) as error:
  rewards={'valid':0,'quick_validation_only':1};report={'quick_validation_only':True,'full_validation':False,'failure_reason':str(error)}
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 (log/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 (log/'reward.json').write_text(json.dumps(rewards,allow_nan=False))
if __name__=='__main__':main()
