"""Independent checks of fresh-input timed training executions.

Intermediate gradient files are temporary verifier scratch, never export assets.
Only compact audit summaries remain after successful root-owned comparison.
"""
import json,math
from pathlib import Path
import torch
from safetensors.torch import load_file
from performance import compute


def validate_execution(folder,seed,device,assets,smoke,tolerance,quick=False):
 folder=Path(folder);measurement=json.loads((folder/'measurement.json').read_text())
 iterations=2 if smoke or quick else 100
 audits=measurement.get('timed_execution_audits',[])
 if measurement.get('iterations')!=iterations:raise ValueError('training timing iteration count differs')
 steps=[item['step'] for item in audits]
 if len(set(steps))!=len(steps) or len(steps)<min(32,iterations) or iterations-1 not in steps:
  raise ValueError('incomplete timed training audits')
 worst=0.;count=0;tokens=None
 for item in audits:
  step=item['step'];rng=item['input_seed']
  if type(step) is not int or not 0<=step<iterations or type(rng) is not int or not 0<=rng<2**63:
   raise ValueError('invalid timed training audit')
  path=folder/f'audit-{step}.safetensors'
  if item['file']!=path.name or path.is_symlink() or not path.is_file():raise ValueError('invalid timed training tensor file')
  actual=load_file(str(path));expected,_,tokens=compute('execution',None,seed,device,assets,smoke,1,[rng])
  names=item['gradient_names']
  if len(names)!=len(set(names)) or set(actual)!={'loss',*names} or not set(actual)<=set(expected):
   raise ValueError('timed training gradient keys differ')
  if step==iterations-1:
   if set(actual)!=set(expected):raise ValueError('final full-gradient audit incomplete')
  elif len(names)!=min(8,len(expected)-1):raise ValueError('timed gradient sample incomplete')
  for name,value in actual.items():
   target=expected[name]
   if value.shape!=target.shape or not torch.isfinite(value).all():raise ValueError('invalid timed gradient '+name)
   error=float((value.float()-target.float()).norm()/target.float().norm().clamp_min(1e-8))
   if not math.isfinite(error) or error>tolerance:raise ValueError('timed training loss/gradient mismatch '+name)
   worst=max(worst,error);count+=1
  if step==iterations-1:
   output=folder/'output.safetensors'
   if output.is_symlink() or not output.is_file():raise ValueError('invalid final training output')
   final=load_file(str(output))
   if set(final)!=set(actual) or any(not torch.equal(final[name],value) for name,value in actual.items()):
    raise ValueError('final training output differs from timed audit')
 # Remove temporary tensors only after every audit has passed. No training
 # checkpoint or intermediate gradient archive is retained or uploaded.
 for item in audits:(folder/item['file']).unlink()
 return tokens,{'timed_calls_checked':len(audits),'total_timed_calls':iterations,'gradient_tensors_checked':count,'worst_nrmse':worst,'fresh_inputs_per_call':True}
