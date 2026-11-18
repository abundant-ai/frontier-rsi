"""Root-owned numerical comparison for every rank and every required tensor."""
import math,json,hashlib
from pathlib import Path
from safetensors.torch import load_file
import torch


def compare_timing(folder,reference_folder,key,seed,mode,index,world=4,smoke=False,quick=False):
 """Compare actual timed fragments with an independent full update trajectory."""
 from distributed_audits import schedule,input_seed
 folder=Path(folder);reference_folder=Path(reference_folder)
 iterations=2 if smoke else (1 if mode=='training' else 2) if quick else 100
 steps=schedule(key,iterations);precision='float32' if smoke or mode=='reduction' and index%2 else 'bfloat16'
 tolerance=.0001 if precision=='float32' and mode=='reduction' else .02
 worst=0.;count=0;hashes=[];temporary=[]
 for rank in range(world):
  manifests=[]
  for phase,directory in [('timing',folder),('timing-reference',reference_folder)]:
   path=directory/f'{phase}-audits-rank{rank}.json'
   if path.is_symlink() or not path.is_file():raise ValueError('missing ordinary timed audit manifest')
   manifest=json.loads(path.read_text());files=manifest.get('files',[])
   if manifest.get('iterations')!=iterations or manifest.get('audited_steps')!=steps or manifest.get('precision')!=precision or manifest.get('audit_key_sha256')!=hashlib.sha256(bytes.fromhex(key)).hexdigest():
    raise ValueError('distributed timed audit contract differs')
   if [item['step'] for item in files]!=steps:raise ValueError('incomplete distributed timed audits')
   for item in files:
    if item['file']!=f'{phase}-audit-{item["step"]}-rank{rank}.safetensors' or item['input_seed']!=input_seed(key,seed,item['step']):
     raise ValueError('distributed timed audit identity differs')
   manifests.append(files)
  for left,right in zip(*manifests):
   paths=[folder/left['file'],reference_folder/right['file']]
   if any(p.is_symlink() or not p.is_file() for p in paths):raise ValueError('timed audits must be ordinary tensor files')
   a,b=[load_file(str(p)) for p in paths]
   if not b or set(a)!=set(b):raise ValueError('timed gradient/output keys differ')
   for name,expected in b.items():
    actual=a[name]
    if actual.shape!=expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():raise ValueError('invalid distributed timed tensor '+name)
    error=float((actual-expected).norm()/expected.norm().clamp_min(1e-8))
    if not math.isfinite(error) or error>tolerance:raise ValueError('distributed timed work mismatch '+name)
    worst=max(worst,error);count+=1
   hashes.append({str(rank)+':'+str(left['step']):[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]});temporary.extend(paths)
 for path in temporary:path.unlink()
 return {'ranks':world,'timed_calls_checked_per_rank':len(steps),'total_timed_calls_per_rank':iterations,'tensor_fragments_checked':count,'worst_nrmse':worst,'precision':precision,'fresh_inputs_per_call':True,'independent_reference_trajectory':True,'temporary_tensor_sha256':hashes}


def compare(folder,world=4,tolerance=.02,reference_folder=None):
 worst=0.;count=0
 reference_folder=Path(reference_folder) if reference_folder is not None else Path(folder)
 for rank in range(world):
  candidate_path=Path(folder)/f'candidate-rank{rank}.safetensors'
  reference_path=reference_folder/f'reference-rank{rank}.safetensors'
  if any(p.is_symlink() or not p.is_file() for p in (candidate_path,reference_path)):raise ValueError('distributed outputs must be ordinary tensor files')
  a=load_file(str(candidate_path));b=load_file(str(reference_path))
  if not a or set(a)!=set(b):raise ValueError('distributed tensor keys differ')
  for name,expected in b.items():
   actual=a[name]
   if actual.shape!=expected.shape or not torch.isfinite(actual).all():raise ValueError('invalid distributed tensor '+name)
   error=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-8));worst=max(worst,error);count+=1
   if not math.isfinite(error) or error>tolerance:raise ValueError('distributed FP32 mismatch '+name)
 return {'ranks':world,'tensors':count,'worst_nrmse':worst,'precision':'float32'}
