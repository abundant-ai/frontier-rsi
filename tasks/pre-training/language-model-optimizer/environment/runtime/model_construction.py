"""No-gradient construction of a fixed-architecture model from supplied states."""
import argparse,ast,json,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from policy import load


def validate_source(path):
 tree=ast.parse(Path(path).read_text())
 for n in ast.walk(tree):
  if isinstance(n,ast.Attribute) and n.attr in ('backward','autograd','optim','enable_grad','set_grad_enabled','requires_grad_'):raise ValueError('gradient training is outside this construction task')


def checkpoint_state(path):
 result={}
 for f in sorted(Path(path).glob('*.safetensors')):result.update(load_file(str(f)))
 if not result:raise ValueError('no checkpoint tensors')
 # Official tied-embedding snapshots may omit the duplicate output matrix.
 if 'model.embed_tokens.weight' in result and 'lm_head.weight' not in result:result['lm_head.weight']=result['model.embed_tokens.weight']
 return result


def construct(candidate,out,assets,smoke=False):
 start=time.monotonic();validate_source(candidate);m=load(candidate,['combine'])
 if smoke:
  generator=torch.Generator().manual_seed(17)
  states=[{'weight':torch.randn(8,8,generator=generator),'bias':torch.randn(8,generator=generator)} for _ in range(4)]
  calibration=[{'prompt':'A calibration question','response':'A training response'}]
 else:
  states=[checkpoint_state(Path(assets)/'checkpoints'/str(i)) for i in range(4)]
  calibration=json.loads((Path(assets)/'construction_calibration.json').read_text())
 expected=states[0]
 for state in states:
  if set(state)!=set(expected) or any(v.shape!=expected[k].shape for k,v in state.items()):raise ValueError('supplied checkpoints do not share one architecture')
 with torch.no_grad():result=m.combine(states,calibration)
 if not isinstance(result,dict) or set(result)!=set(expected):raise ValueError('output checkpoint keys differ')
 for key,tensor in result.items():
  if not isinstance(tensor,torch.Tensor) or tensor.shape!=expected[key].shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():raise ValueError('invalid constructed tensor '+key)
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 save_file({k:v.detach().cpu().contiguous().clone() for k,v in result.items()},str(out/'model.safetensors'))
 (out/'run.json').write_text(json.dumps({'elapsed':time.monotonic()-start,'models':4,'calibration_records':len(calibration)}))


def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);p.add_argument('--assets',default='/opt/assets/public');p.add_argument('--smoke',action='store_true');a=p.parse_args();construct(a.candidate,a.out,a.assets,a.smoke)
if __name__=='__main__':main()
