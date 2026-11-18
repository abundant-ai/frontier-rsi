"""Development command for sequence and learning tasks."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from research_model import ARCH_ROWS
from model import GPT
from evaluate import checked_state,score


def run(c,seconds,seed,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);assets=Path('/opt/assets/public');row=c['source_row']
 common=['--candidate','/app/submission/candidate.py','--row',str(row),'--seed',str(seed),'--out',str(out)]
 def worker(name,args,duration):subprocess.run([sys.executable,'/opt/runtime/'+name,*common,*args],check=True,timeout=duration)
 if row in ARCH_ROWS:
  worker('sequence_worker.py',['--seconds',str(seconds)],seconds+30)
  data=out/'development-input.safetensors';z=torch.tensor(np.load(assets/'dev.npy')[:1025]).long()[None,:];save_file({'input':z[:,:-1].contiguous()},str(data))
  worker('sequence_worker.py',['--weights',str(out),'--input',str(data)],1800)
  logits=load_file(str(out/'predictions.safetensors'))['logits'];loss=float(torch.nn.functional.cross_entropy(logits.flatten(0,1),z[:,-1:].flatten()))
  print(json.dumps({'development_loss':loss,'measurement':json.loads((out/'measurement.json').read_text())}));return
 if row in (20,22,27,29):
  worker('learning_worker.py',['--seconds',str(seconds)],seconds+30)
  if row==22:
   data=out/'questions.json';data.write_text(json.dumps([r['question'] for r in json.loads((assets/'math_dev.json').read_text())[:5]]));worker('learning_worker.py',['--weights',str(out),'--input',str(data)],180)
   from learning_worker import features
   auxiliary=torch.nn.Sequential(torch.nn.Linear(16,128),torch.nn.GELU(),torch.nn.Linear(128,1));auxiliary.load_state_dict(load_file(str(out/'auxiliary.safetensors')));auxiliary.eval()
   diagnostics={}
   with torch.no_grad():
    for name,records in json.loads((assets/'process_development.json').read_text()).items():
     diagnostics[name]=[{'id':r['id'],'label':r['label'],'step_scores':[float(auxiliary(features(step)[-1:]).sigmoid()[0,0]) for step in r['steps']]} for r in records]
   (out/'process_diagnostics.json').write_text(json.dumps(diagnostics))
   print('Development-only position scores and source labels: '+str(out/'process_diagnostics.json'))
  elif row in (27,29):
   student=GPT();checked_state(student,out/'model.safetensors');student.cuda().eval();z=torch.tensor(np.load(assets/'dev.npy')[:1025],device='cuda').long()[None,:]
   with torch.no_grad():print(json.dumps({'development_accuracy':float((student(z[:,:-1]).argmax(-1)==z[:,1:]).float().mean())}))
  else:print(json.dumps({'development_preference_accuracy':score({'family':'post','mode':'adapt'},out/'model.safetensors',assets/'pref_dev.json',assets)}))
  return
 data=out/'development-input.safetensors';extra=[]
 if row==18:
  data=out/'questions.json';data.write_text(json.dumps([r['question'] for r in json.loads((assets/'math_dev.json').read_text())[:5]]))
 elif row in (48,55):
  tokens=np.load(assets/'inference_tokens.npy');data=out/'requests.json';data.write_text(json.dumps([{'input':list(map(int,tokens[i*128:(i+1)*128])),'output_length':32} for i in range(8)]))
 else:
  tokens=np.load(assets/'inference_tokens.npy');save_file({'input':torch.tensor(tokens[:1024]).long()[None,:],'question':torch.tensor(tokens[1022:1024]).long()[None,:]},str(data))
  calibration=out/'calibration.safetensors';save_file({'input':torch.tensor(np.load(assets/'calibration.npy')).long()},str(calibration));extra=['--calibration',str(calibration),'--output-length','32']
 worker('inference_worker.py',['--input',str(data),*extra],1800)
 print((out/'measurement.json').read_text())
