"""Public development CLI. Durations are enforced independently of submissions."""
import argparse,json,subprocess,sys,secrets
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=120);p.add_argument('--seed',type=int,default=101);p.add_argument('--out',default='/app/runs/latest');a=p.parse_args()
 c=json.loads(Path('/app/contract.json').read_text())
 command=[sys.executable]
 if c['family'].startswith('research_'):
  if not 120<=a.seconds<=900:p.error('--seconds must be between 120 and 900')
  from research_develop import run
  run(c,a.seconds,a.seed,a.out)
 elif c['family'] in ('text','post'):
  if not 120<=a.seconds<=900:p.error('--seconds must be between 120 and 900')
  command+=['/opt/runtime/train.py','--seconds',str(a.seconds),'--seed',str(a.seed),'--out',a.out]
  subprocess.run(command,check=True,timeout=a.seconds+120)
  data='/opt/assets/public/dev.npy' if c['family']=='text' else '/opt/assets/public/math_dev.json' if c['mode'] in ('policy','allocation') else '/opt/assets/public/pref_dev.json'
  subprocess.run([sys.executable,'/opt/runtime/evaluate.py','--config','/app/contract.json','--weights',a.out+'/model.safetensors','--data',data],check=True)
 elif c['family']=='merge':
  subprocess.run(command+['/opt/runtime/model_construction.py','--candidate','/app/submission/candidate.py','--out',a.out],check=True,timeout=900)
  subprocess.run([sys.executable,'/opt/runtime/evaluate.py','--config','/app/contract.json','--weights',a.out+'/model.safetensors','--data','/opt/assets/public/pref_dev.json'],check=True)
 elif c['family']=='distributed':
  subprocess.run(command+['/opt/runtime/distributed_ops.py','--candidate','/app/submission/candidate.py','--mode',c['mode'],'--seed',str(a.seed),'--out',a.out,'--audit-key',secrets.token_hex(32)],check=True,timeout=1800)
 elif c['family']=='schedule':
  subprocess.run(command+['/opt/runtime/compiler_suite.py','--candidate','/app/submission/candidate.py','--seed',str(a.seed),'--out',a.out],check=True,timeout=3600)
 elif c['family']=='kernel':
  for index in range(3):
   subprocess.run(command+['/opt/runtime/kernel_worker.py','--candidate','/app/submission/candidate.py','--config','/app/contract.json','--seed',str(a.seed),'--index',str(index),'--out',a.out+f'/{index}'],check=True,timeout=300)
 else:
  subprocess.run(command+['/opt/runtime/performance.py','--candidate','/app/submission/candidate.py','--config','/app/contract.json','--seed',str(a.seed),'--out',a.out],check=True,timeout=900)
if __name__=='__main__':main()
