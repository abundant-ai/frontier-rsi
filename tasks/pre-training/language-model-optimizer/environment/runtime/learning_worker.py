"""Metered generation, teacher-guided training, and trainable solution ranking."""
import argparse,contextlib,json,re,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer
from research_hooks import load
from model import GPT,Config
from inference_core import frozen,reason
from train import logprob,encode_pair

HOOKS=['collect','training_loss','feedback_loss','auxiliary_loss','rank','plan']

def features(text):
 # Public deterministic trajectory features; labels are supplied only for the
 # fixed training records, never inferred from held-out answer correctness.
 lines=text.splitlines() or [text];rows=[]
 for index,line in enumerate(lines):
  numbers=re.findall(r'-?\d+(?:\.\d+)?',line)
  rows.append([1.,index/max(1,len(lines)-1),len(line)/1024,len(numbers)/32,line.count('=')/8,line.count('+')/8,line.count('-')/8,line.count('*')/8]+[sum(map(ord,line[i::8]))%101/100 for i in range(8)])
 return torch.tensor(rows,dtype=torch.float32)


def run(a):
 start=time.monotonic();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assets=Path(a.assets);torch.manual_seed(a.seed)
 hooks=load(a.candidate,HOOKS);teacher,tok=frozen(assets,a.device,a.smoke);teacher.eval().requires_grad_(False)
 if a.row==22:
  auxiliary=torch.nn.Sequential(torch.nn.Linear(16,128),torch.nn.GELU(),torch.nn.Linear(128,1)).to(a.device)
  if a.input:
   auxiliary.load_state_dict(load_file(str(Path(a.weights)/'auxiliary.safetensors')))
   def ranker(text):
    values=auxiliary(features(text).to(a.device)).flatten();result=hooks.rank(values)
    if not isinstance(result,torch.Tensor) or result.numel()!=1 or not torch.isfinite(result):raise ValueError('rank must return a finite scalar')
    return float(result)
   predictions=[];generated_tokens=[]
   for question in json.loads(Path(a.input).read_text()):
    value,count=reason(teacher,tok,hooks,question,a.device,a.smoke,ranker);predictions.append(value);generated_tokens.append(count)
   (out/'measurement.json').write_text(json.dumps({'answers':predictions,'generated_tokens':generated_tokens}));return
  records=[{'text':'2 + 2 = 4','labels':[1.]}]*4 if a.smoke else json.loads((assets/'solution_records.json').read_text())
  if not a.smoke and (len(records)!=10000 or sum(sum(v>=0 for v in r['labels']) for r in records)>20000):raise ValueError('solution/annotation allowance')
  opt=torch.optim.AdamW(auxiliary.parameters(),lr=1e-3);steps=0
  while steps<2 if a.smoke else time.monotonic()-start<a.seconds-15:
   record=records[steps%len(records)];x=features(record['text']).to(a.device);labels=torch.tensor(record['labels'],device=a.device)
   if len(labels)!=len(x):raise ValueError('position annotation mismatch')
   opt.zero_grad();loss=hooks.auxiliary_loss(auxiliary(x).flatten()[labels>=0],labels[labels>=0])
   if not torch.isfinite(loss):raise ValueError('nonfinite auxiliary objective')
   loss.backward();opt.step();steps+=1
  save_file({k:v.detach().cpu().contiguous() for k,v in auxiliary.state_dict().items()},str(out/'auxiliary.safetensors'))
  (out/'measurement.json').write_text(json.dumps({'steps':steps,'parameters':sum(p.numel() for p in auxiliary.parameters())}));return
 if a.row==20:
  # Train the supplied checkpoint; the initial service is released so all
  # generation, judging and updates share one aggregate memory allocation.
  model=teacher.requires_grad_(True);opt=torch.optim.AdamW(model.parameters(),lr=1e-6)
  records=[{'chosen':[],'rejected':[]}]*4 if a.smoke else json.loads((assets/'feedback_labels.json').read_text())
  prompts=['2+2?']*4 if a.smoke else json.loads((assets/'feedback_prompts.json').read_text())
  if not a.smoke and (len(records)!=1000 or len(prompts)!=10000):raise ValueError('feedback data allowance')
  generated=0;steps=0;summary=torch.zeros(len(prompts),2)
  while steps<2 if a.smoke else time.monotonic()-start<a.seconds-30:
   index,count,limit=hooks.collect(summary.clone(),min(1.,(time.monotonic()-start)/a.seconds))
   if type(index)is not int or not 0<=index<len(prompts) or type(count)is not int or not 0<=count<=8 or type(limit)is not int or not 1<=limit<=2048:raise ValueError('data collection request')
   opt.zero_grad()
   if a.smoke:
    x=torch.tensor([[1,2,3,4]],device=a.device);logits=model(x).logits
    pos=logits.log_softmax(-1)[0,-1,4:5];neg=logits.log_softmax(-1)[0,-1,5:6];judgment=torch.zeros(2,device=a.device)
   elif count and generated+count*(limit+1)<=10_000_000:
    x=tok.apply_chat_template([{'role':'user','content':prompts[index]}],add_generation_prompt=True,return_tensors='pt').to(a.device)
    with torch.no_grad():seq=model.generate(x.repeat(count,1),max_new_tokens=limit,do_sample=True,temperature=.8,pad_token_id=tok.eos_token_id)
    generated+=count*(seq.shape[1]-x.shape[1]);responses=[tok.decode(r[x.shape[1]:],skip_special_tokens=True) for r in seq]
    # Judge with this same model; independent final labels never enter here.
    judgment=[]
    for response in responses:
     judge=tok.apply_chat_template([{'role':'user','content':prompts[index]+'\nResponse: '+response+'\nRate correctness from 0 to 9. Return one digit.'}],add_generation_prompt=True,return_tensors='pt').to(a.device)
     with torch.no_grad():rating=model.generate(judge,max_new_tokens=1,do_sample=False,pad_token_id=tok.eos_token_id)
     generated+=1;text=tok.decode(rating[0,judge.shape[1]:]);judgment.append(float(next((ch for ch in text if ch.isdigit()),'0')))
    judgment=torch.tensor(judgment,device=a.device);best=int(judgment.argmax());worst=int(judgment.argmin())
    labels=seq.clone();labels[:,:x.shape[1]]=-100
    pos=logprob(model,seq[best:best+1],labels[best:best+1]);neg=logprob(model,seq[worst:worst+1],labels[worst:worst+1]);summary[index,0]+=count
   else:
    r=records[steps%len(records)];x,labels=encode_pair(tok,r['chosen'],a.device);rx,rl=encode_pair(tok,r['rejected'],a.device)
    pos=logprob(model,x,labels);neg=logprob(model,rx,rl);judgment=torch.tensor([1.,0.],device=a.device)
   loss=hooks.feedback_loss(pos,neg,judgment)
   if not torch.isfinite(loss):raise ValueError('nonfinite feedback objective')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1
  save_file({k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()},str(out/'model.safetensors'))
  (out/'measurement.json').write_text(json.dumps({'steps':steps,'generated_tokens':generated}));return
 # The evaluated student has a fixed architecture/tokenizer. Teacher text is
 # retokenized, rather than equating unrelated GPT-2 and Qwen token indices.
 cfg=Config(vocab=64,width=32,layers=2,heads=4,context=64) if a.smoke else Config()
 student=GPT(cfg).to(a.device);opt=torch.optim.AdamW(student.parameters(),lr=6e-4)
 student_tok=None if a.smoke else AutoTokenizer.from_pretrained(assets/'tokenizer',local_files_only=True)
 pool=np.random.default_rng(a.seed).integers(4,cfg.vocab,8192,dtype=np.int32) if a.smoke else np.load(assets/'train.npy',mmap_mode='r')
 if a.row==29:pool=pool[:1_000_000]
 generated=0;queried=0;steps=0;length=16 if a.smoke else 1024;g=np.random.default_rng(a.seed);summary=torch.zeros(1,2)
 while steps<2 if a.smoke else time.monotonic()-start<a.seconds-30:
  offset=int(g.integers(0,len(pool)-length-1));z=torch.tensor(np.array(pool[offset:offset+length+1]),device=a.device).long()[None,:]
  index,count,limit=hooks.collect(summary.clone(),min(1.,(time.monotonic()-start)/a.seconds))
  if type(count)is not int or not 0<=count<=8 or type(limit)is not int or not 1<=limit<=2048:raise ValueError('teacher service request')
  teacher_ids=z[:,1:].clone()
  cap=5_000_000 if a.row==29 else 2_000_000
  if count and generated+count*limit<=cap and (a.row==29 or queried<10_000_000):
   if a.smoke:synthetic=z
   else:
    text=student_tok.decode(z[0,:min(128,z.shape[1])]);prompt=tok(text,return_tensors='pt').input_ids.to(a.device).repeat(count,1)
    if a.row==27 and queried+prompt.numel()>10_000_000:raise ValueError('teacher input-token allowance')
    queried+=prompt.numel()
    with torch.no_grad():seq=teacher.generate(prompt,max_new_tokens=limit,do_sample=True,temperature=.8,pad_token_id=tok.eos_token_id)
    generated+=seq.shape[0]*(seq.shape[1]-prompt.shape[1]);text='\n'.join(tok.decode(r[prompt.shape[1]:],skip_special_tokens=True) for r in seq)
    ids=student_tok.encode(text,add_special_tokens=False)[:length+1]
    synthetic=torch.tensor([ids],device=a.device).long() if len(ids)>1 else z
   if a.row==29:z=synthetic
   else:
    teacher_ids=synthetic[:,1:];z=synthetic
  summary[0]=torch.tensor([queried,generated]);opt.zero_grad();logits=student(z[:,:-1]);loss=hooks.training_loss(logits,z[:,1:],teacher_ids)
  if not torch.isfinite(loss):raise ValueError('nonfinite student objective')
  loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.);opt.step();steps+=1
 save_file({k:v.detach().cpu().contiguous().clone() for k,v in student.state_dict().items()},str(out/'model.safetensors'))
 (out/'measurement.json').write_text(json.dumps({'steps':steps,'queried_input_tokens':queried,'generated_tokens':generated}))

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--row',required=True,type=int);p.add_argument('--seed',default=101,type=int);p.add_argument('--seconds',default=120,type=int);p.add_argument('--device',default='cuda');p.add_argument('--assets',default='/opt/assets/public');p.add_argument('--out',required=True);p.add_argument('--input');p.add_argument('--weights');p.add_argument('--smoke',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
