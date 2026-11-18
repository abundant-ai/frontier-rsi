"""Root-owned orchestrator; candidate imports run only in unprivileged children.

Performance edits use the restricted numerical API; timers are outside that API.
Training final scores are recomputed here from tensor-only model artifacts.
"""
import argparse,functools,hashlib,json,math,os,shutil,signal,statistics,subprocess,sys,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from safetensors import SafetensorError
from evaluate import score
from kernel_ops import inputs,output_and_gradients,reference
from performance import compute
from packing_validation import validate as validate_packing


class Invalid(Exception):pass
class Infrastructure(Exception):pass


def visible_gpu_memory_mib(gpus=1):
    try:
        # cudaMemGetInfo reports global device usage without starting a process
        # on every poll. The per-device baseline below still measures only the
        # candidate worker's increment on the exclusive allocation.
        if torch.cuda.device_count()!=gpus:raise ValueError('exclusive allocation with expected GPU count required')
        used=[]
        for device in range(gpus):
            free,total=torch.cuda.mem_get_info(device)
            if not 0<=free<=total:raise ValueError('invalid CUDA memory accounting')
            used.append((total-free)/(1024**2))
        return max(used)
    except (RuntimeError,ValueError,AssertionError) as e:raise Infrastructure('GPU memory accounting unavailable or allocation not exclusive') from e


def aggregate(values):
    """Keep the natural seed measurements; final scoring uses complete saved records."""
    if len(values) != 2 or not all(math.isfinite(value) and value >= 0 for value in values):
        raise Invalid('both finite nonnegative seed scores required')
    mean = statistics.mean(values)
    return {'valid': 1, 'reward': -1 * mean}, {
        'seed_scores': values, 'mean': mean,
        'sample_stddev': statistics.stdev(values), 'raw_reward': -1 * mean,
    }


def clear_worker_temporary_files():
    """Remove files left by earlier seed workers before starting another seed."""
    if os.geteuid()!=0:return
    for root in (Path('/tmp'),Path('/var/tmp'),Path('/dev/shm')):
        if not root.exists():continue
        for directory,dirs,files in os.walk(root,topdown=False,followlinks=False):
            for name in (*files,*dirs):
                path=Path(directory)/name
                try:
                    if path.lstat().st_uid not in (1001,1002):continue
                    if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
                    else:path.unlink()
                except FileNotFoundError:continue
    runner_home=Path('/home/runner')
    if runner_home.exists():runner_home.chmod(0o700)


def execute(command,out,timeout,uid=1001,memory_gib=None,gpus=1):
    if torch.cuda.is_available():torch.cuda.empty_cache()
    out.mkdir(parents=True,exist_ok=True)
    if os.geteuid()==0:
        os.chown(out,uid,uid);os.chmod(out,0o700)
        (out/'tmp').mkdir(mode=0o700,exist_ok=True)
        os.chown(out/'tmp',uid,uid)
        command=['setpriv',f'--reuid={uid}',f'--regid={uid}','--clear-groups','--no-new-privs',*command]
    env={k:v for k,v in os.environ.items() if k in ('PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES')}
    env.update(HOME=str(out),TMPDIR=str(out/'tmp'),PYTHONPATH='/opt/runtime',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',TORCHINDUCTOR_CACHE_DIR=str(out/'compile'))
    baseline_memory=visible_gpu_memory_mib(gpus) if memory_gib is not None else 0
    with (out/'worker.log').open('w+b') as log:
        started=time.monotonic()
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            deadline=time.monotonic()+timeout
            while p.poll() is None:
                if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(command,timeout)
                if memory_gib is not None:
                    used=max(0.,visible_gpu_memory_mib(gpus)-baseline_memory)
                    if used>memory_gib*1024:
                        os.killpg(p.pid,signal.SIGKILL);p.wait();raise Invalid('observed GPU memory cap exceeded')
                time.sleep(.1)
            code=p.returncode
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise Invalid('candidate failed to return within its execution deadline')
        finally:
            # Kill all members of the untrusted worker UID, including detached children.
            if os.geteuid()==0:subprocess.run(['pkill','-KILL','-u',str(uid)],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if os.geteuid()==0:out.chmod(0o700)
            # Keep diagnostics from the original open descriptor, so a candidate
            # cannot redirect a privileged read by replacing a path or symlink.
            if os.geteuid()==0:
                log.flush();log.seek(0,os.SEEK_END);size=log.tell();log.seek(max(0,size-262144))
                diagnostics=Path('/logs/verifier/worker-logs');diagnostics.mkdir(parents=True,exist_ok=True)
                name=hashlib.sha256(str(out).encode()).hexdigest()[:16]
                (diagnostics/(name+'.log')).write_bytes(log.read())
                (diagnostics/(name+'.json')).write_text(json.dumps({'worker_directory':str(out),'command':command,'bytes_total':size}))
    if code!=0:raise Invalid(f'candidate execution exited {code}; see worker.log')
    return time.monotonic()-started


def nrmse(actual,expected):
    if actual.shape!=expected.shape or not torch.isfinite(actual).all():return float('inf')
    return float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-8))


def read_outputs(out):
    path=out/'output.safetensors'
    if path.is_symlink() or not path.is_file():raise Invalid('missing ordinary tensor output file')
    return load_file(str(path),device='cpu')


def grade(c,tests,submission,log,smoke=False,device='cuda'):
    seeds=json.loads((tests/'seeds.json').read_text())
    if len(seeds)!=2 or len(set(seeds))!=2 or c.get('grading_seed_count')!=2:raise Infrastructure('grading requires exactly two distinct frozen grading seeds')
    scores=[];details=[]
    assets=Path('/opt/assets/public')
    if not smoke and (device!='cuda' or not torch.cuda.is_available()):raise Infrastructure('one H100 required; CPU smoke is not grading')
    if not smoke and c.get('gpu_type','H100') not in torch.cuda.get_device_name(0):raise Infrastructure('wrong GPU; '+c.get('gpu_type','H100')+' required')
    if c['family'] in ('kernel','performance','schedule','distributed') and c.get('candidate_language')!='restricted-numerical-python-v1':
        raise Infrastructure('performance measurement requires the restricted numerical edit surface')
    workspace=Path('/run/rsi-workers') if os.geteuid()==0 else log/'workers'
    workspace.mkdir(parents=True,exist_ok=True)
    if os.geteuid()==0:workspace.chmod(0o711)
    for seed_index,seed in enumerate(seeds):
        clear_worker_temporary_files()
        seed_execute=functools.partial(execute,uid=1001+seed_index)
        out=workspace/f'worker-{seed}'
        if out.exists():shutil.rmtree(out)
        base=[sys.executable,'/opt/runtime/train.py','--config','/app/contract.json','--candidate',str(submission),'--out',str(out),'--seed',str(seed),'--device',device]
        if c['family'].startswith('research_'):
            from research_grade import grade_seed
            try:value,measurement=grade_seed(c,submission,out,seed,smoke,device,seed_execute)
            except (ValueError,RuntimeError,SafetensorError,OSError) as e:raise Invalid(str(e)) from e
            scores.append(value);details.append({'seed':seed,'metric':value,'research_validation':measurement})
        elif c['family'] in ('text','post'):
            seconds=2 if smoke else 1800
            observed_seconds=seed_execute(base+['--seconds',str(seconds)]+(['--smoke'] if smoke else []),out,60 if smoke else seconds,memory_gib=None if smoke else c['memory_gib'])
            if not smoke:
                if observed_seconds<seconds-15:raise Invalid('training worker exited before its fixed duration')
                if observed_seconds>seconds:raise Invalid('training worker exceeded its fixed duration including export')
                record=out/'run.json'
                if record.is_symlink() or not record.is_file():raise Invalid('missing ordinary training record')
                if record.stat().st_size>65536:raise Invalid('invalid ordinary training record')
                try:measurement=json.loads(record.read_text())
                except (OSError,UnicodeError,json.JSONDecodeError) as error:
                    raise Invalid('invalid ordinary training record') from error
                if not isinstance(measurement,dict):raise Invalid('invalid ordinary training record')
                # Root measures the full worker lifetime, including model setup
                # and compilation; the loop's own elapsed clock starts later.
                if type(measurement.get('steps')) is not int or measurement['steps']<1:
                    raise Invalid('fixed training completed no updates')
            weights=out/'model.safetensors'
            if weights.is_symlink() or not weights.is_file():raise Invalid('missing regular checkpoint')
            if weights.stat().st_size>(1 if c['family']=='text' else 8)*1024**3:raise Invalid('checkpoint exceeds fixed-architecture serialization limit')
            if smoke and c['family']=='post':
                # Only smoke validates hook execution; no substitute toy grading score.
                details.append({'seed':seed,'smoke_only':True});continue
            data=Path('/opt/assets/private')/('eval.npy' if c['family']=='text' else 'math_eval.json' if c['mode'] in ('policy','allocation') else 'pref_eval.json')
            evaluation_measurement={}
            try:value=score(c,weights,data,assets,device,smoke,measurement=evaluation_measurement)
            except (ValueError,RuntimeError,SafetensorError,OSError) as e:raise Invalid(str(e)) from e
            if not smoke and c['mode']=='preferences':
                regression={**c,'mode':'policy'};regression_data=Path('/opt/assets/private/math_eval.json')
                baseline_file=log/'regression-baseline.json'
                if baseline_file.exists():baseline=json.loads(baseline_file.read_text())['accuracy']
                else:
                    baseline=score(regression,None,regression_data,assets,device,False)
                    baseline_file.write_text(json.dumps({'accuracy':baseline}))
                candidate_accuracy=score(regression,weights,regression_data,assets,device,False)
                if candidate_accuracy<baseline-.02:raise Invalid('capability-regression accuracy drop exceeds 2 percentage points')
            scores.append(value);details.append({'seed':seed,'metric':value,'evaluation':evaluation_measurement,'training':measurement if not smoke else {'smoke_only':True}})
        elif c['family']=='merge':
            command=[sys.executable,'/opt/runtime/model_construction.py','--candidate',str(submission),'--out',str(out)]
            seed_execute(command+(['--smoke'] if smoke else []),out,60 if smoke else 900,memory_gib=None if smoke else c['memory_gib'])
            weights=out/'model.safetensors'
            if weights.is_symlink() or not weights.is_file() or weights.stat().st_size>8*1024**3:raise Invalid('invalid constructed checkpoint')
            if smoke:
                state=load_file(str(weights));
                if set(state)!={'weight','bias'} or state['weight'].shape!=(8,8):raise Invalid('constructed smoke state differs')
                details.append({'seed':seed,'smoke_only':True});continue
            try:
                reasoning=score({**c,'family':'post','mode':'policy'},weights,Path('/opt/assets/private/math_eval.json'),assets,device,False)
                preference=score({**c,'family':'post','mode':'adapt'},weights,Path('/opt/assets/private/pref_eval.json'),assets,device,False)
            except (ValueError,RuntimeError,SafetensorError,OSError) as e:raise Invalid(str(e)) from e
            value=(reasoning+preference)/2;scores.append(value);details.append({'seed':seed,'metric':value,'reasoning_accuracy':reasoning,'preference_accuracy':preference})
        elif c['family']=='distributed':
            from distributed_validation import compare,compare_timing
            import secrets
            latencies=[];checks=[]
            for index in range(1 if smoke else 3 if c['mode']=='training' else 30):
                case=out/f'case-{index}'
                audit_key=secrets.token_hex(32)
                for phase in ('reference','candidate','timing-reference','timing'):
                    phase_out=case/phase
                    command=[sys.executable,'/opt/runtime/distributed_ops.py','--candidate',str(submission),'--mode',c['mode'],'--seed',str(seed),'--index',str(index),'--out',str(phase_out),'--phase',phase]
                    if phase.startswith('timing'):command+=['--audit-key',audit_key]
                    seed_execute(command+(['--smoke'] if smoke else []),phase_out,180 if smoke else 1800,memory_gib=None if smoke else c['memory_gib'],gpus=4)
                    # Candidate and timing callbacks run under the same worker
                    # UID. Seal each completed phase before another callback
                    # can access it; references never share a writable folder.
                    if os.geteuid()==0:os.chown(phase_out,0,0);phase_out.chmod(0o700)
                try:
                    proof=compare(case/'candidate',reference_folder=case/'reference',tolerance=.02 if c['mode']=='training' else .0001)
                    proof['timed_execution']=compare_timing(case/'timing',case/'timing-reference',audit_key,seed,c['mode'],index,smoke=smoke)
                    checks.append(proof)
                except ValueError as e:raise Invalid(str(e)) from e
                measurement=json.loads((case/'timing/timing-measurement.json').read_text());ms=float(measurement['latency_ms'])
                if measurement['ranks']!=4 or not math.isfinite(ms) or ms<=0:raise Invalid('invalid distributed measurement')
                latencies.append(ms)
            value=math.exp(statistics.mean(math.log(x) for x in latencies));scores.append(value);details.append({'seed':seed,'metric':value,'numerical_validation':checks})
        elif c['family']=='schedule':
            command=[sys.executable,'/opt/runtime/compiler_suite.py','--candidate',str(submission),'--out',str(out),'--seed',str(seed),'--device',device]
            seed_execute(command+(['--smoke'] if smoke else []),out,60 if smoke else 3600,memory_gib=None if smoke else c['memory_gib'])
            measurement=json.loads((out/'measurement.json').read_text())
            if measurement['cases']!=(40 if smoke else 800) or measurement['max_nrmse']>1e-4:raise Invalid('incomplete or incorrect compiler suite')
            if measurement['peak_allocated_bytes']>c['memory_gib']*1024**3:raise Invalid('compiler suite memory cap exceeded')
            value=float(measurement['metric'])
            if not math.isfinite(value) or value<=0:raise Invalid('invalid compiler suite speedup')
            scores.append(value);details.append({'seed':seed,'metric':value,'cases':measurement['cases']})
        elif c['family']=='kernel':
            latencies=[]
            for index in range(2 if smoke else c['workloads']):
                case=out/f'case-{index}';case.parent.mkdir(parents=True,exist_ok=True)
                if os.geteuid()==0:os.chown(case.parent,1001+seed_index,1001+seed_index)
                command=[sys.executable,'/opt/runtime/kernel_worker.py','--config','/app/contract.json','--candidate',str(submission),'--out',str(case),'--seed',str(seed),'--index',str(index),'--device',device]
                seed_execute(command+(['--smoke'] if smoke else []),case,300,memory_gib=None if smoke else c['memory_gib'])
                actual=read_outputs(case);a=inputs(c['operation'],seed,index,device,smoke)
                expected=output_and_gradients(lambda x:reference(c['operation'],x),a,c.get('gradients',False))
                if set(actual)!={f'output_{i}' for i in range(len(expected))}:raise Invalid('output tensor count differs')
                if c['operation']=='op16':
                    from kernel_checks import sampling
                    try:sampling([actual[f'output_{i}'] for i in range(len(expected))],[x.cpu() for x in expected],{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in a.items()},smoke)
                    except ValueError as e:raise Invalid(str(e)) from e
                for i,e in enumerate(expected):
                    if c['operation']=='op16':continue
                    tolerance=c['tolerance']
                    if c['operation'] in ('op07','op08','op10'):tolerance=.0001 if e.dtype==torch.float32 else .02
                    if c['operation']=='op15' and i==0 and not torch.equal(actual[f'output_{i}'],e.cpu()):raise Invalid('indexed copies must be bitwise exact')
                    if not e.is_floating_point() and not torch.equal(actual[f'output_{i}'],e.cpu()):raise Invalid('discrete outputs differ')
                    error=abs(float(actual[f'output_{i}'])-float(e)) if c['operation']=='op09' and i==0 else nrmse(actual[f'output_{i}'],e.cpu())
                    if error>(.001 if c['operation']=='op09' and i==0 else tolerance):raise Invalid(f'output/gradient mismatch case {index}, tensor {i}')
                measurement=json.loads((case/'measurement.json').read_text())
                if measurement['peak_allocated_bytes']>c['memory_gib']*1024**3:raise Invalid('memory cap exceeded')
                ms=float(measurement['latency_ms'])
                if not math.isfinite(ms) or ms<=0:raise Invalid('invalid latency')
                metric=ms
                if c['operation']=='op02':
                    ref_ms=float(measurement['reference_latency_ms'])
                    if not math.isfinite(ref_ms) or ref_ms<=0:raise Infrastructure('invalid fixed reference timing')
                    metric=ref_ms/ms
                latencies.append(metric)
            value=math.exp(statistics.mean(math.log(v) for v in latencies));scores.append(value);details.append({'seed':seed,'metric':value,'unit':'speedup' if c['operation']=='op02' else 'milliseconds'})
        else:
            command=[sys.executable,'/opt/runtime/performance.py','--config','/app/contract.json','--candidate',str(submission),'--out',str(out),'--seed',str(seed),'--device',device]
            seed_execute(command+(['--smoke'] if smoke else []),out,600,memory_gib=None if smoke else c['memory_gib'])
            actual=read_outputs(out)
            validation=None
            if c['mode']=='batches':
                try:validation=validate_packing(actual,seed,device,assets,smoke,c['tolerance'])
                except ValueError as e:raise Invalid(str(e)) from e
                except torch.cuda.OutOfMemoryError as e:raise Infrastructure('trusted FP32 packing validation ran out of memory') from e
                tokens=validation['tokens']
            else:
                from performance_validation import validate_execution
                try:tokens,validation=validate_execution(out,seed,device,assets,smoke,c['tolerance'])
                except ValueError as e:raise Invalid(str(e)) from e
            measurement=json.loads((out/'measurement.json').read_text())
            if measurement['peak_allocated_bytes']>c['memory_gib']*1024**3:raise Invalid('memory cap exceeded')
            ms=float(measurement['latency_ms'])
            if not math.isfinite(ms) or ms<=0:raise Invalid('invalid latency')
            value=tokens/(ms/1000) if c['mode']=='batches' else ms
            scores.append(value);details.append({'seed':seed,'metric':value})
            if validation:details[-1]['numerical_validation']=validation
    if smoke:return {'valid':1,'smoke_only':1},{'smoke_only':True,'seeds':details}
    rewards,report=aggregate(scores);report['runs']=details
    from gpu_timing import hardware_snapshot
    queued=c['family']=='kernel' or c['family']=='research_arch' and c['source_row'] in (7,11) or c['family']=='research_inference' and c['source_row']==32
    protocol='queued-cuda-events-v1' if queued else 'synchronized-wall-clock-v1'
    report['hardware']=hardware_snapshot(timing_protocol=protocol)
    return rewards,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--tests',default='/tests');p.add_argument('--log',default='/logs/verifier');p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args()
    tests=Path(a.tests);log=Path(a.log);log.mkdir(parents=True,exist_ok=True)
    c=json.loads((tests/'contract.json').read_text())
    # An unresolved infrastructure error must not leave a normal reward artifact.
    for name in ('reward.json','reward.txt'):(log/name).unlink(missing_ok=True)
    try:rewards,report=grade(c,tests,Path('/opt/submission/candidate.py'),log,a.smoke,a.device)
    except Infrastructure as e:
        (log/'infrastructure_error.json').write_text(json.dumps({'error':str(e)}));raise
    except Invalid as e:rewards={'valid':0};report={'valid':False,'failure_reason':str(e)}
    if not a.smoke and not report.get('partial'):
        from score import finalize_metrics, score_metrics
        report=finalize_metrics(report, rewards.get('valid')==1, tests, Path('/opt/submission/candidate.py'))
        rewards=score_metrics(report, json.loads((tests/'reward_config.json').read_text()))
    (log/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (log/'reward.json').write_text(json.dumps(rewards,allow_nan=False))
if __name__=='__main__':main()
