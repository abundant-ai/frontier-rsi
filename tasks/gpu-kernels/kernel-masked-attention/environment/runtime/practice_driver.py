"""Adapt public custom inputs to each task's existing execution and checking APIs."""
from __future__ import annotations
import argparse
import gc
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from graph_checks import compare
from kernel_ops import output_and_gradients, reference
from practice_cases import Case,read_cases
try:
    from practice_inputs import Values,make_inputs
except ImportError:
    from case_inputs import Values,make_inputs
from sweep_protocol import read_contract


class InputSuite:
    """Provide freshly generated values to the unchanged sweep worker."""
    def __init__(self, kind: str, cases: list[Case]) -> None:
        """Store validated public metadata.

        Args:
            kind: Operation identifier.
            cases: Shape specifications in execution order.
        """
        self.kind,self.cases=kind,cases

    def __call__(self, kind: str, seed: int, index: int, device: str='cpu', smoke: bool=False) -> Values:
        """Generate one case using the runtime's input factory interface.

        Args:
            kind: Must match this suite's operation.
            seed: Tensor-value seed.
            index: Position in the suite.
            device: Torch allocation device.
            smoke: Must be false to preserve selected dimensions.

        Returns:
            Fresh input tensors and scalar settings.
        """
        if kind!=self.kind or smoke or not 0<=index<len(self.cases):raise ValueError('Invalid input request')
        return make_inputs(kind,self.cases[index],seed,device)


def main() -> None:
    """Run preparation, untimed output generation, timing, or independent checking."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('specs','outputs','measure','check'))
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();contract=read_contract(args.config);kind=contract['operation']
    cases=read_cases(args.cases,kind);suite=InputSuite(kind,cases)
    if args.phase=='specs':
        specs=[]
        for c in cases:
            m,n,k=c['m'],c['n'],c['k']
            spec=[m,k,n,1 if c['transpose_a'] else k,m if c['transpose_a'] else 1,
                  1 if c['transpose_b'] else n,k if c['transpose_b'] else 1]
            if spec not in specs:specs.append(spec)
        (args.out/'specs.json').write_text(json.dumps(specs));return
    if args.phase=='measure':
        if kind=='op11':import development_sweep_worker as worker
        else:import sweep_worker as worker
        worker.inputs=suite
        interface=['--plan',str(args.out/'plan')] if kind=='op11' else ['--candidate','/app/submission/candidate.py']
        sys.argv=['worker',*interface,'--config',str(args.config),'--seed',str(args.seed),'--out',str(args.out)]
        worker.main();return
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if args.phase=='outputs':
        torch.cuda.set_per_process_memory_fraction(min(1.,contract['memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory))
        if kind=='op11':
            from plan_executor import Executor
            fn=Executor(args.out/'plan').run
        else:
            from restricted import load
            fn=load('/app/submission/candidate.py',['run']).run
        torch.cuda.reset_peak_memory_stats()
        for i,c in enumerate(cases):
            values=make_inputs(kind,c,args.seed)
            if kind=='op10':
                from sweep_worker import optimizer_chain
                optimizer_chain(fn,values,contract)
            outputs=output_and_gradients(fn,values,contract['gradients'])
            if any(type(v) is not torch.Tensor or v.device.type!='cuda' for v in outputs):raise ValueError('Outputs must be CUDA tensors')
            save_file({f'output_{j}':v.cpu().contiguous().clone() for j,v in enumerate(outputs)},str(args.out/f'case-{i}.safetensors'))
            del values,outputs;gc.collect();torch.cuda.empty_cache()
        (args.out/'execution.json').write_text(json.dumps({'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'timed':False}));return

    def numerical_reference(values: Values) -> torch.Tensor | tuple[torch.Tensor,...] | list[torch.Tensor]:
        """Recompute the public operation without loading candidate code.

        Args:
            values: Fresh input dictionary.

        Returns:
            Reference output tensors in their documented order.
        """
        return reference(kind,values)

    results=[];failed=False
    for i,c in enumerate(cases):
        values=make_inputs(kind,c,args.seed)
        expected=output_and_gradients(numerical_reference,values,contract['gradients'])
        saved=load_file(str(args.out/f'case-{i}.safetensors'),device='cuda')
        result={'index':i,'case':c,'passed':True}
        try:
            if set(saved)!={f'output_{j}' for j in range(len(expected))}:raise ValueError('Output keys differ')
            actual=[saved[f'output_{j}'] for j in range(len(expected))]
            compare(actual,expected,contract,values)
        except ValueError as error:
            result.update(passed=False,error=str(error));failed=True
        errors={}
        for j,target in enumerate(expected):
            value=saved.get(f'output_{j}');error=None
            if value is not None and value.shape==target.shape and target.is_floating_point() and bool(torch.isfinite(value).all()):
                error=float((value.double()-target.double()).norm()/target.double().norm().clamp_min(1e-8))
            errors[f'output_{j}']=error
        result['nrmse_vs_reference']=errors;results.append(result)
        del values,expected,saved
        if 'actual' in locals():del actual
        gc.collect();torch.cuda.empty_cache()
        print(json.dumps({'case':i,'passed':result['passed']}),flush=True)
    report={'correctness_passed':not failed,'cases':results}
    if kind=='op16':report['note']='NRMSE against the reference is diagnostic; acceptance uses the returned-prefix boundary, CDF and distribution checks.'
    (args.out/'correctness.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
