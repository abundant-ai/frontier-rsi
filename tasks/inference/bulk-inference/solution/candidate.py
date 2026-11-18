"""Reference implementation: independent sequential decoding in four GPU workers."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def _generate_chunk(requests, model_path, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, local_files_only=True
    ).to(device).eval()
    result = []
    with torch.inference_mode():
        for request in requests:
            ids = tokenizer(request["prompt"], return_tensors="pt")["input_ids"].to(device)
            length = int(request["max_new_tokens"])
            output = model.generate(
                input_ids=ids,
                do_sample=False,
                min_new_tokens=length,
                max_new_tokens=length,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            result.append(output[0, ids.shape[1]:].tolist())
    return result


def generate_all(requests, model_path, device):
    worker_count = 4
    chunks = [[requests[i] for i in range(worker, len(requests), worker_count)]
              for worker in range(worker_count)]
    with tempfile.TemporaryDirectory(prefix="bulk-workers-", dir="/app/runs") as directory:
        root = Path(directory)
        processes = []
        for worker, chunk in enumerate(chunks):
            input_path = root / f"input-{worker}.json"
            output_path = root / f"output-{worker}.json"
            input_path.write_text(json.dumps(chunk))
            process = subprocess.Popen([
                sys.executable, "/app/submission/candidate.py", "--worker",
                str(input_path), str(output_path), model_path, device,
            ])
            processes.append((process, output_path))
        chunk_outputs = []
        for process, output_path in processes:
            if process.wait() != 0:
                raise RuntimeError(f"reference worker exited with {process.returncode}")
            chunk_outputs.append(json.loads(output_path.read_text()))
    outputs = [None] * len(requests)
    for worker, chunk in enumerate(chunk_outputs):
        for offset, answer in enumerate(chunk):
            outputs[worker + offset * worker_count] = answer
    print("BULK_ORACLE_DIAGNOSTICS " + json.dumps({"workers": worker_count}), flush=True)
    return outputs


if __name__ == "__main__" and len(sys.argv) == 6 and sys.argv[1] == "--worker":
    _, _, input_file, output_file, model_path, device = sys.argv
    answers = _generate_chunk(json.loads(Path(input_file).read_text()), model_path, device)
    Path(output_file).write_text(json.dumps(answers, separators=(",", ":")))
