"""Correct sequential starting point for bulk greedy generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_all(requests, model_path, device):
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
