# Schedule prompt processing and generation across two GPUs

## Objective

Minimize average stream duration by scheduling prompt processing and token generation across two GPUs.

## Workload

Two H100 GPUs each hold the same frozen Qwen2.5-1.5B-Instruct model in FP32. Each GPU has one execution worker; the workers can run concurrently. A stream contains 24 requests with prompt lengths of 512, 2,048 or 4,096 tokens and output lengths of 16 or 32 tokens. The first four requests arrive immediately; later requests arrive approximately 30 milliseconds apart. Generation is greedy and always produces the required number of tokens, even after an end-of-sequence token.

## Timing

Stream duration runs from stream start until its last request finishes.

Time includes scheduling, queue waits, prompt processing, generation, state transfers and synchronization. Model loading, warm-up and independent reference computation happen outside this interval.

Each workload starts with a fresh namespace and empty request state. Required preparation and compilation count toward elapsed time. CPU tensor operations use one intra-operation and one inter-operation thread.

## Correctness and resource limits

Persistent request-state reservations across both GPUs are limited to 2 GiB. `bytes_per_token` gives the fixed KV storage per cached token. `free_bytes` is the remaining reservation capacity, accounting for both waiting and executing requests. For a row with `g` generated tokens, `r` required tokens and prompt length `p`, an action of quantum `q` reserves the larger of its current state bytes and `(p + g + s - 1) * bytes_per_token`, where `s = q - 1` if the action completes the request, otherwise `s = q`. When `s = 0`, use current state bytes alone. Subtract only the increase over current state bytes from `free_bytes` when selecting each request. This reservation covers the largest retained state before the action's final release. Temporary batching and transfer buffers count toward the separate total GPU-memory limit of 16 GiB per device, including weights.

Every token must exactly match independently executed uncached output from the frozen model. Every request must finish within 120 seconds. Nearest-rank p95 time from arrival to first token must be at most 20 seconds; nearest-rank p95 gap between consecutive output tokens must be at most two seconds. Tokens are timestamped when their IDs become available on the CPU. Missing outputs, invalid actions, numerical differences or resource violations invalidate the submission.

The policy uses the pure-Python subset in `/opt/runtime/policy.py`. Imports, file/process/network APIs, reflection, private timers, annotations, decorators, defaults and persistent module state are unavailable. Source is limited to 64 KiB and each policy call to 100 milliseconds. No model training or checkpoint export is needed.

## Implementation and local runs

Edit only `/app/submission/candidate.py`, keeping `plan(now, ready, busy, free_bytes, bytes_per_token)`. The arguments contain immutable numerical summaries. `now` is real elapsed workload time. `ready` contains arrived requests that are neither complete nor executing, with rows `(request_id, prompt_length, generated_tokens, required_tokens, state_gpu, state_bytes, arrival_time)`. A request without generated tokens has `state_gpu = -1` and no retained KV state. `busy` contains two booleans for GPU 0 and GPU 1. Future requests are not exposed.

Return a list of up to two actions `(gpu, request_ids, quantum)`. Each action selects one idle GPU, a list or tuple of one to eight distinct ready request IDs, and a positive integer number of output tokens to generate for every selected request. Select each GPU and request at most once per call. Quantum cannot exceed any selected request's remaining output count. A batch must consist entirely of unstarted requests sharing a prompt length, or entirely of started requests sharing a KV-cache length. You may select any subset of such requests; you do not have to include every ready request with the same length. A started request's cache length is `prompt_length + generated_tokens - 1`. New and started requests cannot share a batch. You may leave one GPU idle while the other is busy, but cannot idle both when requests are ready.

For an unstarted request, the worker evaluates the full prompt to produce its first token. For a started request, it continues from the retained state. State may move between GPUs when an action assigns a request to a different device. Transfers copy the actual KV tensors and are timed. The worker performs batched model calls, retains separate owned state for each unfinished request and releases completed request state. You control scheduling, batching, output quantum and placement; model weights, numerical computation and state contents are fixed.

Run `/opt/conda/bin/python /opt/runtime/develop.py --out /app/runs/test-01` for a shorter workload of six requests and four output tokens each. The four-hour research budget includes all experiments and uses the same two GPUs.
