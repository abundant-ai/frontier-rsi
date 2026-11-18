# Bulk inference

Improve the supplied program for generating responses to a collection of prompts with a frozen language model. Your goal is to minimize the total wall-clock time needed to complete the collection on one H100.

The model is Qwen2.5-1.5B-Instruct. The model weights, tokenizer, prompt text, requested output lengths, and greedy decoding rule are fixed. Your program must return exactly the same token IDs as the supplied sequential program for every request, in the original request order. Every request specifies a positive `max_new_tokens`; generate exactly that many tokens, including after an EOS token.

Edit `/app/submission/candidate.py`. The entry point is `generate_all(requests, model_path, device)`, returning a list of lists of generated token IDs. Each request is a dictionary with `prompt` (text) and `max_new_tokens` (integer). You may change the implementation and its imports, but do not change any files outside `/app/submission`. Do not download or modify model assets.

The workspace provides 100 example requests in `/app/sample_prompts.json`. Test your implementation with `python /opt/runtime/develop.py --count 10`; use `--count 100` to test every example. The production workload contains 500 requests with input lengths of about 64, 128, 256, or 512 tokens and output lengths of 16, 32, or 64 tokens. Both input and output lengths vary across requests.

The measured interval includes process startup, model loading, input loading, tokenization, inference, and writing outputs. A 500-request run must finish within one hour; incorrect, incomplete, or timed-out runs fail. You have two hours for development.
