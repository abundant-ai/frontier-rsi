# Improve the supplied implementation

Minimize final next-token validation loss, measured in nats per token, by improving the supplied training computation and objective while exporting the unchanged deployed model architecture.

Final evaluation trains from scratch under two independent seeds and measures held-out next-token loss. Minimize the arithmetic mean of the two losses.

## Environment

You have 1 H100 GPU(s), 16 CPU cores, 64 GiB host RAM, and four hours for all research activity. Internet access and external model APIs are unavailable. Dependencies and permitted data/model assets are installed under `/opt/assets/public`. `/opt/runtime` is read-only. Development feedback uses separate development data; held-out evaluation inputs are unavailable during research.

## Deliverable and editing boundary

Edit only `/app/submission/candidate.py`. The submitted file must be no larger than 256 KiB. Keep these public entrypoints and their signatures:
- `configure`
- `training_loss`

The starter source and fixed runner define their argument/return contracts. You may add undecorated helper functions within the bounded stateless numerical API. Do not change the fixed model architecture, data/evaluation logic, numerical contract, or resource limits unless the task explicitly permits it. No external files or trained weights from research are accepted as the final submission. The submitted source is rerun from scratch.

Checkpoint serialization is limited to 1 GiB for the small model and 8 GiB for the pretrained model. The fixed four-hour research time is separate from final execution.

## Constraints

1× H100 80 GB; fixed deployed 125M-parameter GPT and tokenizer; 100M-token FineWeb pool; 1,024-token context; development training budget: agent chooses 2–15 min/run; two independent 30-minute training runs, including all training-only computation; ≤250M total training parameters; 64 GiB peak GPU memory.

The limits above apply to final execution. Training-data reuse is permitted. Held-out evaluation items may not be used for training or checkpoint selection. The runner manages temporary tensor weights needed for evaluation. Output tensors must be finite. Any memory/indexing error, incorrect result/gradient, or resource violation invalidates the submission.

Development runs may use 120–900 seconds, chosen by you. The submitted code must reproduce its results from the fixed initialization in two independent 1,800-second runs. Loading, compilation, data preparation, reference queries, and all training count against the run budget. The final completed checkpoint is measured on held-out data.

## Objective and validity

Your goal is to minimize final next-token validation loss, while satisfying every quality, correctness and resource constraint above. Both independent runs must succeed. A failure caused by submitted code invalidates the result. Reaching a prescribed training deadline is normal completion if a usable final completed checkpoint exists.

`configure(shapes)` receives names and integer shapes of the fixed model parameters. Return up to 64 auxiliary tensor declarations, each with `shape` (1–4 positive dimensions) and `std` (normal initialization standard deviation, 0–1). The runner counts the fixed model plus every auxiliary scalar before allocating auxiliaries; total training parameters must be at most 250M. `training_loss(forward,inputs,targets,auxiliary)` returns one finite scalar loss. The trusted `forward(ids)` callable returns `(logits,hidden)`, where hidden contains the final normalized model features; you may call it multiple times and build training-only components from the allocated auxiliary tensors. The runner owns model initialization, AdamW, backward, clipping and original-architecture export. Hooks use the stateless numerical surface in `/opt/runtime/training_objective.py` and `/opt/runtime/adapter.py`; module/optimizer access, mutation and custom export are unavailable. All added computation counts against the training deadline.

## Development

Run `python3 /opt/runtime/develop.py --seconds 120 --seed 101 --out /app/runs/experiment-01`. Choose a development training duration between 120 and 900 seconds. Use new output directories to retain experiment records. Development measurements use public inputs and may differ from held-out results.

The bounded hook interface accepts undecorated top-level functions only. Retained closures, lambdas, computed annotations and hidden callable state are unavailable. Only the runner may create trainable leaf tensors; `requires_grad` arguments and keyword unpacking are unavailable.
