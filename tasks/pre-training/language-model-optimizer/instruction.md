# Improve the supplied implementation

Minimize final next-token validation loss, measured in nats per token, by improving the supplied parameter-update implementation for the fixed language model.

Final evaluation trains from scratch under two independent seeds and measures held-out next-token loss. Minimize the arithmetic mean of the two losses.

## Environment

You have 1 H100 GPU(s), 16 CPU cores, 64 GiB host RAM, and four hours for all research activity. Internet access and external model APIs are unavailable. Dependencies and permitted data/model assets are installed under `/opt/assets/public`. `/opt/runtime` is read-only. Development feedback uses separate development data; held-out inputs are unavailable during development.

## Deliverable and editing boundary

Edit only `/app/submission/candidate.py`. The submitted file must be no larger than 256 KiB. Keep these public entrypoints and their signatures:
- `optimizer`

The starter source and fixed runner define their argument/return contracts. You may add helper functions or classes inside that file. Do not change the fixed model architecture, data/evaluation logic, numerical contract, or resource limits unless the task explicitly permits it. No external files or trained weights from research are accepted as the final submission. The submitted source is rerun from scratch.

Checkpoint serialization is limited to 1 GiB for the small model and 8 GiB for the pretrained model. The four-hour research limit is separate from the final execution budget.

## Constraints

1× H100 80 GB; fixed 125M-parameter GPT, 1,024-token context, 65,536 tokens/update, BF16 model computation; development training budget: agent chooses 2–15 min/run; two independent 30-minute training runs; 64 GiB peak GPU memory; fixed initialization seeds and 100M-token FineWeb pool.

The submitted code must stay within the limits above. Training-data reuse is permitted. Final-evaluation items may not be used for training or checkpoint selection. The runner manages temporary tensor weights needed for evaluation. Output tensors must be finite. Any memory/indexing error, incorrect result/gradient, or resource violation invalidates the submission.

Development runs may use 120–900 seconds, chosen by you. The submitted code must train from the supplied initialization for 1,800 seconds per run. Loading, compilation, data preparation, reference queries, and training all count toward that limit. The final completed checkpoint must be usable for held-out loss evaluation.

## Objective and validity

Your goal is to minimize final validation loss, while satisfying every quality, correctness and resource constraint above. The submission must complete successfully for both independent runs. A failure caused by submitted code invalidates the result. Each run must finish and export a usable final checkpoint within its 1,800-second budget.

## Development

Run `python /opt/runtime/develop.py --seconds 120 --seed 101 --out /app/runs/experiment-01`. For model training, choose a duration between 120 and 900 seconds. Use new output directories to retain experiment records. Development measurements use public inputs and do not establish held-out loss.
