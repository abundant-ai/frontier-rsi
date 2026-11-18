FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false DO_NOT_TRACK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git procps util-linux ca-certificates curl gzip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 safetensors==0.6.2
RUN useradd -m -u 1001 runner
COPY . /tests/
RUN ln -sf /opt/conda/bin/python /usr/local/bin/python3 && chmod -R go-rwx /tests && mkdir -p /app/submission /app/runs
ENV HOME=/root HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/tests
WORKDIR /app
