FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends git procps ca-certificates python3 curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 safetensors==0.6.2 transformers==4.56.2 huggingface-hub==0.35.3
COPY --from=node:22.20.0-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22.20.0-bookworm-slim /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && npm install --prefix /opt/rsi-codex --no-audit --no-fund @openai/codex@0.159.2 && ln -s /opt/rsi-codex/node_modules/.bin/codex /usr/local/bin/codex
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', local_dir='/opt/assets/model', allow_patterns=['*.json','*.safetensors','*.txt'])"
RUN useradd -m -u 1000 agent
COPY runtime/ /opt/runtime/
COPY app/ /app/
RUN chmod -R a-w,a+rX /opt/runtime /opt/assets /app && chown -R agent:agent /app/submission && chmod -R u+rwX /app/submission && install -d -o agent -g agent /app/runs
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/opt/runtime
WORKDIR /app
