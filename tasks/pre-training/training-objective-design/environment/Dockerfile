FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false DO_NOT_TRACK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git procps util-linux ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 safetensors==0.6.2 transformers==4.56.2 datasets==4.1.1 huggingface-hub==0.35.3
RUN useradd -m -u 1000 agent && useradd -m -u 1001 runner && useradd -m -u 1002 runner2
COPY --from=node:22.20.0-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22.20.0-bookworm-slim /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && npm install --prefix /opt/rsi-codex --no-audit --no-fund @openai/codex@0.159.2
COPY runtime/prepare_assets.py runtime/assets.lock.json /opt/runtime/
ARG ASSET_FAMILY=text
RUN python /opt/runtime/prepare_assets.py --family text


COPY runtime/ /opt/runtime/
RUN rm -f /opt/runtime/prepare_assets.py /opt/runtime/prepare_research_assets.py
COPY app/ /app/
RUN ln -sf /opt/conda/bin/python /usr/local/bin/python3 && install -m 755 /opt/runtime/codex_guard.py /usr/local/bin/codex && codex --version && rm /opt/runtime/export_gzip.py /opt/runtime/codex_guard.py
RUN chmod -R a-w,a+rX /opt/runtime /opt/assets/public /app && chmod -R go-rwx /opt/assets/private && chown -R agent:agent /app/submission && chmod -R u+rwX /app/submission && install -d -o agent -g agent /app/runs
ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/opt/runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
