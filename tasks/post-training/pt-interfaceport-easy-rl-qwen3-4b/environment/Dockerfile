# Task image: the model to post-train (pinned revision, read-only, its chat template pinned by pin_base.py), vLLM, the
# training libraries and the posttrain-kit's kernels. Network is available while the image is built, not at run time.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/opt/hf_cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 PATH="/root/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev python3-pip python3-venv \
      build-essential curl ca-certificates procps git tzdata \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/local/bin/python
RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
RUN /root/.local/bin/uv pip install --system --break-system-packages --no-cache "vllm==0.29.0" "transformers==5.19.0" \
      "trl==1.15.0" "peft==0.21.2" "accelerate==1.15.0" "datasets==5.1.0" "bitsandbytes==0.50.2" \
      "huggingface_hub[hf_transfer]" "tzdata" \
    && python3 -c "import vllm, torch, transformers, trl, peft; print(vllm.__version__, torch.__version__, transformers.__version__, trl.__version__, peft.__version__)"
# posttrain-kit speed-ups: Liger Kernel and the FlashAttention 3 / 2 kernels, fetched now (no network at run time)
# into /opt/kit-kernels for this torch; the kit's accel.py and transformers' kernels loader read them there
RUN /root/.local/bin/uv pip install --system --break-system-packages --no-cache "liger-kernel==0.8.5" "kernels==0.17.3" \
    && HF_HUB_OFFLINE=0 KERNELS_CACHE=/tmp/kernels-cache python3 -c "import pathlib, shutil, kernels.hf_hub as H, kernels.install as I; from huggingface_hub import HfApi; H._get_hf_api = I._get_hf_api = lambda user_agent=None: HfApi(user_agent='posttrain-kit'); [shutil.copytree(p, '/opt/kit-kernels/' + r.split('/')[1] + '/build/' + p.name, dirs_exist_ok=True) for r, v in (('kernels-community/vllm-flash-attn3', 1), ('kernels-community/flash-attn2', 3)) for p in [pathlib.Path(I.install_kernel(r, version=v, trust_remote_code=[r]))]]" \
    && rm -rf /tmp/kernels-cache && chmod -R a+rX /opt/kit-kernels \
    && python3 -c "import glob, torch, liger_kernel.transformers as L, kernels, transformers; from transformers.utils.import_utils import is_kernels_available; assert is_kernels_available(), 'transformers does not accept this kernels version'; assert callable(L.apply_liger_kernel_to_qwen3); b = sorted(glob.glob('/opt/kit-kernels/*/build/*/metadata.json')); assert len(b) == 2, b; print('liger-kernel ok, kernels', kernels.__version__, 'torch', torch.__version__, *b)"
ENV KIT_KERNELS=/opt/kit-kernels LOCAL_KERNELS=kernels-community/vllm-flash-attn3=/opt/kit-kernels/vllm-flash-attn3:kernels-community/flash-attn2=/opt/kit-kernels/flash-attn2
# The model, at a pinned revision: weights, config, tokenizer and chat template; then its template is pinned and the
# directory made root-owned and read-only.
COPY build/pin_base.py /opt/pin_base.py
RUN HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "from huggingface_hub import snapshot_download as s; \
s('Qwen/Qwen3-4B-Instruct-2507', revision='cdbee75f17c01a7cc42f958dc650907174af0554', local_dir='/opt/base', allow_patterns=['*.safetensors', '*.json', '*.jinja', '*.txt'])" \
    && python3 /opt/pin_base.py /opt/base 2024 7 26 \
    && rm -rf /opt/base/.cache /opt/hf_cache/hub /root/.cache/huggingface && chmod -R a-w,a+rX /opt/base
RUN (userdel -r ubuntu 2>/dev/null || true) && useradd -m -u 1000 agent && id nobody
COPY app/ /app/
RUN mkdir -p /app/rl_pool && chmod +x /app/timer.sh && chown -R agent:agent /app && chown -R root:root /app/data && chmod -R a-w,a+rX /app/data
# No file in the image is writable by other users, so the agent cannot change a file it does not own,
# and no program runs with its owner's rights (setuid/setgid: su, mount, passwd...), so the agent cannot become root
RUN find / -xdev -type f -perm -0002 -exec chmod o-w {} + 2>/dev/null; \
    find / -xdev -type f -perm /6000 -exec chmod ug-s {} + 2>/dev/null; true
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
WORKDIR /app
