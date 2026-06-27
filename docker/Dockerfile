# Generic LLM QLoRA/LoRA training image
# Stack: Unsloth + torch 2.6 (cu124) + xformers, sobre Python 3.11 / CUDA 12.4.
# Serve para fine-tune de qualquer modelo suportado pelo Unsloth
# (Qwen, Llama, Mistral, Gemma, Phi, ...). Validado em RTX 4090 / A5000 / L4.
#
# Build/push: via GitHub Actions (.github/workflows/build-image.yml) -> ghcr.io
# Uso no Runpod: imagem do pod = ghcr.io/<owner>/qwen-trainer-image:latest
#
# Base runpod/pytorch garante Python 3.11 + CUDA 12.4 devel + SSH do Runpod.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Commit fixo do Unsloth para builds reprodutiveis
ARG UNSLOTH_COMMIT=1fcd69e662f233896e075fc03472f45ac60e8819

RUN pip install -U pip

# 1) Unsloth (puxa transformers/trl/peft/bnb/datasets/unsloth_zoo)
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@${UNSLOTH_COMMIT}"

# 2) Fixa a stack torch 2.6 + cu124 (sobrepoe o que o unsloth puxou)
RUN pip install -U torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3) xformers (atencao memory-efficient) compativel com torch 2.6
RUN pip install --no-deps xformers==0.0.29.post3

# 4) torchao quebra (torch.int1) e nao e usado -> remover
RUN pip uninstall -y torchao || true

# 5) Sanidade em build-time (sem importar unsloth: ele exige GPU, ausente no runner).
#    O import do unsloth e validado em runtime, no pod com GPU.
RUN python -c "import torch, transformers, trl, peft, bitsandbytes, xformers; \
print('OK torch', torch.__version__, '| transformers', transformers.__version__, \
'| trl', trl.__version__, '| xformers', xformers.__version__)"

WORKDIR /workspace
