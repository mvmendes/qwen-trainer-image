# qwen-trainer-image

Imagem Docker **genérica** para fine-tune **QLoRA/LoRA** de LLMs com [Unsloth](https://github.com/unslothai/unsloth).
Duas variantes disponíveis: uma para GPUs **legacy** (RTX 3090/4090/L4) e outra para **Blackwell** (RTX 5090).

## Variantes e Tags

| Tag | GPU | Torch | CUDA | Base |
|---|---|---|---|---|
| `:1.0` / `:torch2.6-unsloth` | RTX 3090, 4090, L4, T4 | 2.6 + cu124 | 12.4 | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| `:2.0-blackwell` | RTX 5090 | 2.8 + cu128 | 12.8 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |

### Versão Legacy (`:1.0`)

Stack fixada e validada (RTX 4090 / A5000 / L4):

| Componente | Versão |
|---|---|
| torch | 2.6.0 + cu124 |
| torchvision / torchaudio | 0.21.0 / 2.6.0 |
| xformers | 0.0.29.post3 |
| Unsloth | commit fixo (reprodutível) |
| transformers / trl / peft / bitsandbytes | puxados pelo Unsloth |
| torchao | removido (quebra com `torch.int1`) |

### Versão Blackwell (`:2.0-blackwell`)

| Componente | Versão |
|---|---|
| torch | 2.8.0 + cu128 |
| CUDA Toolkit | 12.8.1 |
| triton | 3.3.1+ |
| bitsandbytes | 0.45.3+ |
| Unsloth | commit fixo (reprodutível) |
| SSH RunPod | suporta `PUBLIC_KEY` (base moderna) |

Serve para qualquer modelo suportado pelo Unsloth: **Qwen, Llama, Mistral, Gemma, Phi, ...**

## Como a imagem é construída

Push na branch `main` (ou *Run workflow* manual) dispara o **GitHub Actions**, que builda
nos runners do GitHub e publica em `ghcr.io`. **Você não faz upload pesado.**

Imagens publicadas:
- `ghcr.io/<owner>/qwen-trainer-image:1.0`
- `ghcr.io/<owner>/qwen-trainer-image:torch2.6-unsloth`
- `ghcr.io/<owner>/qwen-trainer-image:2.0-blackwell`

> Após o primeiro build, deixe o pacote **público** em GitHub → Packages → Package settings,
> para o Runpod puxar sem credenciais. (Ou configure registry credentials no Runpod.)

## Como usar no Runpod

1. Crie o template ou Pod com **Container Image**:
   - RTX 3090/4090/L4: `ghcr.io/<owner>/qwen-trainer-image:1.0`
   - RTX 5090: `ghcr.io/<owner>/qwen-trainer-image:2.0-blackwell`
2. Monte um Network Volume em `/workspace`
3. Expanda **TCP Ports**: Add port → Label `SSH`, Port `22`
4. SSH, suba seu script de treino e dataset, rode em `tmux`

O ambiente já vem pronto — **sem setup de venv, sem whack-a-mole de versão**.

## Por que esta stack específica

- torch **2.6** (legacy): compatível com todas as GPUs Ada/Ampere
- torch **2.8 + cu128** (Blackwell): suporte a `sm_120` da RTX 5090
- **xformers** obrigatório: sem ele o SDPA materializa a atenção O(seq²) → OOM
- **torchao removido**: incompatível com a versão de torch via `torch.int1`
- Unsloth em **commit fixo**: evita quebras de API entre releases
