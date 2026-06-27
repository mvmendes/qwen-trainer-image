# qwen-trainer-image

Imagem Docker **genérica** para fine-tune **QLoRA/LoRA** de LLMs com [Unsloth](https://github.com/unslothai/unsloth).

Stack fixada e validada (RTX 4090 / A5000 / L4):

| Componente | Versão |
|---|---|
| Base | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| torch | 2.6.0 + cu124 |
| torchvision / torchaudio | 0.21.0 / 2.6.0 |
| xformers | 0.0.29.post3 (atenção memory-efficient) |
| Unsloth | commit fixo (reprodutível) |
| transformers / trl / peft / bitsandbytes | versões puxadas pelo Unsloth |
| torchao | **removido** (quebra com `torch.int1`; não usado) |

Serve para qualquer modelo suportado pelo Unsloth: **Qwen, Llama, Mistral, Gemma, Phi, ...**

## Como a imagem é construída

Push na branch `main` (ou *Run workflow* manual) dispara o **GitHub Actions**, que builda
nos runners do GitHub e publica em `ghcr.io`. **Você não faz upload pesado.**

Imagens publicadas:
- `ghcr.io/<owner>/qwen-trainer-image:latest`
- `ghcr.io/<owner>/qwen-trainer-image:torch2.6-unsloth`

> Após o primeiro build, deixe o pacote **público** em GitHub → Packages → Package settings,
> para o Runpod puxar sem credenciais. (Ou configure registry credentials no Runpod.)

## Como usar no Runpod

1. Crie o pod com **Container Image** = `ghcr.io/<owner>/qwen-trainer-image:latest`
2. Monte um Network Volume em `/workspace`
3. SSH, suba seu script de treino e dataset, rode em `tmux`

O ambiente já vem pronto — **sem setup de venv, sem whack-a-mole de versão**.

## Por que esta stack específica

Cada versão foi fixada após depurar conflitos reais:
- torch **2.6** (não 2.4): `nn.Module.set_submodule`, `torch._inductor.config`, `torch.int1`
- **xformers** obrigatório: sem ele o SDPA materializa a atenção O(seq²) → OOM
- **torchao removido**: incompatível com a versão de torch via `torch.int1`
- Unsloth em **commit fixo**: evita quebras de API entre releases
