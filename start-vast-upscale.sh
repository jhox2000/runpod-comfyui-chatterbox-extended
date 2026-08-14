#!/bin/bash
# ============================================================================
# start-vast-upscale.sh — Template VAST.AI: POD DEDICADO DE UPSCALE
# ----------------------------------------------------------------------------
# ESCOPO: SOMENTE upscaling cinematografico. Este pod NAO tem geracao de
# imagem (Qwen), video (Wan I2V), personagens em lote nem TTS.
#
# Pipeline: Estagio 1 SeedVR2 7B fp16 (deblur/resolucao)
#           Estagio 2 SRPO Q8 GGUF *ou* Wan 2.2 T2V low-noise fp8 (textura)
#           Estagio 3 Grading (post-processing nodes, sem download de modelo)
#
# Config Vast.ai (mesmo padrao do start-vast.sh de producao):
#   - Base image: vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311
#   - GPU: RTX 5090 (Blackwell sm_120) | Container disk: 120 GB
#   - Porta 8188, "Jupyter direct HTTPS" DESMARCADO
#   - HF_TOKEN como variavel de ambiente (NUNCA hardcoded aqui)
# Tempo de boot estimado: ~10-20 min (~72GB de modelos + torch).
# ============================================================================
set -e
export DEBIAN_FRONTEND=noninteractive

# Ambiente Python da imagem da Vast: tudo roda dentro do venv /venv/main
[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate || true

# Log de boot: tudo que aparece aqui tambem vai para /workspace/logs
mkdir -p /workspace/logs
exec > >(tee -a /workspace/logs/boot-upscale.log) 2>&1
echo "[boot] ====== POD DE UPSCALE — $(date) ======"

# ----------------------------------------------------------------------------
# TESTE DE CONECTIVIDADE (roda ANTES de qualquer instalacao)
# Mesma regra do pipeline de producao: melhor falhar em 30 segundos do que
# no minuto 5 com host que nao alcanca GitHub/HuggingFace.
# ----------------------------------------------------------------------------
echo "[boot] Testando acesso ao GitHub e ao HuggingFace..."
if ! curl -sI --max-time 15 https://github.com >/dev/null; then
    echo "[boot] ERRO: este host NAO alcanca github.com."
    echo "[boot] >>> DESTRUA esta instancia e alugue outra em US/EU. <<<"
    exit 1
fi
if ! curl -sI --max-time 15 https://huggingface.co >/dev/null; then
    echo "[boot] ERRO: este host NAO alcanca huggingface.co."
    echo "[boot] >>> DESTRUA esta instancia e alugue outra em US/EU. <<<"
    exit 1
fi
echo "[boot] Conectividade OK."

# HuggingFace: desligar hf_transfer (deprecado) e usar XET (rapido e moderno)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_HIGH_PERFORMANCE=1

# ----------------------------------------------------------------------------
# (1/5) Dependencias de sistema + PyTorch cu128
# A imagem base da Vast NAO traz PyTorch: instalar a versao cu128 (Blackwell)
# ANTES do requirements do ComfyUI, para o pip nao escolher um build errado.
# ----------------------------------------------------------------------------
echo "[boot] (1/5) Dependencias de sistema + PyTorch cu128..."
apt-get update -qq
apt-get install -y -qq ffmpeg htop tmux build-essential lsof
pip install -q -U huggingface_hub hf_xet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ----------------------------------------------------------------------------
# (2/5) ComfyUI + custom nodes
# Clones CONDICIONAIS (o volume /workspace persiste), pip install SEMPRE
# (container efemero perde os pacotes pip no restart — licao do pipeline).
# ----------------------------------------------------------------------------
if [ ! -d /workspace/ComfyUI ]; then
    echo "[boot] (2/5) Clonando ComfyUI..."
    cd /workspace
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI
fi
CN=/workspace/ComfyUI/custom_nodes
cd "$CN"
[ -d ComfyUI-Manager ]                  || git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager
[ -d ComfyUI-SeedVR2_VideoUpscaler ]    || git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
[ -d ComfyUI_UltimateSDUpscale ]        || git clone --depth 1 --recursive https://github.com/ssitu/ComfyUI_UltimateSDUpscale
[ -d ComfyUI-GGUF ]                     || git clone --depth 1 https://github.com/city96/ComfyUI-GGUF
[ -d ComfyUI-post-processing-nodes ]    || git clone --depth 1 https://github.com/EllangoK/ComfyUI-post-processing-nodes

echo "[boot] (2/5) Instalando dependencias Python (SEMPRE, todo boot)..."
pip install -r /workspace/ComfyUI/requirements.txt
pip install -r "$CN/ComfyUI-Manager/requirements.txt" || true
pip install -r "$CN/ComfyUI-SeedVR2_VideoUpscaler/requirements.txt" || echo "[boot] AVISO: requirements do SeedVR2 falhou — verificar log"
pip install -r "$CN/ComfyUI-GGUF/requirements.txt" || echo "[boot] AVISO: requirements do GGUF falhou — verificar log"
pip uninstall -y xformers 2>/dev/null || true   # vem na imagem base; incompativel com Blackwell

# ----------------------------------------------------------------------------
# (3/5) Modelos (~72GB) via hf_hub_download com local_dir + mv.
# Os arquivos moram DIRETO no /workspace (volume persistente) — sem symlink
# para o cache HF, que fica no disco efemero e quebraria em todo restart.
# Cada download tem checagem PROPRIA por arquivo (licao da LoRA 8-steps:
# checar o arquivo especifico, nao o diretorio/modelo pai).
# ----------------------------------------------------------------------------
M=/workspace/ComfyUI/models
mkdir -p "$M"/{SEEDVR2,unet,diffusion_models,text_encoders,vae}

# --- Estagio 1: SeedVR2 7B fp16 + variante sharp (teste A/B do T1) + VAE ---
if [ ! -f "$M/SEEDVR2/seedvr2_ema_7b_fp16.safetensors" ]; then
    echo "[boot] (3/5) Baixando SeedVR2 7B fp16 (~16.5GB)..."
    python3 - << 'PYDL1'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='numz/SeedVR2_comfyUI',
    filename='seedvr2_ema_7b_fp16.safetensors', local_dir='/workspace/_dl_seedvr')
PYDL1
    mv /workspace/_dl_seedvr/seedvr2_ema_7b_fp16.safetensors "$M/SEEDVR2/"
fi
if [ ! -f "$M/SEEDVR2/seedvr2_ema_7b_sharp_fp16.safetensors" ]; then
    echo "[boot] (3/5) Baixando SeedVR2 7B SHARP fp16 (~16.5GB, teste A/B)..."
    python3 - << 'PYDL2'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='numz/SeedVR2_comfyUI',
    filename='seedvr2_ema_7b_sharp_fp16.safetensors', local_dir='/workspace/_dl_seedvr')
PYDL2
    mv /workspace/_dl_seedvr/seedvr2_ema_7b_sharp_fp16.safetensors "$M/SEEDVR2/"
fi
if [ ! -f "$M/SEEDVR2/ema_vae_fp16.safetensors" ]; then
    echo "[boot] (3/5) Baixando VAE do SeedVR2 (~0.5GB)..."
    python3 - << 'PYDL3'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='numz/SeedVR2_comfyUI',
    filename='ema_vae_fp16.safetensors', local_dir='/workspace/_dl_seedvr')
PYDL3
    mv /workspace/_dl_seedvr/ema_vae_fp16.safetensors "$M/SEEDVR2/"
fi
rm -rf /workspace/_dl_seedvr

# --- Estagio 2, candidato A: SRPO Q8_0 GGUF (quantizacao correta na origem) ---
# REGRA: NUNCA converter FP32->FP8 em runtime no ComfyUI (a Tencent documenta
# denoise incompleto). Q8_0 pre-quantizado e a via correta para 32GB.
# O nome exato do arquivo pode variar entre revisoes do repo, entao a busca e
# por padrao (*q8_0*.gguf), com fallback no repo befox/SRPO-GGUF. Se nada
# casar, o boot FALHA com a lista de arquivos no log (nada de falha muda).
if ! ls "$M"/unet/*[qQ]8_0*.gguf >/dev/null 2>&1; then
    echo "[boot] (3/5) Baixando SRPO Q8_0 GGUF (~13GB)..."
    python3 - << 'PYDL4'
import fnmatch, sys
from huggingface_hub import hf_hub_download, list_repo_files
repos = ['wikeeyang/SRPO-Refine-Quantized-v1.0', 'befox/SRPO-GGUF']
for repo in repos:
    try:
        files = list_repo_files(repo)
    except Exception as e:
        print(f'[boot] AVISO: nao consegui listar {repo}: {e}')
        continue
    matches = sorted([f for f in files if fnmatch.fnmatch(f.lower(), '*q8_0*.gguf')], key=len)
    if matches:
        print(f'[boot] SRPO Q8_0 encontrado em {repo}: {matches[0]}')
        hf_hub_download(repo_id=repo, filename=matches[0], local_dir='/workspace/_dl_srpo')
        sys.exit(0)
    print(f'[boot] Nenhum *q8_0*.gguf em {repo}. Arquivos: {files}')
sys.exit('[boot] ERRO: SRPO Q8_0 nao encontrado em nenhum repositorio.')
PYDL4
    find /workspace/_dl_srpo -name '*.gguf' -exec mv {} "$M/unet/" \;
    rm -rf /workspace/_dl_srpo
fi

# Encoders e VAE do FLUX (necessarios para o SRPO). T5 em fp8_scaled: quase
# sem perda e ~5GB a menos de VRAM que o fp16 — folga para tile de 1536.
if [ ! -f "$M/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors" ]; then
    echo "[boot] (3/5) Baixando T5-XXL fp8 + CLIP-L (~5.3GB)..."
    python3 - << 'PYDL5'
from huggingface_hub import hf_hub_download
for f in ['t5xxl_fp8_e4m3fn_scaled.safetensors', 'clip_l.safetensors']:
    hf_hub_download(repo_id='comfyanonymous/flux_text_encoders',
        filename=f, local_dir='/workspace/_dl_flux_te')
PYDL5
    mv /workspace/_dl_flux_te/*.safetensors "$M/text_encoders/"
    rm -rf /workspace/_dl_flux_te
fi
# VAE do FLUX (ae.safetensors). Fonte: repo repackaged da Comfy-Org que
# distribui o mesmo ae.safetensors sem gate de licenca. Se este caminho
# mudar, alternativa: black-forest-labs/FLUX.1-dev (gated, exige aceitar a
# licenca na conta do HF_TOKEN).
if [ ! -f "$M/vae/flux_ae.safetensors" ]; then
    echo "[boot] (3/5) Baixando VAE do FLUX (~0.3GB)..."
    python3 - << 'PYDL6'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Comfy-Org/Lumina_Image_2.0_Repackaged',
    filename='split_files/vae/ae.safetensors', local_dir='/workspace/_dl_flux_vae')
PYDL6
    mv /workspace/_dl_flux_vae/split_files/vae/ae.safetensors "$M/vae/flux_ae.safetensors"
    rm -rf /workspace/_dl_flux_vae
fi

# --- Estagio 2, candidato B: Wan 2.2 T2V LOW-NOISE fp8 (refinador de textura) ---
# ATENCAO: e o T2V low-noise — arquivo DIFERENTE do I2V do pod de producao.
# O high-noise NAO e baixado: refino de imagem usa somente o expert que
# poli textura/iluminacao. Mesmo repo Comfy-Org ja validado no pipeline.
if [ ! -f "$M/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" ]; then
    echo "[boot] (3/5) Baixando Wan 2.2 T2V low-noise fp8 + encoder + VAE (~20GB)..."
    python3 - << 'PYDL7'
from huggingface_hub import hf_hub_download
REPO = 'Comfy-Org/Wan_2.2_ComfyUI_Repackaged'
for f in [
    'split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors',
    'split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors',
    'split_files/vae/wan_2.1_vae.safetensors',
]:
    print('  baixando:', f.split('/')[-1])
    hf_hub_download(repo_id=REPO, filename=f, local_dir='/workspace/_dl_wan')
PYDL7
    mv /workspace/_dl_wan/split_files/diffusion_models/*.safetensors "$M/diffusion_models/"
    mv /workspace/_dl_wan/split_files/text_encoders/*.safetensors    "$M/text_encoders/"
    mv /workspace/_dl_wan/split_files/vae/*.safetensors              "$M/vae/"
    rm -rf /workspace/_dl_wan
fi

# ----------------------------------------------------------------------------
# (4/5) Verificacao de sanidade — tamanhos minimos.
# Download truncado (rede de host ruim) e pego AQUI, nao no meio do teste.
# ----------------------------------------------------------------------------
echo "[boot] (4/5) Verificacao de sanidade dos modelos..."
check_size() {
    local f="$1"; local min_gb="$2"
    if [ ! -f "$f" ]; then echo "[boot] ERRO: $f nao existe."; exit 1; fi
    local size_gb=$(du -BG "$f" | cut -f1 | tr -d 'G')
    if [ "$size_gb" -lt "$min_gb" ]; then
        echo "[boot] ERRO: $f tem ${size_gb}GB, esperado >= ${min_gb}GB (download truncado?)"; exit 1
    fi
    echo "[boot]   [ok] $(basename "$f") — ${size_gb}GB"
}
check_size "$M/SEEDVR2/seedvr2_ema_7b_fp16.safetensors" 15
check_size "$M/SEEDVR2/seedvr2_ema_7b_sharp_fp16.safetensors" 15
check_size "$M/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" 12
SRPO_FILE=$(ls "$M"/unet/*[qQ]8_0*.gguf 2>/dev/null | head -n1)
if [ -z "$SRPO_FILE" ]; then echo "[boot] ERRO: SRPO Q8_0 ausente em models/unet."; exit 1; fi
check_size "$SRPO_FILE" 10

# Pastas de trabalho do lote de upscale
mkdir -p /workspace/ComfyUI/input/upscale_in /workspace/ComfyUI/output/upscale_out

# ----------------------------------------------------------------------------
# (5/5) Subir o ComfyUI na porta 8188
# ----------------------------------------------------------------------------
echo "[boot] (5/5) Boot concluido. Subindo ComfyUI :8188"
cd /workspace/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188
