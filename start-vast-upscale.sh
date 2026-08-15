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
# (4.5/5) Workflow T3 pre-instalado + script de lote.
# Gravados em TODO boot (idempotente, sobrescreve com a versao validada):
#   - T3 na barra lateral do ComfyUI (SeedVR2 3072 -> SRPO 0.28, tile_debug
#     como string "false" — correcao validada em 14/08/2026)
#   - /workspace/lote_upscale.py: processa todas as imagens de
#     input/upscale_in pelo pipeline completo, com retomada automatica.
# ----------------------------------------------------------------------------
echo "[boot] (4.5/5) Instalando workflow T3 e script de lote..."
mkdir -p /workspace/ComfyUI/user/default/workflows
cat > "/workspace/ComfyUI/user/default/workflows/T3 - Pipeline SeedVR2 + SRPO.json" << 'T3JSONEOF'
{"id": "T3 - Pipeline SeedVR2 + SRPO", "revision": 0, "last_node_id": 18, "last_link_id": 16, "nodes": [{"id": 1, "type": "SeedVR2LoadDiTModel", "pos": [40, 40], "size": [340, 200], "flags": {}, "order": 1, "mode": 0, "inputs": [{"name": "torch_compile_args", "type": "TORCH_COMPILE_ARGS", "link": null}], "outputs": [{"name": "SEEDVR2_DIT", "type": "SEEDVR2_DIT", "links": [2], "slot_index": 0}], "properties": {"Node name for S&R": "SeedVR2LoadDiTModel"}, "widgets_values": ["seedvr2_ema_7b_fp16.safetensors", "cuda:0", 36, false, "cpu", false, "sdpa"], "title": "E1 - DiT fp16"}, {"id": 2, "type": "SeedVR2LoadVAEModel", "pos": [440, 40], "size": [340, 200], "flags": {}, "order": 2, "mode": 0, "inputs": [{"name": "torch_compile_args", "type": "TORCH_COMPILE_ARGS", "link": null}], "outputs": [{"name": "SEEDVR2_VAE", "type": "SEEDVR2_VAE", "links": [3], "slot_index": 0}], "properties": {"Node name for S&R": "SeedVR2LoadVAEModel"}, "widgets_values": ["ema_vae_fp16.safetensors", "cuda:0", true, 1024, 128, true, 1024, 128, "false", "cpu", false], "title": "E1 - VAE"}, {"id": 3, "type": "LoadImage", "pos": [840, 40], "size": [340, 200], "flags": {}, "order": 3, "mode": 0, "inputs": [], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1], "slot_index": 0}, {"name": "MASK", "type": "MASK", "links": [], "slot_index": 1}], "properties": {"Node name for S&R": "LoadImage"}, "widgets_values": ["example.png", "image"], "title": "ENTRADA - imagem crua do GPT"}, {"id": 4, "type": "SeedVR2VideoUpscaler", "pos": [1240, 40], "size": [340, 200], "flags": {}, "order": 4, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 1}, {"name": "dit", "type": "SEEDVR2_DIT", "link": 2}, {"name": "vae", "type": "SEEDVR2_VAE", "link": 3}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [4, 5], "slot_index": 0}], "properties": {"Node name for S&R": "SeedVR2VideoUpscaler"}, "widgets_values": [42, "fixed", 3072, 3072, 1, false, "lab", 0, 0, 0.0, 0.0, "cpu", false], "title": "E1 - Upscale 3072"}, {"id": 5, "type": "SaveImage", "pos": [1640, 40], "size": [340, 200], "flags": {}, "order": 5, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 4}], "outputs": [], "properties": {"Node name for S&R": "SaveImage"}, "widgets_values": ["T3_etapa1_seedvr2"], "title": "Salvar intermediaria (E1)"}, {"id": 6, "type": "UnetLoaderGGUF", "pos": [40, 340], "size": [340, 200], "flags": {}, "order": 6, "mode": 0, "inputs": [], "outputs": [{"name": "MODEL", "type": "MODEL", "links": [10], "slot_index": 0}], "properties": {"Node name for S&R": "UnetLoaderGGUF"}, "widgets_values": ["Flux1-Dev-SRPO-v1-Q8_0.gguf"], "title": "E2 - SRPO Q8"}, {"id": 7, "type": "DualCLIPLoader", "pos": [440, 340], "size": [340, 200], "flags": {}, "order": 7, "mode": 0, "inputs": [], "outputs": [{"name": "CLIP", "type": "CLIP", "links": [7], "slot_index": 0}], "properties": {"Node name for S&R": "DualCLIPLoader"}, "widgets_values": ["t5xxl_fp8_e4m3fn_scaled.safetensors", "clip_l.safetensors", "flux"]}, {"id": 8, "type": "VAELoader", "pos": [840, 340], "size": [340, 200], "flags": {}, "order": 8, "mode": 0, "inputs": [], "outputs": [{"name": "VAE", "type": "VAE", "links": [6, 15], "slot_index": 0}], "properties": {"Node name for S&R": "VAELoader"}, "widgets_values": ["flux_ae.safetensors"]}, {"id": 9, "type": "CLIPTextEncode", "pos": [1240, 340], "size": [340, 200], "flags": {}, "order": 9, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 7}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [8, 9], "slot_index": 0}], "properties": {"Node name for S&R": "CLIPTextEncode"}, "widgets_values": ["RAW photograph, film still captured on a cinema camera, natural skin texture with visible pores, individual hair strands, realistic fabric weave, natural lighting, photorealistic fine detail"], "title": "Prompt (captura)"}, {"id": 10, "type": "FluxGuidance", "pos": [1640, 340], "size": [340, 200], "flags": {}, "order": 10, "mode": 0, "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": 8}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [11], "slot_index": 0}], "properties": {"Node name for S&R": "FluxGuidance"}, "widgets_values": [3.5]}, {"id": 11, "type": "ConditioningZeroOut", "pos": [40, 640], "size": [340, 200], "flags": {}, "order": 11, "mode": 0, "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": 9}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [12], "slot_index": 0}], "properties": {"Node name for S&R": "ConditioningZeroOut"}, "widgets_values": []}, {"id": 12, "type": "VAEEncode", "pos": [440, 640], "size": [340, 200], "flags": {}, "order": 12, "mode": 0, "inputs": [{"name": "pixels", "type": "IMAGE", "link": 5}, {"name": "vae", "type": "VAE", "link": 6}], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [13], "slot_index": 0}], "properties": {"Node name for S&R": "VAEEncode"}, "widgets_values": []}, {"id": 13, "type": "KSampler", "pos": [840, 640], "size": [340, 200], "flags": {}, "order": 13, "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 10}, {"name": "positive", "type": "CONDITIONING", "link": 11}, {"name": "negative", "type": "CONDITIONING", "link": 12}, {"name": "latent_image", "type": "LATENT", "link": 13}], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [14], "slot_index": 0}], "properties": {"Node name for S&R": "KSampler"}, "widgets_values": [42, "fixed", 20, 1.0, "euler", "simple", 0.28], "title": "E2 - denoise 0.28 NAO MEXER"}, {"id": 14, "type": "VAEDecode", "pos": [1240, 640], "size": [340, 200], "flags": {}, "order": 14, "mode": 0, "inputs": [{"name": "samples", "type": "LATENT", "link": 14}, {"name": "vae", "type": "VAE", "link": 15}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16], "slot_index": 0}], "properties": {"Node name for S&R": "VAEDecode"}, "widgets_values": []}, {"id": 15, "type": "SaveImage", "pos": [1640, 640], "size": [340, 200], "flags": {}, "order": 15, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 16}], "outputs": [], "properties": {"Node name for S&R": "SaveImage"}, "widgets_values": ["T3_final"], "title": "SAIDA FINAL"}, {"id": 16, "type": "Note", "pos": [40, -560], "size": [440, 480], "flags": {}, "order": 16, "mode": 0, "inputs": [], "outputs": [], "title": "LEIA-ME: 1 imagem", "properties": {}, "widgets_values": ["═══ COMO USAR — 1 IMAGEM (por aqui, na interface) ═══\n\n1. No no \"ENTRADA - imagem crua do GPT\":\n   clique em \"escolher arquivo para enviar\" e suba a imagem\n2. Clique em EXECUTAR\n3. AGUARDE: fica parado em 0% por 3-6 min na primeira\n   execucao (carregando modelos). NAO e travamento.\n   Total: ~5-8 min por imagem.\n4. Resultados na pasta output:\n   - T3_etapa1_seedvr2_*.png  (intermediaria, conferencia)\n   - T3_final_*.png           (a imagem pronta)\n5. Confira a final em 100% de ZOOM no PC.\n\nNAO MEXER: denoise 0.28 / resolution 3072 / seed 42\n(parametros travados da calibragem de 14/08/2026)"], "color": "#432", "bgcolor": "#653"}, {"id": 17, "type": "Note", "pos": [520, -560], "size": [440, 480], "flags": {}, "order": 17, "mode": 0, "inputs": [], "outputs": [], "title": "LEIA-ME: LOTE", "properties": {}, "widgets_values": ["═══ COMO USAR — LOTE (varias imagens de uma vez) ═══\n\n1. Abra o JupyterLab da instancia (botao Open na Vast)\n2. No painel de arquivos, navegue ate:\n   ComfyUI > input > upscale_in\n3. ARRASTE todas as imagens cruas para dentro da pasta\n4. Abra um Terminal (File > New > Terminal) e rode:\n\n   python3 /workspace/lote_upscale.py\n\n5. Acompanhe o progresso: [1/20] nome.png ... ok em 95s\n6. Resultados em: ComfyUI > output > upscale_out\n   - nome_E1_*.png     = so SeedVR2 (conferencia)\n   - nome_final_*.png  = pronta\n7. Baixe, confira em 100% de zoom, descarte alucinadas.\n\nDICAS:\n- Interrompeu? Rode o comando de novo: ele PULA as\n  ja prontas e continua de onde parou.\n- Imagem com erro e pulada; o lote nao para.\n- A primeira imagem e a mais lenta (carga dos modelos)."], "color": "#432", "bgcolor": "#653"}, {"id": 18, "type": "Note", "pos": [1000, -560], "size": [440, 480], "flags": {}, "order": 18, "mode": 0, "inputs": [], "outputs": [], "title": "LEIA-ME: problemas", "properties": {}, "widgets_values": ["═══ SE DER PROBLEMA ═══\n\nVer o que esta acontecendo (terminal do Jupyter):\n   tail -f /workspace/logs/boot-upscale.log\n\nGPU esta trabalhando?\n   nvidia-smi\n   (memoria subindo ou GPU-Util alto = trabalhando, espere)\n\nParado em 0%? Ate ~6 min e NORMAL (carga silenciosa\ndos modelos). So suspeite depois de 10 min sem nada\nnovo no log.\n\nGuia completo: /workspace/GUIA_UPSCALE.md"], "color": "#432", "bgcolor": "#653"}], "links": [[1, 3, 0, 4, 0, "IMAGE"], [2, 1, 0, 4, 1, "SEEDVR2_DIT"], [3, 2, 0, 4, 2, "SEEDVR2_VAE"], [4, 4, 0, 5, 0, "IMAGE"], [5, 4, 0, 12, 0, "IMAGE"], [6, 8, 0, 12, 1, "VAE"], [7, 7, 0, 9, 0, "CLIP"], [8, 9, 0, 10, 0, "CONDITIONING"], [9, 9, 0, 11, 0, "CONDITIONING"], [10, 6, 0, 13, 0, "MODEL"], [11, 10, 0, 13, 1, "CONDITIONING"], [12, 11, 0, 13, 2, "CONDITIONING"], [13, 12, 0, 13, 3, "LATENT"], [14, 13, 0, 14, 0, "LATENT"], [15, 8, 0, 14, 1, "VAE"], [16, 14, 0, 15, 0, "IMAGE"]], "groups": [], "config": {}, "extra": {}, "version": 0.4}
T3JSONEOF

cat > /workspace/lote_upscale.py << 'LOTEPYEOF'
#!/usr/bin/env python3
# ============================================================================
# lote_upscale.py — Processa TODAS as imagens de upscale_in pelo pipeline
# completo (SeedVR2 3072 -> SRPO denoise 0.28) via API do ComfyUI.
#
# USO:
#   1. Suba as imagens cruas em:  /workspace/ComfyUI/input/upscale_in/
#   2. Rode:                      python3 /workspace/lote_upscale.py
#   3. Resultados em:             /workspace/ComfyUI/output/upscale_out/
#      - <nome>_E1_*.png    = intermediaria (so SeedVR2, para conferencia)
#      - <nome>_final_*.png = final (SeedVR2 + SRPO)
#
# Comportamento:
#   - Sequencial, uma imagem por vez (modelos ficam em cache entre imagens;
#     a primeira e lenta, as seguintes sao bem mais rapidas).
#   - Idempotente: se ja existe *_final_* para uma imagem, ela e PULADA.
#     Interrompeu no meio? Rode de novo e ele continua de onde parou.
#   - Parametros travados da calibragem: 3072 / batch 1 / denoise 0.28 /
#     guidance 3.5 / seed 42. Mudanca de parametro = mudanca aqui, de caso
#     pensado, nunca no improviso.
# ============================================================================
import json
import os
import sys
import time
import urllib.request
import urllib.error

API = "http://127.0.0.1:8188"
IN_DIR = "/workspace/ComfyUI/input/upscale_in"
OUT_DIR = "/workspace/ComfyUI/output/upscale_out"
EXTS = (".png", ".jpg", ".jpeg", ".webp")

PROMPT_TXT = ("RAW photograph, film still captured on a cinema camera, "
              "natural skin texture with visible pores, individual hair strands, "
              "realistic fabric weave, natural lighting, photorealistic fine detail")


def grafo(nome_arquivo, stem):
    """Grafo do T3 em formato de API. nome_arquivo relativo a pasta input."""
    return {
        # ---------- Estagio 1: SeedVR2 ----------
        "1": {"class_type": "SeedVR2LoadDiTModel", "inputs": {
            "model": "seedvr2_ema_7b_fp16.safetensors",
            "device": "cuda:0",
            "blocks_to_swap": 36,
            "swap_io_components": False,
            "offload_device": "cpu",
            "cache_model": False,
            "attention_mode": "sdpa"}},
        "2": {"class_type": "SeedVR2LoadVAEModel", "inputs": {
            "model": "ema_vae_fp16.safetensors",
            "device": "cuda:0",
            "encode_tiled": True,
            "encode_tile_size": 1024,
            "encode_tile_overlap": 128,
            "decode_tiled": True,
            "decode_tile_size": 1024,
            "decode_tile_overlap": 128,
            "tile_debug": "false",
            "offload_device": "cpu",
            "cache_model": False}},
        "3": {"class_type": "LoadImage", "inputs": {
            "image": nome_arquivo, "upload": "image"}},
        "4": {"class_type": "SeedVR2VideoUpscaler", "inputs": {
            "image": ["3", 0], "dit": ["1", 0], "vae": ["2", 0],
            "seed": 42,
            "resolution": 3072,
            "max_resolution": 3072,
            "batch_size": 1,
            "uniform_batch_size": False,
            "color_correction": "lab",
            "temporal_overlap": 0,
            "prepend_frames": 0,
            "input_noise_scale": 0.0,
            "latent_noise_scale": 0.0,
            "offload_device": "cpu",
            "enable_debug": False}},
        "5": {"class_type": "SaveImage", "inputs": {
            "images": ["4", 0],
            "filename_prefix": f"upscale_out/{stem}_E1"}},
        # ---------- Estagio 2: SRPO Q8 ----------
        "6": {"class_type": "UnetLoaderGGUF", "inputs": {
            "unet_name": "Flux1-Dev-SRPO-v1-Q8_0.gguf"}},
        "7": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "t5xxl_fp8_e4m3fn_scaled.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux"}},
        "8": {"class_type": "VAELoader", "inputs": {
            "vae_name": "flux_ae.safetensors"}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["7", 0], "text": PROMPT_TXT}},
        "10": {"class_type": "FluxGuidance", "inputs": {
            "conditioning": ["9", 0], "guidance": 3.5}},
        "11": {"class_type": "ConditioningZeroOut", "inputs": {
            "conditioning": ["9", 0]}},
        "12": {"class_type": "VAEEncode", "inputs": {
            "pixels": ["4", 0], "vae": ["8", 0]}},
        "13": {"class_type": "KSampler", "inputs": {
            "model": ["6", 0], "positive": ["10", 0], "negative": ["11", 0],
            "latent_image": ["12", 0],
            "seed": 42, "steps": 20, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "denoise": 0.28}},
        "14": {"class_type": "VAEDecode", "inputs": {
            "samples": ["13", 0], "vae": ["8", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {
            "images": ["14", 0],
            "filename_prefix": f"upscale_out/{stem}_final"}},
    }


def api(caminho, dados=None, timeout=30):
    req = urllib.request.Request(API + caminho)
    if dados is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(dados).encode()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    # ComfyUI esta de pe?
    try:
        api("/system_stats")
    except Exception:
        sys.exit("ERRO: ComfyUI nao respondeu em 127.0.0.1:8188. O pod terminou de subir?")

    os.makedirs(IN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    imagens = sorted(f for f in os.listdir(IN_DIR)
                     if f.lower().endswith(EXTS) and os.path.isfile(os.path.join(IN_DIR, f)))
    if not imagens:
        sys.exit(f"Nenhuma imagem em {IN_DIR}. Suba os arquivos la e rode de novo.")

    ja_prontas = os.listdir(OUT_DIR) if os.path.isdir(OUT_DIR) else []
    fila = []
    for img in imagens:
        stem = os.path.splitext(img)[0]
        if any(x.startswith(stem + "_final") for x in ja_prontas):
            print(f"[pula] {img} — final ja existe")
            continue
        fila.append(img)

    total = len(fila)
    print(f"=== LOTE: {total} imagem(ns) a processar, {len(imagens)-total} ja pronta(s) ===")

    for i, img in enumerate(fila, 1):
        stem = os.path.splitext(img)[0]
        print(f"\n[{i}/{total}] {img} ...", flush=True)
        t0 = time.time()
        try:
            r = api("/prompt", {"prompt": grafo(f"upscale_in/{img}", stem)})
        except urllib.error.HTTPError as e:
            corpo = e.read().decode(errors="replace")
            print(f"  ERRO na validacao do ComfyUI (imagem {img}):")
            print("  " + corpo[:3000])
            print("  -> Pulando esta imagem; o lote continua.")
            continue
        pid = r.get("prompt_id")
        if not pid:
            print(f"  ERRO: sem prompt_id na resposta: {r}")
            continue
        # aguardar concluir (poll no /history)
        while True:
            time.sleep(5)
            try:
                h = api(f"/history/{pid}")
            except Exception:
                continue
            if pid in h:
                status = h[pid].get("status", {})
                if status.get("completed"):
                    print(f"  ok em {time.time()-t0:.0f}s")
                    break
                if status.get("status_str") == "error":
                    print(f"  ERRO na execucao (ver log do ComfyUI). Pulando.")
                    break

    print(f"\n=== LOTE CONCLUIDO. Resultados em {OUT_DIR} ===")
    print("Confira as *_final_* em 100% de zoom; descarte alucinadas (as *_E1_* ajudam a comparar).")


if __name__ == "__main__":
    main()

LOTEPYEOF
chmod +x /workspace/lote_upscale.py

cat > /workspace/GUIA_UPSCALE.md << 'GUIAEOF'
# GUIA — POD DE UPSCALE CINEMATOGRAFICO
Pipeline: SeedVR2 7B fp16 (3072px) -> SRPO Q8 (denoise 0.28)
Calibrado e validado em 14/08/2026. Parametros travados: nao mexer sem decisao registrada.

## SUBIR O POD (do zero)
1. Vast.ai > Search > selecione o template `upscale-pod` NO TOPO antes de alugar
2. GPU RTX 5090, host US/EU, filtros: cuda_max_good>=12.8 inet_down>500 inet_up>100 verified=true
3. Rent > aguarde 10-20 min de boot (primeira vez baixa ~72GB)
4. Acompanhar: abra o Jupyter (Open) > Terminal > `tail -f /workspace/logs/boot-upscale.log`
5. Pronto quando aparecer: `(5/5) Boot concluido. Subindo ComfyUI :8188`
6. Pod reaproveitado (Reboot): 3-5 min, modelos ja estao no disco

## MODO 1 — UMA IMAGEM (interface do ComfyUI)
1. Abra o ComfyUI (porta 8188)
2. Barra lateral > Workflows > "T3 - Pipeline SeedVR2 + SRPO"
3. No no ENTRADA: "escolher arquivo para enviar" > suba a imagem crua
4. EXECUTAR. Parado em 0% por 3-6 min = NORMAL (carga dos modelos)
5. Saida em ComfyUI/output: T3_etapa1_seedvr2_* (conferencia) e T3_final_* (pronta)

## MODO 2 — LOTE (varias imagens)
1. Jupyter > painel de arquivos > ComfyUI/input/upscale_in > arraste as imagens
2. Terminal:
       python3 /workspace/lote_upscale.py
3. Progresso na tela: [3/20] nome.png ... ok em 95s
4. Saida em ComfyUI/output/upscale_out: nome_E1_* e nome_final_*
5. Interrompeu? Rode de novo: pula as prontas e continua.

## CONFERENCIA (sempre)
- Abrir as *_final_* em 100% de zoom no PC (NUNCA julgar por miniatura)
- Checar: pele com poros, fios de cabelo, trama de tecido, MESMO rosto, sem alucinacao
- Descartar alucinadas; as *_E1_* servem de comparacao

## DIAGNOSTICO
    tail -f /workspace/logs/boot-upscale.log    # o que o ComfyUI esta fazendo
    nvidia-smi                                   # GPU trabalhando? memoria/util
    df -h /workspace                             # espaco em disco
- 0% ate ~6 min: normal. So suspeitar apos 10 min sem linha nova no log.
- Erro "tile_debug ... nao esta disponivel": campo deve ser "false" (texto, via setinha)
- 504 no Jupyter: Reboot da instancia resolve (modelos ficam intactos)

## REGRAS TRAVADAS (calibragem 14/08/2026)
- SeedVR2: 7B fp16 (NUNCA fp8), resolution 3072, batch 1, no de torch.compile DESLIGADO (roxo)
- SRPO: Q8_0 GGUF (NUNCA converter fp8 em runtime), denoise 0.28 MAXIMO, guidance 3.5, cfg 1.0, seed 42
- Julgamento de qualidade: PNG individual a 100%, nunca mosaico/miniatura
GUIAEOF


# ----------------------------------------------------------------------------
# (5/5) Subir o ComfyUI na porta 8188
# ----------------------------------------------------------------------------
echo "[boot] (5/5) Boot concluido. Subindo ComfyUI :8188"
cd /workspace/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188
