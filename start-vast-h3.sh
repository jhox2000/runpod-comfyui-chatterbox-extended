#!/bin/bash
# ============================================================================
# start-vast-h3.sh — Template VAST.AI: POD DE TESTE DO MINIMAX H3
# ----------------------------------------------------------------------------
# ESCOPO: SOMENTE geracao de video com o MiniMax H3 (open weights, audio
# estereo nativo). Este pod NAO tem Qwen-Image, Wan 2.2, lote de personagens
# nem TTS — e um pod dedicado para testar o H3 contra o pipeline atual.
#
# O que sobe:
#   - ComfyUI (versao mais recente — o H3 precisa de >= 0.30.0, nos nativos)
#   - Modelos H3 (repackaged da Comfy-Org):
#       FL2VA  -> text-to-video / image-to-video / first-last-frame  (I2V)
#       REF2VA -> reference-to-video (identidade, estilo, voz)      (R2V)
#       Text encoder Qwen3-VL 32B (nvfp4_awq — precisa de Blackwell/5090)
#       VAE de video + VAE de audio (os DOIS sao obrigatorios)
#       LoRAs turbo (4/8 steps) — o workflow escaneia por elas, mas ficam
#       desligadas por padrao (turbo_mode = false, 20 steps = RAW)
#   - 3 workflows oficiais da Comfy-Org na barra lateral:
#       1 - H3 Imagem para Video (I2V)
#       2 - H3 Referencia para Video (R2V)
#       3 - H3 Texto para Video (T2V)
#
# Config Vast.ai (mesmo padrao do start-vast.sh de producao):
#   - Base image: vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311
#   - GPU: RTX 5090 (Blackwell sm_120) — OBRIGATORIO para o text encoder
#     nvfp4. Em 4090 troque H3_TEXT_ENCODER para int8 (ver abaixo).
#   - Container disk: 150 GB
#   - Porta 8188 (Docker Options: -p 8188:8188)
#   - HF_TOKEN como variavel de ambiente (NUNCA hardcoded aqui)
#
# Variaveis opcionais (Docker Options: -e NOME=valor):
#   H3_PRECISION     pruned_int8 (padrao, ~21GB, cabe folgado em 32GB)
#                    pruned_bf16 (~40GB, qualidade maxima teorica, roda com
#                                 offload dinamico do ComfyUI — mais lento)
#   H3_TEXT_ENCODER  nvfp4 (padrao, ~16GB, so Blackwell)
#                    int8  (~27GB, funciona em Ada/4090)
#   H3_SKIP_R2V=1    nao baixa o REF2VA (economiza ~21GB se so quiser I2V)
#
# On-start script na Vast:
#   bash -c "mkdir -p /workspace && curl -fsSL https://raw.githubusercontent.com/jhox2000/runpod-comfyui-chatterbox-extended/refs/heads/main/start-vast-h3.sh -o /workspace/start-vast-h3.sh && bash /workspace/start-vast-h3.sh"
#
# Tempo de boot estimado: ~10-20 min em host >= 1 Gbps (~68GB de modelos).
# ============================================================================
set -e
export DEBIAN_FRONTEND=noninteractive

H3_PRECISION="${H3_PRECISION:-pruned_int8}"
H3_TEXT_ENCODER="${H3_TEXT_ENCODER:-nvfp4}"
H3_SKIP_R2V="${H3_SKIP_R2V:-0}"

# Ambiente Python da imagem da Vast: tudo roda dentro do venv /venv/main
[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate || true

# Log de boot: tudo que aparece aqui tambem vai para /workspace/logs
mkdir -p /workspace/logs
exec > >(tee -a /workspace/logs/boot-h3.log) 2>&1
echo "[boot] ====== POD MINIMAX H3 — $(date) ======"
echo "[boot] H3_PRECISION=$H3_PRECISION  H3_TEXT_ENCODER=$H3_TEXT_ENCODER  H3_SKIP_R2V=$H3_SKIP_R2V"

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

# Checagem da GPU: o text encoder nvfp4 exige Blackwell (sm_120).
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "desconhecida")
echo "[boot] GPU detectada: $GPU_NAME"
if [ "$H3_TEXT_ENCODER" = "nvfp4" ] && ! echo "$GPU_NAME" | grep -qiE "50[0-9]{2}|B200|B100|RTX PRO"; then
    echo "[boot] AVISO: GPU nao parece ser Blackwell. O text encoder nvfp4 pode falhar."
    echo "[boot] AVISO: Se der erro no CLIPLoader, suba de novo com -e H3_TEXT_ENCODER=int8"
fi

# HuggingFace: desligar hf_transfer (deprecado) e usar XET (rapido e moderno)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_HIGH_PERFORMANCE=1

# ----------------------------------------------------------------------------
# (1/6) Dependencias de sistema + PyTorch cu128
# A imagem base da Vast NAO traz PyTorch: instalar a versao cu128 (Blackwell)
# ANTES do requirements do ComfyUI, para o pip nao escolher um build errado.
# ----------------------------------------------------------------------------
echo "[boot] (1/6) Dependencias de sistema + PyTorch cu128..."
apt-get update -qq
apt-get install -y -qq ffmpeg htop tmux build-essential lsof
pip install -q -U huggingface_hub hf_xet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ----------------------------------------------------------------------------
# (2/6) ComfyUI + custom nodes
# Clone CONDICIONAL (o volume /workspace persiste), mas o ComfyUI e ATUALIZADO
# em todo boot (git pull) — os nos do H3 sao nativos e recebem correcoes
# frequentes (guias em qualquer frame, embeddings, mascaras). pip install
# SEMPRE (container efemero perde os pacotes pip no restart).
# ----------------------------------------------------------------------------
if [ ! -d /workspace/ComfyUI ]; then
    echo "[boot] (2/6) Clonando ComfyUI..."
    cd /workspace
    git clone https://github.com/comfyanonymous/ComfyUI
else
    echo "[boot] (2/6) Atualizando ComfyUI (git pull)..."
    cd /workspace/ComfyUI
    git pull --ff-only || echo "[boot] AVISO: git pull falhou, seguindo com a versao local"
fi
CN=/workspace/ComfyUI/custom_nodes
mkdir -p "$CN"
cd "$CN"
[ -d ComfyUI-Manager ] || git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager
# KJNodes: traz o "Patch Sage Attention KJ" (dobra a velocidade do H3 se o
# sageattention estiver instalado). Opcional — se nao instalar o wheel do
# sageattention, o no simplesmente nao e usado.
[ -d ComfyUI-KJNodes ] || git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes

echo "[boot] (2/6) Instalando dependencias Python (SEMPRE, todo boot)..."
pip install -r /workspace/ComfyUI/requirements.txt
pip install -r "$CN/ComfyUI-Manager/requirements.txt" || true
pip install -r "$CN/ComfyUI-KJNodes/requirements.txt" || echo "[boot] AVISO: requirements do KJNodes falhou (opcional)"
pip uninstall -y xformers 2>/dev/null || true   # vem na imagem base; incompativel com Blackwell

# Versao do ComfyUI (o H3 precisa de >= 0.30.0)
COMFY_VER=$(grep -oE '__version__ = "[^"]+"' /workspace/ComfyUI/comfyui_version.py 2>/dev/null | cut -d'"' -f2 || echo "?")
echo "[boot] ComfyUI versao: $COMFY_VER"

# ----------------------------------------------------------------------------
# (3/6) Modelos (~68GB) via hf_hub_download com local_dir + mv.
# Os arquivos moram DIRETO no /workspace (volume persistente) — sem symlink
# para o cache HF, que fica no disco efemero e quebraria em todo restart.
# Cada download tem checagem PROPRIA por arquivo.
#
# Fonte: Comfy-Org/MiniMax-H3 (repackaged oficial do ComfyUI) +
#        lightx2v/Minimax-h3-Turbo (LoRA turbo 8-step referenciada no template)
# ----------------------------------------------------------------------------
M=/workspace/ComfyUI/models
mkdir -p "$M"/{diffusion_models,text_encoders,vae,loras,embeddings}

# Funcao unica de download: repo, arquivo-no-repo, pasta-destino
hf_get() {
    local repo="$1"; local rfile="$2"; local dest="$3"
    local fname=$(basename "$rfile")
    if [ -f "$dest/$fname" ]; then
        echo "[boot]   [ja existe] $fname"
        return 0
    fi
    echo "[boot]   baixando $fname ..."
    python3 - "$repo" "$rfile" << 'PYDL'
import sys
from huggingface_hub import hf_hub_download
repo, rfile = sys.argv[1], sys.argv[2]
hf_hub_download(repo_id=repo, filename=rfile, local_dir='/workspace/_dl_h3')
PYDL
    mv "/workspace/_dl_h3/$rfile" "$dest/$fname"
}

# --- Escolha dos arquivos conforme as variaveis ---
case "$H3_PRECISION" in
    pruned_int8) DM_SUFFIX="pruned_int8_convrot"; DM_MIN_GB=19 ;;
    pruned_bf16) DM_SUFFIX="pruned_bf16";         DM_MIN_GB=38 ;;
    *) echo "[boot] ERRO: H3_PRECISION invalido: $H3_PRECISION (use pruned_int8 ou pruned_bf16)"; exit 1 ;;
esac
case "$H3_TEXT_ENCODER" in
    nvfp4) TE_FILE="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";    TE_MIN_GB=14 ;;
    int8)  TE_FILE="qwen3vl_32b_minimax_h3_int8_convrot.safetensors"; TE_MIN_GB=25 ;;
    *) echo "[boot] ERRO: H3_TEXT_ENCODER invalido: $H3_TEXT_ENCODER (use nvfp4 ou int8)"; exit 1 ;;
esac
FL2VA_FILE="minimax_h3_fl2va_${DM_SUFFIX}.safetensors"
REF2VA_FILE="minimax_h3_ref2va_${DM_SUFFIX}.safetensors"

echo "[boot] (3/6) Baixando modelos do MiniMax H3..."
H3REPO="Comfy-Org/MiniMax-H3"

# Diffusion models
hf_get "$H3REPO" "diffusion_models/$FL2VA_FILE" "$M/diffusion_models"
if [ "$H3_SKIP_R2V" != "1" ]; then
    hf_get "$H3REPO" "diffusion_models/$REF2VA_FILE" "$M/diffusion_models"
fi

# Text encoder (Qwen3-VL 32B)
hf_get "$H3REPO" "text_encoders/$TE_FILE" "$M/text_encoders"

# VAEs — os DOIS sao obrigatorios (video + audio)
hf_get "$H3REPO" "vae/minimax_h3_video_vae_fp16.safetensors" "$M/vae"
hf_get "$H3REPO" "vae/minimax_h3_audio_vae_fp32.safetensors" "$M/vae"

# LoRAs turbo — o template escaneia por elas; ficam desligadas (RAW por padrao)
hf_get "$H3REPO" "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" "$M/loras"
hf_get "$H3REPO" "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors" "$M/loras"
if [ "$H3_SKIP_R2V" != "1" ]; then
    hf_get "$H3REPO" "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" "$M/loras"
fi

# Embedding de estilo referenciado pelo template (pequeno, evita aviso de
# arquivo faltando no scan do workflow)
hf_get "$H3REPO" "embeddings/minimaxh3_art_is_explosion.safetensors" "$M/embeddings"

rm -rf /workspace/_dl_h3

# ----------------------------------------------------------------------------
# (4/6) Verificacao de sanidade — tamanhos minimos.
# Download truncado (rede de host ruim) e pego AQUI, nao no meio do teste.
# ----------------------------------------------------------------------------
echo "[boot] (4/6) Verificacao de sanidade dos modelos..."
check_size() {
    local f="$1"; local min_gb="$2"
    if [ ! -f "$f" ]; then echo "[boot] ERRO: $f nao existe."; exit 1; fi
    local size_gb=$(du -BG "$f" | cut -f1 | tr -d 'G')
    if [ "$size_gb" -lt "$min_gb" ]; then
        echo "[boot] ERRO: $f tem ${size_gb}GB, esperado >= ${min_gb}GB (download truncado?)"; exit 1
    fi
    echo "[boot]   [ok] $(basename "$f") — ${size_gb}GB"
}
check_size "$M/diffusion_models/$FL2VA_FILE" "$DM_MIN_GB"
[ "$H3_SKIP_R2V" != "1" ] && check_size "$M/diffusion_models/$REF2VA_FILE" "$DM_MIN_GB"
check_size "$M/text_encoders/$TE_FILE" "$TE_MIN_GB"
check_size "$M/vae/minimax_h3_video_vae_fp16.safetensors" 4
# VAE de audio tem 0.6GB — checagem so de existencia
[ -f "$M/vae/minimax_h3_audio_vae_fp32.safetensors" ] || { echo "[boot] ERRO: VAE de audio ausente."; exit 1; }
echo "[boot]   [ok] minimax_h3_audio_vae_fp32.safetensors"

# Pastas de trabalho
mkdir -p /workspace/ComfyUI/input /workspace/ComfyUI/output/h3

# ----------------------------------------------------------------------------
# (5/6) Workflows oficiais da Comfy-Org na barra lateral.
# Baixados em TODO boot (idempotente) do repo workflow_templates — assim
# qualquer correcao de template chega no pod. Se voce customizar um workflow
# dentro do ComfyUI, SALVE COM OUTRO NOME.
# Sao renomeados para aparecerem na ordem de uso e em portugues. Os arquivos
# de modelo que os templates esperam sao EXATAMENTE os baixados acima
# (fl2va/ref2va pruned_int8_convrot + nvfp4_awq). Se voce subiu com
# H3_PRECISION=pruned_bf16 ou H3_TEXT_ENCODER=int8, os nomes sao trocados
# dentro do JSON aqui para casarem.
# ----------------------------------------------------------------------------
echo "[boot] (5/6) Instalando workflows do H3 na barra lateral..."
WF_DIR=/workspace/ComfyUI/user/default/workflows
mkdir -p "$WF_DIR"
TPL="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates"
get_wf() {
    local src="$1"; local dst="$2"
    if curl -fsSL "$TPL/$src.json" -o "$WF_DIR/$dst.json"; then
        # Ajusta nomes de modelo se a precisao/encoder escolhidos nao forem os do template
        sed -i \
            -e "s/minimax_h3_fl2va_pruned_int8_convrot\.safetensors/$FL2VA_FILE/g" \
            -e "s/minimax_h3_ref2va_pruned_int8_convrot\.safetensors/$REF2VA_FILE/g" \
            -e "s/qwen3vl_32b_minimax_h3_nvfp4_awq\.safetensors/$TE_FILE/g" \
            -e 's#"video/MiniMax_H3"#"h3/H3"#g' \
            "$WF_DIR/$dst.json"
        echo "[boot]   [ok] $dst"
    else
        echo "[boot] AVISO: nao consegui baixar o template $src — use Template Library > Video > MiniMax H3"
    fi
}
get_wf video_minimax_h3_i2v "1 - H3 Imagem para Video (I2V)"
[ "$H3_SKIP_R2V" != "1" ] && get_wf video_minimax_h3_r2v "2 - H3 Referencia para Video (R2V)"
get_wf video_minimax_h3_t2v "3 - H3 Texto para Video (T2V)"

# Guia rapido em portugues, tambem na barra lateral
cat > "$WF_DIR/0 - LEIA-ME (H3).json" << 'WORKFLOW_EOF_README'
{"id":"h3-readme-0001","revision":0,"last_node_id":1,"last_link_id":0,"nodes":[{"id":1,"type":"MarkdownNote","pos":[0,0],"size":[720,980],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"title":"GUIA — POD MINIMAX H3","properties":{},"widgets_values":["# GUIA — POD MINIMAX H3\n\nModelo de video **open weights** com **audio estereo nativo** (fala, efeitos e musica gerados junto). Ate 15 s, 24 fps, canvas nativo **768p** (1344x768 em 16:9).\n\n## Workflows\n- **1 - I2V**: sua imagem vira video. Conecte em `first_frame`; `last_frame` e opcional (controla o quadro final).\n- **2 - R2V**: referencia de personagem/estilo/voz. Ate 9 imagens, 3 videos, 3 audios. Cite cada referencia por tag no prompt: `<Picture 1>`, `<Video 1>`, `<Audio 1>` — e diga o que cada uma controla (identidade, estilo, camera, voz).\n- **3 - T2V**: so texto.\n\n## Qualidade MAXIMA (padrao deste pod)\n- `turbo_mode` = **false** (LoRA turbo desligada)\n- steps = **20** (suba para **25** para movimento melhor)\n- sampler `res_multistep` + scheduler `simple`\n- Resolucao: no *Resolution Selector* use **Megapixels 0.98** para 16:9 (=1344x768). NAO use 1.0 (passa do limite do modelo).\n- Duracao: o modelo trabalha em blocos de 17 frames (17k+5). O campo `duration` arredonda sozinho.\n\n## Prompt (o que faz diferenca)\n1. Cena inteira primeiro (lugar, personagem, o que acontece)\n2. Depois os planos com tempo: `0-3s: ...`, `3-7s: ...`\n3. Camera (dolly, pan, handheld...)\n4. **Audio na mesma frase**: `Audio: rain on windows, soft piano, she says in Spanish: \"...\"`\n\nO H3 fala 11 idiomas (inclui **portugues e espanhol**). Para narracao externa (TTS), peca `no dialogue, ambient sound only` e use so o ambiente por baixo da sua voz.\n\n## Modo rapido (para iterar)\nLigue `turbo_mode` = true -> 8 steps com a LoRA turbo. Perde um pouco de audio e movimento. Bom para achar o prompt certo; depois desliga e gera a final.\n\n## Onde saem os videos\n`ComfyUI/output/h3/` -> `H3_00001_.mp4` (com audio embutido).\nBaixar tudo: botao direito na pasta `output` no Jupyter -> *Download as Archive*.\n\n## Se der erro\n- `CLIPLoader` / nvfp4: GPU nao e Blackwell. Suba de novo com `-e H3_TEXT_ENCODER=int8`.\n- Sem audio no mp4: confira se os DOIS VAEs (video fp16 + audio fp32) estao carregados e se o `VAEDecodeAudio` esta ligado no `SaveVideo`.\n- OOM: reduza para 0.5 Megapixels ou 5-8 s de duracao.\n\n## Licenca\nO H3 usa a *MiniMax H3 Community License* (nao e Apache 2.0). Uso comercial dos videos gerados localmente exige licenca comercial da MiniMax (vendida via Comfy). Leia antes de colocar em canal monetizado."],"color":"#432","bgcolor":"#653"}],"links":[],"groups":[],"config":{},"extra":{},"version":0.4}
WORKFLOW_EOF_README

# ----------------------------------------------------------------------------
# (6/6) Subir o ComfyUI na porta 8188
#   --enable-cors-header: necessario atras do proxy da Vast.
#   Sage Attention NAO e ligado por padrao (exige wheel compilado para a
#   versao exata de torch/cuda). Para ligar: pip install <wheel do
#   woct0rdho/SageAttention> e adicione --use-sage-attention aqui.
# ----------------------------------------------------------------------------
echo "[boot] (6/6) Boot concluido. Subindo ComfyUI :8188"
cd /workspace/ComfyUI
nohup python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header \
    > /workspace/logs/comfyui.log 2>&1 &

echo "[boot] ============================================================"
echo "[boot]  TUDO PRONTO!"
echo "[boot]   ComfyUI -> porta 8188  (log: /workspace/logs/comfyui.log)"
echo "[boot]   Jupyter -> botao Open da instancia (fornecido pela Vast)"
echo "[boot]   Workflows: barra lateral -> 0 LEIA-ME / 1 I2V / 2 R2V / 3 T2V"
echo "[boot]   Modelos: $FL2VA_FILE + $TE_FILE"
echo "[boot]  O ComfyUI leva mais ~1-2 min apos esta mensagem para"
echo "[boot]  terminar de carregar."
echo "[boot] ============================================================"
