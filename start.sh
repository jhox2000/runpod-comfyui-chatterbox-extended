#!/bin/bash
# ============================================================================
# start.sh — Template RunPod 1-CLICK: ComfyUI (Wan 2.2 I2V + Qwen-Image ger/edit) + Chatterbox
# ----------------------------------------------------------------------------
# AUTOSSUFICIENTE: nao precisa de upload manual de nenhum arquivo.
# Cole este script inteiro no campo "Container Start Command" do template.
# No deploy ele instala tudo, baixa os modelos e sobe os dois servicos sozinho.
#
# Imagem base esperada: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
# GPU alvo: RTX 5090 (Blackwell sm_120) ou superior, com 24GB+ de VRAM e CUDA 12.8.
# Portas HTTP: 7860 (Chatterbox) | 8188 (ComfyUI) | 8888 (JupyterLab)
# Tempo de boot estimado: ~25-35 min (download de ~88GB de modelos: Wan + Qwen).
# ============================================================================
set -e
export DEBIAN_FRONTEND=noninteractive

# HuggingFace: desligar hf_transfer (deprecado/ausente) e usar XET (rapido e moderno)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p /workspace/logs

# ----------------------------------------------------------------------------
# JupyterLab: sobe logo no inicio, sem senha, para servir como ferramenta
# de debug em tempo real enquanto o resto do boot roda. A imagem base ja tem
# o pacote instalado; so precisamos iniciar.
# ----------------------------------------------------------------------------
echo "[boot] (0/7) Iniciando JupyterLab na porta 8888..."
nohup jupyter lab \
    --ip=0.0.0.0 --port=8888 --allow-root --no-browser \
    --ServerApp.token='' --ServerApp.password='' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=True \
    --ServerApp.root_dir=/workspace \
    > /workspace/logs/jupyter.log 2>&1 &

echo "[boot] (1/7) Dependencias de sistema..."
apt-get update -qq
apt-get install -y -qq \
    ffmpeg libsndfile1 sox git-lfs aria2 \
    python3.10 python3.10-venv python3.10-dev \
    htop tmux build-essential lsof
git-lfs install --skip-repo
pip install -q -U huggingface_hub hf_xet

# ----------------------------------------------------------------------------
# (2/7) ComfyUI + ComfyUI-Manager
# ----------------------------------------------------------------------------
if [ ! -d /workspace/ComfyUI ]; then
    echo "[boot] (2/7) Clonando ComfyUI..."
    cd /workspace
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI
    cd ComfyUI
    pip install -r requirements.txt
    pip uninstall -y xformers 2>/dev/null || true   # incompativel com Blackwell
    cd custom_nodes
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager
    cd ComfyUI-Manager && pip install -r requirements.txt || true
fi

# ----------------------------------------------------------------------------
# (3/7) Modelos Wan 2.2 I2V 14B — versao fp8 (rapida) + LoRAs lightx2v 4steps
#       ~35GB total. fp8 e ~5x mais rapido que fp16 e cabe folgado na 5090.
# ----------------------------------------------------------------------------
if [ ! -f /workspace/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors ]; then
    echo "[boot] (3/7) Baixando modelos Wan 2.2 fp8 + LoRAs (~35GB)..."
    mkdir -p /workspace/_dl_stage
    mkdir -p /workspace/ComfyUI/models/{diffusion_models,text_encoders,vae,loras}
    python3 - << 'PYDL'
from huggingface_hub import hf_hub_download
REPO = 'Comfy-Org/Wan_2.2_ComfyUI_Repackaged'
files = [
    'split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors',
    'split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors',
    'split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors',
    'split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors',
    'split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors',
    'split_files/vae/wan_2.1_vae.safetensors',
]
for f in files:
    print('  baixando:', f.split('/')[-1])
    hf_hub_download(repo_id=REPO, filename=f, local_dir='/workspace/_dl_stage')
PYDL
    mv /workspace/_dl_stage/split_files/diffusion_models/*.safetensors /workspace/ComfyUI/models/diffusion_models/
    mv /workspace/_dl_stage/split_files/loras/*.safetensors            /workspace/ComfyUI/models/loras/
    mv /workspace/_dl_stage/split_files/text_encoders/*.safetensors    /workspace/ComfyUI/models/text_encoders/
    mv /workspace/_dl_stage/split_files/vae/*.safetensors              /workspace/ComfyUI/models/vae/
    rm -rf /workspace/_dl_stage
fi

# ----------------------------------------------------------------------------
# (3.5/7) Modelos Qwen-Image — GERADOR (2512) + EDITOR (2511), versao FP8 + Lightning 4-steps.
#         Gerador: texto->imagem (cenas, paisagens, estilos).
#         Editor: mantem o MESMO rosto e junta 2 personagens na mesma imagem.
#         Encoder e VAE sao COMPARTILHADOS com o gerador (baixados 1x so).
#         Custom node Comfyui-QwenEditUtils e obrigatorio para o editor.
#         ~53GB adicionais. Precisa de 24GB+ de VRAM (roda na 5090).
# ----------------------------------------------------------------------------
if [ ! -f /workspace/ComfyUI/models/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors ]; then
    echo "[boot] (3.5/7) Baixando modelos Qwen-Image (gerador + editor)..."
    mkdir -p /workspace/_dl_qwen
    mkdir -p /workspace/ComfyUI/models/{diffusion_models,text_encoders,vae,loras}
    python3 - << 'PYBLOCK_QWEN'
from huggingface_hub import hf_hub_download

# Gerador 2512 + encoder + VAE (repo oficial Comfy-Org)
GEN_REPO = 'Comfy-Org/Qwen-Image_ComfyUI'
for f in [
    'split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors',
    'split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors',
    'split_files/vae/qwen_image_vae.safetensors',
]:
    print('  [gerador]', f.split('/')[-1])
    hf_hub_download(repo_id=GEN_REPO, filename=f, local_dir='/workspace/_dl_qwen')

# LoRA Lightning 4-steps do gerador
print('  [gerador] Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors')
hf_hub_download(repo_id='lightx2v/Qwen-Image-2512-Lightning',
    filename='Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors',
    local_dir='/workspace/_dl_qwen')

# Editor 2511 (FP8 + Lightning 4-steps embutido)
print('  [editor] qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors')
hf_hub_download(repo_id='lightx2v/Qwen-Image-Edit-2511-Lightning',
    filename='qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors',
    local_dir='/workspace/_dl_qwen')
PYBLOCK_QWEN
    # Move tudo para as pastas certas (gerador + editor + compartilhados)
    find /workspace/_dl_qwen -name '*.safetensors' -path '*diffusion_models*' -exec mv -n {} /workspace/ComfyUI/models/diffusion_models/ \; 2>/dev/null || true
    find /workspace/_dl_qwen -name '*.safetensors' -path '*text_encoders*'    -exec mv -n {} /workspace/ComfyUI/models/text_encoders/ \;    2>/dev/null || true
    find /workspace/_dl_qwen -name '*.safetensors' -path '*vae*'              -exec mv -n {} /workspace/ComfyUI/models/vae/ \;              2>/dev/null || true
    find /workspace/_dl_qwen -name '*Lightning*4steps*.safetensors'           -exec mv -n {} /workspace/ComfyUI/models/loras/ \;            2>/dev/null || true
    find /workspace/_dl_qwen -name 'qwen_image_edit_2511*lightning*.safetensors' -exec mv -n {} /workspace/ComfyUI/models/diffusion_models/ \; 2>/dev/null || true
    rm -rf /workspace/_dl_qwen
fi

# Custom node obrigatorio para o workflow do EDITOR (no TextEncodeQwenImageEditPlus)
if [ ! -d /workspace/ComfyUI/custom_nodes/Comfyui-QwenEditUtils ]; then
    echo "[boot] (3.5/7) Instalando custom node Comfyui-QwenEditUtils..."
    cd /workspace/ComfyUI/custom_nodes
    git clone --depth 1 https://github.com/lrzjason/Comfyui-QwenEditUtils
    [ -f Comfyui-QwenEditUtils/requirements.txt ] && pip install -r Comfyui-QwenEditUtils/requirements.txt || true
fi

# WAS Node Suite — le prompts linha-a-linha (lote de imagem) e carrega imagens em sequencia (lote de video)
if [ ! -d /workspace/ComfyUI/custom_nodes/was-node-suite-comfyui ]; then
    echo "[boot] (3.5/7) Instalando custom node WAS Node Suite..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
    git -C was-node-suite-comfyui checkout ea935d1044ae5a26efa54ebeb18fe9020af49a45 || true
    [ -f was-node-suite-comfyui/requirements.txt ] && pip install -r was-node-suite-comfyui/requirements.txt || true
fi

# VideoHelperSuite — componente de video (lote de video)
if [ ! -d /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite ]; then
    echo "[boot] (3.5/7) Instalando custom node VideoHelperSuite..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    git -C ComfyUI-VideoHelperSuite checkout 4ee72c065db22c9d96c2427954dc69e7b908444b || true
    [ -f ComfyUI-VideoHelperSuite/requirements.txt ] && pip install -r ComfyUI-VideoHelperSuite/requirements.txt || true
fi

# ----------------------------------------------------------------------------
# (3.6/7) Workflows prontos na barra lateral do ComfyUI.
#         Os 3 ja vem ajustados (rosto fixo, lanczos, CLIP qwen_image, etc).
#         Ao abrir o ComfyUI eles aparecem em "Workflows" > pasta do usuario,
#         a um clique cada, cada um abrindo em sua propria aba.
# ----------------------------------------------------------------------------
echo "[boot] (3.6/7) Instalando workflows prontos (Wan / Gerador / Editor)..."
WF_DIR=/workspace/ComfyUI/user/default/workflows
mkdir -p "$WF_DIR"

cat > "$WF_DIR/1 - Video (Wan 2.2).json" << 'WORKFLOW_EOF_WAN'
{"id":"ec7da562-7e21-4dac-a0d2-f4441e1efd3b","revision":0,"last_node_id":9999,"last_link_id":246,"nodes":[{"id":97,"type":"LoadImage","pos":[90,250],"size":[610,490],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"IMAGE","type":"IMAGE","slot_index":0,"links":[245]},{"name":"MASK","type":"MASK","slot_index":1,"links":null}],"title":"Start Frame Image","properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"LoadImage","ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":["03_video_wan2_2_14B_i2v_subgraphed_input_image.png","image"]},{"id":108,"type":"SaveVideo","pos":[1300,250],"size":[680,530],"flags":{},"order":4,"mode":0,"inputs":[{"name":"video","type":"VIDEO","link":246}],"outputs":[],"properties":{"cnr_id":"comfy-core","ver":"0.3.49","Node name for S&R":"SaveVideo","ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":["video/Wan2.2_image_to_video","auto","auto"]},{"id":118,"type":"MarkdownNote","pos":[-450,250],"size":[480,460],"flags":{},"order":1,"mode":0,"inputs":[],"outputs":[],"title":"Note: how to use this workflow","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":["**Wan2.2 Image to Video(Subgraph)** node is a subgraph, which is converted from the Wan2.2 image to video workflow.\n\nYou can find the tutorial and original workflow [here](https://docs.comfy.org/tutorials/video/wan/wan2_2).\n\n\nThis workflow requires high VRAM (More than 20GB). If you are running this workflow locally, please make sure you have enough VRAM.\n\n## For Comfy Cloud users\n\nIf you are using [cloud.comfy.org](https://cloud.comfy.org/):\n\n1. Since the workflow in the Cloud will have the input image ready, for the first run, you can just click the run button to see what happens. \n\n2. Try to upload your own image to the **Start Frame Image**, and then try to describe the video you want to generate in the **Wan2.2 Image to Video(Subgraph)**."],"color":"#222","bgcolor":"#000"},{"id":119,"type":"MarkdownNote","pos":[-450,700],"size":[480,1070],"flags":{},"order":2,"mode":0,"inputs":[],"outputs":[],"title":"For local ComfyUI users","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["[Tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2\n)\n\n## VRAM Usage\n\nThe following data was tested using an RTX 4090 24GB\n\n| Model            | Size |VRAM Usage | 1st Generation | 2nd Generation |\n|---------------------|-------|-----------|---------------|-----------------|\n| fp8_scaled               |640*640| 84%               | ≈  536s              | ≈ 513s                   |\n| fp8_scaled +  4steps LoRA  | 640*640  | 83%                | ≈ 97s               | ≈ 71s                   |\n\n**Diffusion Model**\n- [wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors)\n- [wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors)\n\n**LoRA**\n- [wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors)\n- [wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors)\n\n**VAE**\n- [wan_2.1_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors)\n\n**Text Encoder**   \n- [umt5_xxl_fp8_e4m3fn_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors)\n\n\nFile save location\n\n```\nComfyUI/\n├───📂 models/\n│   ├───📂 diffusion_models/\n│   │   ├─── wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors\n│   │   └─── wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors\n│   ├───📂 loras/\n│   │   ├─── wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors\n│   │   └─── wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors\n│   ├───📂 text_encoders/\n│   │   └─── umt5_xxl_fp8_e4m3fn_scaled.safetensors \n│   └───📂 vae/\n│       └── wan_2.1_vae.safetensors\n```\n\nIf you have any problems running this workflow, please report template-related issues via this link: [report the template issue here](https://github.com/Comfy-Org/workflow_templates/issues)"],"color":"#222","bgcolor":"#000"},{"id":130,"type":"d2ac71a3-c7a1-48fa-abea-0aa3f97d7bf2","pos":[800,250],"size":[400,720],"flags":{"collapsed":false},"order":3,"mode":0,"inputs":[{"label":"start image","name":"start_image","shape":7,"type":"IMAGE","link":245},{"label":"prompt","name":"text","type":"STRING","widget":{"name":"text"},"link":null},{"label":"high_noise_unet","name":"unet_name","type":"COMBO","widget":{"name":"unet_name"},"link":null},{"label":"high_noise_lora","name":"lora_name","type":"COMBO","widget":{"name":"lora_name"},"link":null},{"label":"low_noise_unet","name":"unet_name_1","type":"COMBO","widget":{"name":"unet_name_1"},"link":null},{"label":"low_noise_lora","name":"lora_name_1","type":"COMBO","widget":{"name":"lora_name_1"},"link":null}],"outputs":[{"name":"VIDEO","type":"VIDEO","links":[246]}],"properties":{"proxyWidgets":[["107","text"],["128","width"],["128","height"],["128","length"],["122","unet_name"],["126","lora_name"],["123","unet_name"],["127","lora_name"],["105","clip_name"],["106","vae_name"],["110","noise_seed"],["110","control_after_generate"]],"cnr_id":"comfy-core","ver":"0.11.0","ue_properties":{"widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":[]},{"id":9001,"type":"MarkdownNote","pos":[-520,250],"size":[420,300],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"title":"Proporcoes (video)","properties":{},"widgets_values":["## Proporcoes (video) - Wan 2.2\n\nAjuste em **width** e **height** na caixa do Wan. Maior = mais lento.\n\n- 16:9 rapido: 832 x 480\n- 16:9 HD: 1280 x 720\n- 9:16 rapido: 480 x 832\n- 9:16 HD: 720 x 1280\n- 1:1: 720 x 720\n\nDica: use a mesma proporcao das imagens."],"color":"#432","bgcolor":"#653"}],"links":[[245,97,0,130,0,"IMAGE"],[246,130,0,108,0,"VIDEO"]],"groups":[],"definitions":{"subgraphs":[{"id":"d2ac71a3-c7a1-48fa-abea-0aa3f97d7bf2","version":1,"state":{"lastGroupId":16,"lastNodeId":130,"lastLinkId":246,"lastRerouteId":0},"revision":0,"config":{},"name":"Image to Video (Wan 2.2)","inputNode":{"id":-10,"bounding":[-350,610,139.435546875,268]},"outputNode":{"id":-20,"bounding":[1723.4786916118696,716.3650158766799,128,68]},"inputs":[{"id":"69d8b033-5601-446e-9634-f5cafbd373e2","name":"start_image","type":"IMAGE","linkIds":[186],"localized_name":"start_image","label":"start image","shape":7,"pos":[-234.564453125,634]},{"id":"88ae2af6-63c1-41be-90e8-6359f4d5f133","name":"text","type":"STRING","linkIds":[222],"label":"prompt","pos":[-234.564453125,654]},{"id":"fad9d346-653e-4be5-9e52-38cef6fa59f3","name":"width","type":"INT","linkIds":[223],"pos":[-234.564453125,674]},{"id":"a4f34897-8063-4613-a2eb-6c2503167eb1","name":"height","type":"INT","linkIds":[224],"pos":[-234.564453125,694]},{"id":"dc4d4472-cff7-41e0-9a4a-d118fcd4a21a","name":"length","type":"INT","linkIds":[225],"pos":[-234.564453125,714]},{"id":"f7317e79-4a52-460b-9d71-89ec450dc333","name":"unet_name","type":"COMBO","linkIds":[226],"label":"high_noise_unet","pos":[-234.564453125,734]},{"id":"7a470f86-503a-474f-9571-830c8eb99231","name":"lora_name","type":"COMBO","linkIds":[227],"label":"high_noise_lora","pos":[-234.564453125,754]},{"id":"1d88c531-f68e-41b9-95c5-16f944a55b7d","name":"unet_name_1","type":"COMBO","linkIds":[228],"label":"low_noise_unet","pos":[-234.564453125,774]},{"id":"67a79742-33e5-4c38-89d8-ecb021d067c8","name":"lora_name_1","type":"COMBO","linkIds":[229],"label":"low_noise_lora","pos":[-234.564453125,794]},{"id":"9d184b83-37c6-4891-bbdf-ffcdf5ab2016","name":"clip_name","type":"COMBO","linkIds":[230],"pos":[-234.564453125,814]},{"id":"24c568ec-aeb2-4c31-9f87-54ee9099d55f","name":"vae_name","type":"COMBO","linkIds":[231],"pos":[-234.564453125,834]}],"outputs":[{"id":"994c9c48-5f35-48ed-8c9d-0f2b21990cb6","name":"VIDEO","type":"VIDEO","linkIds":[221],"pos":[1747.4786916118696,740.3650158766799]}],"widgets":[],"nodes":[{"id":105,"type":"CLIPLoader","pos":[60,180],"size":[350,169.328125],"flags":{},"order":4,"mode":0,"inputs":[{"localized_name":"clip_name","name":"clip_name","type":"COMBO","widget":{"name":"clip_name"},"link":230}],"outputs":[{"localized_name":"clip","name":"CLIP","type":"CLIP","slot_index":0,"links":[178,181]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"CLIPLoader","models":[{"name":"umt5_xxl_fp8_e4m3fn_scaled.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors","directory":"text_encoders"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["umt5_xxl_fp8_e4m3fn_scaled.safetensors","wan","default"]},{"id":106,"type":"VAELoader","pos":[60,400],"size":[350,134.65625],"flags":{},"order":5,"mode":0,"inputs":[{"localized_name":"vae_name","name":"vae_name","type":"COMBO","widget":{"name":"vae_name"},"link":231}],"outputs":[{"localized_name":"vae","name":"VAE","type":"VAE","slot_index":0,"links":[176,185]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"VAELoader","models":[{"name":"wan_2.1_vae.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors","directory":"vae"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["wan_2.1_vae.safetensors"]},{"id":122,"type":"UNETLoader","pos":[60,-210],"size":[350,134.65625],"flags":{},"order":11,"mode":0,"inputs":[{"localized_name":"unet_name","name":"unet_name","type":"COMBO","widget":{"name":"unet_name"},"link":226}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","slot_index":0,"links":[194]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"UNETLoader","models":[{"name":"wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors","directory":"diffusion_models"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors","default"]},{"id":123,"type":"UNETLoader","pos":[60,-20],"size":[350,134.65625],"flags":{},"order":12,"mode":0,"inputs":[{"localized_name":"unet_name","name":"unet_name","type":"COMBO","widget":{"name":"unet_name"},"link":228}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","slot_index":0,"links":[196]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"UNETLoader","models":[{"name":"wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors","directory":"diffusion_models"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors","default"]},{"id":124,"type":"ModelSamplingSD3","pos":[750,-10],"size":[225,104],"flags":{"collapsed":false},"order":13,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":189}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","slot_index":0,"links":[192]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"ModelSamplingSD3","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":[5.000000000000001]},{"id":107,"type":"CLIPTextEncode","pos":[450,240],"size":[510,140],"flags":{},"order":6,"mode":0,"inputs":[{"localized_name":"clip","name":"clip","type":"CLIP","link":181},{"localized_name":"texto","name":"text","type":"STRING","widget":{"name":"text"},"link":222}],"outputs":[{"localized_name":"CONDICIONAMENTO","name":"CONDITIONING","type":"CONDITIONING","slot_index":0,"links":[183]}],"title":"CLIP Text Encode (Positive Prompt)","properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"CLIPTextEncode","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["A felt-style little eagle cashier greeting, waving, and smiling at the camera."],"color":"#232","bgcolor":"#353"},{"id":125,"type":"CLIPTextEncode","pos":[450,440],"size":[510,140],"flags":{},"order":14,"mode":0,"inputs":[{"localized_name":"clip","name":"clip","type":"CLIP","link":178}],"outputs":[{"localized_name":"CONDICIONAMENTO","name":"CONDITIONING","type":"CONDITIONING","slot_index":0,"links":[184]}],"title":"CLIP Text Encode (Negative Prompt)","properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"CLIPTextEncode","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"],"color":"#322","bgcolor":"#533"},{"id":126,"type":"LoraLoaderModelOnly","pos":[440,-230],"size":[280,166.65625],"flags":{},"order":15,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":194},{"localized_name":"nome_do_lora","name":"lora_name","type":"COMBO","widget":{"name":"lora_name"},"link":227}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","links":[190]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.49","Node name for S&R":"LoraLoaderModelOnly","models":[{"name":"wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors","directory":"loras"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",1.0000000000000002]},{"id":127,"type":"LoraLoaderModelOnly","pos":[450,-20],"size":[280,166.65625],"flags":{},"order":16,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":196},{"localized_name":"nome_do_lora","name":"lora_name","type":"COMBO","widget":{"name":"lora_name"},"link":229}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","links":[189]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.49","Node name for S&R":"LoraLoaderModelOnly","models":[{"name":"wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors","directory":"loras"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",1.0000000000000002]},{"id":109,"type":"ModelSamplingSD3","pos":[730,-230],"size":[225,104],"flags":{},"order":7,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":190}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","slot_index":0,"links":[195]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"ModelSamplingSD3","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":[5.000000000000001]},{"id":128,"type":"WanImageToVideo","pos":[550,760],"size":[350,296],"flags":{},"order":17,"mode":0,"inputs":[{"localized_name":"positivo","name":"positive","type":"CONDITIONING","link":183},{"localized_name":"negativo","name":"negative","type":"CONDITIONING","link":184},{"localized_name":"vae","name":"vae","type":"VAE","link":185},{"localized_name":"clip_vision_output","name":"clip_vision_output","shape":7,"type":"CLIP_VISION_OUTPUT","link":null},{"localized_name":"imagem_inicial","name":"start_image","shape":7,"type":"IMAGE","link":186},{"localized_name":"largura","name":"width","type":"INT","widget":{"name":"width"},"link":223},{"localized_name":"altura","name":"height","type":"INT","widget":{"name":"height"},"link":224},{"localized_name":"duração","name":"length","type":"INT","widget":{"name":"length"},"link":225}],"outputs":[{"localized_name":"positivo","name":"positive","type":"CONDITIONING","slot_index":0,"links":[168,172]},{"localized_name":"negativo","name":"negative","type":"CONDITIONING","slot_index":1,"links":[169,173]},{"localized_name":"latente","name":"latent","type":"LATENT","slot_index":2,"links":[174]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"WanImageToVideo","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":[720,720,81,1]},{"id":110,"type":"KSamplerAdvanced","pos":[1030,-230],"size":[310,340],"flags":{},"order":8,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":195},{"localized_name":"positivo","name":"positive","type":"CONDITIONING","link":172},{"localized_name":"negativo","name":"negative","type":"CONDITIONING","link":173},{"localized_name":"imagem_latente","name":"latent_image","type":"LATENT","link":174}],"outputs":[{"localized_name":"LATENT","name":"LATENT","type":"LATENT","links":[170]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"KSamplerAdvanced","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["enable",0,"randomize",4,1,"euler","simple",0,2,"enable"]},{"id":111,"type":"KSamplerAdvanced","pos":[1370,-230],"size":[310,340],"flags":{},"order":9,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":192},{"localized_name":"positivo","name":"positive","type":"CONDITIONING","link":168},{"localized_name":"negativo","name":"negative","type":"CONDITIONING","link":169},{"localized_name":"imagem_latente","name":"latent_image","type":"LATENT","link":170}],"outputs":[{"localized_name":"LATENT","name":"LATENT","type":"LATENT","links":[175]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"KSamplerAdvanced","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["disable",0,"fixed",4,1,"euler","simple",2,4,"disable"]},{"id":67,"type":"Note","pos":[1180,780],"size":[390,116],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"title":"Video Size","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["By default, we set the video to a smaller size for users with low VRAM. If you have enough VRAM, you can change the size"],"color":"#222","bgcolor":"#000"},{"id":112,"type":"MarkdownNote","pos":[-850,420],"size":[480,258.78125],"flags":{},"order":1,"mode":0,"inputs":[],"outputs":[],"title":"VRAM Usage","properties":{"ue_properties":{"version":"7.1","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":["## GPU:RTX4090D 24GB\n\n| Model            | Size |VRAM Usage | 1st Generation | 2nd Generation |\n|---------------------|-------|-----------|---------------|-----------------|\n| fp8_scaled               |640*640| 84%               | ≈  536s              | ≈ 513s                   |\n| fp8_scaled +  4steps LoRA  | 640*640  | 83%                | ≈ 97s               | ≈ 71s                   |"],"color":"#222","bgcolor":"#000"},{"id":66,"type":"MarkdownNote","pos":[-850,-380],"size":[480,755.5625],"flags":{},"order":2,"mode":0,"inputs":[],"outputs":[],"title":"Model Links","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["[Tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2\n)\n\n**Diffusion Model**\n- [wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors)\n- [wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors)\n\n**LoRA**\n- [wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors)\n- [wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors)\n\n**VAE**\n- [wan_2.1_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors)\n\n**Text Encoder**   \n- [umt5_xxl_fp8_e4m3fn_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors)\n\n\nFile save location\n\n```\nComfyUI/\n├───📂 models/\n│   ├───📂 diffusion_models/\n│   │   ├─── wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors\n│   │   └─── wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors\n│   ├───📂 loras/\n│   │   ├─── wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors\n│   │   └─── wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors\n│   ├───📂 text_encoders/\n│   │   └─── umt5_xxl_fp8_e4m3fn_scaled.safetensors \n│   └───📂 vae/\n│       └── wan_2.1_vae.safetensors\n```\n"],"color":"#222","bgcolor":"#000"},{"id":115,"type":"Note","pos":[30,-530],"size":[360,116],"flags":{},"order":3,"mode":0,"inputs":[],"outputs":[],"title":"About 4 Steps LoRA","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["Using the Wan2.2 Lighting LoRA will result in the loss of video dynamics, but it will reduce the generation time. This template provides two workflows, and you can enable one as needed."],"color":"#222","bgcolor":"#000"},{"id":117,"type":"CreateVideo","pos":[1370,620],"size":[270,128],"flags":{},"order":10,"mode":0,"inputs":[{"localized_name":"imagens","name":"images","type":"IMAGE","link":220},{"localized_name":"áudio","name":"audio","shape":7,"type":"AUDIO","link":null}],"outputs":[{"localized_name":"VÍDEO","name":"VIDEO","type":"VIDEO","links":[221]}],"properties":{"cnr_id":"comfy-core","ver":"0.11.0","Node name for S&R":"CreateVideo","ue_properties":{"widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":[16]},{"id":129,"type":"VAEDecode","pos":[1370,470],"size":[225,96],"flags":{},"order":18,"mode":0,"inputs":[{"localized_name":"amostras","name":"samples","type":"LATENT","link":175},{"localized_name":"vae","name":"vae","type":"VAE","link":176}],"outputs":[{"localized_name":"IMAGEM","name":"IMAGE","type":"IMAGE","slot_index":0,"links":[220]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"VAEDecode","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":[]}],"groups":[{"id":15,"title":"fp8_scaled +  4steps LoRA","bounding":[30,-350,1650,1510],"color":"#444","flags":{}},{"id":11,"title":"Step1 - Load models","bounding":[40,-310,380,910],"color":"#444","flags":{}},{"id":13,"title":"Step4 -  Prompt","bounding":[440,170,540,470],"color":"#444","flags":{}},{"id":14,"title":"Step3 - Video size & length","bounding":[440,660,530,470],"color":"#444","flags":{}},{"id":16,"title":"Lightx2v 4steps LoRA","bounding":[430,-310,550,460],"color":"#444","flags":{}}],"links":[{"id":189,"origin_id":127,"origin_slot":0,"target_id":124,"target_slot":0,"type":"MODEL"},{"id":181,"origin_id":105,"origin_slot":0,"target_id":107,"target_slot":0,"type":"CLIP"},{"id":178,"origin_id":105,"origin_slot":0,"target_id":125,"target_slot":0,"type":"CLIP"},{"id":194,"origin_id":122,"origin_slot":0,"target_id":126,"target_slot":0,"type":"MODEL"},{"id":196,"origin_id":123,"origin_slot":0,"target_id":127,"target_slot":0,"type":"MODEL"},{"id":190,"origin_id":126,"origin_slot":0,"target_id":109,"target_slot":0,"type":"MODEL"},{"id":183,"origin_id":107,"origin_slot":0,"target_id":128,"target_slot":0,"type":"CONDITIONING"},{"id":184,"origin_id":125,"origin_slot":0,"target_id":128,"target_slot":1,"type":"CONDITIONING"},{"id":185,"origin_id":106,"origin_slot":0,"target_id":128,"target_slot":2,"type":"VAE"},{"id":175,"origin_id":111,"origin_slot":0,"target_id":129,"target_slot":0,"type":"LATENT"},{"id":176,"origin_id":106,"origin_slot":0,"target_id":129,"target_slot":1,"type":"VAE"},{"id":195,"origin_id":109,"origin_slot":0,"target_id":110,"target_slot":0,"type":"MODEL"},{"id":172,"origin_id":128,"origin_slot":0,"target_id":110,"target_slot":1,"type":"CONDITIONING"},{"id":173,"origin_id":128,"origin_slot":1,"target_id":110,"target_slot":2,"type":"CONDITIONING"},{"id":174,"origin_id":128,"origin_slot":2,"target_id":110,"target_slot":3,"type":"LATENT"},{"id":192,"origin_id":124,"origin_slot":0,"target_id":111,"target_slot":0,"type":"MODEL"},{"id":168,"origin_id":128,"origin_slot":0,"target_id":111,"target_slot":1,"type":"CONDITIONING"},{"id":169,"origin_id":128,"origin_slot":1,"target_id":111,"target_slot":2,"type":"CONDITIONING"},{"id":170,"origin_id":110,"origin_slot":0,"target_id":111,"target_slot":3,"type":"LATENT"},{"id":186,"origin_id":-10,"origin_slot":0,"target_id":128,"target_slot":4,"type":"IMAGE"},{"id":220,"origin_id":129,"origin_slot":0,"target_id":117,"target_slot":0,"type":"IMAGE"},{"id":221,"origin_id":117,"origin_slot":0,"target_id":-20,"target_slot":0,"type":"VIDEO"},{"id":222,"origin_id":-10,"origin_slot":1,"target_id":107,"target_slot":1,"type":"STRING"},{"id":223,"origin_id":-10,"origin_slot":2,"target_id":128,"target_slot":5,"type":"INT"},{"id":224,"origin_id":-10,"origin_slot":3,"target_id":128,"target_slot":6,"type":"INT"},{"id":225,"origin_id":-10,"origin_slot":4,"target_id":128,"target_slot":7,"type":"INT"},{"id":226,"origin_id":-10,"origin_slot":5,"target_id":122,"target_slot":0,"type":"COMBO"},{"id":227,"origin_id":-10,"origin_slot":6,"target_id":126,"target_slot":1,"type":"COMBO"},{"id":228,"origin_id":-10,"origin_slot":7,"target_id":123,"target_slot":0,"type":"COMBO"},{"id":229,"origin_id":-10,"origin_slot":8,"target_id":127,"target_slot":1,"type":"COMBO"},{"id":230,"origin_id":-10,"origin_slot":9,"target_id":105,"target_slot":0,"type":"COMBO"},{"id":231,"origin_id":-10,"origin_slot":10,"target_id":106,"target_slot":0,"type":"COMBO"}],"extra":{"ds":{"scale":0.7926047855889957,"offset":[-30.12529469925767,690.3829855122884]},"frontendVersion":"1.37.11","workflowRendererVersion":"LG","VHS_latentpreview":false,"VHS_latentpreviewrate":0,"VHS_MetadataImage":true,"VHS_KeepIntermediate":true,"ue_links":[]}}]},"config":{},"extra":{"ds":{"scale":0.43519108280254776,"offset":[2176.23673618734,51.66666666666674]},"frontendVersion":"1.44.19","workflowRendererVersion":"LG","VHS_latentpreview":false,"VHS_latentpreviewrate":0,"VHS_MetadataImage":true,"VHS_KeepIntermediate":true,"ue_links":[]},"version":0.4}
WORKFLOW_EOF_WAN

cat > "$WF_DIR/2 - Gerar Imagem (Qwen).json" << 'WORKFLOW_EOF_GER'
{"id":"91f6bbe2-ed41-4fd6-bac7-71d5b5864ecb","revision":0,"last_node_id":263,"last_link_id":375,"nodes":[{"id":67,"type":"MarkdownNote","pos":[-390.00012263971144,39.99993780414644],"size":[570,977.15625],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"title":"Model links","properties":{"ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":["Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)\n\n## Model Links (for Local Users)\n\n**diffusion_models**\n\n- [qwen_image_2512_fp8_e4m3fn.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors)\n\n**text_encoders**\n\n- [qwen_2.5_vl_7b_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors)\n\n**vae**\n\n- [qwen_image_vae.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors)\n\n**loras**\n\n- [Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/resolve/main/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors)\n\n\n## Model Storage Location\n\n```\n📂 ComfyUI/\n├── 📂 models/\n│   ├── 📂 diffusion_models/\n│   │   └── qwen_image_2512_fp8_e4m3fn.safetensors\n│   ├── 📂 text_encoders/\n│   │   └── qwen_2.5_vl_7b_fp8_scaled.safetensors\n│   ├── 📂 vae/\n│   │   └── qwen_image_vae.safetensors\n│   └── 📂 loras/\n│       └── Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors\n```\n\n## Report Issue\n\nNote: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud will be updated after the stable release; nightly-supported models may not be included yet, please wait for the next stable release.\n\n- Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)\n- UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)\n- Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)\n"],"color":"#222","bgcolor":"#000"},{"id":60,"type":"SaveImage","pos":[699.9997512165859,49.9999399941413],"size":[520,550],"flags":{},"order":3,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":339}],"outputs":[],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"Node name for S&R":"SaveImage","ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":["Qwen-Image-2512"]},{"id":94,"type":"MarkdownNote","pos":[259.9997407046104,739.9998335603921],"size":[270,264.734375],"flags":{},"order":1,"mode":0,"inputs":[],"outputs":[],"title":"Aspect Ratios","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":["\n- 1:1 : 1328x1328\n- 16:9 : 1664x928\n- 9:16 : 928x1664\n- 4:3 : 1472x1104\n- 3:4 : 1104x1472\n- 3:2 : 1584x1056\n- 2:3 : 1056x1584"],"color":"#222","bgcolor":"#000"},{"id":238,"type":"c3c58f7e-2004-43ae-8b06-a956294bf7f4","pos":[259.9998265524083,39.999959266095914],"size":[400,560],"flags":{},"order":2,"mode":0,"inputs":[{"label":"enable_turbo_mode","name":"value","type":"BOOLEAN","widget":{"name":"value"},"link":null}],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[339]}],"properties":{"proxyWidgets":[["227","text"],["232","width"],["232","height"],["229","value"],["230","seed"],["226","unet_name"],["219","clip_name"],["220","vae_name"],["221","lora_name"]],"cnr_id":"comfy-core","ver":"0.16.4","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{"value":true},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":[]}],"links":[[339,238,0,60,0,"IMAGE"]],"groups":[],"definitions":{"subgraphs":[{"id":"c3c58f7e-2004-43ae-8b06-a956294bf7f4","version":1,"state":{"lastGroupId":7,"lastNodeId":263,"lastLinkId":375,"lastRerouteId":0},"revision":0,"config":{},"name":"Text to Image (Qwen-Image 2512)","inputNode":{"id":-10,"bounding":[-1080,1480,151.744140625,220]},"outputNode":{"id":-20,"bounding":[1550,1460,120,60]},"inputs":[{"id":"74d26021-a723-4a90-8e33-5d805a7e5deb","name":"text","type":"STRING","linkIds":[360],"pos":[-948.255859375,1500]},{"id":"b55f69e6-c7cb-4641-9e1f-2cb1c1942ed0","name":"width","type":"INT","linkIds":[361],"pos":[-948.255859375,1520]},{"id":"3e80284d-aba3-43cd-ab7b-ac2a619ef18c","name":"height","type":"INT","linkIds":[362],"pos":[-948.255859375,1540]},{"id":"de06e137-6cec-4cb3-a6bb-737022310a7b","name":"value","type":"BOOLEAN","linkIds":[370],"label":"enable_turbo_mode","pos":[-948.255859375,1560]},{"id":"9e500dee-a5b9-481b-ac46-64bab4bd3530","name":"seed","type":"INT","linkIds":[371],"pos":[-948.255859375,1580]},{"id":"33422b12-24e5-41c6-96fc-f9a8dadd5d94","name":"unet_name","type":"COMBO","linkIds":[372],"pos":[-948.255859375,1600]},{"id":"5cf753e4-236e-468e-9a06-6b8e238badc8","name":"clip_name","type":"COMBO","linkIds":[373],"pos":[-948.255859375,1620]},{"id":"790e775c-a639-4e5f-9007-e2ee6764dc5e","name":"vae_name","type":"COMBO","linkIds":[374],"pos":[-948.255859375,1640]},{"id":"3ebed521-3fe9-4922-ae26-2483e03d9305","name":"lora_name","type":"COMBO","linkIds":[375],"pos":[-948.255859375,1660]}],"outputs":[{"id":"7db1f9e2-40ee-4f9f-bb24-a0db7b96d45e","name":"IMAGE","type":"IMAGE","linkIds":[333],"localized_name":"IMAGE","pos":[1570,1480]}],"widgets":[],"nodes":[{"id":219,"type":"CLIPLoader","pos":[-590.0000306599279,1369.999917656194],"size":[280,150],"flags":{},"order":4,"mode":0,"inputs":[{"localized_name":"clip_name","name":"clip_name","type":"COMBO","widget":{"name":"clip_name"},"link":373}],"outputs":[{"localized_name":"clip","name":"CLIP","type":"CLIP","slot_index":0,"links":[314,315]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"CLIPLoader","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"models":[{"name":"qwen_2.5_vl_7b_fp8_scaled.safetensors","url":"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors","directory":"text_encoders"},{"name":"qwen_2.5_vl_7b_fp8_scaled.safetensors","url":"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors","directory":"text_encoders"}],"ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":["qwen_2.5_vl_7b_fp8_scaled.safetensors","qwen_image","default"]},{"id":220,"type":"VAELoader","pos":[-580,1620],"size":[280,106.65625],"flags":{},"order":5,"mode":0,"inputs":[{"localized_name":"vae_name","name":"vae_name","type":"COMBO","widget":{"name":"vae_name"},"link":374}],"outputs":[{"localized_name":"vae","name":"VAE","type":"VAE","slot_index":0,"links":[323]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"VAELoader","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"models":[{"name":"qwen_image_vae.safetensors","url":"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors","directory":"vae"},{"name":"qwen_image_vae.safetensors","url":"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors","directory":"vae"}],"ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":["qwen_image_vae.safetensors"]},{"id":222,"type":"ModelSamplingAuraFlow","pos":[1039.9997827525117,1109.9998931282516],"size":[250,104],"flags":{},"order":7,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":367}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","links":[316]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"ModelSamplingAuraFlow","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":[3.1000000000000005]},{"id":226,"type":"UNETLoader","pos":[-590.0000306599279,1139.9998353123879],"size":[280,130],"flags":{},"order":8,"mode":0,"inputs":[{"localized_name":"unet_name","name":"unet_name","type":"COMBO","widget":{"name":"unet_name"},"link":372}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","slot_index":0,"links":[312,324]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"UNETLoader","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"models":[{"name":"qwen_image_2512_fp8_e4m3fn.safetensors","url":"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors","directory":"diffusion_models"},{"name":"qwen_image_2512_fp8_e4m3fn.safetensors","url":"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors","directory":"diffusion_models"}],"ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":["qwen_image_2512_fp8_e4m3fn.safetensors","default"]},{"id":227,"type":"CLIPTextEncode","pos":[-200,1140],"size":[360,420],"flags":{},"order":9,"mode":0,"inputs":[{"localized_name":"clip","name":"clip","type":"CLIP","link":314},{"localized_name":"texto","name":"text","type":"STRING","widget":{"name":"text"},"link":360}],"outputs":[{"localized_name":"CONDICIONAMENTO","name":"CONDITIONING","type":"CONDITIONING","slot_index":0,"links":[317]}],"title":"CLIP Text Encode (Positive Prompt)","properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"CLIPTextEncode","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":["Urban alleyway at dusk. Tall, statuesque high-fashion model striding elegantly, mid distant full body shot from an angular perspective, cinematic/editorial with bold contrasts and tactile materials. They wear a rose-gold metallic trench coat with deconstructed elements over a black long-sleeved turtleneck with subtle texture; paired with forest-green pleated pants with raw hems and a soft texture. Long braided dark hair, medium complexion. They carry a vibrant yellow designer handbag with geometric details and a structured silhouette. White architectural sneakers with bold geometric cutouts. Bold, high-contrast, tactile, urban-grit meets high-fashion impact, extreme clarity, extreme layering, post-processing with transparent light-transmitting ultra-smooth high-definition film effect, removing all noise and grain, removing all blur, removing all vintage feel, removing all roughness, drawn with 32K pixel precision, unparalleled fine line drawing of every single detail, the entire image like a brand new photograph, photorealistic\n"],"color":"#232","bgcolor":"#353"},{"id":228,"type":"CLIPTextEncode","pos":[-200,1610],"size":[370,170],"flags":{},"order":10,"mode":0,"inputs":[{"localized_name":"clip","name":"clip","type":"CLIP","link":315}],"outputs":[{"localized_name":"CONDICIONAMENTO","name":"CONDITIONING","type":"CONDITIONING","slot_index":0,"links":[318]}],"title":"CLIP Text Encode (Negative Prompt)","properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"CLIPTextEncode","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":["低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲"],"color":"#322","bgcolor":"#533"},{"id":231,"type":"VAEDecode","pos":[1319.9997582245696,1119.9998738562972],"size":[225,96],"flags":{"collapsed":false},"order":13,"mode":0,"inputs":[{"localized_name":"amostras","name":"samples","type":"LATENT","link":322},{"localized_name":"vae","name":"vae","type":"VAE","link":323}],"outputs":[{"localized_name":"IMAGEM","name":"IMAGE","type":"IMAGE","slot_index":0,"links":[333]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"VAEDecode","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":[]},{"id":232,"type":"EmptySD3LatentImage","pos":[-550,1930],"size":[230,168],"flags":{},"order":14,"mode":0,"inputs":[{"localized_name":"largura","name":"width","type":"INT","widget":{"name":"width"},"link":361},{"localized_name":"altura","name":"height","type":"INT","widget":{"name":"height"},"link":362}],"outputs":[{"localized_name":"LATENT","name":"LATENT","type":"LATENT","links":[319]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"EmptySD3LatentImage","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":[1328,1328,1]},{"id":230,"type":"KSampler","pos":[1039.9997827525117,1249.9998808642806],"size":[250,341.3125],"flags":{},"order":12,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":316},{"localized_name":"positivo","name":"positive","type":"CONDITIONING","link":317},{"localized_name":"negativo","name":"negative","type":"CONDITIONING","link":318},{"localized_name":"imagem_latente","name":"latent_image","type":"LATENT","link":319},{"localized_name":"semente","name":"seed","type":"INT","widget":{"name":"seed"},"link":371},{"localized_name":"passos","name":"steps","type":"INT","widget":{"name":"steps"},"link":368},{"localized_name":"cfg","name":"cfg","type":"FLOAT","widget":{"name":"cfg"},"link":369}],"outputs":[{"localized_name":"LATENT","name":"LATENT","type":"LATENT","slot_index":0,"links":[322]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"KSampler","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"version":"7.7","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":[464857551335368,"randomize",50,4,"euler","simple",1]},{"id":224,"type":"PrimitiveInt","pos":[300,1150],"size":[230,104],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"localized_name":"INTEIRO","name":"INT","type":"INT","links":[355]}],"title":"Int (Steps)","properties":{"cnr_id":"comfy-core","ver":"0.12.3","Node name for S&R":"PrimitiveInt","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":[50,"fixed"]},{"id":223,"type":"PrimitiveFloat","pos":[300,1290],"size":[230,104],"flags":{},"order":1,"mode":0,"inputs":[],"outputs":[{"localized_name":"PONTO FLUTUANTE","name":"FLOAT","type":"FLOAT","links":[357]}],"title":"Float (CFG)","properties":{"cnr_id":"comfy-core","ver":"0.12.3","Node name for S&R":"PrimitiveFloat","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":[4]},{"id":229,"type":"PrimitiveBoolean","pos":[300,2060],"size":[230,100],"flags":{},"order":11,"mode":0,"inputs":[{"localized_name":"valor","name":"value","type":"BOOLEAN","widget":{"name":"value"},"link":370}],"outputs":[{"localized_name":"BOOLEANO","name":"BOOLEAN","type":"BOOLEAN","links":[326,358,359]}],"title":"Enable 4 Steps LoRA?","properties":{"cnr_id":"comfy-core","ver":"0.12.3","Node name for S&R":"PrimitiveBoolean","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":[true]},{"id":225,"type":"PrimitiveInt","pos":[290,1540],"size":[230,104],"flags":{},"order":2,"mode":0,"inputs":[],"outputs":[{"localized_name":"INTEIRO","name":"INT","type":"INT","links":[347,354]}],"title":"Int (Steps)","properties":{"cnr_id":"comfy-core","ver":"0.12.3","Node name for S&R":"PrimitiveInt","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":[4,"fixed"]},{"id":218,"type":"PrimitiveFloat","pos":[290,1670],"size":[230,104],"flags":{},"order":3,"mode":0,"inputs":[],"outputs":[{"localized_name":"PONTO FLUTUANTE","name":"FLOAT","type":"FLOAT","links":[356]}],"title":"Float (CFG)","properties":{"cnr_id":"comfy-core","ver":"0.12.3","Node name for S&R":"PrimitiveFloat","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":[1]},{"id":221,"type":"LoraLoaderModelOnly","pos":[240,1820],"size":[330,138.65625],"flags":{},"order":6,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":312},{"localized_name":"nome_do_lora","name":"lora_name","type":"COMBO","widget":{"name":"lora_name"},"link":375}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","links":[325]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.49","Node name for S&R":"LoraLoaderModelOnly","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"models":[{"name":"Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors","url":"https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/resolve/main/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors","directory":"loras"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":["Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",1]},{"id":233,"type":"ComfySwitchNode","pos":[710,1170],"size":[230,124],"flags":{},"order":15,"mode":0,"inputs":[{"localized_name":"falso","name":"on_false","type":"MODEL","link":324},{"localized_name":"verdadeiro","name":"on_true","type":"MODEL","link":325},{"localized_name":"alternar","name":"switch","type":"BOOLEAN","widget":{"name":"switch"},"link":326}],"outputs":[{"localized_name":"saída","name":"output","type":"MODEL","links":[367]}],"title":"Switch (model)","properties":{"cnr_id":"comfy-core","ver":"0.12.3","Node name for S&R":"ComfySwitchNode","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":[false]},{"id":240,"type":"ComfySwitchNode","pos":[710,1415],"size":[230,124],"flags":{},"order":16,"mode":0,"inputs":[{"localized_name":"falso","name":"on_false","type":"INT","link":355},{"localized_name":"verdadeiro","name":"on_true","type":"INT","link":354},{"localized_name":"alternar","name":"switch","type":"BOOLEAN","widget":{"name":"switch"},"link":359}],"outputs":[{"localized_name":"saída","name":"output","type":"INT","links":[368]}],"title":"Switch (steps)","properties":{"cnr_id":"comfy-core","ver":"0.12.3","Node name for S&R":"ComfySwitchNode","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":[false]},{"id":243,"type":"ComfySwitchNode","pos":[710,1660],"size":[230,124],"flags":{},"order":17,"mode":0,"inputs":[{"localized_name":"falso","name":"on_false","type":"FLOAT","link":357},{"localized_name":"verdadeiro","name":"on_true","type":"FLOAT","link":356},{"localized_name":"alternar","name":"switch","type":"BOOLEAN","widget":{"name":"switch"},"link":358}],"outputs":[{"localized_name":"saída","name":"output","type":"FLOAT","links":[369]}],"title":"Switch (cfg)","properties":{"cnr_id":"comfy-core","ver":"0.12.3","Node name for S&R":"ComfySwitchNode","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":[false]}],"groups":[{"id":1,"title":"Model","bounding":[-640,1060,390,740],"color":"#3f789e","flags":{}},{"id":2,"title":"Image size","bounding":[-630,1830,380,290],"color":"#3f789e","flags":{}},{"id":3,"title":"Prompt","bounding":[-220,1060,400,740],"color":"#3f789e","flags":{}},{"id":5,"title":"4-steps LoRA","bounding":[210,1460,410,550],"color":"#3f789e","flags":{}},{"id":6,"title":"Original Settings","bounding":[210,1060,420,370],"color":"#3f789e","flags":{}},{"id":7,"title":"Swtich","bounding":[660,1060,320,750],"color":"#3f789e","flags":{}}],"links":[{"id":312,"origin_id":226,"origin_slot":0,"target_id":221,"target_slot":0,"type":"MODEL"},{"id":314,"origin_id":219,"origin_slot":0,"target_id":227,"target_slot":0,"type":"CLIP"},{"id":315,"origin_id":219,"origin_slot":0,"target_id":228,"target_slot":0,"type":"CLIP"},{"id":322,"origin_id":230,"origin_slot":0,"target_id":231,"target_slot":0,"type":"LATENT"},{"id":323,"origin_id":220,"origin_slot":0,"target_id":231,"target_slot":1,"type":"VAE"},{"id":316,"origin_id":222,"origin_slot":0,"target_id":230,"target_slot":0,"type":"MODEL"},{"id":317,"origin_id":227,"origin_slot":0,"target_id":230,"target_slot":1,"type":"CONDITIONING"},{"id":318,"origin_id":228,"origin_slot":0,"target_id":230,"target_slot":2,"type":"CONDITIONING"},{"id":319,"origin_id":232,"origin_slot":0,"target_id":230,"target_slot":3,"type":"LATENT"},{"id":324,"origin_id":226,"origin_slot":0,"target_id":233,"target_slot":0,"type":"MODEL"},{"id":325,"origin_id":221,"origin_slot":0,"target_id":233,"target_slot":1,"type":"MODEL"},{"id":326,"origin_id":229,"origin_slot":0,"target_id":233,"target_slot":2,"type":"BOOLEAN"},{"id":333,"origin_id":231,"origin_slot":0,"target_id":-20,"target_slot":0,"type":"IMAGE"},{"id":347,"origin_id":225,"origin_slot":0,"target_id":230,"target_slot":4,"type":"INT"},{"id":354,"origin_id":225,"origin_slot":0,"target_id":240,"target_slot":1,"type":"INT"},{"id":355,"origin_id":224,"origin_slot":0,"target_id":240,"target_slot":0,"type":"INT"},{"id":356,"origin_id":218,"origin_slot":0,"target_id":243,"target_slot":1,"type":"FLOAT"},{"id":357,"origin_id":223,"origin_slot":0,"target_id":243,"target_slot":0,"type":"FLOAT"},{"id":358,"origin_id":229,"origin_slot":0,"target_id":243,"target_slot":2,"type":"BOOLEAN"},{"id":359,"origin_id":229,"origin_slot":0,"target_id":240,"target_slot":2,"type":"BOOLEAN"},{"id":360,"origin_id":-10,"origin_slot":0,"target_id":227,"target_slot":1,"type":"STRING"},{"id":361,"origin_id":-10,"origin_slot":1,"target_id":232,"target_slot":0,"type":"INT"},{"id":362,"origin_id":-10,"origin_slot":2,"target_id":232,"target_slot":1,"type":"INT"},{"id":367,"origin_id":233,"origin_slot":0,"target_id":222,"target_slot":0,"type":"MODEL"},{"id":368,"origin_id":240,"origin_slot":0,"target_id":230,"target_slot":5,"type":"INT"},{"id":369,"origin_id":243,"origin_slot":0,"target_id":230,"target_slot":6,"type":"FLOAT"},{"id":370,"origin_id":-10,"origin_slot":3,"target_id":229,"target_slot":0,"type":"BOOLEAN"},{"id":371,"origin_id":-10,"origin_slot":4,"target_id":230,"target_slot":4,"type":"INT"},{"id":372,"origin_id":-10,"origin_slot":5,"target_id":226,"target_slot":0,"type":"COMBO"},{"id":373,"origin_id":-10,"origin_slot":6,"target_id":219,"target_slot":0,"type":"COMBO"},{"id":374,"origin_id":-10,"origin_slot":7,"target_id":220,"target_slot":0,"type":"COMBO"},{"id":375,"origin_id":-10,"origin_slot":8,"target_id":221,"target_slot":1,"type":"COMBO"}],"extra":{"workflowRendererVersion":"Vue-corrected","ue_links":[]}}]},"config":{},"extra":{"ds":{"scale":1.1784176489701548,"offset":[911.217388386522,49.072450008049614]},"frontendVersion":"1.44.19","workflowRendererVersion":"Vue-corrected","ue_links":[],"links_added_by_ue":[],"VHS_latentpreview":false,"VHS_latentpreviewrate":0,"VHS_MetadataImage":true,"VHS_KeepIntermediate":true},"version":0.4}
WORKFLOW_EOF_GER

cat > "$WF_DIR/3 - Editar Rosto 1 e 2 pessoas (Qwen).json" << 'WORKFLOW_EOF_EDIT'
{"id":"91f6bbe2-ed41-4fd6-bac7-71d5b5864ecb","revision":0,"last_node_id":9999,"last_link_id":272,"nodes":[{"id":124,"type":"CFGNorm","pos":[-429.3699296052963,-2.8945202484982815],"size":[290,82],"flags":{},"order":17,"mode":0,"inputs":[{"name":"model","type":"MODEL","link":248}],"outputs":[{"name":"patched_model","type":"MODEL","links":[253]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.50","Node name for S&R":"CFGNorm","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{"strength":true}}},"widgets_values":[1,false]},{"id":131,"type":"ModelSamplingAuraFlow","pos":[-429.3699296052963,-112.89452024849828],"size":[290,60],"flags":{},"order":13,"mode":0,"inputs":[{"name":"model","type":"MODEL","link":251}],"outputs":[{"name":"MODEL","type":"MODEL","links":[248]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"ModelSamplingAuraFlow","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"widget_ue_connectable":{}},"widgets_values":[3]},{"id":134,"type":"MarkdownNote","pos":[-1359.3699296052964,827.1054797515017],"size":[290,140],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"properties":{},"widgets_values":["This node is to avoid bad output results caused by excessively large input image sizes. Because when we input one image, we use the size of that input image.\n\nThe **TextEncodeQwenImageEditPlus** will scale your input to 1024×104 pixels. We use the size of your first input image. This node is to avoid having an input image size that is too large (such as 3000×3000 pixels), which could bring bad results."],"color":"#432","bgcolor":"#653"},{"id":138,"type":"VAEDecode","pos":[-79.3699296052963,-102.89452024849828],"size":[210,46],"flags":{"collapsed":false},"order":19,"mode":0,"inputs":[{"name":"samples","type":"LATENT","link":249},{"name":"vae","type":"VAE","link":250}],"outputs":[{"name":"IMAGE","type":"IMAGE","slot_index":0,"links":[264,270]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"VAEDecode","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"widget_ue_connectable":{}},"widgets_values":[]},{"id":135,"type":"VAEEncode","pos":[-862.7844426869905,547.7855335231267],"size":[140,46],"flags":{},"order":14,"mode":0,"inputs":[{"name":"pixels","type":"IMAGE","link":256},{"name":"vae","type":"VAE","link":257}],"outputs":[{"name":"LATENT","type":"LATENT","links":[]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.50","Node name for S&R":"VAEEncode","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{}}},"widgets_values":[]},{"id":130,"type":"MarkdownNote","pos":[-247.0169761719179,673.3001277687473],"size":[300,160],"flags":{},"order":1,"mode":0,"inputs":[],"outputs":[],"title":"Note: KSampler settings","properties":{},"widgets_values":["You can test and find the best setting by yourself. The following table is for reference.\n\n| Model            | Steps | CFG |\n|---------------------|---------------|---------------|\n| Offical             | 50               | 4.0               \n| fp8_e4m3fn             | 20                | 2.5               |\n| fp8_e4m3fn + 4steps LoRA    | 4               | 1.0               |\n"],"color":"#432","bgcolor":"#653"},{"id":139,"type":"PreviewImage","pos":[332.3650411258537,-773.5846443567503],"size":[714.5306396484375,742.0773315429688],"flags":{},"order":21,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":270}],"outputs":[],"properties":{"cnr_id":"comfy-core","ver":"0.3.65","Node name for S&R":"PreviewImage"},"widgets_values":[]},{"id":129,"type":"MarkdownNote","pos":[-147.0480726900406,876.8977591573284],"size":[330,90],"flags":{},"order":2,"mode":0,"inputs":[],"outputs":[],"title":"Note: About image size","properties":{},"widgets_values":["You can use the latent from the **EmptySD3LatentImage** to replace **VAE Encode**, so you can customize the image size."],"color":"#432","bgcolor":"#653"},{"id":140,"type":"LoraLoaderModelOnly","pos":[-1418.0486387104302,50.355372669731224],"size":[310,82],"flags":{},"order":11,"mode":4,"inputs":[{"name":"model","type":"MODEL","link":252}],"outputs":[{"name":"MODEL","type":"MODEL","links":[251]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.50","Node name for S&R":"LoraLoaderModelOnly","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{"lora_name":true,"strength_model":true}}},"widgets_values":["Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",1]},{"id":60,"type":"SaveImage","pos":[287.7538654938035,52.942383018037035],"size":[962.8776041666667,623.0208333333334],"flags":{},"order":20,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":264}],"outputs":[],"properties":{"cnr_id":"comfy-core","ver":"0.3.65","Node name for S&R":"SaveImage"},"widgets_values":["ComfyUI"]},{"id":119,"type":"MarkdownNote","pos":[-2226.907467472407,-160.9675931139252],"size":[660,1115.2083333333335],"flags":{},"order":3,"mode":4,"inputs":[],"outputs":[],"title":"Note: how to use this workflow","properties":{},"widgets_values":["[Tutorial](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit) \n\n## About Qwen Image Edit model\n\n**Qwen Image Edit** is an image-editing model. It supports image editing tasks such as changing the background of an image, replacing something on it, changing a character's outfit, and so on.\n\nIt supports up to 3 images as input. So, you can use one image as the main image, and the others as style references, background, or something.\n\n**Qwen Image Edit 2511 (subgraph)** node is a subgraph, which is converted from the Qwen-Image workflow.\n\n## For Comfy Cloud Users\n\nIf you are using [cloud.comfy.org](https://cloud.comfy.org/):\n\n1. Since the workflow in the Cloud will have the input image ready, for the first run, you can just click the run button to see what happens. \n\n2. Update the text (prompt) in the **Qwen Image Edit 2511 (subgraph)** node. Try the following prompts:\n  - \"Remove the yellow balloon\"\n  - \"Change the balloon's color to blue.\"\n  - \"Replace the man with a child, keep the same oil-painting style.\" \n\n3. Try to upload your own image and then try to apply some changes to it.\n\n## For Local Users\n\nThis workflow requires high VRAM(More than 16GB). If you are using the ComfyUI for the first time, you can get started here: [Basic text-to-image workflow for beginners\n](https://docs.comfy.org/tutorials/basic/text-to-image)\n\n## Model links\n\nYou can find all the models on [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main) and  [Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI) \n\n**Diffusion model**\n\n- [qwen_image_edit_2511_fp8_e4m3fn.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8_e4m3fn.safetensors)\n\n**LoRA**\n\n- [Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors](https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2511/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors)\n\n**Text encoder**\n\n- [qwen_2.5_vl_7b_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors)\n\n**VAE**\n\n- [qwen_image_vae.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors)\n\nModel Storage Location\n\n```\n📂 ComfyUI/\n├── 📂 models/\n│   ├── 📂 diffusion_models/\n│   │   └── qwen_image_edit_2511_fp8_e4m3fn.safetensors\n│   ├── 📂 loras/\n│   │   └── Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors\n│   ├── 📂 vae/\n│   │   └── qwen_image_vae.safetensors\n│   └── 📂 text_encoders/\n│       └── qwen_2.5_vl_7b_fp8_scaled.safetensors\n```\n"],"color":"#222","bgcolor":"#000"},{"id":126,"type":"CLIPLoader","pos":[-1419.3699296052964,177.10547975150172],"size":[330,110],"flags":{},"order":4,"mode":0,"inputs":[],"outputs":[{"name":"CLIP","type":"CLIP","slot_index":0,"links":[258,261]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"CLIPLoader","models":[{"name":"qwen_2.5_vl_7b_fp8_scaled.safetensors","url":"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors","directory":"text_encoders"}],"enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"widget_ue_connectable":{}},"widgets_values":["qwen_2.5_vl_7b_fp8_scaled.safetensors","qwen_image","default"]},{"id":127,"type":"UNETLoader","pos":[-1419.3699296052973,-84.82312227186128],"size":[330,90],"flags":{},"order":5,"mode":0,"inputs":[],"outputs":[{"name":"MODEL","type":"MODEL","slot_index":0,"links":[252]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"UNETLoader","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"widget_ue_connectable":{}},"widgets_values":["qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors","fp8_e4m3fn"]},{"id":125,"type":"VAELoader","pos":[-1419.3699296052964,337.1054797515017],"size":[330,60],"flags":{},"order":6,"mode":0,"inputs":[],"outputs":[{"name":"VAE","type":"VAE","slot_index":0,"links":[250,257,259,262]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"VAELoader","models":[{"name":"qwen_image_vae.safetensors","url":"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors","directory":"vae"}],"enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"widget_ue_connectable":{}},"widgets_values":["qwen_image_vae.safetensors"]},{"id":136,"type":"ImageScaleToTotalPixels","pos":[-1402.3773577085658,540.5893855277511],"size":[327.96640625,106],"flags":{},"order":12,"mode":0,"inputs":[{"name":"image","type":"IMAGE","link":265}],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[256,260,263]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.50","Node name for S&R":"ImageScaleToTotalPixels","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{"upscale_method":true,"megapixels":true}}},"widgets_values":["lanczos",1.5,1]},{"id":133,"type":"EmptySD3LatentImage","pos":[-417.2560354532317,837.1750065077929],"size":[270,106],"flags":{},"order":7,"mode":0,"inputs":[],"outputs":[{"name":"LATENT","type":"LATENT","links":[272]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.59","Node name for S&R":"EmptySD3LatentImage"},"widgets_values":[2560,1440,1]},{"id":137,"type":"KSampler","pos":[-436.151057679435,129.94346089921282],"size":[300,474],"flags":{},"order":18,"mode":0,"inputs":[{"name":"model","type":"MODEL","link":253},{"name":"positive","type":"CONDITIONING","link":254},{"name":"negative","type":"CONDITIONING","link":255},{"name":"latent_image","type":"LATENT","link":272}],"outputs":[{"name":"LATENT","type":"LATENT","slot_index":0,"links":[249]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.48","Node name for S&R":"KSampler","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"widget_ue_connectable":{}},"widgets_values":[914804090745153,"randomize",4,1,"euler","simple",1]},{"id":128,"type":"TextEncodeQwenImageEditPlus","pos":[819.6981440632765,735.2023016614373],"size":[400,200],"flags":{},"order":15,"mode":0,"inputs":[{"name":"clip","type":"CLIP","link":258},{"name":"vae","shape":7,"type":"VAE","link":259},{"name":"image1","shape":7,"type":"IMAGE","link":260},{"name":"image2","shape":7,"type":"IMAGE","link":267},{"name":"image3","shape":7,"type":"IMAGE","link":269}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[255]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.59","Node name for S&R":"TextEncodeQwenImageEditPlus"},"widgets_values":[""],"color":"#223","bgcolor":"#335"},{"id":132,"type":"TextEncodeQwenImageEditPlus","pos":[372.7447103837626,736.8344021374161],"size":[400,200],"flags":{},"order":16,"mode":0,"inputs":[{"name":"clip","type":"CLIP","link":261},{"name":"vae","shape":7,"type":"VAE","link":262},{"name":"image1","shape":7,"type":"IMAGE","link":263},{"name":"image2","shape":7,"type":"IMAGE","link":266},{"name":"image3","shape":7,"type":"IMAGE","link":268}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[254]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.59","Node name for S&R":"TextEncodeQwenImageEditPlus"},"widgets_values":["mulher andando em um museu. camera visao de cima cinematografico"],"color":"#232","bgcolor":"#353"},{"id":78,"type":"LoadImage","pos":[1599.787012413809,113.18650565034997],"size":[580,441.14583333333337],"flags":{},"order":8,"mode":0,"inputs":[],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[265]},{"name":"MASK","type":"MASK","links":null}],"properties":{"cnr_id":"comfy-core","ver":"0.3.50","Node name for S&R":"LoadImage","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{"image":true,"upload":true}}},"widgets_values":["example.png","image"]},{"id":121,"type":"LoadImage","pos":[1292.4399372544672,417.5670810701894],"size":[265.408203125,325.98958333333337],"flags":{},"order":9,"mode":0,"inputs":[],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[268,269]},{"name":"MASK","type":"MASK","links":null}],"properties":{"cnr_id":"comfy-core","ver":"0.3.50","Node name for S&R":"LoadImage","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{"image":true,"upload":true}}},"widgets_values":["example.png","image"]},{"id":120,"type":"LoadImage","pos":[1280.8956181350848,55.08959928910269],"size":[280,325.98958333333337],"flags":{},"order":10,"mode":0,"inputs":[],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[266,267]},{"name":"MASK","type":"MASK","links":null}],"properties":{"cnr_id":"comfy-core","ver":"0.3.50","Node name for S&R":"LoadImage","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65,"ue_properties":{"widget_ue_connectable":{"image":true,"upload":true}}},"widgets_values":["example.png","image"]},{"id":9001,"type":"MarkdownNote","pos":[-520,-160],"size":[420,300],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"title":"Proporcoes (imagem)","properties":{},"widgets_values":["## Proporcoes (imagem)\n\nAjuste em **largura** e **altura** (EmptySD3LatentImage).\n\n- 1:1 : 1328x1328\n- 16:9 : 1664x928\n- 9:16 : 928x1664\n- 4:3 : 1472x1104\n- 3:4 : 1104x1472\n- 3:2 : 1584x1056\n- 2:3 : 1056x1584"],"color":"#432","bgcolor":"#653"}],"links":[[248,131,0,124,0,"MODEL"],[249,137,0,138,0,"LATENT"],[250,125,0,138,1,"VAE"],[251,140,0,131,0,"MODEL"],[252,127,0,140,0,"MODEL"],[253,124,0,137,0,"MODEL"],[254,132,0,137,1,"CONDITIONING"],[255,128,0,137,2,"CONDITIONING"],[256,136,0,135,0,"IMAGE"],[257,125,0,135,1,"VAE"],[258,126,0,128,0,"CLIP"],[259,125,0,128,1,"VAE"],[260,136,0,128,2,"IMAGE"],[261,126,0,132,0,"CLIP"],[262,125,0,132,1,"VAE"],[263,136,0,132,2,"IMAGE"],[264,138,0,60,0,"IMAGE"],[265,78,0,136,0,"IMAGE"],[266,120,0,132,3,"IMAGE"],[267,120,0,128,3,"IMAGE"],[268,121,0,132,4,"IMAGE"],[269,121,0,128,4,"IMAGE"],[270,138,0,139,0,"IMAGE"],[272,133,0,137,3,"LATENT"]],"groups":[{"id":1,"title":"More inputs (Select and Ctrl-B to enable)","bounding":[-968.6666343961194,665.3333301817496,590,409.6],"color":"#3f789e","flags":{}},{"id":1,"title":"Step1 - Load models","bounding":[-1439.3699296052964,-152.89452024849828,370,570],"color":"#3f789e","flags":{}},{"id":2,"title":"Step 2 - Scale Image","bounding":[-1439.3699296052964,447.1054797515017,970,550],"color":"#3f789e","flags":{}},{"id":3,"title":"Step 4 - Prompt","bounding":[-1039.3699296052964,-152.89452024849828,570,570],"color":"#3f789e","flags":{}},{"id":4,"title":"Step3 - Image Size","bounding":[-439.3699296052963,797.1054797515017,310,200],"color":"#3f789e","flags":{}}],"config":{},"extra":{"ds":{"scale":0.7082990135767593,"offset":[1455.7450360462149,-18.605598054109542]},"frontendVersion":"1.44.19","ue_links":[],"links_added_by_ue":[],"VHS_latentpreview":false,"VHS_latentpreviewrate":0,"VHS_MetadataImage":true,"VHS_KeepIntermediate":true,"workflowRendererVersion":"LG"},"version":0.4}
WORKFLOW_EOF_EDIT

cat > "$WF_DIR/4 - Gerar Imagens em Lote (Qwen).json" << 'WORKFLOW_EOF_BATCHIMG'
{"id":"batch-img-0001","revision":0,"last_node_id":9999,"last_link_id":12,"nodes":[{"id":1,"type":"CLIPLoader","pos":[20,40],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"CLIP","type":"CLIP","links":[4,5],"slot_index":0}],"properties":{"Node name for S&R":"CLIPLoader"},"widgets_values":["qwen_2.5_vl_7b_fp8_scaled.safetensors","qwen_image","default"]},{"id":2,"type":"UNETLoader","pos":[20,220],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"MODEL","type":"MODEL","links":[1],"slot_index":0}],"properties":{"Node name for S&R":"UNETLoader"},"widgets_values":["qwen_image_2512_fp8_e4m3fn.safetensors","default"]},{"id":3,"type":"LoraLoaderModelOnly","pos":[20,400],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[{"name":"model","type":"MODEL","link":1}],"outputs":[{"name":"MODEL","type":"MODEL","links":[2],"slot_index":0}],"properties":{"Node name for S&R":"LoraLoaderModelOnly"},"widgets_values":["Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",1.0]},{"id":4,"type":"ModelSamplingAuraFlow","pos":[20,580],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[{"name":"model","type":"MODEL","link":2}],"outputs":[{"name":"MODEL","type":"MODEL","links":[3],"slot_index":0}],"properties":{"Node name for S&R":"ModelSamplingAuraFlow"},"widgets_values":[3.1]},{"id":5,"type":"VAELoader","pos":[20,720],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"VAE","type":"VAE","links":[10],"slot_index":0}],"properties":{"Node name for S&R":"VAELoader"},"widgets_values":["qwen_image_vae.safetensors"]},{"id":6,"type":"Text Load Line From File","pos":[360,40],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"line_text","type":"STRING","links":[6],"slot_index":0},{"name":"dictionary","type":"DICT","links":[],"slot_index":0}],"properties":{"Node name for S&R":"Text Load Line From File"},"widgets_values":["/workspace/ComfyUI/input/prompts.txt","[filename]","bible_batch","automatic",0],"title":"Ler prompts (linha a linha)"},{"id":7,"type":"CLIPTextEncode","pos":[360,260],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[{"name":"clip","type":"CLIP","link":4},{"name":"text","type":"STRING","link":6,"widget":{"name":"text"}}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[7],"slot_index":0}],"properties":{"Node name for S&R":"CLIPTextEncode"},"widgets_values":[],"title":"Prompt POSITIVO (vem do arquivo)"},{"id":8,"type":"CLIPTextEncode","pos":[360,440],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[{"name":"clip","type":"CLIP","link":5}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[8],"slot_index":0}],"properties":{"Node name for S&R":"CLIPTextEncode"},"widgets_values":["低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲"],"title":"Prompt NEGATIVO (fixo)"},{"id":9,"type":"EmptySD3LatentImage","pos":[360,640],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"LATENT","type":"LATENT","links":[9],"slot_index":0}],"properties":{"Node name for S&R":"EmptySD3LatentImage"},"widgets_values":[1328,1328,1]},{"id":10,"type":"KSampler","pos":[720,260],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[{"name":"model","type":"MODEL","link":3},{"name":"positive","type":"CONDITIONING","link":7},{"name":"negative","type":"CONDITIONING","link":8},{"name":"latent_image","type":"LATENT","link":9}],"outputs":[{"name":"LATENT","type":"LATENT","links":[11],"slot_index":0}],"properties":{"Node name for S&R":"KSampler"},"widgets_values":[0,"randomize",4,1.0,"euler","simple",1.0]},{"id":11,"type":"VAEDecode","pos":[1040,260],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[{"name":"samples","type":"LATENT","link":11},{"name":"vae","type":"VAE","link":10}],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[12],"slot_index":0}],"properties":{"Node name for S&R":"VAEDecode"},"widgets_values":[]},{"id":12,"type":"SaveImage","pos":[1240,260],"size":[300,120],"flags":{},"order":0,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":12}],"outputs":[],"properties":{"Node name for S&R":"SaveImage"},"widgets_values":["lote"]},{"id":9001,"type":"MarkdownNote","pos":[1500,40],"size":[420,300],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"title":"Proporcoes (imagem)","properties":{},"widgets_values":["## Proporcoes (imagem)\n\nAjuste em **largura** e **altura** (EmptySD3LatentImage).\n\n- 1:1 : 1328x1328\n- 16:9 : 1664x928\n- 9:16 : 928x1664\n- 4:3 : 1472x1104\n- 3:4 : 1104x1472\n- 3:2 : 1584x1056\n- 2:3 : 1056x1584"],"color":"#432","bgcolor":"#653"},{"id":9002,"type":"MarkdownNote","pos":[1500,400],"size":[420,300],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"title":"Como usar o lote","properties":{},"widgets_values":["## Como usar o LOTE de imagens\n\n**1) Colocar os prompts**\nJupyterLab -> pasta `ComfyUI/input` -> arraste `prompts.txt` (1 prompt por linha, texto puro).\n\n**2) Rodar**\n- Troque o campo `label` (ex.: v2) para zerar a contagem\n- `batch count` = numero de prompts\n- Run uma vez\n\n**3) Onde ficam as imagens**\n`ComfyUI/output` -> `lote_00001_.png`, `lote_00002_.png` ...\n\n**4) Baixar tudo**\nBotao direito na pasta `output` -> *Download as Archive* (.zip)"],"color":"#432","bgcolor":"#653"}],"links":[[1,2,0,3,0,"MODEL"],[2,3,0,4,0,"MODEL"],[3,4,0,10,0,"MODEL"],[4,1,0,7,0,"CLIP"],[5,1,0,8,0,"CLIP"],[6,6,0,7,1,"STRING"],[7,7,0,10,1,"CONDITIONING"],[8,8,0,10,2,"CONDITIONING"],[9,9,0,10,3,"LATENT"],[10,5,0,11,1,"VAE"],[11,10,0,11,0,"LATENT"],[12,11,0,12,0,"IMAGE"]],"groups":[],"config":{},"extra":{},"version":0.4}
WORKFLOW_EOF_BATCHIMG

cat > "$WF_DIR/5 - Gerar Videos em Lote (Wan).json" << 'WORKFLOW_EOF_BATCHVID'
{"id":"ec7da562-7e21-4dac-a0d2-f4441e1efd3b","revision":0,"last_node_id":9999,"last_link_id":246,"nodes":[{"id":108,"type":"SaveVideo","pos":[1300,250],"size":[680,530],"flags":{},"order":4,"mode":0,"inputs":[{"name":"video","type":"VIDEO","link":246}],"outputs":[],"properties":{"cnr_id":"comfy-core","ver":"0.3.49","Node name for S&R":"SaveVideo","ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":["video/Wan2.2_image_to_video","auto","auto"]},{"id":118,"type":"MarkdownNote","pos":[-450,250],"size":[480,460],"flags":{},"order":1,"mode":0,"inputs":[],"outputs":[],"title":"Note: how to use this workflow","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.7","input_ue_unconnectable":{}}},"widgets_values":["**Wan2.2 Image to Video(Subgraph)** node is a subgraph, which is converted from the Wan2.2 image to video workflow.\n\nYou can find the tutorial and original workflow [here](https://docs.comfy.org/tutorials/video/wan/wan2_2).\n\n\nThis workflow requires high VRAM (More than 20GB). If you are running this workflow locally, please make sure you have enough VRAM.\n\n## For Comfy Cloud users\n\nIf you are using [cloud.comfy.org](https://cloud.comfy.org/):\n\n1. Since the workflow in the Cloud will have the input image ready, for the first run, you can just click the run button to see what happens. \n\n2. Try to upload your own image to the **Start Frame Image**, and then try to describe the video you want to generate in the **Wan2.2 Image to Video(Subgraph)**."],"color":"#222","bgcolor":"#000"},{"id":119,"type":"MarkdownNote","pos":[-450,700],"size":[480,1070],"flags":{},"order":2,"mode":0,"inputs":[],"outputs":[],"title":"For local ComfyUI users","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["[Tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2\n)\n\n## VRAM Usage\n\nThe following data was tested using an RTX 4090 24GB\n\n| Model            | Size |VRAM Usage | 1st Generation | 2nd Generation |\n|---------------------|-------|-----------|---------------|-----------------|\n| fp8_scaled               |640*640| 84%               | ≈  536s              | ≈ 513s                   |\n| fp8_scaled +  4steps LoRA  | 640*640  | 83%                | ≈ 97s               | ≈ 71s                   |\n\n**Diffusion Model**\n- [wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors)\n- [wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors)\n\n**LoRA**\n- [wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors)\n- [wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors)\n\n**VAE**\n- [wan_2.1_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors)\n\n**Text Encoder**   \n- [umt5_xxl_fp8_e4m3fn_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors)\n\n\nFile save location\n\n```\nComfyUI/\n├───📂 models/\n│   ├───📂 diffusion_models/\n│   │   ├─── wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors\n│   │   └─── wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors\n│   ├───📂 loras/\n│   │   ├─── wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors\n│   │   └─── wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors\n│   ├───📂 text_encoders/\n│   │   └─── umt5_xxl_fp8_e4m3fn_scaled.safetensors \n│   └───📂 vae/\n│       └── wan_2.1_vae.safetensors\n```\n\nIf you have any problems running this workflow, please report template-related issues via this link: [report the template issue here](https://github.com/Comfy-Org/workflow_templates/issues)"],"color":"#222","bgcolor":"#000"},{"id":130,"type":"d2ac71a3-c7a1-48fa-abea-0aa3f97d7bf2","pos":[800,250],"size":[400,720],"flags":{"collapsed":false},"order":3,"mode":0,"inputs":[{"label":"start image","name":"start_image","shape":7,"type":"IMAGE","link":245},{"label":"prompt","name":"text","type":"STRING","widget":{"name":"text"},"link":null},{"label":"high_noise_unet","name":"unet_name","type":"COMBO","widget":{"name":"unet_name"},"link":null},{"label":"high_noise_lora","name":"lora_name","type":"COMBO","widget":{"name":"lora_name"},"link":null},{"label":"low_noise_unet","name":"unet_name_1","type":"COMBO","widget":{"name":"unet_name_1"},"link":null},{"label":"low_noise_lora","name":"lora_name_1","type":"COMBO","widget":{"name":"lora_name_1"},"link":null}],"outputs":[{"name":"VIDEO","type":"VIDEO","links":[246]}],"properties":{"proxyWidgets":[["107","text"],["128","width"],["128","height"],["128","length"],["122","unet_name"],["126","lora_name"],["123","unet_name"],["127","lora_name"],["105","clip_name"],["106","vae_name"],["110","noise_seed"],["110","control_after_generate"]],"cnr_id":"comfy-core","ver":"0.11.0","ue_properties":{"widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":[]},{"id":200,"type":"Load Image Batch","pos":[90,250],"size":[400,260],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[245],"slot_index":0},{"name":"filename_text","type":"STRING","links":[],"slot_index":1}],"title":"Puxar imagens da pasta (em sequência)","properties":{"Node name for S&R":"Load Image Batch"},"widgets_values":["incremental_image",0,"fixed",0,"video_batch","/workspace/ComfyUI/output","lote*","false","true"]},{"id":9001,"type":"MarkdownNote","pos":[90,560],"size":[420,300],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"title":"Proporcoes (video)","properties":{},"widgets_values":["## Proporcoes (video) - Wan 2.2\n\nAjuste em **width** e **height** na caixa do Wan. Maior = mais lento.\n\n- 16:9 rapido: 832 x 480\n- 16:9 HD: 1280 x 720\n- 9:16 rapido: 480 x 832\n- 9:16 HD: 720 x 1280\n- 1:1: 720 x 720\n\nDica: use a mesma proporcao das imagens."],"color":"#432","bgcolor":"#653"},{"id":9002,"type":"MarkdownNote","pos":[90,920],"size":[420,300],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"title":"Como usar o lote","properties":{},"widgets_values":["## Como usar o LOTE de videos\n\n**Antes:** gere as imagens no '4 - Gerar Imagens em Lote'.\n\n**1) Rodar**\n- O bloco 'Puxar imagens da pasta' le de `ComfyUI/output` (padrao `lote*`)\n- Troque o `label` (ex.: v2) para comecar da 1a imagem\n- `batch count` = numero de imagens\n- Run uma vez (video e lento: ~1 min cada)\n\n**2) Onde ficam os videos**\n`ComfyUI/output/video` -> `Wan2.2_..._00001_.mp4` ...\n\n**3) Baixar tudo**\nBotao direito na pasta `output` -> *Download as Archive* (.zip)"],"color":"#432","bgcolor":"#653"}],"links":[[245,200,0,130,0,"IMAGE"],[246,130,0,108,0,"VIDEO"]],"groups":[],"definitions":{"subgraphs":[{"id":"d2ac71a3-c7a1-48fa-abea-0aa3f97d7bf2","version":1,"state":{"lastGroupId":16,"lastNodeId":130,"lastLinkId":246,"lastRerouteId":0},"revision":0,"config":{},"name":"Image to Video (Wan 2.2)","inputNode":{"id":-10,"bounding":[-350,610,139.435546875,268]},"outputNode":{"id":-20,"bounding":[1723.4786916118696,716.3650158766799,128,68]},"inputs":[{"id":"69d8b033-5601-446e-9634-f5cafbd373e2","name":"start_image","type":"IMAGE","linkIds":[186],"localized_name":"start_image","label":"start image","shape":7,"pos":[-234.564453125,634]},{"id":"88ae2af6-63c1-41be-90e8-6359f4d5f133","name":"text","type":"STRING","linkIds":[222],"label":"prompt","pos":[-234.564453125,654]},{"id":"fad9d346-653e-4be5-9e52-38cef6fa59f3","name":"width","type":"INT","linkIds":[223],"pos":[-234.564453125,674]},{"id":"a4f34897-8063-4613-a2eb-6c2503167eb1","name":"height","type":"INT","linkIds":[224],"pos":[-234.564453125,694]},{"id":"dc4d4472-cff7-41e0-9a4a-d118fcd4a21a","name":"length","type":"INT","linkIds":[225],"pos":[-234.564453125,714]},{"id":"f7317e79-4a52-460b-9d71-89ec450dc333","name":"unet_name","type":"COMBO","linkIds":[226],"label":"high_noise_unet","pos":[-234.564453125,734]},{"id":"7a470f86-503a-474f-9571-830c8eb99231","name":"lora_name","type":"COMBO","linkIds":[227],"label":"high_noise_lora","pos":[-234.564453125,754]},{"id":"1d88c531-f68e-41b9-95c5-16f944a55b7d","name":"unet_name_1","type":"COMBO","linkIds":[228],"label":"low_noise_unet","pos":[-234.564453125,774]},{"id":"67a79742-33e5-4c38-89d8-ecb021d067c8","name":"lora_name_1","type":"COMBO","linkIds":[229],"label":"low_noise_lora","pos":[-234.564453125,794]},{"id":"9d184b83-37c6-4891-bbdf-ffcdf5ab2016","name":"clip_name","type":"COMBO","linkIds":[230],"pos":[-234.564453125,814]},{"id":"24c568ec-aeb2-4c31-9f87-54ee9099d55f","name":"vae_name","type":"COMBO","linkIds":[231],"pos":[-234.564453125,834]}],"outputs":[{"id":"994c9c48-5f35-48ed-8c9d-0f2b21990cb6","name":"VIDEO","type":"VIDEO","linkIds":[221],"pos":[1747.4786916118696,740.3650158766799]}],"widgets":[],"nodes":[{"id":105,"type":"CLIPLoader","pos":[60,180],"size":[350,169.328125],"flags":{},"order":4,"mode":0,"inputs":[{"localized_name":"clip_name","name":"clip_name","type":"COMBO","widget":{"name":"clip_name"},"link":230}],"outputs":[{"localized_name":"clip","name":"CLIP","type":"CLIP","slot_index":0,"links":[178,181]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"CLIPLoader","models":[{"name":"umt5_xxl_fp8_e4m3fn_scaled.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors","directory":"text_encoders"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["umt5_xxl_fp8_e4m3fn_scaled.safetensors","wan","default"]},{"id":106,"type":"VAELoader","pos":[60,400],"size":[350,134.65625],"flags":{},"order":5,"mode":0,"inputs":[{"localized_name":"vae_name","name":"vae_name","type":"COMBO","widget":{"name":"vae_name"},"link":231}],"outputs":[{"localized_name":"vae","name":"VAE","type":"VAE","slot_index":0,"links":[176,185]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"VAELoader","models":[{"name":"wan_2.1_vae.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors","directory":"vae"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["wan_2.1_vae.safetensors"]},{"id":122,"type":"UNETLoader","pos":[60,-210],"size":[350,134.65625],"flags":{},"order":11,"mode":0,"inputs":[{"localized_name":"unet_name","name":"unet_name","type":"COMBO","widget":{"name":"unet_name"},"link":226}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","slot_index":0,"links":[194]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"UNETLoader","models":[{"name":"wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors","directory":"diffusion_models"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors","default"]},{"id":123,"type":"UNETLoader","pos":[60,-20],"size":[350,134.65625],"flags":{},"order":12,"mode":0,"inputs":[{"localized_name":"unet_name","name":"unet_name","type":"COMBO","widget":{"name":"unet_name"},"link":228}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","slot_index":0,"links":[196]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"UNETLoader","models":[{"name":"wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors","directory":"diffusion_models"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors","default"]},{"id":124,"type":"ModelSamplingSD3","pos":[750,-10],"size":[225,104],"flags":{"collapsed":false},"order":13,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":189}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","slot_index":0,"links":[192]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"ModelSamplingSD3","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":[5.000000000000001]},{"id":107,"type":"CLIPTextEncode","pos":[450,240],"size":[510,140],"flags":{},"order":6,"mode":0,"inputs":[{"localized_name":"clip","name":"clip","type":"CLIP","link":181},{"localized_name":"texto","name":"text","type":"STRING","widget":{"name":"text"},"link":222}],"outputs":[{"localized_name":"CONDICIONAMENTO","name":"CONDITIONING","type":"CONDITIONING","slot_index":0,"links":[183]}],"title":"CLIP Text Encode (Positive Prompt)","properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"CLIPTextEncode","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["Gentle, subtle natural motion; slow, smooth cinematic camera movement; the scene quietly comes to life; realistic, high quality, no abrupt changes."],"color":"#232","bgcolor":"#353"},{"id":125,"type":"CLIPTextEncode","pos":[450,440],"size":[510,140],"flags":{},"order":14,"mode":0,"inputs":[{"localized_name":"clip","name":"clip","type":"CLIP","link":178}],"outputs":[{"localized_name":"CONDICIONAMENTO","name":"CONDITIONING","type":"CONDITIONING","slot_index":0,"links":[184]}],"title":"CLIP Text Encode (Negative Prompt)","properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"CLIPTextEncode","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"],"color":"#322","bgcolor":"#533"},{"id":126,"type":"LoraLoaderModelOnly","pos":[440,-230],"size":[280,166.65625],"flags":{},"order":15,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":194},{"localized_name":"nome_do_lora","name":"lora_name","type":"COMBO","widget":{"name":"lora_name"},"link":227}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","links":[190]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.49","Node name for S&R":"LoraLoaderModelOnly","models":[{"name":"wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors","directory":"loras"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",1.0000000000000002]},{"id":127,"type":"LoraLoaderModelOnly","pos":[450,-20],"size":[280,166.65625],"flags":{},"order":16,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":196},{"localized_name":"nome_do_lora","name":"lora_name","type":"COMBO","widget":{"name":"lora_name"},"link":229}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","links":[189]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.49","Node name for S&R":"LoraLoaderModelOnly","models":[{"name":"wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors","url":"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors","directory":"loras"}],"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",1.0000000000000002]},{"id":109,"type":"ModelSamplingSD3","pos":[730,-230],"size":[225,104],"flags":{},"order":7,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":190}],"outputs":[{"localized_name":"MODELO","name":"MODEL","type":"MODEL","slot_index":0,"links":[195]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"ModelSamplingSD3","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":[5.000000000000001]},{"id":128,"type":"WanImageToVideo","pos":[550,760],"size":[350,296],"flags":{},"order":17,"mode":0,"inputs":[{"localized_name":"positivo","name":"positive","type":"CONDITIONING","link":183},{"localized_name":"negativo","name":"negative","type":"CONDITIONING","link":184},{"localized_name":"vae","name":"vae","type":"VAE","link":185},{"localized_name":"clip_vision_output","name":"clip_vision_output","shape":7,"type":"CLIP_VISION_OUTPUT","link":null},{"localized_name":"imagem_inicial","name":"start_image","shape":7,"type":"IMAGE","link":186},{"localized_name":"largura","name":"width","type":"INT","widget":{"name":"width"},"link":223},{"localized_name":"altura","name":"height","type":"INT","widget":{"name":"height"},"link":224},{"localized_name":"duração","name":"length","type":"INT","widget":{"name":"length"},"link":225}],"outputs":[{"localized_name":"positivo","name":"positive","type":"CONDITIONING","slot_index":0,"links":[168,172]},{"localized_name":"negativo","name":"negative","type":"CONDITIONING","slot_index":1,"links":[169,173]},{"localized_name":"latente","name":"latent","type":"LATENT","slot_index":2,"links":[174]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"WanImageToVideo","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":[720,720,81,1]},{"id":110,"type":"KSamplerAdvanced","pos":[1030,-230],"size":[310,340],"flags":{},"order":8,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":195},{"localized_name":"positivo","name":"positive","type":"CONDITIONING","link":172},{"localized_name":"negativo","name":"negative","type":"CONDITIONING","link":173},{"localized_name":"imagem_latente","name":"latent_image","type":"LATENT","link":174}],"outputs":[{"localized_name":"LATENT","name":"LATENT","type":"LATENT","links":[170]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"KSamplerAdvanced","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["enable",0,"randomize",4,1,"euler","simple",0,2,"enable"]},{"id":111,"type":"KSamplerAdvanced","pos":[1370,-230],"size":[310,340],"flags":{},"order":9,"mode":0,"inputs":[{"localized_name":"modelo","name":"model","type":"MODEL","link":192},{"localized_name":"positivo","name":"positive","type":"CONDITIONING","link":168},{"localized_name":"negativo","name":"negative","type":"CONDITIONING","link":169},{"localized_name":"imagem_latente","name":"latent_image","type":"LATENT","link":170}],"outputs":[{"localized_name":"LATENT","name":"LATENT","type":"LATENT","links":[175]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"KSamplerAdvanced","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["disable",0,"fixed",4,1,"euler","simple",2,4,"disable"]},{"id":67,"type":"Note","pos":[1180,780],"size":[390,116],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[],"title":"Video Size","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["By default, we set the video to a smaller size for users with low VRAM. If you have enough VRAM, you can change the size"],"color":"#222","bgcolor":"#000"},{"id":112,"type":"MarkdownNote","pos":[-850,420],"size":[480,258.78125],"flags":{},"order":1,"mode":0,"inputs":[],"outputs":[],"title":"VRAM Usage","properties":{"ue_properties":{"version":"7.1","widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":["## GPU:RTX4090D 24GB\n\n| Model            | Size |VRAM Usage | 1st Generation | 2nd Generation |\n|---------------------|-------|-----------|---------------|-----------------|\n| fp8_scaled               |640*640| 84%               | ≈  536s              | ≈ 513s                   |\n| fp8_scaled +  4steps LoRA  | 640*640  | 83%                | ≈ 97s               | ≈ 71s                   |"],"color":"#222","bgcolor":"#000"},{"id":66,"type":"MarkdownNote","pos":[-850,-380],"size":[480,755.5625],"flags":{},"order":2,"mode":0,"inputs":[],"outputs":[],"title":"Model Links","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["[Tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2\n)\n\n**Diffusion Model**\n- [wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors)\n- [wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors)\n\n**LoRA**\n- [wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors)\n- [wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors)\n\n**VAE**\n- [wan_2.1_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors)\n\n**Text Encoder**   \n- [umt5_xxl_fp8_e4m3fn_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors)\n\n\nFile save location\n\n```\nComfyUI/\n├───📂 models/\n│   ├───📂 diffusion_models/\n│   │   ├─── wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors\n│   │   └─── wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors\n│   ├───📂 loras/\n│   │   ├─── wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors\n│   │   └─── wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors\n│   ├───📂 text_encoders/\n│   │   └─── umt5_xxl_fp8_e4m3fn_scaled.safetensors \n│   └───📂 vae/\n│       └── wan_2.1_vae.safetensors\n```\n"],"color":"#222","bgcolor":"#000"},{"id":115,"type":"Note","pos":[30,-530],"size":[360,116],"flags":{},"order":3,"mode":0,"inputs":[],"outputs":[],"title":"About 4 Steps LoRA","properties":{"ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":["Using the Wan2.2 Lighting LoRA will result in the loss of video dynamics, but it will reduce the generation time. This template provides two workflows, and you can enable one as needed."],"color":"#222","bgcolor":"#000"},{"id":117,"type":"CreateVideo","pos":[1370,620],"size":[270,128],"flags":{},"order":10,"mode":0,"inputs":[{"localized_name":"imagens","name":"images","type":"IMAGE","link":220},{"localized_name":"áudio","name":"audio","shape":7,"type":"AUDIO","link":null}],"outputs":[{"localized_name":"VÍDEO","name":"VIDEO","type":"VIDEO","links":[221]}],"properties":{"cnr_id":"comfy-core","ver":"0.11.0","Node name for S&R":"CreateVideo","ue_properties":{"widget_ue_connectable":{},"input_ue_unconnectable":{}}},"widgets_values":[16]},{"id":129,"type":"VAEDecode","pos":[1370,470],"size":[225,96],"flags":{},"order":18,"mode":0,"inputs":[{"localized_name":"amostras","name":"samples","type":"LATENT","link":175},{"localized_name":"vae","name":"vae","type":"VAE","link":176}],"outputs":[{"localized_name":"IMAGEM","name":"IMAGE","type":"IMAGE","slot_index":0,"links":[220]}],"properties":{"cnr_id":"comfy-core","ver":"0.3.45","Node name for S&R":"VAEDecode","ue_properties":{"widget_ue_connectable":{},"version":"7.1","input_ue_unconnectable":{}}},"widgets_values":[]}],"groups":[{"id":15,"title":"fp8_scaled +  4steps LoRA","bounding":[30,-350,1650,1510],"color":"#444","flags":{}},{"id":11,"title":"Step1 - Load models","bounding":[40,-310,380,910],"color":"#444","flags":{}},{"id":13,"title":"Step4 -  Prompt","bounding":[440,170,540,470],"color":"#444","flags":{}},{"id":14,"title":"Step3 - Video size & length","bounding":[440,660,530,470],"color":"#444","flags":{}},{"id":16,"title":"Lightx2v 4steps LoRA","bounding":[430,-310,550,460],"color":"#444","flags":{}}],"links":[{"id":189,"origin_id":127,"origin_slot":0,"target_id":124,"target_slot":0,"type":"MODEL"},{"id":181,"origin_id":105,"origin_slot":0,"target_id":107,"target_slot":0,"type":"CLIP"},{"id":178,"origin_id":105,"origin_slot":0,"target_id":125,"target_slot":0,"type":"CLIP"},{"id":194,"origin_id":122,"origin_slot":0,"target_id":126,"target_slot":0,"type":"MODEL"},{"id":196,"origin_id":123,"origin_slot":0,"target_id":127,"target_slot":0,"type":"MODEL"},{"id":190,"origin_id":126,"origin_slot":0,"target_id":109,"target_slot":0,"type":"MODEL"},{"id":183,"origin_id":107,"origin_slot":0,"target_id":128,"target_slot":0,"type":"CONDITIONING"},{"id":184,"origin_id":125,"origin_slot":0,"target_id":128,"target_slot":1,"type":"CONDITIONING"},{"id":185,"origin_id":106,"origin_slot":0,"target_id":128,"target_slot":2,"type":"VAE"},{"id":175,"origin_id":111,"origin_slot":0,"target_id":129,"target_slot":0,"type":"LATENT"},{"id":176,"origin_id":106,"origin_slot":0,"target_id":129,"target_slot":1,"type":"VAE"},{"id":195,"origin_id":109,"origin_slot":0,"target_id":110,"target_slot":0,"type":"MODEL"},{"id":172,"origin_id":128,"origin_slot":0,"target_id":110,"target_slot":1,"type":"CONDITIONING"},{"id":173,"origin_id":128,"origin_slot":1,"target_id":110,"target_slot":2,"type":"CONDITIONING"},{"id":174,"origin_id":128,"origin_slot":2,"target_id":110,"target_slot":3,"type":"LATENT"},{"id":192,"origin_id":124,"origin_slot":0,"target_id":111,"target_slot":0,"type":"MODEL"},{"id":168,"origin_id":128,"origin_slot":0,"target_id":111,"target_slot":1,"type":"CONDITIONING"},{"id":169,"origin_id":128,"origin_slot":1,"target_id":111,"target_slot":2,"type":"CONDITIONING"},{"id":170,"origin_id":110,"origin_slot":0,"target_id":111,"target_slot":3,"type":"LATENT"},{"id":186,"origin_id":-10,"origin_slot":0,"target_id":128,"target_slot":4,"type":"IMAGE"},{"id":220,"origin_id":129,"origin_slot":0,"target_id":117,"target_slot":0,"type":"IMAGE"},{"id":221,"origin_id":117,"origin_slot":0,"target_id":-20,"target_slot":0,"type":"VIDEO"},{"id":222,"origin_id":-10,"origin_slot":1,"target_id":107,"target_slot":1,"type":"STRING"},{"id":223,"origin_id":-10,"origin_slot":2,"target_id":128,"target_slot":5,"type":"INT"},{"id":224,"origin_id":-10,"origin_slot":3,"target_id":128,"target_slot":6,"type":"INT"},{"id":225,"origin_id":-10,"origin_slot":4,"target_id":128,"target_slot":7,"type":"INT"},{"id":226,"origin_id":-10,"origin_slot":5,"target_id":122,"target_slot":0,"type":"COMBO"},{"id":227,"origin_id":-10,"origin_slot":6,"target_id":126,"target_slot":1,"type":"COMBO"},{"id":228,"origin_id":-10,"origin_slot":7,"target_id":123,"target_slot":0,"type":"COMBO"},{"id":229,"origin_id":-10,"origin_slot":8,"target_id":127,"target_slot":1,"type":"COMBO"},{"id":230,"origin_id":-10,"origin_slot":9,"target_id":105,"target_slot":0,"type":"COMBO"},{"id":231,"origin_id":-10,"origin_slot":10,"target_id":106,"target_slot":0,"type":"COMBO"}],"extra":{"ds":{"scale":0.7926047855889957,"offset":[-30.12529469925767,690.3829855122884]},"frontendVersion":"1.37.11","workflowRendererVersion":"LG","VHS_latentpreview":false,"VHS_latentpreviewrate":0,"VHS_MetadataImage":true,"VHS_KeepIntermediate":true,"ue_links":[]}}]},"config":{},"extra":{"ds":{"scale":0.43519108280254776,"offset":[2176.23673618734,51.66666666666674]},"frontendVersion":"1.44.19","workflowRendererVersion":"LG","VHS_latentpreview":false,"VHS_latentpreviewrate":0,"VHS_MetadataImage":true,"VHS_KeepIntermediate":true,"ue_links":[]},"version":0.4}
WORKFLOW_EOF_BATCHVID



# ----------------------------------------------------------------------------
# (4/7) Chatterbox-TTS-Extended (multilingual)
# ----------------------------------------------------------------------------
if [ ! -d /workspace/Chatterbox-TTS-Extended ]; then
    echo "[boot] (4/7) Instalando Chatterbox-TTS-Extended..."
    cd /workspace
    git clone --depth 1 https://github.com/petermg/Chatterbox-TTS-Extended
    cd Chatterbox-TTS-Extended
    python3.10 -m venv venv
    source venv/bin/activate
    pip install -q -U pip setuptools wheel

    # Torch Blackwell-compatible (sm_120) PRIMEIRO
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

    # requirements sem torch/xformers
    grep -viE '^(torch|torchaudio|torchvision|xformers)([=<>!]|$)' requirements.txt > req_filt.txt
    pip install -r req_filt.txt

    # chatterbox-tts traz o multilingual (mtl_tts) mas forca downgrade do torch
    pip install chatterbox-tts

    # Limpa libs CUDA 12.4 orfas e reinstala torch 12.8 (|| true evita abortar)
    pip uninstall -y \
        nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 \
        nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
        nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
        nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 triton \
        torch torchaudio || true
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

    pip install torchcodec
    python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

    # ---- PATCH t3.py: desabilitar alignment_stream_analyzer ----
    # Bug em chatterbox-tts 0.1.7 quando usado via ChatterboxMultilingualTTS:
    # ~95% dos chunks crasham em textos longos (audiolivros). O proprio codigo
    # da biblioteca ja tem o caminho "analyzer desligado" implementado; este
    # patch apenas forca esse caminho. Ver PATCH_02_alignment_stream_analyzer.md
    # Trava de seguranca: se a Resemble AI mudar a linha em versao futura,
    # o assert ABORTA o boot com erro claro (em vez de patchear errado).
    echo "[boot] Aplicando patch t3.py (alignment_stream_analyzer)..."
    python << 'PYPATCH'
import ast, shutil
import chatterbox.models.t3.t3 as _m
path = _m.__file__
shutil.copy(path, path + '.original.bak')
with open(path) as f:
    src = f.read()
old = 'if self.patched_model.alignment_stream_analyzer is not None:'
new = 'if False:  # PATCHED: alignment_stream_analyzer disabled (bug in multilingual t3 inference)'
assert old in src, f"Marker nao encontrado em {path} - versao da chatterbox-tts mudou? Revisar PATCH_02."
src = src.replace(old, new, 1)
with open(path, 'w') as f:
    f.write(src)
ast.parse(open(path).read())  # confirma sintaxe Python ok pos-patch
print("  Patch t3.py aplicado e validado.")
PYPATCH
    # smoke test: importar ChatterboxMultilingualTTS apos o patch
    python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; print('  Import OK pos-patch.')"

    # ---- (5/7) Gravar o Chatter.py JA PATCHEADO (multilingual) ----
    echo "[boot] (5/7) Gravando Chatter.py patcheado (multilingual)..."
    cat > /workspace/Chatterbox-TTS-Extended/Chatter.py << 'CHATTER_PATCHED_EOF_MARKER_9f3a'
import random
import numpy as np
import torch
import os
import re
import datetime
import torchaudio
import gradio as gr
import spaces
import subprocess
from pydub import AudioSegment
import ffmpeg
import librosa
import string
import difflib
import time
import gc
# PATCHED: pip-installed chatterbox + multilingual
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from concurrent.futures import ThreadPoolExecutor, as_completed
import whisper
import nltk
from nltk.tokenize import sent_tokenize
from faster_whisper import WhisperModel as FasterWhisperModel
import json
import csv
import argparse
import soundfile as sf
import inspect, traceback
from chatterbox.vc import ChatterboxVC
try:
    import pyrnnoise
    _PYRNNOISE_AVAILABLE = True
except Exception:
    _PYRNNOISE_AVAILABLE = False


SETTINGS_PATH = "settings.json"

# === PATCH multilingual ===
SELECTED_LANGUAGE = "es"  # default; alterado pelo dropdown na UI
SUPPORTED_LANG_CHOICES = [
    ("Spanish (es)", "es"), ("English (en)", "en"), ("Portuguese (pt)", "pt"),
    ("Italian (it)", "it"), ("French (fr)", "fr"), ("German (de)", "de"),
    ("Dutch (nl)", "nl"), ("Polish (pl)", "pl"), ("Russian (ru)", "ru"),
    ("Greek (el)", "el"), ("Turkish (tr)", "tr"), ("Arabic (ar)", "ar"),
    ("Hebrew (he)", "he"), ("Hindi (hi)", "hi"), ("Japanese (ja)", "ja"),
    ("Korean (ko)", "ko"), ("Chinese (zh)", "zh"), ("Danish (da)", "da"),
    ("Finnish (fi)", "fi"), ("Norwegian (no)", "no"), ("Swedish (sv)", "sv"),
    ("Swahili (sw)", "sw"), ("Malay (ms)", "ms"),
]
def set_language(lang_code):
    global SELECTED_LANGUAGE
    SELECTED_LANGUAGE = lang_code
    print(f"[PATCH] SELECTED_LANGUAGE = {lang_code}")
    return None

#THIS IS THE START
def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                d = default_settings()
                d.update(data)
                return d
            except Exception:
                return default_settings()
    else:
        return default_settings()

def save_settings(mapping):
    # Ensure "whisper_model_dropdown" is always saved as the label, not code
    whisper_model_map = {
        "tiny (~1 GB VRAM OpenAI / ~0.5 GB faster-whisper)": "tiny",
        "base (~1.2–2 GB OpenAI / ~0.7–1 GB faster-whisper)": "base",
        "small (~2–3 GB OpenAI / ~1.2–1.7 GB faster-whisper)": "small",
        "medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)": "medium",
        "large (~10–13 GB OpenAI / ~4.5–6.5 GB faster-whisper)": "large"
    }
    v = mapping.get("whisper_model_dropdown", "")
    if v not in whisper_model_map:
        label = next((k for k, code in whisper_model_map.items() if code == v), v)
        mapping["whisper_model_dropdown"] = label

    # --- Add the extra "per-generation" fields for full compatibility ---
    if "input_basename" not in mapping:
        mapping["input_basename"] = "text_input_"
    if "audio_prompt_path_input" not in mapping:
        mapping["audio_prompt_path_input"] = None
    if "generation_time" not in mapping:
        import datetime
        mapping["generation_time"] = datetime.datetime.now().isoformat()
    if "output_audio_files" not in mapping:
        mapping["output_audio_files"] = []

    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
        
def save_settings_csv(settings_dict, output_audio_files, csv_path):
    """
    Save a dict of settings and a list of output audio files to a one-row CSV.
    """
    # Prepare a flattened settings dict for CSV
    flat_settings = {}
    for k, v in settings_dict.items():
        if isinstance(v, (list, tuple)):
            flat_settings[k] = '|'.join(map(str, v))
        else:
            flat_settings[k] = v
    flat_settings['output_audio_files'] = '|'.join(output_audio_files)
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_settings.keys()))
        writer.writeheader()
        writer.writerow(flat_settings)

def save_settings_json(settings_dict, json_path):
    """
    Save the settings dict as a JSON file.
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=2, ensure_ascii=False)
        
        
# === VC TAB (NEW) ===

VC_MODEL = None  # Reuse the global DEVICE defined earlier

def get_or_load_vc_model():
    global VC_MODEL
    if VC_MODEL is None:
        VC_MODEL = ChatterboxVC.from_pretrained(DEVICE)
    return VC_MODEL



def voice_conversion(input_audio_path, target_voice_audio_path, chunk_sec=60, overlap_sec=0.1, disable_watermark=True, pitch_shift=0):
    vc_model = get_or_load_vc_model()
    model_sr = vc_model.sr

    wav, sr = sf.read(input_audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != model_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=model_sr)
        sr = model_sr

    total_sec = len(wav) / model_sr

    if total_sec <= chunk_sec:
        wav_out = vc_model.generate(
            input_audio_path,
            target_voice_path=target_voice_audio_path,
            apply_watermark=not disable_watermark,
            pitch_shift=pitch_shift
        )
        out_wav = wav_out.squeeze(0).numpy()
        return model_sr, out_wav

    # chunking logic for long files
    chunk_samples = int(chunk_sec * model_sr)
    overlap_samples = int(overlap_sec * model_sr)
    step_samples = chunk_samples - overlap_samples

    out_chunks = []
    for start in range(0, len(wav), step_samples):
        end = min(start + chunk_samples, len(wav))
        chunk = wav[start:end]
        temp_chunk_path = f"temp_vc_chunk_{start}_{end}.wav"
        sf.write(temp_chunk_path, chunk, model_sr)
        out_chunk = vc_model.generate(
            temp_chunk_path,
            target_voice_path=target_voice_audio_path,
            apply_watermark=not disable_watermark,
            pitch_shift=pitch_shift
        )
        out_chunk_np = out_chunk.squeeze(0).numpy()
        out_chunks.append(out_chunk_np)
        os.remove(temp_chunk_path)

    # Crossfade join as before...
    result = out_chunks[0]
    for i in range(1, len(out_chunks)):
        overlap = min(overlap_samples, len(out_chunks[i]), len(result))
        if overlap > 0:
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            result[-overlap:] = result[-overlap:] * fade_out + out_chunks[i][:overlap] * fade_in
            result = np.concatenate([result, out_chunks[i][overlap:]])
        else:
            result = np.concatenate([result, out_chunks[i]])
    return model_sr, result

def default_settings():
    return {
        "text_input": """Three Rings for the Elven-kings under the sky,

Seven for the Dwarf-lords in their halls of stone,

Nine for Mortal Men doomed to die,

One for the Dark Lord on his dark throne

In the Land of Mordor where the Shadows lie.

One Ring to rule them all, One Ring to find them,

One Ring to bring them all and in the darkness bind them

In the Land of Mordor where the Shadows lie.""",
        "separate_files_checkbox": False,
        "export_format_checkboxes": ["flac", "mp3"],
        "disable_watermark_checkbox": True,
        "num_generations_input": 1,
        "num_candidates_slider": 1,
        "max_attempts_slider": 2,
        "bypass_whisper_checkbox": False,
        "whisper_model_dropdown": "medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)",
        "use_faster_whisper_checkbox": True,
        "enable_parallel_checkbox": True,
        "use_longest_transcript_on_fail_checkbox": True,
        "num_parallel_workers_slider": 4,
        "exaggeration_slider": 0.5,
        "cfg_weight_slider": 1.0,
        "temp_slider": 0.75,
        "seed_input": 0,
        "enable_batching_checkbox": False,
        "smart_batch_short_sentences_checkbox": True,
        "to_lowercase_checkbox": True,
        "normalize_spacing_checkbox": True,
        "fix_dot_letters_checkbox": True,
        "remove_reference_numbers_checkbox": True,
        "use_auto_editor_checkbox": False,
        "keep_original_checkbox": False,
        "threshold_slider": 0.06,
        "margin_slider": 0.2,
        "normalize_audio_checkbox": False,
        "normalize_method_dropdown": "ebu",
        "normalize_level_slider": -24,
        "normalize_tp_slider": -2,
        "normalize_lra_slider": 7,
        "sound_words_field": "",
        "use_pyrnnoise_checkbox": False,
    }
        
settings = load_settings()        
# Download both punkt and punkt_tab if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
#try:
#    nltk.data.find('tokenizers/punkt_tab')
#except LookupError:
#    nltk.download('punkt_tab')

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Select device: Apple Silicon GPU (MPS) if available, else fallback to CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"🚀 Running on device: {DEVICE}")
# ---- Determinism (CUDA / PyTorch) ----
import os as _os, torch as _torch
_torch.backends.cudnn.benchmark = False
if hasattr(_torch.backends.cudnn, "deterministic"):
    _torch.backends.cudnn.deterministic = True
try:
    _torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
if DEVICE == "cuda":
    _torch.backends.cuda.matmul.allow_tf32 = False
    _torch.backends.cudnn.allow_tf32 = False
# --------------------------------------

MODEL = None

def _free_vram():
    """
    Best-effort VRAM/RAM cleanup before (re)initializing heavy models.
    Safe to call on CPU-only systems.
    """
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def load_whisper_backend(model_name, use_faster_whisper, device):
    """
    Load Whisper with VRAM-friendly fallbacks:
      CUDA: try float16 -> int8_float16 -> int8
      non-CUDA: try int8 -> float32
    """
    if use_faster_whisper:
        _free_vram()  # free memory before constructing Faster-Whisper
        if device == "cuda":
            candidates = ["float16", "int8_float16", "int8"]
        else:
            candidates = ["int8", "float32"]

        last_err = None
        for ct in candidates:
            try:
                print(f"[DEBUG] Loading faster-whisper model: {model_name} (device={device}, compute_type={ct})")
                return FasterWhisperModel(model_name, device=device, compute_type=ct)
            except Exception as e:
                last_err = e
                print(f"[WARN] Failed loading faster-whisper ({ct}): {e}")

        raise RuntimeError(
            f"Failed to load Faster-Whisper '{model_name}' on device={device}. "
            f"Tried compute_types={candidates}. Last error: {last_err}"
        )
    else:
        print(f"[DEBUG] Loading openai-whisper model: {model_name}")
        _free_vram()  # also free before OpenAI-whisper to reduce fragmentation
        return whisper.load_model(model_name, device=device)


def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)  # PATCHED
        if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
            MODEL.to(DEVICE)
        if hasattr(MODEL, "eval"):
            MODEL.eval()
        print(f"Model loaded on device: {getattr(MODEL, 'device', 'unknown')}")
    return MODEL

try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model. Error: {e}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def derive_seed(base_seed: int, chunk_idx: int, cand_idx: int, attempt_idx: int) -> int:
    """
    Deterministically derive a 32-bit seed for each (chunk, candidate, attempt)
    from the user-supplied base seed. This avoids any use of global random().
    """
    # use 64-bit mixing then clamp to 32-bit
    mix = (np.uint64(base_seed) * np.uint64(1000003)
           + np.uint64(chunk_idx) * np.uint64(10007)
           + np.uint64(cand_idx) * np.uint64(10009)
           + np.uint64(attempt_idx) * np.uint64(101))
    s = int(mix & np.uint64(0xFFFFFFFF))
    return s if s != 0 else 1


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s{2,}', ' ', text.strip())

def replace_letter_period_sequences(text: str) -> str:
    def replacer(match):
        cleaned = match.group(0).rstrip('.')
        letters = cleaned.split('.')
        return ' '.join(letters)
    return re.sub(r'\b(?:[A-Za-z]\.){2,}', replacer, text)
    
def remove_inline_reference_numbers(text):
    # Remove reference numbers after sentence-ending punctuation, but keep the punctuation
    pattern = r'([.!?,\"\'”’)\]])(\d+)(?=\s|$)'
    return re.sub(pattern, r'\1', text)


def split_into_sentences(text):
    # NLTK's Punkt tokenizer handles abbreviations and common English quirks
    return sent_tokenize(text)

def split_long_sentence(sentence, max_len=300, seps=None):
    """
    Recursively split a sentence into chunks of <= max_len using a sequence of separators.
    Tries each separator in order, splitting further as needed.
    """
    if seps is None:
        seps = [';', ':', '-', ',', ' ']

    sentence = sentence.strip()
    if len(sentence) <= max_len:
        return [sentence]

    if not seps:
        # Fallback: force split every max_len chars
        return [sentence[i:i+max_len].strip() for i in range(0, len(sentence), max_len)]

    sep = seps[0]
    parts = sentence.split(sep)

    if len(parts) == 1:
        # Separator not found, try next separator
        return split_long_sentence(sentence, max_len, seps=seps[1:])

    # Now recursively process each part, joining separator back except for the first
    chunks = []
    current = parts[0].strip()
    for part in parts[1:]:
        candidate = (current + sep + part).strip()
        if len(candidate) > max_len:
            # Split current chunk further with the next separator
            chunks.extend(split_long_sentence(current.strip(), max_len, seps=seps[1:]))
            current = part.strip()
        else:
            current = candidate
    # Process the last current
    if current:
        if len(current) > max_len:
            chunks.extend(split_long_sentence(current.strip(), max_len, seps=seps[1:]))
        else:
            chunks.append(current.strip())

    return chunks

    # Fallback: force split every max_len chars
    #return [sentence[i:i+max_len].strip() for i in range(0, len(sentence), max_len)]

def group_sentences(sentences, max_chars=300):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if not sentence:
            print(f"\033[32m[DEBUG] Skipping empty sentence\033[0m")
            continue
        sentence = sentence.strip()
        sentence_len = len(sentence)

        print(f"\033[32m[DEBUG] Processing sentence: len={sentence_len}, content='\033[33m{sentence}...'\033[0m")

        if sentence_len > 300:
            print(f"\033[32m[DEBUG] Splitting overlong sentence of {sentence_len} chars\033[0m")
            for chunk in split_long_sentence(sentence, 300):
                if len(chunk) > max_chars:
                    # For extremely long non-breakable segments, just chunk them
                    for i in range(0, len(chunk), max_chars):
                        chunks.append(chunk[i:i+max_chars])
                else:
                    chunks.append(chunk)
            current_chunk = []
            current_length = 0
            continue  # Skip the rest of the loop for this sentence

        if sentence_len > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                print(f"\033[32m[DEBUG] Finalized chunk: {' '.join(current_chunk)}...\033[0m")
            chunks.append(sentence)
            print(f"\033[32m[DEBUG] Added long sentence as chunk: {sentence}...\033[0m")
            current_chunk = []
            current_length = 0
        elif current_length + sentence_len + (1 if current_chunk else 0) <= max_chars:
            current_chunk.append(sentence)
            current_length += sentence_len + (1 if current_chunk else 0)
            print(f"\033[32m[DEBUG] Adding sentence to chunk: {sentence}...\033[0m")
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                print(f"\033[32m[DEBUG] Finalized chunk: {' '.join(current_chunk)}...\033[0m")
            current_chunk = [sentence]
            current_length = sentence_len
            print(f"\033[32m[DEBUG] Starting new chunk with: {sentence}...\033[0m")

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        print(f"\033[32m[DEBUG] Finalized final chunk: {' '.join(current_chunk)}...\033[0m")

    print(f"\033[32m[DEBUG] Total chunks created: {len(chunks)}\033[0m")
    for i, chunk in enumerate(chunks):
        print(f"\033[32m[DEBUG] Chunk {i}: len={len(chunk)}, content='\033[33m{chunk}...'\033[0m")

    return chunks

def smart_append_short_sentences(sentences, max_chars=300):
    new_groups = []
    i = 0
    while i < len(sentences):
        current = sentences[i].strip()
        if len(current) >= 20:
            new_groups.append(current)
            i += 1
        else:
            appended = False
            if i + 1 < len(sentences):
                next_sentence = sentences[i + 1].strip()
                if len(current + " " + next_sentence) <= max_chars:
                    new_groups.append(current + " " + next_sentence)
                    i += 2
                    appended = True
            if not appended and new_groups:
                if len(new_groups[-1] + " " + current) <= max_chars:
                    new_groups[-1] += " " + current
                    i += 1
                    appended = True
            if not appended:
                new_groups.append(current)
                i += 1
    return new_groups

def normalize_with_ffmpeg(input_wav, output_wav, method="ebu", i=-24, tp=-2, lra=7):
    if method == "ebu":
        loudnorm = f"loudnorm=I={i}:TP={tp}:LRA={lra}"
        (
            ffmpeg
            .input(input_wav)
            .output(output_wav, af=loudnorm)
            .overwrite_output()
            .run(quiet=True)
        )
    elif method == "peak":
        (
            ffmpeg
            .input(input_wav)
            .output(output_wav, af="alimiter=limit=-2dB")
            .overwrite_output()
            .run(quiet=True)
        )

    else:
        raise ValueError("Unknown normalization method.")
    os.replace(output_wav, input_wav)

def _convert_to_pcm48k_mono(input_wav, output_wav, sr=48000):
    """
    Convert to 48kHz, mono, s16 PCM for RNNoise (pyrnnoise) best compatibility.
    """
    subprocess.run([
        "ffmpeg", "-y", "-i", input_wav,
        "-ac", "2", "-ar", str(sr), "-sample_fmt", "s16", output_wav
    ], check=True)


def _run_pyrnnoise(input_wav, output_wav):
    """
    Try the pyrnnoise CLI ('denoise') first; if missing or fails, fall back to Python API.
    """
    if not _PYRNNOISE_AVAILABLE:
        print("[DENOISE] pyrnnoise not available; skipping.")
        return False

    print("[DENOISE] Running pyrnnoise (RNNoise)…")
    # Prefer CLI if present (often faster and lighter on Python mem)
    try:
        result = subprocess.run(["denoise", input_wav, output_wav], capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 1024:
            print(f"[DENOISE] Saved: {output_wav}")
            return True
        else:
            print("[DENOISE] pyrnnoise CLI failed, falling back to Python API…")
    except FileNotFoundError:
        print("[DENOISE] pyrnnoise CLI not found, using Python API…")

    # Python API fallback
    rate, data = sf.read(input_wav)
    denoiser = pyrnnoise.RNNoise(rate)
    denoised = denoiser.process_buffer(data)
    sf.write(output_wav, denoised, rate)
    print(f"[DENOISE] Saved: {output_wav}")
    return True


def _apply_pyrnnoise_in_place(wav_output_path):
    """
    Denoise wav_output_path with RNNoise, preserving the original path.
    Converts to 48k mono s16 for processing, then converts back to the original sample rate.
    """
    try:
        original_sr = librosa.get_samplerate(wav_output_path)
    except Exception:
        # Fallback if librosa can't read it
        original_sr = None

    tmp_48kmono = wav_output_path.replace(".wav", "_48kmono.wav")
    tmp_dn = wav_output_path.replace(".wav", "_dn.wav")
    tmp_back = wav_output_path.replace(".wav", "_dn_resamp.wav")

    try:
        _convert_to_pcm48k_mono(wav_output_path, tmp_48kmono)
        ok = _run_pyrnnoise(tmp_48kmono, tmp_dn)
        if not ok:
            return False

        # Convert back to original sample rate (if known), keep mono
        if original_sr:
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_dn, "-ar", str(original_sr), "-ac", "1", tmp_back
            ], check=True)
            os.replace(tmp_back, wav_output_path)
        else:
            # If we don't know SR, just adopt the denoised file
            os.replace(tmp_dn, wav_output_path)

        print(f"[DENOISE] Denoised in-place: {wav_output_path}")
        return True
    except Exception as e:
        print(f"[DENOISE] RNNoise failed: {e}")
        return False
    finally:
        for p in [tmp_48kmono, tmp_dn, tmp_back]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def get_wav_duration(path):
    try:
        return librosa.get_duration(filename=path)
    except Exception as e:
        print(f"[ERROR] librosa.get_duration failed: {e}")
        return float('inf')

def normalize_for_compare_all_punct(text):
    text = re.sub(r'[–—-]', ' ', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def fuzzy_match(text1, text2, threshold=0.85):
    t1 = normalize_for_compare_all_punct(text1)
    t2 = normalize_for_compare_all_punct(text2)
    seq = difflib.SequenceMatcher(None, t1, t2)
    return seq.ratio() >= threshold

def parse_sound_word_field(user_input):
    # Accepts comma or newline separated, allows 'sound=>replacement'
    lines = [l.strip() for l in user_input.split('\n') if l.strip()]
    result = []
    for line in lines:
        if '=>' in line:
            pattern, replacement = line.split('=>', 1)
            result.append((pattern.strip(), replacement.strip()))
        else:
            result.append((line, ''))  # Remove (replace with empty string)
    return result

def smart_remove_sound_words(text, sound_words):
    for pattern, replacement in sound_words:
        if replacement:
            # 1. Handle possessive: "Baggins’" or "Baggins'" (optionally with s or S after apostrophe)
            text = re.sub(
                r'(?i)(%s)([’\']s?)' % re.escape(pattern),
                lambda m: replacement + "'s" if m.group(2) else replacement,
                text
            )
            # 2. Replace word in quotes
            text = re.sub(
                r'(["\'])%s(["\'])' % re.escape(pattern),
                lambda m: f"{m.group(1)}{replacement}{m.group(2)}",
                text,
                flags=re.IGNORECASE
            )
            # If pattern is a punctuation character (like dash), replace all
            if all(char in "-–—" for char in pattern.strip()):
                text = re.sub(re.escape(pattern), replacement, text)
            else:
                # 3. Replace as whole word (not in quotes)
                text = re.sub(
                    r'\b%s\b' % re.escape(pattern),
                    replacement,
                    text,
                    flags=re.IGNORECASE
                )
        else:
            # Remove only the pattern itself, not adjacent spaces
            text = re.sub(
                r'%s' % re.escape(pattern),
                '',
                text,
                flags=re.IGNORECASE
            )

    # --- Fix accidental joining of words caused by quote removal ---
    # Add a space if a letter is next to a letter and was separated by removed quote
    #text = re.sub(r'(\w)([’\'"“”‘’])(\w)', r'\1 \3', text)
    # Add a space between lowercase and uppercase, likely joined words (e.g., rainbowPride)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # --- Clean up doubled-up commas and extra spaces ---
    text = re.sub(r'([,\s]+,)+', ',', text)
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(\s+,|,\s+)', ', ', text)
    text = re.sub(r'(^|[\.!\?]\s*),+', r'\1', text)
    text = re.sub(r',+\s*([\.!\?])', r'\1', text)
    return text.strip()


def whisper_check_mp(candidate_path, target_text, whisper_model, use_faster_whisper=False):
    import difflib
    import re
    import string
    import os

    try:
        print(f"\033[32m[DEBUG] Whisper checking: {candidate_path}\033[0m")
        if use_faster_whisper:
            segments, info = whisper_model.transcribe(candidate_path)
            transcribed = "".join([seg.text for seg in segments]).strip().lower()
        else:
            result = whisper_model.transcribe(candidate_path)
            transcribed = result['text'].strip().lower()
        print(f"\033[32m[DEBUG] Whisper transcription: '\033[33m{transcribed}' for candidate '{os.path.basename(candidate_path)}'\033[0m")
        score = difflib.SequenceMatcher(
            None,
            normalize_for_compare_all_punct(transcribed),
            normalize_for_compare_all_punct(target_text.strip().lower())
        ).ratio()
        print(f"\033[32m[DEBUG] Score: {score:.3f} (target: '\033[33m{target_text}')\033[0m")
        return (candidate_path, score, transcribed)
    except Exception as e:
        print(f"[ERROR] Whisper transcription failed for {candidate_path}: {e}")
        return (candidate_path, 0.0, f"ERROR: {e}")
        
        
def process_one_chunk(
    model, sentence_group, idx, gen_index, this_seed,
    audio_prompt_path_input, exaggeration_input, temperature_input, cfgw_input,
    disable_watermark, num_candidates_per_chunk, max_attempts_per_candidate,
    bypass_whisper_checking,
    retry_attempt_number=1
):
    candidates = []
    try:
        if not sentence_group.strip():
            print(f"\033[32m[DEBUG] Skipping empty sentence group at index {idx}\033[0m")
            return (idx, candidates)
        if len(sentence_group) > 300:
            print(f"\033[33m[WARNING] Very long sentence group at index {idx} (len={len(sentence_group)}); proceeding anyway.\033[0m")

        print(f"\033[32m[DEBUG] Processing group {idx}: len={len(sentence_group)}:\033[33m {sentence_group}\033[0m")

        for cand_idx in range(num_candidates_per_chunk):
            for attempt in range(max_attempts_per_candidate):
                candidate_seed = derive_seed(this_seed, idx, cand_idx, attempt)
                set_seed(candidate_seed)
                try:
                    print(f"\033[32m[DEBUG] Generating candidate {cand_idx+1} attempt {attempt+1} for chunk {idx}...\033[0m")
#                    print(f"[TTS DEBUG] audio_prompt_path passed: {audio_prompt_path_input!r}")
                    wav = model.generate(
                        sentence_group,
                        audio_prompt_path=audio_prompt_path_input,
                        exaggeration=min(exaggeration_input, 1.0),
                        temperature=temperature_input,
                        cfg_weight=cfgw_input,
                        language_id=SELECTED_LANGUAGE,  # PATCHED
                    )
                    

                    candidate_path = f"temp/gen{gen_index+1}_chunk_{idx:03d}_cand_{cand_idx+1}_try{retry_attempt_number}_seed{candidate_seed}.wav"
                    torchaudio.save(candidate_path, wav, model.sr)
                    for _ in range(10):
                        if os.path.exists(candidate_path) and os.path.getsize(candidate_path) > 1024:
                            break
                        time.sleep(0.05)
                    duration = get_wav_duration(candidate_path)
                    print(f"\033[32m[DEBUG] Saved candidate {cand_idx+1}, attempt {attempt+1}, duration={duration:.3f}s: {candidate_path}\033[0m")
                    candidates.append({
                        'path': candidate_path,
                        'duration': duration,
                        'sentence_group': sentence_group,
                        'cand_idx': cand_idx,
                        'attempt': attempt,
                        'seed': candidate_seed,
                    })
                    break
                except Exception as e:
                    print(f"[ERROR] Candidate {cand_idx+1} generation attempt {attempt+1} failed: {e}")
    except Exception as exc:
        print(f"[ERROR] Exception in chunk {idx}: {exc}")
    return (idx, candidates)

def process_one_chunk_deterministic(
    model, sentence_group, idx, gen_index, this_seed,
    audio_prompt_path_input, exaggeration_input, temperature_input, cfgw_input,
    disable_watermark, num_candidates_per_chunk, max_attempts_per_candidate,
    bypass_whisper_checking,
    retry_attempt_number=1
):
    """
    Deterministic per-chunk generation that does NOT mutate global RNG.
    - If model.generate supports `generator`, use a per-call torch.Generator.
    - Else, fallback to a forked RNG scope + manual seeds (still thread-local).
    Also logs full tracebacks on failure so we can see the exact cause.
    """
    import inspect, traceback

    candidates = []
    try:
        if not sentence_group.strip():
            print(f"\033[32m[DEBUG] Skipping empty sentence group at index {idx}\033[0m")
            return (idx, candidates)
        if len(sentence_group) > 300:
            print(f"\033[33m[WARNING] Very long sentence group at index {idx} (len={len(sentence_group)}); proceeding anyway.\033[0m")

        print(f"\033[32m[DEBUG] [DET] Processing group {idx}: len={len(sentence_group)}:\033[33m {sentence_group}\033[0m")

        # Detect whether model.generate accepts a `generator` argument
        supports_generator = False
        try:
            sig = inspect.signature(model.generate)
            supports_generator = ("generator" in sig.parameters)
        except Exception:
            supports_generator = False

        model_device = str(getattr(model, "device", "cpu"))
        on_cuda = torch.cuda.is_available() and (model_device == "cuda")
        devices = [torch.cuda.current_device()] if on_cuda else []

        for cand_idx in range(num_candidates_per_chunk):
            for attempt in range(max_attempts_per_candidate):
                candidate_seed = derive_seed(this_seed, idx, cand_idx, attempt)
                print(f"\033[32m[DEBUG] [DET] Generating cand {cand_idx+1} attempt {attempt+1} for chunk {idx} (seed={candidate_seed}).\033[0m")

                try:
                    if supports_generator and (model_device != "mps"):
                        # Use a per-call generator on the matching device (CUDA→cuda, otherwise CPU)
                        gen_device = "cuda" if on_cuda else "cpu"
                        gen = torch.Generator(device=gen_device)
                        gen.manual_seed(int(candidate_seed) & 0xFFFFFFFFFFFFFFFF)

                        wav = model.generate(
                            sentence_group,
                            audio_prompt_path=audio_prompt_path_input,
                            exaggeration=min(exaggeration_input, 1.0),
                            temperature=temperature_input,
                            cfg_weight=cfgw_input,
                            generator=gen,  # isolated RNG
                            language_id=SELECTED_LANGUAGE,  # PATCHED
                        )
                    else:
                        # Fallback: fork RNG state locally and seed inside the scope
                        with torch.random.fork_rng(devices=devices, enabled=True):
                            torch.manual_seed(int(candidate_seed))
                            if on_cuda:
                                torch.cuda.manual_seed_all(int(candidate_seed))
                            wav = model.generate(
                                sentence_group,
                                audio_prompt_path=audio_prompt_path_input,
                                exaggeration=min(exaggeration_input, 1.0),
                                temperature=temperature_input,
                                cfg_weight=cfgw_input,
                                language_id=SELECTED_LANGUAGE,  # PATCHED
                            )

                    candidate_path = f"temp/gen{gen_index+1}_chunk_{idx:03d}_cand_{cand_idx+1}_try{retry_attempt_number}_seed{candidate_seed}.wav"
                    torchaudio.save(candidate_path, wav, model.sr)

                    # Wait briefly for filesystem consistency
                    for _ in range(10):
                        if os.path.exists(candidate_path) and os.path.getsize(candidate_path) > 1024:
                            break
                        time.sleep(0.05)

                    duration = get_wav_duration(candidate_path)
                    print(f"\033[32m[DEBUG] [DET] Saved cand {cand_idx+1}, attempt {attempt+1}, duration={duration:.3f}s: {candidate_path}\033[0m")
                    candidates.append({
                        'path': candidate_path,
                        'duration': duration,
                        'sentence_group': sentence_group,
                        'cand_idx': cand_idx,
                        'attempt': attempt,
                        'seed': candidate_seed,
                    })

                    # If bypass is ON we can short-circuit after first successful candidate
                    if bypass_whisper_checking:
                        break

                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"[ERROR] Deterministic generation failed for chunk {idx}, cand {cand_idx+1}, attempt {attempt+1}: {e}\n{tb}")
                    # Continue to next attempt/candidate

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] process_one_chunk_deterministic failed for index {idx}: {e}\n{tb}")

    return (idx, candidates)





def generate_and_preview(*args):

    output_paths = generate_batch_tts(*args)
    audio_files = [p for p in output_paths if os.path.splitext(p)[1].lower() in [".wav", ".mp3", ".flac"]]
    dropdown_value = audio_files[0] if audio_files else None
    return output_paths, gr.update(choices=audio_files, value=dropdown_value), dropdown_value
    

def update_audio_preview(selected_path):
    return selected_path

@spaces.GPU
def generate_batch_tts(
    text: str,
    text_file,
    audio_prompt_path_input,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    use_pyrnnoise: bool,
    use_auto_editor: bool,
    ae_threshold: float,
    ae_margin: float,
    export_formats: list,
    enable_batching: bool,
    to_lowercase: bool,
    normalize_spacing: bool,
    fix_dot_letters: bool,
    remove_reference_numbers: bool,
    keep_original_wav: bool,
    smart_batch_short_sentences: bool,
    disable_watermark: bool,
    num_generations: int,
    normalize_audio: bool,
    normalize_method: str,
    normalize_level: float,
    normalize_tp: float,
    normalize_lra: float,
    num_candidates_per_chunk: int,
    max_attempts_per_candidate: int,
    bypass_whisper_checking: bool,
    whisper_model_name: str,
    enable_parallel: bool = True,
    num_parallel_workers: int = 4,
    use_longest_transcript_on_fail: bool = False,
    sound_words_field: str = "",
    use_faster_whisper: bool = False,
    generate_separate_audio_files: bool = False,
) -> list[str]:
    print(f"[DEBUG] Received audio_prompt_path_input: {audio_prompt_path_input!r}")

    if not audio_prompt_path_input or (isinstance(audio_prompt_path_input, str) and not os.path.isfile(audio_prompt_path_input)):
        audio_prompt_path_input = None
    model = get_or_load_model()

    # PATCH: Get file basename (to prepend) if a text file was uploaded
    # Support for multiple file uploads
    # PATCH: Get file basename (to prepend) if a text file was uploaded
    # Support for multiple file uploads
    input_basename = ""

    # Robust handling for Gradio's file input (can be None, False, or list containing such)
    files = []
    if text_file:
        files = text_file if isinstance(text_file, list) else [text_file]
        # Remove any entry that's not a file-like object with a .name attribute (filters out None, False, bool)
        files = [f for f in files if hasattr(f, "name") and isinstance(getattr(f, "name", None), str)]

    if files:
        # If generating separate audio files per text file:
        if generate_separate_audio_files:
            all_jobs = []
            for fobj in files:
                try:
                    fname = os.path.basename(fobj.name)
                    base = os.path.splitext(fname)[0]
                    base = re.sub(r'[^a-zA-Z0-9_\-]', '_', base + "_")
                    with open(fobj.name, "r", encoding="utf-8") as f:
                        file_text = f.read()
                    all_jobs.append((file_text, base))
                except Exception as e:
                    print(f"[ERROR] Failed to read file: {getattr(fobj, 'name', repr(fobj))} | {e}")
            # Now process each file separately and collect outputs
            all_outputs = []
            for job_text, base in all_jobs:
                output_paths = process_text_for_tts(
                    job_text, base,
                    audio_prompt_path_input,
                    exaggeration_input, temperature_input, seed_num_input, cfgw_input,
                    use_pyrnnoise,  # <-- add this
                    use_auto_editor, ae_threshold, ae_margin, export_formats, enable_batching,
                    to_lowercase, normalize_spacing, fix_dot_letters, remove_reference_numbers, keep_original_wav,
                    smart_batch_short_sentences, disable_watermark, num_generations,
                    normalize_audio, normalize_method, normalize_level, normalize_tp,
                    normalize_lra, num_candidates_per_chunk, max_attempts_per_candidate,
                    bypass_whisper_checking, whisper_model_name, enable_parallel,
                    num_parallel_workers, use_longest_transcript_on_fail, sound_words_field, use_faster_whisper
                )
                all_outputs.extend(output_paths)
            return all_outputs  # Return list of output files

        # ELSE (default: join all text files as one, as before)
        all_text = []
        basenames = []
        for fobj in files:
            try:
                fname = os.path.basename(fobj.name)
                base = os.path.splitext(fname)[0]
                base = re.sub(r'[^a-zA-Z0-9_\-]', '_', base)
                basenames.append(base)
                with open(fobj.name, "r", encoding="utf-8") as f:
                    all_text.append(f.read())
            except Exception as e:
                print(f"[ERROR] Failed to read file: {getattr(fobj, 'name', repr(fobj))} | {e}")
        text = "\n\n".join(all_text)
        input_basename = "_".join(basenames) + "_"

        return process_text_for_tts(
            text, input_basename, audio_prompt_path_input,
            exaggeration_input, temperature_input, seed_num_input, cfgw_input,
            use_pyrnnoise,
            use_auto_editor, ae_threshold, ae_margin, export_formats, enable_batching,
            to_lowercase, normalize_spacing, fix_dot_letters, remove_reference_numbers, keep_original_wav,
            smart_batch_short_sentences, disable_watermark, num_generations,
            normalize_audio, normalize_method, normalize_level, normalize_tp,
            normalize_lra, num_candidates_per_chunk, max_attempts_per_candidate,
            bypass_whisper_checking, whisper_model_name, enable_parallel,
            num_parallel_workers, use_longest_transcript_on_fail, sound_words_field, use_faster_whisper
        )
    else:
        # No text file: just process the Text Input box as one job
        input_basename = "text_input_"
        return process_text_for_tts(
            text, input_basename, audio_prompt_path_input,
            exaggeration_input, temperature_input, seed_num_input, cfgw_input,
            use_pyrnnoise,
            use_auto_editor, ae_threshold, ae_margin, export_formats, enable_batching,
            to_lowercase, normalize_spacing, fix_dot_letters, remove_reference_numbers, keep_original_wav,
            smart_batch_short_sentences, disable_watermark, num_generations,
            normalize_audio, normalize_method, normalize_level, normalize_tp,
            normalize_lra, num_candidates_per_chunk, max_attempts_per_candidate,
            bypass_whisper_checking, whisper_model_name, enable_parallel,
            num_parallel_workers, use_longest_transcript_on_fail, sound_words_field, use_faster_whisper
        )

def process_text_for_tts(
    text,
    input_basename,
    audio_prompt_path_input,
    exaggeration_input,
    temperature_input,
    seed_num_input,
    cfgw_input,
    use_pyrnnoise,
    use_auto_editor,
    ae_threshold,
    ae_margin,
    export_formats,
    enable_batching,
    to_lowercase,
    normalize_spacing,
    fix_dot_letters,
    remove_reference_numbers,
    keep_original_wav,
    smart_batch_short_sentences,
    disable_watermark,
    num_generations,
    normalize_audio,
    normalize_method,
    normalize_level,
    normalize_tp,
    normalize_lra,
    num_candidates_per_chunk,
    max_attempts_per_candidate,
    bypass_whisper_checking,
    whisper_model_name,
    enable_parallel,
    num_parallel_workers,
    use_longest_transcript_on_fail,
    sound_words_field,
    use_faster_whisper=False,
):

    

    model = get_or_load_model()
    whisper_model = None
    if not text or len(text.strip()) == 0:
        raise ValueError("No text provided.")
    
    # ---- NEW: Apply sound word removals/replacements ----
    if sound_words_field and sound_words_field.strip():
        sound_words = parse_sound_word_field(sound_words_field)
        if sound_words:
            text = smart_remove_sound_words(text, sound_words)

    if to_lowercase:
        text = text.lower()
    if normalize_spacing:
        text = normalize_whitespace(text)
    if fix_dot_letters:
        text = replace_letter_period_sequences(text)
    if remove_reference_numbers:
        text = remove_inline_reference_numbers(text)

    print("[DEBUG] After reference number removal:", repr(text))  # <--- ADD THIS LINE HERE

    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    for f in os.listdir("temp"):
        os.remove(os.path.join("temp", f))

    sentences = split_into_sentences(text)
    print(f"\033[32m[DEBUG] Split text into {len(sentences)} sentences.\033[0m")

    def enforce_min_chunk_length(chunks, min_len=20, max_len=300):
        out = []
        i = 0
        while i < len(chunks):
            current = chunks[i].strip()
            if len(current) >= min_len or i == len(chunks) - 1:
                out.append(current)
                i += 1
            else:
                # Try to merge with the next chunk if possible
                if i + 1 < len(chunks):
                    merged = current + " " + chunks[i + 1]
                    if len(merged) <= max_len:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(current)
                        i += 1
                else:
                    out.append(current)
                    i += 1
        return out

    sentence_groups = None
    if enable_batching:
        sentence_groups = group_sentences(sentences, max_chars=300)
        if smart_batch_short_sentences:  # NEW: now works as post-processing!
            sentence_groups = enforce_min_chunk_length(sentence_groups)
    elif smart_batch_short_sentences:
        sentence_groups = smart_append_short_sentences(sentences)
        sentence_groups = enforce_min_chunk_length(sentence_groups)
    else:
        sentence_groups = sentences

    output_paths = []
    for gen_index in range(num_generations):
        if seed_num_input == 0:
            this_seed = random.randint(1, 2**32 - 1)
        else:
            this_seed = int(seed_num_input) + gen_index
        set_seed(this_seed)

        print(f"\033[43m[DEBUG] Starting generation {gen_index+1}/{num_generations} with seed {this_seed}\033[0m")

        chunk_candidate_map = {}
        waveform_list = []  # Initialize waveform_list here to ensure it’s defined

        # -------- CHUNK GENERATION --------
        if enable_parallel:
            total_chunks = len(sentence_groups)
            completed = 0
            with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
                futures = [
                    executor.submit(
                        process_one_chunk_deterministic,
                        model, group, idx, gen_index, this_seed,
                        audio_prompt_path_input, exaggeration_input, temperature_input, cfgw_input,
                        disable_watermark, num_candidates_per_chunk, max_attempts_per_candidate, bypass_whisper_checking
                    )
                    for idx, group in enumerate(sentence_groups)
                ]
                for future in as_completed(futures):
                    idx, candidates = future.result()
                    chunk_candidate_map[idx] = candidates
                    completed += 1
                    percent = int(100 * completed / total_chunks)
                    print(f"\033[36m[PROGRESS] Generated chunk {completed}/{total_chunks} ({percent}%)\033[0m")
        else:
            # Sequential mode: Process chunks one by one
            for idx, group in enumerate(sentence_groups):
                idx, candidates = process_one_chunk_deterministic(
                    model, group, idx, gen_index, this_seed,
                    audio_prompt_path_input, exaggeration_input, temperature_input, cfgw_input,
                    disable_watermark, num_candidates_per_chunk, max_attempts_per_candidate, bypass_whisper_checking
                )
                chunk_candidate_map[idx] = candidates

        # -------- WHISPER VALIDATION --------
        if not bypass_whisper_checking:
            print(f"\033[32m[DEBUG] Validating all candidates with Whisper for all chunks (sequentially)...\033[0m")

            # Purge as much memory as possible before initializing Whisper
            _free_vram()

            model_key = whisper_model_map.get(whisper_model_name, "medium")
            whisper_model = load_whisper_backend(model_key, use_faster_whisper, DEVICE)

            try:
                all_candidates = []
                for chunk_idx, candidates in chunk_candidate_map.items():
                    for cand in candidates:
                        all_candidates.append((chunk_idx, cand))

                chunk_validations = {chunk_idx: [] for chunk_idx in chunk_candidate_map}
                chunk_failed_candidates = {chunk_idx: [] for chunk_idx in chunk_candidate_map}

                # Initial sequential Whisper validation
                for chunk_idx, cand in all_candidates:
                    candidate_path = cand['path']
                    sentence_group = cand['sentence_group']
                    try:
                        if not os.path.exists(candidate_path) or os.path.getsize(candidate_path) < 1024:
                            print(f"[ERROR] Candidate file missing or too small: {candidate_path}")
                            chunk_failed_candidates[chunk_idx].append((0.0, candidate_path, ""))
                            continue
                        path, score, transcribed = whisper_check_mp(candidate_path, sentence_group, whisper_model, use_faster_whisper)
                        print(f"\033[32m[DEBUG] [Chunk {chunk_idx}] {os.path.basename(candidate_path)}: score={score:.3f}, transcript=\033[33m'{transcribed}'\033[0m")
                        if score >= 0.85:
                            chunk_validations[chunk_idx].append((cand['duration'], cand['path']))
                        else:
                            chunk_failed_candidates[chunk_idx].append((score, cand['path'], transcribed))
                    except Exception as e:
                        print(f"[ERROR] Whisper transcription failed for {candidate_path}: {e}")
                        chunk_failed_candidates[chunk_idx].append((0.0, candidate_path, ""))

                # Retry block for failed chunks
                retry_queue = [chunk_idx for chunk_idx in sorted(chunk_candidate_map.keys()) if not chunk_validations[chunk_idx]]
                chunk_attempts = {chunk_idx: 1 for chunk_idx in retry_queue}

                while retry_queue:
                    still_need_retry = [
                        chunk_idx for chunk_idx in retry_queue
                        if chunk_attempts[chunk_idx] < max_attempts_per_candidate
                    ]
                    if not still_need_retry:
                        break

                    print(f"\033[33m[RETRY] Retrying {len(still_need_retry)} chunks, attempt {chunk_attempts[still_need_retry[0]]+1} of {max_attempts_per_candidate}\033[0m")

                    retry_candidate_map = {}
                    with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
                        futures = [
                            executor.submit(
                                process_one_chunk_deterministic,
                                model,
                                chunk_candidate_map[chunk_idx][0]['sentence_group'] if chunk_candidate_map[chunk_idx] else sentence_groups[chunk_idx],
                                chunk_idx,
                                gen_index,
                                this_seed,  # base; per-candidate attempts derive inside deterministic function
                                audio_prompt_path_input, exaggeration_input, temperature_input, cfgw_input,
                                disable_watermark, num_candidates_per_chunk, 1,
                                bypass_whisper_checking,
                                chunk_attempts[chunk_idx] + 1
                            )
                            for chunk_idx in still_need_retry
                        ]
                        for future in as_completed(futures):
                            idx, candidates = future.result()
                            retry_candidate_map[idx] = candidates

                    for chunk_idx, candidates in retry_candidate_map.items():
                        for cand in candidates:
                            candidate_path = cand['path']
                            sentence_group = cand['sentence_group']
                            try:
                                if not os.path.exists(candidate_path) or os.path.getsize(candidate_path) < 1024:
                                    print(f"[ERROR] Retry candidate file missing or too small: {candidate_path}")
                                    chunk_failed_candidates[chunk_idx].append((0.0, candidate_path, ""))
                                    continue
                                path, score, transcribed = whisper_check_mp(candidate_path, sentence_group, whisper_model, use_faster_whisper)
                                print(f"\033[32m[DEBUG] [Chunk {chunk_idx}] RETRY {os.path.basename(candidate_path)}: score={score:.3f}, transcript=\033[33m'{transcribed}'\033[0m")
                                if score >= 0.95:
                                    chunk_validations[chunk_idx].append((cand['duration'], cand['path']))
                                else:
                                    chunk_failed_candidates[chunk_idx].append((score, cand['path'], transcribed))
                            except Exception as e:
                                print(f"[ERROR] Whisper transcription failed for retry {candidate_path}: {e}")
                                chunk_failed_candidates[chunk_idx].append((0.0, candidate_path, ""))

                    retry_queue = [chunk_idx for chunk_idx in still_need_retry if not chunk_validations[chunk_idx]]
                    for chunk_idx in still_need_retry:
                        chunk_attempts[chunk_idx] += 1

                # Assemble waveform list
                for chunk_idx in sorted(chunk_candidate_map.keys()):
                    if chunk_validations[chunk_idx]:
                        best_path = sorted(chunk_validations[chunk_idx], key=lambda x: x[0])[0][1]
                        print(f"\033[32m[DEBUG] Selected {best_path} as best candidate for chunk {chunk_idx} \033[1;33m(PASSED Whisper check)\033[0m")
                        waveform, sr = torchaudio.load(best_path)
                        waveform_list.append(waveform)
                    elif chunk_failed_candidates[chunk_idx]:
                        if use_longest_transcript_on_fail:
                            best_failed = max(chunk_failed_candidates[chunk_idx], key=lambda x: len(x[2]))
                            print(f"\033[33m[WARNING] No candidate passed for chunk {chunk_idx}. Using failed candidate with longest transcript: {best_failed[1]} (len={len(best_failed[2])})\033[0m")
                        else:
                            best_failed = max(chunk_failed_candidates[chunk_idx], key=lambda x: x[0])
                            print(f"\033[33m[WARNING] No candidate passed for chunk {chunk_idx}. Using failed candidate with highest score: {best_failed[1]} (score={best_failed[0]:.3f})\033[0m")
                        waveform, sr = torchaudio.load(best_failed[1])
                        waveform_list.append(waveform)
                    else:
                        print(f"[ERROR] No candidates were generated for chunk {chunk_idx}.")
            finally:
                # Clean up Whisper model
                try:
                    del whisper_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    print("\033[32m[DEBUG] Whisper model deleted and VRAM cache cleared.\033[0m")
                except Exception as e:
                    print(f"\033[32m[DEBUG] Could not delete Whisper model: {e}\033[0m")
        else:
            # Bypass Whisper: pick shortest duration per chunk
            for chunk_idx in sorted(chunk_candidate_map.keys()):
                candidates = chunk_candidate_map[chunk_idx]
                # Only consider candidates whose files exist and are > 1024 bytes
                valid_candidates = [
                    c for c in candidates
                    if os.path.exists(c['path']) and os.path.getsize(c['path']) > 1024
                ]
                if valid_candidates:
                    # Prefer the primary seeded candidate deterministically (cand_idx=0, attempt=0)
                    if all(('cand_idx' in c and 'attempt' in c) for c in valid_candidates):
                        best = sorted(valid_candidates, key=lambda c: (c['cand_idx'], c['attempt']))[0]
                    else:
                        best = min(valid_candidates, key=lambda c: c['duration'])

                    print(f"\033[32m[DEBUG] [Bypass Whisper] Selected {best['path']} as shortest candidate for chunk {chunk_idx}\033[0m")
                    waveform, sr = torchaudio.load(best['path'])
                    waveform_list.append(waveform)
                else:
                    print(f"\033[33m[WARNING] No valid candidates found for chunk {chunk_idx} (all generations failed)\033[0m")


        if not waveform_list:
            print(f"\033[33m[WARNING] No audio generated in generation {gen_index+1}\033[0m")
            continue

        full_audio = torch.cat(waveform_list, dim=1)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
        filename_suffix = f"{timestamp}_gen{gen_index+1}_seed{this_seed}"
        wav_output = f"output/{input_basename}audio_{filename_suffix}.wav"
        torchaudio.save(wav_output, full_audio, model.sr)
        print(f"\33[104m[DEBUG] \33[5mFinal audio concatenated, output file: {wav_output}\033[0m")

        # --- DENOISE (optional, before Auto-Editor) ---
        if use_pyrnnoise:
            if _PYRNNOISE_AVAILABLE:
                try:
                    if _apply_pyrnnoise_in_place(wav_output):
                        print(f"\033[32m[DEBUG] Denoised with RNNoise before Auto-Editor: {wav_output}\033[0m")
                    else:
                        print(f"\033[33m[WARNING] RNNoise returned False; continuing without denoise.\033[0m")
                except Exception as e:
                    print(f"[ERROR] RNNoise failed: {e}")
            else:
                print("[WARNING] pyrnnoise not installed; skipping denoise.")
                
        if use_auto_editor:
            try:
                cleaned_output = wav_output.replace(".wav", "_cleaned.wav")
                if keep_original_wav:
                    backup_path = wav_output.replace(".wav", "_original.wav")
                    os.rename(wav_output, backup_path)
                    auto_editor_input = backup_path
                else:
                    auto_editor_input = wav_output

                auto_editor_cmd = [
                    "auto-editor",
                    "--edit", f"audio:threshold={ae_threshold}",
                    "--margin", f"{ae_margin}s",
                    "--export", "audio",
                    auto_editor_input,
                    "-o", cleaned_output
                ]

                subprocess.run(auto_editor_cmd, check=True)

                if os.path.exists(cleaned_output):
                    os.replace(cleaned_output, wav_output)
                    print(f"\033[32m[DEBUG] Post-processed with auto-editor: {wav_output}\033[0m")
            except Exception as e:
                print(f"[ERROR] Auto-editor post-processing failed: {e}")

        if normalize_audio:
            try:
                norm_temp = wav_output.replace(".wav", "_norm.wav")
                normalize_with_ffmpeg(
                    wav_output,
                    norm_temp,
                    method=normalize_method,
                    i=normalize_level,
                    tp=normalize_tp,
                    lra=normalize_lra,
                )
                print(f"\033[32m[DEBUG] Post-processed with ffmpeg normalization: {wav_output}\033[0m")
            except Exception as e:
                print(f"[ERROR] ffmpeg normalization failed: {e}")

        gen_outputs = []
        for export_format in export_formats:
            if export_format.lower() == "wav":
                gen_outputs.append(wav_output)
            else:
                audio = AudioSegment.from_wav(wav_output)
                final_output = wav_output.replace(".wav", f".{export_format}")
                export_kwargs = {}
                if export_format.lower() == "mp3":
                    export_kwargs["bitrate"] = "320k"
                audio.export(final_output, format=export_format, **export_kwargs)
                gen_outputs.append(final_output)

        output_paths.extend(gen_outputs)

        if "wav" not in [fmt.lower() for fmt in export_formats]:
            try:
                os.remove(wav_output)
            except Exception as e:
                print(f"[ERROR] Could not remove temp wav file: {e}")
                
            # === Save settings CSV and JSON for this generation ===
        # Only include relevant fields and NOT the raw text_input
        settings_to_save = {
            "text_input": "",  # Intentionally blank for privacy
            "exaggeration_slider": exaggeration_input,
            "temp_slider": temperature_input,
            "seed_input": this_seed,
            "cfg_weight_slider": cfgw_input,
            "use_pyrnnoise_checkbox": use_pyrnnoise,
            "use_auto_editor_checkbox": use_auto_editor,
            "threshold_slider": ae_threshold,
            "margin_slider": ae_margin,
            "export_format_checkboxes": export_formats,
            "enable_batching_checkbox": enable_batching,
            "to_lowercase_checkbox": to_lowercase,
            "normalize_spacing_checkbox": normalize_spacing,
            "fix_dot_letters_checkbox": fix_dot_letters,
            "remove_reference_numbers_checkbox": remove_reference_numbers,
            "keep_original_checkbox": keep_original_wav,
            "smart_batch_short_sentences_checkbox": smart_batch_short_sentences,
            "disable_watermark_checkbox": disable_watermark,
            "num_generations_input": num_generations,
            "normalize_audio_checkbox": normalize_audio,
            "normalize_method_dropdown": normalize_method,
            "normalize_level_slider": normalize_level,
            "normalize_tp_slider": normalize_tp,
            "normalize_lra_slider": normalize_lra,
            "num_candidates_slider": num_candidates_per_chunk,
            "max_attempts_slider": max_attempts_per_candidate,
            "bypass_whisper_checkbox": bypass_whisper_checking,
            "whisper_model_dropdown": next((k for k, v in whisper_model_map.items() if v == whisper_model_name), whisper_model_name),
            "enable_parallel_checkbox": enable_parallel,
            "num_parallel_workers_slider": num_parallel_workers,
            "use_longest_transcript_on_fail_checkbox": use_longest_transcript_on_fail,
            "sound_words_field": sound_words_field,
            "use_faster_whisper_checkbox": use_faster_whisper,
            "separate_files_checkbox": False,  # Or True, if that option was used for this job
            "input_basename": input_basename,  # Additional info, optional
            "audio_prompt_path_input": audio_prompt_path_input,  # Additional info, optional
            "generation_time": datetime.datetime.now().isoformat(),
            #"output_audio_files": gen_outputs,  # Add this so each settings.json also points to its outputs!
        }

        # Name settings file after the first output audio file (base)
        base_out = gen_outputs[0].rsplit('.', 1)[0]  # E.g., output/audiofile_gen1_seedXXXXX
        csv_path = base_out + ".settings.csv"
        json_path = base_out + ".settings.json"

        # Save CSV (no output_audio_files in dict)
        save_settings_csv(settings_to_save, gen_outputs, csv_path)

        # Save JSON (add output_audio_files to dict)
        settings_for_json = settings_to_save.copy()
        settings_for_json["output_audio_files"] = gen_outputs
        save_settings_json(settings_for_json, json_path)

    print(f"\033[1;36m[DEBUG] \33[6;4;3;34;102mALL GENERATIONS COMPLETE. Outputs:\033[0m\n" + "\n".join(output_paths))
    return output_paths

# ----- UI SECTION -----
whisper_model_choices = [
    "tiny (~1 GB VRAM OpenAI / ~0.5 GB faster-whisper)",
    "base (~1.2–2 GB OpenAI / ~0.7–1 GB faster-whisper)",
    "small (~2–3 GB OpenAI / ~1.2–1.7 GB faster-whisper)",
    "medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)",
    "large (~10–13 GB OpenAI / ~4.5–6.5 GB faster-whisper)",
]

whisper_model_map = {
    "tiny (~1 GB VRAM OpenAI / ~0.5 GB faster-whisper)": "tiny",
    "base (~1.2–2 GB OpenAI / ~0.7–1 GB faster-whisper)": "base",
    "small (~2–3 GB OpenAI / ~1.2–1.7 GB faster-whisper)": "small",
    "medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)": "medium",
    "large (~10–13 GB OpenAI / ~4.5–6.5 GB faster-whisper)": "large"
}


def apply_settings_json(settings_json):
    import json
    if not settings_json:
        return [gr.update() for _ in range(36)]
    try:
        with open(settings_json.name, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # --- helpers for coercion/back-compat ---
        def _float(x, default):
            try:
                return float(x)
            except Exception:
                return default

        def _int(x, default):
            try:
                return int(x)
            except Exception:
                return default

        def _bool(x, default):
            if isinstance(x, bool):
                return x
            if isinstance(x, (int, float)):
                return bool(x)
            if isinstance(x, str):
                return x.strip().lower() in {"1", "true", "yes", "on"}
            return default

        # Map whisper model code -> label if needed
        wm = loaded.get(
            "whisper_model_dropdown",
            "medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)"
        )
        if wm not in whisper_model_map:  # if a code like "medium" was saved
            inv = {v: k for k, v in whisper_model_map.items()}
            wm = inv.get(wm, "medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)")

        # Guard normalize method against legacy/bad numeric values
        nm = loaded.get("normalize_method_dropdown", "ebu")
        if isinstance(nm, (int, float)) or nm not in {"ebu", "peak"}:
            nm = "ebu"

        # --- CRITICAL: return values in EXACT outputs order (36) ---
        return [
            loaded.get("text_input", ""),                              # 0
            None,                                                      # 1 text_file_input (cannot load)
            _bool(loaded.get("separate_files_checkbox", False), False),# 2
            loaded.get("audio_prompt_path_input", ""),                 # 3 ref_audio_input (filepath string)
            loaded.get("export_format_checkboxes", ["wav"]),           # 4
            _bool(loaded.get("disable_watermark_checkbox", False), False), # 5
            _int(loaded.get("num_generations_input", 1), 1),           # 6
            _int(loaded.get("num_candidates_slider", 3), 3),           # 7
            _int(loaded.get("max_attempts_slider", 3), 3),             # 8
            _bool(loaded.get("bypass_whisper_checkbox", False), False),# 9
            wm,                                                        # 10 whisper_model_dropdown (label)
            _bool(loaded.get("use_faster_whisper_checkbox", True), True), # 11
            _bool(loaded.get("enable_parallel_checkbox", True), True), # 12
            _bool(loaded.get("use_longest_transcript_on_fail_checkbox", True), True), # 13
            _int(loaded.get("num_parallel_workers_slider", 4), 4),     # 14
            _float(loaded.get("exaggeration_slider", 0.5), 0.5),       # 15
            _float(loaded.get("cfg_weight_slider", 1.0), 1.0),         # 16
            _float(loaded.get("temp_slider", 0.75), 0.75),             # 17
            _int(loaded.get("seed_input", 0), 0),                      # 18
            _bool(loaded.get("enable_batching_checkbox", False), False), # 19
            _bool(loaded.get("smart_batch_short_sentences_checkbox", True), True), # 20
            _bool(loaded.get("to_lowercase_checkbox", True), True),    # 21
            _bool(loaded.get("normalize_spacing_checkbox", True), True),# 22
            _bool(loaded.get("fix_dot_letters_checkbox", True), True), # 23
            _bool(loaded.get("remove_reference_numbers_checkbox", True), True), # 24
            _bool(loaded.get("use_pyrnnoise_checkbox", False), False), # 25  ✅ position fixed
            _bool(loaded.get("use_auto_editor_checkbox", False), False),# 26
            _bool(loaded.get("keep_original_checkbox", False), False), # 27
            _float(loaded.get("threshold_slider", 0.06), 0.06),        # 28
            _float(loaded.get("margin_slider", 0.2), 0.2),             # 29
            _bool(loaded.get("normalize_audio_checkbox", False), False),# 30
            nm,                                                        # 31 normalize_method_dropdown  ✅
            _float(loaded.get("normalize_level_slider", -24), -24),    # 32
            _float(loaded.get("normalize_tp_slider", -2), -2),         # 33
            _float(loaded.get("normalize_lra_slider", 7), 7),          # 34
            loaded.get("sound_words_field", ""),                       # 35
        ]
    except Exception as e:
        print(f"[ERROR] Failed to load settings JSON: {e}")
        return [gr.update() for _ in range(36)]





def main(server_name=None, server_port=None, share=False):
    with gr.Blocks() as demo:
        gr.Markdown("# 🎧 Chatterbox TTS Extended")
        with gr.Tabs():
            # TTS Tab (your original interface)
            with gr.Tab("TTS & Multi-Gen"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(label="Text Input", lines=6, value=settings["text_input"])
                        text_file_input = gr.File(label="Text File(s) (.txt)", file_types=[".txt"], file_count="multiple")
                        separate_files_checkbox = gr.Checkbox(label="Generate separate audio files per text file", value=settings["separate_files_checkbox"])
                        ref_audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio (Optional)")
                        export_format_checkboxes = gr.CheckboxGroup(
                            choices=["wav", "mp3", "flac"],
                            value=settings["export_format_checkboxes"],  # default selection
                            label="Export Format(s): Select one or more"
                        )
                        disable_watermark_checkbox = gr.Checkbox(label="Disable Perth Watermark", value=settings["disable_watermark_checkbox"], visible=False)
                        num_generations_input = gr.Number(value=settings["num_generations_input"], precision=0, label="Number of Generations")
                        num_candidates_slider = gr.Slider(1, 10, value=settings["num_candidates_slider"], step=1, label="Number of Candidates Per Chunk (after batching) - [reduces the chance of artifacts and hallucinations]")
                        max_attempts_slider = gr.Slider(1, 10, value=settings["max_attempts_slider"], step=1, label="Max Attempts Per Candidate (Whisper check retries)")
                        bypass_whisper_checkbox = gr.Checkbox(label="Bypass Whisper Checking (pick shortest candidate regardless of transcription)", value=settings["bypass_whisper_checkbox"])
                        whisper_model_dropdown = gr.Dropdown(
                            choices=whisper_model_choices,
                            value=settings["whisper_model_dropdown"],
                            label="Whisper Sync Model (with VRAM requirements)",
                            info="Select a Whisper model for sync/transcription; smaller models use less VRAM but are less accurate."
                        )
                        use_faster_whisper_checkbox = gr.Checkbox(
                            label="Use faster-whisper (SYSTRAN) backend for Whisper validation (much faster, less VRAM, almost as accurate)",
                            value=settings["use_faster_whisper_checkbox"]
                        )
                        enable_parallel_checkbox = gr.Checkbox(label="Enable Parallel Chunk Processing", value=settings["enable_parallel_checkbox"], visible=False)
                        use_longest_transcript_on_fail_checkbox = gr.Checkbox(
                        label="When all candidates fail Whisper check, pick candidate with longest transcript (not highest fuzzy match score)",
                        value=settings["use_longest_transcript_on_fail_checkbox"]
                        )
                        num_parallel_workers_slider = gr.Slider(1, 8, value=settings["num_parallel_workers_slider"], step=1, label="Parallel Workers - set to 1 for sequential processing")
                        load_settings_file = gr.File(label="Load Settings (.json)", file_types=[".json"])

                        run_button = gr.Button("Generate")
                    with gr.Column():
                        language_dropdown = gr.Dropdown(
                            choices=SUPPORTED_LANG_CHOICES,
                            value="es",
                            label="✨ Language / Idioma",
                            info="Idioma do texto. A mudança vale imediatamente para a próxima geração."
                        )
                        language_dropdown.change(fn=set_language, inputs=[language_dropdown], outputs=[])
                        exaggeration_slider = gr.Slider(0.0, 2.0, value=settings["exaggeration_slider"], step=0.1, label="Emotion Exaggeration")
                        cfg_weight_slider = gr.Slider(0.1, 1.0, value=settings["cfg_weight_slider"], step=0.01, label="CFG Weight/Pace")
                        temp_slider = gr.Slider(0.01, 5.0, value=settings["temp_slider"], step=0.05, label="Temperature")
                        seed_input = gr.Number(value=settings["seed_input"], label="Random Seed (0 for random)")
                        enable_batching_checkbox = gr.Checkbox(label="Enable Sentence Batching (Max 300 chars)", value=settings["enable_batching_checkbox"])
                        smart_batch_short_sentences_checkbox = gr.Checkbox(label="Smart-append short sentences (if batching is off)", value=settings["smart_batch_short_sentences_checkbox"])
                        to_lowercase_checkbox = gr.Checkbox(label="Convert input text to lowercase", value=settings["to_lowercase_checkbox"])
                        normalize_spacing_checkbox = gr.Checkbox(label="Normalize spacing (remove extra newlines and spaces)", value=settings["normalize_spacing_checkbox"])
                        fix_dot_letters_checkbox = gr.Checkbox(label="Convert 'J.R.R.' style input to 'J R R'", value=settings["fix_dot_letters_checkbox"])
                        remove_reference_numbers_checkbox = gr.Checkbox(
                            label="Remove inline reference numbers after sentences (e.g., '.188', '.”3')",
                            value=settings.get("remove_reference_numbers_checkbox", True)
                        )
                        
                        use_pyrnnoise_checkbox = gr.Checkbox(
                            label="Denoise with RNNoise (pyrnnoise) before Auto-Editor",
                            value=settings["use_pyrnnoise_checkbox"]
                        )

                        use_auto_editor_checkbox = gr.Checkbox(label="Post-process with Auto-Editor", value=settings["use_auto_editor_checkbox"])
                        keep_original_checkbox = gr.Checkbox(label="Keep original WAV (before Auto-Editor)", value=settings["keep_original_checkbox"])
                        threshold_slider = gr.Slider(0.01, 0.5, value=settings["threshold_slider"], step=0.01, label="Auto-Editor Volume Threshold")
                        margin_slider = gr.Slider(0.0, 2.0, value=settings["margin_slider"], step=0.1, label="Auto-Editor Margin (seconds)")

                        normalize_audio_checkbox = gr.Checkbox(label="Normalize with ffmpeg (loudness/peak)", value=settings["normalize_audio_checkbox"])
                        normalize_method_dropdown = gr.Dropdown(
                            choices=["ebu", "peak"], value=settings["normalize_method_dropdown"], label="Normalization Method"
                        )
                        normalize_level_slider = gr.Slider(
                            -70, -5, value=settings["normalize_level_slider"], step=1, label="EBU Target Integrated Loudness (I, dB, ebu only)"
                        )
                        normalize_tp_slider = gr.Slider(
                            -9, 0, value=settings["normalize_tp_slider"], step=1, label="EBU True Peak (TP, dB, ebu only)"
                        )
                        normalize_lra_slider = gr.Slider(
                            1, 50, value=settings["normalize_lra_slider"], step=1, label="EBU Loudness Range (LRA, ebu only)"
                        )


                        sound_words_field = gr.Textbox(
                            label="Remove/Replace Words/Sounds (newline separated or 'sound=>replacement')",
                            lines=2,
                            info="Examples: sss, ss, ahh=>um, hmm (removes/replace as standalone or quoted; not in words)",
                            value=settings["sound_words_field"]
                        )
                        # === LOAD SETTINGS FROM JSON FEATURE ===
                        load_settings_file.change(
                            fn=apply_settings_json,
                            inputs=[load_settings_file],
                            outputs=[
                                text_input,                          # 0
                                text_file_input,                     # 1
                                separate_files_checkbox,             # 2
                                ref_audio_input,                     # 3
                                export_format_checkboxes,            # 4
                                disable_watermark_checkbox,          # 5
                                num_generations_input,               # 6
                                num_candidates_slider,               # 7
                                max_attempts_slider,                 # 8
                                bypass_whisper_checkbox,             # 9
                                whisper_model_dropdown,              # 10
                                use_faster_whisper_checkbox,         # 11
                                enable_parallel_checkbox,            # 12
                                use_longest_transcript_on_fail_checkbox, # 13
                                num_parallel_workers_slider,         # 14
                                exaggeration_slider,                 # 15
                                cfg_weight_slider,                   # 16
                                temp_slider,                         # 17
                                seed_input,                          # 18
                                enable_batching_checkbox,            # 19
                                smart_batch_short_sentences_checkbox,# 20
                                to_lowercase_checkbox,               # 21
                                normalize_spacing_checkbox,          # 22
                                fix_dot_letters_checkbox,            # 23
                                remove_reference_numbers_checkbox,   # 24
                                use_pyrnnoise_checkbox,              # 25  <-- added
                                use_auto_editor_checkbox,            # 26
                                keep_original_checkbox,              # 27
                                threshold_slider,                    # 28
                                margin_slider,                       # 29
                                normalize_audio_checkbox,            # 30
                                normalize_method_dropdown,           # 31
                                normalize_level_slider,              # 32
                                normalize_tp_slider,                 # 33
                                normalize_lra_slider,                # 34
                                sound_words_field,                   # 35
                            ]
                        )

                        
                        

                        output_audio = gr.Files(label="Download Final Audio File(s)")
                        audio_dropdown = gr.Dropdown(label="Click to Preview Any Generated File")
                        audio_preview = gr.Audio(label="Audio Preview", interactive=True)
                        audio_dropdown.change(fn=update_audio_preview, inputs=audio_dropdown, outputs=audio_preview)

            def collect_ui_settings(*vals):
                keys = [
                    "text_input",
                    "exaggeration_slider",
                    "temp_slider",
                    "seed_input",
                    "cfg_weight_slider",
                    "use_pyrnnoise_checkbox",
                    "use_auto_editor_checkbox",
                    "threshold_slider",
                    "margin_slider",
                    "export_format_checkboxes",
                    "enable_batching_checkbox",
                    "to_lowercase_checkbox",
                    "normalize_spacing_checkbox",
                    "fix_dot_letters_checkbox",
                    "remove_reference_numbers_checkbox",
                    "keep_original_checkbox",
                    "smart_batch_short_sentences_checkbox",
                    "disable_watermark_checkbox",
                    "num_generations_input",
                    "normalize_audio_checkbox",
                    "normalize_method_dropdown",
                    "normalize_level_slider",
                    "normalize_tp_slider",
                    "normalize_lra_slider",
                    "num_candidates_slider",
                    "max_attempts_slider",
                    "bypass_whisper_checkbox",
                    "whisper_model_dropdown",
                    "enable_parallel_checkbox",
                    "num_parallel_workers_slider",
                    "use_longest_transcript_on_fail_checkbox",
                    "sound_words_field",
                    "use_faster_whisper_checkbox",
                    "separate_files_checkbox",
                ]
                if len(keys) != len(vals):
                    raise ValueError(f"[SETTINGS ERROR] collect_ui_settings: Number of values ({len(vals)}) does not match keys ({len(keys)})!")
                mapping = dict(zip(keys, vals))
                save_settings(mapping)
                return
             
            

            run_button.click(
                fn=lambda *args: (
                    collect_ui_settings(*([args[0]] + list(args[3:]))),  # text_input + rest of option fields (skipping file/audio)
                    generate_and_preview(*args)
                )[1],
                inputs=[
                    text_input,                   # 0
                    text_file_input,              # 1
                    ref_audio_input,              # 2
                    exaggeration_slider,          # 3
                    temp_slider,                  # 4
                    seed_input,                   # 5
                    cfg_weight_slider,            # 6
                    use_pyrnnoise_checkbox,       # 7  (NEW)
                    use_auto_editor_checkbox,     # 8
                    threshold_slider,             # 9
                    margin_slider,                #10
                    export_format_checkboxes,     #11
                    enable_batching_checkbox,     #12
                    to_lowercase_checkbox,        #13
                    normalize_spacing_checkbox,   #14
                    fix_dot_letters_checkbox,     #15
                    remove_reference_numbers_checkbox,   #16
                    keep_original_checkbox,       #17
                    smart_batch_short_sentences_checkbox,#18
                    disable_watermark_checkbox,   #19
                    num_generations_input,        #20
                    normalize_audio_checkbox,     #21
                    normalize_method_dropdown,    #22
                    normalize_level_slider,       #23
                    normalize_tp_slider,          #24
                    normalize_lra_slider,         #25
                    num_candidates_slider,        #26
                    max_attempts_slider,          #27
                    bypass_whisper_checkbox,      #28
                    whisper_model_dropdown,       #29
                    enable_parallel_checkbox,     #30
                    num_parallel_workers_slider,  #31
                    use_longest_transcript_on_fail_checkbox, #32
                    sound_words_field,            #33
                    use_faster_whisper_checkbox,  #34
                    separate_files_checkbox       #35
                ],
                outputs=[output_audio, audio_dropdown, audio_preview],
            )


            # === VC TAB: Voice Conversion Tab ===
            with gr.Tab("Voice Conversion (VC)"):
                gr.Markdown("## Voice Conversion\nConvert one speaker's voice to sound like another speaker using a target/reference voice audio.")
                with gr.Row():
                    vc_input_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input Audio (to convert)")
                    vc_target_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target Voice Audio")
                vc_pitch_shift = gr.Number(value=0, label="Pitch", step=0.5, interactive=True)
                vc_convert_btn = gr.Button("Run Voice Conversion")
                vc_output_files = gr.Files(label="Converted VC Audio File(s)")
                vc_output_audio = gr.Audio(label="VC Output Preview", interactive=True)

                def _vc_wrapper(input_audio_path, target_voice_audio_path, disable_watermark, pitch_shift):
                    # Defensive: None means Gradio didn't get file yet
                    if not input_audio_path or not os.path.exists(input_audio_path):
                        raise gr.Error("Please upload or record an input audio file.")
                    if not target_voice_audio_path or not os.path.exists(target_voice_audio_path):
                        raise gr.Error("Please upload or record a target/reference voice audio file.")

                    sr, out_wav = voice_conversion(
                        input_audio_path,
                        target_voice_audio_path,
                        disable_watermark=disable_watermark,
                        pitch_shift=pitch_shift
                    )
                    os.makedirs("output", exist_ok=True)
                    base = os.path.splitext(os.path.basename(input_audio_path))[0]
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
                    out_path = f"output/{base}_vc_{timestamp}.wav"
                    sf.write(out_path, out_wav, sr)
                    return [out_path], out_path  # Files and preview

                vc_convert_btn.click(
                    fn=_vc_wrapper,
                    inputs=[vc_input_audio, vc_target_audio, disable_watermark_checkbox, vc_pitch_shift],
                    outputs=[vc_output_files, vc_output_audio],
                )

        with gr.Accordion("Show Help / Instructions", open=False):
            gr.Markdown(
            """
            **What do all the main sliders and settings do?**
            ---

            ### **Text & Reference Input**
            - **Text Input:**  
              Enter the text you want to convert to speech. This can be any length, but for best results, keep sentences concise.  
            - **Text File(s) (.txt):**  
              Upload one or more plain text files. If files are uploaded, their contents override the text box input.  
              - *Tip: You can drag-and-drop multiple `.txt` files. If you do, you can choose to generate either one combined audio file, or separate audio files for each text file (see below).*
            - **Generate Separate Audio Files Per Text File:**  
              If checked, each uploaded text file will result in a separate audio file.  
              If unchecked, all text files are merged (in alphabetical order) and a single audio file is generated.
            - **Reference Audio:**  
              (Optional) Upload or record a sample of the target voice or style. The model will attempt to mimic this reference in generated speech.

            ---

            ### **TTS Voice/Emotion Controls**
            - **Emotion Exaggeration:**  
              Controls how dramatically emotions (like excitement, sadness, etc.) are expressed.  
              - *Low values* = more monotone/neutral  
              - *1.0* = model's default expressiveness  
              - *Above 1.0* = extra dramatic
            - **CFG Weight (Classifier-Free Guidance):**  
              Governs how strictly the output should follow the input text vs. being natural and expressive.  
              - *Higher values* = more literal, less expressive  
              - *Lower values* = more natural, possibly less faithful to the input
            - **Temperature:**  
              Adds randomness/variety to speech.  
              - *Low (0.1–0.5)* = more predictable, less expressive  
              - *High (0.7–1.2)* = more variety and unpredictability in speech patterns

            - **Random Seed (0 for random):**  
              Sets the base for the random number generator.  
              - *0* = pick a new random seed each time (unique results)  
              - *Any other number* = repeatable generations (for reproducibility/debugging)

            ---

            ### **Text Processing Options**
            - **Enable Sentence Batching (Max 300 chars):**  
              Chunks the input into groups of sentences, up to the specified maximum character length per batch.  
              - *Improves natural phrasing and makes TTS more efficient.*
            - **Smart-Append Short Sentences (if batching is off):**  
              If sentence batching is disabled, this option intelligently merges very short sentences together for smoother prosody.
            - **Convert Input Text to Lowercase:**  
              Automatically lowercases the input before synthesis.  
              - *May improve consistency in pronunciation for some models.*
            - **Normalize Spacing:**  
              Removes redundant spaces and blank lines, creating cleaner input for the model.
            - **Convert 'J.R.R.' to 'J R R':**  
              Automatically converts abbreviations written with periods to a spaced-out format (improves pronunciation of initials/names).

            ---

            ### **Audio Post-Processing**
            - **Post-process with Auto-Editor:**  
              Uses [auto-editor](https://github.com/WyattBlue/auto-editor) to automatically trim silences and clean up the audio, reducing stutters and small TTS artifacts.
            - **Auto-Editor Volume Threshold:**  
              Sets the loudness level below which audio is considered silence and removed.  
              - *Higher values = more aggressive trimming.*
            - **Auto-Editor Margin (seconds):**  
              Adds a buffer before and after detected audio to avoid cutting words or breaths.
            - **Keep Original WAV (before Auto-Editor):**  
              If enabled, the unprocessed audio is also saved, alongside the cleaned-up version.
            - **Normalize with ffmpeg (loudness/peak):**  
              Uses `ffmpeg` to adjust output volume.  
              - *Loudness normalization* matches the volume across different audio files.  
              - *Peak normalization* ensures audio doesn't exceed a certain volume.
            - **Normalization Method:**  
              - *ebu*: Broadcast-standard loudness normalization (good for consistent perceived loudness).  
              - *peak*: Simple normalization so the loudest part is at a fixed level.
            - **EBU Target Integrated Loudness (I, dB, ebu only):**  
              Target average loudness in decibels (usually -24 dB for TV, -16 dB for podcasts).
            - **EBU True Peak (TP, dB, ebu only):**  
              Maximum peak volume in dB (e.g., -2 dB to avoid digital clipping).
            - **EBU Loudness Range (LRA, ebu only):**  
              Controls the dynamic range of the output.  
              - *Lower values* = more compressed sound; *higher values* = more dynamic range.

            ---

            ### **Output & Export Options**
            - **Export Format:**  
              Choose one or more audio formats for export:  
              - *WAV*: Uncompressed, highest quality  
              - *MP3*: Compressed, smaller files, near-universal support  
              - *FLAC*: Lossless compression, smaller than WAV but no loss in quality  
              - *Tip: You can select multiple formats to export all at once.*
            - **Disable Perth Watermark:**  
              If enabled, disables the PerthNet audio watermarking (if the model applies it by default).  
              - *Recommended for privacy or when watermarking is not needed.*

            ---

            ### **Generation Controls**
            - **Number of Generations:**  
              Produces multiple unique audio outputs in one click (for variety or "takes").  
              - *All generations will have different random seeds (unless a fixed seed is set).*
            - **Number of Candidates Per Chunk:**  
              For each chunk, generate this many TTS variants and pick the best one (based on Whisper check or duration).  
              - *More candidates can reduce artifacts, but increases processing time and VRAM use.*
            - **Max Attempts Per Candidate (Whisper check retries):**  
              How many times to retry each candidate if the Whisper sync check fails.  
              - Will keep trying new variations up to this number per candidate when failing Whisper Sync validation.  
            - **Bypass Whisper Checking:**  
              If enabled, skips speech-to-text validation (faster but riskier—may allow more TTS mistakes).  
              - *When off, each candidate is checked using Whisper for accuracy.*

            ---

            ### **Whisper Sync Options**
            - **Whisper Sync Model (with VRAM requirements):**  
              Choose which Whisper model to use for automatic speech-to-text checking (to validate each TTS chunk and reduce artifacts). There are **two different backends** you can select:

              **1. OpenAI Whisper (official, more VRAM required):**
                - *OpenAI's original Whisper models offer high accuracy, but use more VRAM, especially at larger sizes.*
                - **VRAM usage (approximate, CUDA/float16):**
                    - tiny: ~1 GB
                    - base: ~1.2–2 GB
                    - small: ~2–3 GB
                    - medium: ~5–8 GB
                    - large: ~10–13 GB
                - *medium* (~5–8 GB VRAM) is a good compromise between speed and accuracy for most users.
                - **Use this if:**  
                  - You want the "classic" Whisper experience, or your GPU has ample VRAM.

              **2. faster-whisper (SYSTRAN, highly optimized):**
                - *This is a fast, memory-efficient reimplementation of Whisper. It is nearly as accurate as the official version, but uses far less VRAM and runs significantly faster, especially on modern NVIDIA GPUs.*
                - **VRAM usage (approximate, CUDA/float16):**
                    - tiny: ~0.5 GB
                    - base: ~0.7–1.0 GB
                    - small: ~1.2–1.7 GB
                    - medium: ~2.5–4.5 GB
                    - large: ~4.5–6.5 GB
                - *Even "large" can run comfortably on a 6 GB GPU!*
                - **Use this if:**  
                  - You want faster processing and/or have limited VRAM.

            - **Accuracy/Speed Tips:**
                - **tiny**/**base** are fastest but less accurate (good for quick checks, not critical applications).
                - **small**/**medium** are a good balance for most TTS validation use-cases.
                - **large** offers best accuracy, but is only practical on powerful GPUs.

            - **Which backend should I choose?**
                - **faster-whisper** is highly recommended for most users.  
                  It will check the "Use faster-whisper (SYSTRAN) backend" box.  
                  It is typically 2× faster and uses 30–60% less VRAM than official Whisper.
                - If you experience VRAM errors with OpenAI Whisper, switch to faster-whisper or a smaller model.
                - If you want to exactly match results from the original Whisper repo, use the OpenAI Whisper backend.

            - **Note:**  
                - Model size can affect TTS generation time and GPU memory use. If you get CUDA out-of-memory errors, try a smaller model or enable "faster-whisper".

            ---

            **Summary Table: Whisper Model VRAM Usage**

            | Model   | OpenAI Whisper VRAM | faster-whisper VRAM |
            |---------|---------------------|--------------------|
            | tiny    | ~1 GB               | ~0.5 GB            |
            | base    | ~1.2–2 GB           | ~0.7–1.0 GB        |
            | small   | ~2–3 GB             | ~1.2–1.7 GB        |
            | medium  | ~5–8 GB             | ~2.5–4.5 GB        |
            | large   | ~10–13 GB           | ~4.5–6.5 GB        |

            ---

            ### **Parallel Processing & Performance**
            - **Enable Parallel Chunk Processing:**  
              Speeds up synthesis by generating multiple audio chunks at the same time.  
              - *Uses more VRAM; can speed up batch synthesis a lot on powerful GPUs.*
            - **Parallel Workers:**  
              How many chunks to process in parallel.  
              - *Set to 1 for full sequential processing (lower VRAM, slower).*
              - *Higher = more speed, but may hit VRAM limits on consumer GPUs.*

            ---

            ### **How Candidate Selection Works**
            - For each chunk, the model creates the specified number of candidate audio variations.
            - If Whisper checking is enabled:  
              - Each candidate is transcribed, and the one with the closest match to the input text is chosen.
            - If Whisper is bypassed:  
              - The shortest-duration candidate is chosen (assumed best).
            - If all candidates fail validation after retries:  
              - The candidate with the highest Whisper score is used, or the one with the most text characters, depending on user settings.

            ---

            ### **Sound Words / Replacement (Advanced)**
            - **Sound Word List:**  
              (Advanced) Supply a list of word replacements in the provided format to automatically substitute or remove problematic words during synthesis.
              - *Format: "original=>replacement, nextword=>newword"*  
              - Can be used to fix tricky pronunciations or remove unwanted sound cues from the text.

            ---

            ### **Tips & Troubleshooting**
            - If you experience **slow Whisper checking or VRAM errors**, try:
              - Reducing the number of parallel workers
              - Switching to a smaller Whisper model
              - Reducing the number of candidates per chunk
            - If audio sounds choppy or cut off, try **raising the Auto-Editor margin**, or lowering the volume threshold.

            ---

            **Still have questions?**  
            This interface aims to expose every option for maximum control, but if you’re unsure, try using defaults for most sliders and options.
            """,
            elem_classes=["gr-text-center"]

            )

        # Pass through host/port/share from CLI if provided
        demo.launch(server_name='0.0.0.0', server_port=7860, share=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chatterbox-TTS Extended UI")
    parser.add_argument("--host", default=None, help="Host/IP to bind (e.g., 0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (e.g., 7860)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--public", action="store_true",
                        help="Shortcut for --host 0.0.0.0 (bind all interfaces)")

    args = parser.parse_args()

    # --public is a convenience alias
    if args.public and not args.host:
        args.host = "0.0.0.0"

    main(server_name=args.host, server_port=args.port, share=args.share)
CHATTER_PATCHED_EOF_MARKER_9f3a

    # Validar que o Chatter.py patcheado compila
    if python -m py_compile /workspace/Chatterbox-TTS-Extended/Chatter.py; then
        echo "[boot] OK: Chatter.py patcheado gravado e compila."
    else
        echo "[boot][ERRO] Chatter.py patcheado NAO compila!"
    fi

    deactivate
fi

# ----------------------------------------------------------------------------
# (6/7) Subir os servicos
#   ComfyUI usa --enable-cors-header (senao o proxy do RunPod devolve 403).
#   Os modelos so ocupam VRAM ao GERAR: use UM servico de cada vez p/ nao
#   estourar os 32GB da 5090 (gerar video E audio ao mesmo tempo = OOM).
# ----------------------------------------------------------------------------
echo "[boot] (6/7) Iniciando ComfyUI na porta 8188..."
cd /workspace/ComfyUI
nohup python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header \
    > /workspace/logs/comfyui.log 2>&1 &

echo "[boot] (6/7) Iniciando Chatterbox-TTS-Extended na porta 7860..."
cd /workspace/Chatterbox-TTS-Extended
source venv/bin/activate
nohup python Chatter.py \
    > /workspace/logs/chatterbox.log 2>&1 &

echo "[boot] ============================================================"
echo "[boot]  TUDO PRONTO!"
echo "[boot]   ComfyUI    -> porta 8188  (log: /workspace/logs/comfyui.log)"
echo "[boot]   Chatterbox -> porta 7860  (log: /workspace/logs/chatterbox.log)"
echo "[boot]  Acesse pelo botao Connect do pod. Os servicos levam mais"
echo "[boot]  ~1-2 min apos esta mensagem para terminarem de carregar."
echo "[boot] ============================================================"
tail -f /dev/null
