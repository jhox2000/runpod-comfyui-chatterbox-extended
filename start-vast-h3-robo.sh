#!/bin/bash
# =============================================================================
# start-vast-h3-robo.sh — boot 1-click na Vast.ai
# ComfyUI + MiniMax H3 + ROBO de blocos encadeados + VIGIA (relanca o robo, reinicia o
# ComfyUI, backup de hora em hora no Google Drive via rclone), tudo pronto ao ligar.
#
# On-start do template Vast (uma linha):
# bash -c "mkdir -p /workspace && curl -fsSL https://raw.githubusercontent.com/jhox2000/runpod-comfyui-chatterbox-extended/refs/heads/main/start-vast-h3-robo.sh -o /workspace/boot.sh && bash /workspace/boot.sh"
#
# Variaveis opcionais (definir no template se precisar):
#   H3_PRECISION=pruned_int8_convrot | pruned_bf16    (padrao: pruned_int8_convrot)
#   H3_TEXT_ENCODER=nvfp4 | int8                      (padrao: nvfp4 na 5090, int8 nas demais)
#   H3_SKIP_R2V=1        pula o checkpoint REF2VA     (NAO use: a continuacao precisa dele)
# =============================================================================
mkdir -p /workspace/logs /workspace/scripts /workspace/projeto/imagens /workspace/filme /workspace/projeto/vozes
exec >> /workspace/logs/boot-h3.log 2>&1
set -e
echo "[boot] ====== $(date) ======"

echo "[boot] (1/9) Conectividade..."
curl -fsS -o /dev/null https://github.com || { echo "[boot] ERRO: sem acesso ao GitHub"; exit 1; }
curl -fsS -o /dev/null https://huggingface.co || { echo "[boot] ERRO: sem acesso ao HuggingFace"; exit 1; }
[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate || true
unset HF_HUB_ENABLE_HF_TRANSFER; export HF_XET_HIGH_PERFORMANCE=1

echo "[boot] (2/9) GPU e dependencias base..."
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo desconhecida)
echo "[boot]   GPU: $GPU"
case "$GPU" in *5090*|*B200*|*B100*|*RTX\ PRO*) BLACKWELL=1;; *) BLACKWELL=0;; esac
TE="${H3_TEXT_ENCODER:-}"
if [ -z "$TE" ]; then [ "$BLACKWELL" = "1" ] && TE=nvfp4 || TE=int8; fi
[ "$BLACKWELL" = "0" ] && echo "[boot]   AVISO: GPU nao e Blackwell -> text encoder $TE"
python3 -c "import torch" 2>/dev/null || pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) || echo "[boot] AVISO: ffmpeg nao instalou (o robo precisa dele)"
command -v rclone >/dev/null || (curl -fsSL https://rclone.org/install.sh | bash >/dev/null 2>&1 </dev/null) || echo "[boot] AVISO: rclone nao instalou (backup no Drive desligado)"

echo "[boot] (3/9) ComfyUI (o H3 recebe correcoes semanais, sempre atualiza)..."
cd /workspace
[ -d ComfyUI ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
cd /workspace/ComfyUI
git fetch --depth=1 origin master && git reset --hard FETCH_HEAD
pip install -q -r requirements.txt
pip install -q -U huggingface_hub hf_xet
grep -oE '__version__ = "[^"]+"' comfyui_version.py || true

echo "[boot] (4/9) Modelos do H3 (~68GB, pula o que ja existe)..."
M=/workspace/ComfyUI/models
mkdir -p "$M"/{diffusion_models,text_encoders,vae,loras,embeddings}
hf_get() {
  local f=$(basename "$2")
  [ -f "$3/$f" ] && { echo "[boot]   [ja existe] $f"; return 0; }
  echo "[boot]   baixando $f ..."
  python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$1', filename='$2', local_dir='/workspace/_dl_h3')"
  mv "/workspace/_dl_h3/$2" "$3/$f"
}
R=Comfy-Org/MiniMax-H3
PREC="${H3_PRECISION:-pruned_int8_convrot}"
hf_get $R "diffusion_models/minimax_h3_fl2va_${PREC}.safetensors" "$M/diffusion_models"
if [ "${H3_SKIP_R2V:-0}" != "1" ]; then
  hf_get $R "diffusion_models/minimax_h3_ref2va_${PREC}.safetensors" "$M/diffusion_models"
fi
if [ "$TE" = "nvfp4" ]; then
  hf_get $R text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors "$M/text_encoders"
else
  hf_get $R text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors "$M/text_encoders"
fi
hf_get $R vae/minimax_h3_video_vae_fp16.safetensors "$M/vae"
hf_get $R vae/minimax_h3_audio_vae_fp32.safetensors "$M/vae"
hf_get $R loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors "$M/loras"
hf_get $R loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors "$M/loras"
hf_get $R loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors "$M/loras"
hf_get $R embeddings/minimaxh3_art_is_explosion.safetensors "$M/embeddings"
rm -rf /workspace/_dl_h3

echo "[boot] (5/9) Conferindo tamanhos..."
chk() { local g=$(du -BG "$1" | cut -f1 | tr -d G); [ "$g" -ge "$2" ] && echo "[boot]   [ok] $(basename $1) ${g}GB" || { echo "[boot] ERRO: $1 truncado (${g}GB)"; exit 1; }; }
chk "$M/diffusion_models/minimax_h3_fl2va_${PREC}.safetensors" 19
[ "${H3_SKIP_R2V:-0}" != "1" ] && chk "$M/diffusion_models/minimax_h3_ref2va_${PREC}.safetensors" 19
chk "$M/vae/minimax_h3_video_vae_fp16.safetensors" 4

echo "[boot] (6/9) Workflows oficiais..."
W=/workspace/ComfyUI/user/default/workflows
mkdir -p "$W" /workspace/ComfyUI/output/robo
T=https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates
curl -fsSL $T/video_minimax_h3_i2v.json | sed 's#"video/MiniMax_H3"#"robo/H3"#g' > "$W/1 - H3 Imagem para Video (I2V).json" || true
curl -fsSL $T/video_minimax_h3_r2v.json | sed 's#"video/MiniMax_H3"#"robo/H3"#g' > "$W/2 - H3 Referencia para Video (R2V).json" || true
curl -fsSL $T/video_minimax_h3_t2v.json | sed 's#"video/MiniMax_H3"#"robo/H3"#g' > "$W/3 - H3 Texto para Video (T2V).json" || true

echo "[boot] (7/9) Instalando o ROBO e buscando os workflows do robo no seu repo..."
cat > /workspace/scripts/robo_h3.py << 'EOF_ROBO'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robo_h3.py — Robô de produção por BLOCOS ENCADEADOS no MiniMax H3 (ComfyUI).

Lê o prompts_videos.txt no formato [N] / +, gera cada bloco (abertura + extensões
encadeadas por cauda de vídeo+áudio), nomeia B035_2.mp4, loga falhas, pula o resto
do bloco quando um elo falha, tem checkpoint pra retomar, e no fim empacota tudo
num .tar pra baixar com um clique.

FORMATO DO TXT:
    [001] @10 prompt de abertura do bloco 1...
    + prompt da extensao 1...
    + @8 prompt da extensao 2...
    [002] prompt de abertura do bloco 2...
  - [N]  abre bloco novo e usa a imagem N (001.png etc.) da pasta de imagens
  - [N] vozes=sol,nico  -> abertura com VOZ UNICA por personagem (receita E1D):
        usa o wf_abertura_ref.json; a ordem dos nomes = <Audio 1>, <Audio 2>
        (= ordem em que falam no clipe). Os arquivos vem do vozes.json ao lado
        do prompts_videos.txt: {"sol": "vozes/sol.wav", "nico": "vozes/nico.wav"}.
        Bloco sem vozes= roda exatamente como sempre. Continuacoes nunca mudam
        (a voz clonada atravessa a emenda pela cauda).
  - +    extensao do bloco aberto (continua do fim do clipe anterior)
  - @N   (opcional, logo depois do marcador) duracao em segundos daquele clipe
  - linhas vazias e linhas comecando com # sao ignoradas

MODOS:
    python3 robo_h3.py --validar          -> so confere txt + imagens, nao gera nada
    python3 robo_h3.py                    -> rodada completa
    python3 robo_h3.py --refazer refazer.txt  -> refaz so os blocos listados (1 numero por linha)
    python3 robo_h3.py --apenas 35,78     -> roda so esses blocos (bom pra teste)

SETUP UNICO (uma vez, na interface do ComfyUI):
  1. Deixe o workflow de I2V funcionando e exporte:  Workflow -> Export (API)
     Salve como wf_abertura.json
  2. Monte o workflow de continuacao (LoadVideo -> MiniMaxH3AddGuide no frame 0)
     e exporte igual. Salve como wf_continuacao.json
  3. Nos DOIS workflows, renomeie os titulos dos nos (botao direito -> Title):
       ROBO_PROMPT   -> no do texto do prompt (CLIPTextEncode positivo)
       ROBO_IMAGEM   -> no LoadImage (so no de abertura)
       ROBO_VIDEO    -> no LoadVideo da cauda (so no de continuacao)
       ROBO_DURACAO  -> no que tem o campo duration
     (SaveVideo e seeds o robo acha sozinho)
"""

import argparse, json, os, re, shutil, subprocess, sys, time, random, glob, datetime
import urllib.request, urllib.error

COMFY = "http://127.0.0.1:8188"
VID_EXT = (".mp4", ".webm", ".mov", ".mkv")
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")

# ----------------------------------------------------------------------------- util

def agora():
    return datetime.datetime.now().strftime("%H:%M:%S")

class Log:
    def __init__(self, caminho):
        self.f = open(caminho, "a", encoding="utf-8")
    def __call__(self, msg):
        linha = f"[{agora()}] {msg}"
        print(linha, flush=True)
        self.f.write(linha + "\n"); self.f.flush()

def api(caminho, dados=None, timeout=60):
    url = COMFY + caminho
    if dados is not None:
        req = urllib.request.Request(url, json.dumps(dados).encode("utf-8"),
                                     {"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        corpo = r.read()
        return json.loads(corpo) if corpo else {}

# ----------------------------------------------------------------------------- parser do txt

RE_BLOCO = re.compile(r"^\[(\d+)\]\s*(.*)$")
RE_DUR = re.compile(r"^@(\d+(?:\.\d+)?)\s+")

def parse_txt(caminho, dur_padrao):
    blocos = []   # [{num:int, clipes:[{prompt, dur}]}]
    atual = None
    with open(caminho, "r", encoding="utf-8-sig") as f:
        for n, linha in enumerate(f, 1):
            linha = linha.rstrip("\n").rstrip("\r").strip()  # mata o \r do Windows
            if not linha or linha.startswith("#"):
                continue
            m = RE_BLOCO.match(linha)
            if m:
                atual = {"num": int(m.group(1)), "clipes": []}
                blocos.append(atual)
                resto = m.group(2).strip()
                if not resto:
                    raise SystemExit(f"ERRO linha {n}: bloco [{m.group(1)}] sem prompt de abertura na mesma linha.")
                atual["clipes"].append(_clipe(resto, dur_padrao))
            elif linha.startswith("+"):
                if atual is None:
                    raise SystemExit(f"ERRO linha {n}: linha '+' antes de qualquer bloco [N].")
                c = _clipe(linha[1:].strip(), dur_padrao)
                if c["vozes"]:
                    raise SystemExit(f"ERRO linha {n}: vozes= so vale na abertura [N]; continuacao (+) herda a voz pela cauda.")
                atual["clipes"].append(c)
            else:
                raise SystemExit(f"ERRO linha {n}: linha nao comeca com [N] nem com + :\n  {linha[:80]}")
    nums = [b["num"] for b in blocos]
    dup = {x for x in nums if nums.count(x) > 1}
    if dup:
        raise SystemExit(f"ERRO: numero de bloco repetido no txt: {sorted(dup)}")
    return blocos

RE_VOZES = re.compile(r"^vozes\s*=\s*([A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*)\s*", re.I)

def _clipe(texto, dur_padrao):
    dur = dur_padrao
    vozes = []
    # @N e vozes= podem vir em qualquer ordem, logo depois do marcador
    for _ in range(2):
        m = RE_DUR.match(texto)
        if m:
            dur = float(m.group(1))
            texto = texto[m.end():].lstrip()
            continue
        m = RE_VOZES.match(texto)
        if m:
            vozes = [v.strip().lower() for v in m.group(1).split(",") if v.strip()]
            texto = texto[m.end():].lstrip()
    if not texto:
        raise SystemExit("ERRO: prompt vazio depois do marcador de duracao/vozes.")
    if len(vozes) > 3:
        raise SystemExit(f"ERRO: vozes={','.join(vozes)} tem {len(vozes)} vozes (maximo 3 por clipe).")
    if len(set(vozes)) != len(vozes):
        raise SystemExit(f"ERRO: nome repetido em vozes={','.join(vozes)}.")
    return {"prompt": texto.replace(" || ", "\n\n").strip(), "dur": dur, "vozes": vozes}

def achar_imagem(pasta, num):
    for padrao in (f"{num:04d}", f"{num:03d}", f"{num:02d}", f"{num}"):
        for ext in IMG_EXT:
            hits = sorted(glob.glob(os.path.join(pasta, padrao + "*" + ext)))
            for h in hits:
                base = os.path.basename(h)
                pref = base.split(".")[0]
                if pref in (f"{num:04d}", f"{num:03d}", f"{num:02d}", str(num)):
                    return h
    return None

# ----------------------------------------------------------------------------- patch de workflow

def _por_titulo(wf, titulo):
    for nid, no in wf.items():
        if no.get("_meta", {}).get("title", "").strip().upper() == titulo:
            return nid, no
    return None, None

def _por_classe(wf, trechos):
    achados = [(nid, no) for nid, no in wf.items()
               if any(t.lower() in no.get("class_type", "").lower() for t in trechos)]
    return achados

def _set_input(no, chaves, valor):
    for c in chaves:
        if c in no.get("inputs", {}):
            no["inputs"][c] = valor
            return True
    return False

def patch(wf_base, prompt, dur, prefixo, imagem=None, video=None):
    wf = json.loads(json.dumps(wf_base))  # copia funda

    # prompt
    nid, no = _por_titulo(wf, "ROBO_PROMPT")
    if no is None:
        cands = _por_classe(wf, ["CLIPTextEncode"])
        if len(cands) == 1:
            no = cands[0][1]
        else:
            raise SystemExit("ERRO: nao achei o no do prompt. Renomeie o titulo dele para ROBO_PROMPT no workflow.")
    if not _set_input(no, ["text", "prompt", "string", "value"], prompt):
        raise SystemExit("ERRO: no ROBO_PROMPT nao tem campo de texto reconhecivel.")

    # duracao
    nid, no = _por_titulo(wf, "ROBO_DURACAO")
    if no is not None:
        if not _set_input(no, ["duration", "seconds", "length", "value"], dur):
            raise SystemExit("ERRO: no ROBO_DURACAO nao tem campo duration/seconds/length/value.")
    else:
        ok = False
        for nid2, no2 in _por_classe(wf, ["MiniMaxH3"]):
            if _set_input(no2, ["duration"], dur):
                ok = True
        if not ok:
            raise SystemExit("ERRO: nao achei onde setar a duracao. Renomeie o no certo para ROBO_DURACAO.")

    # imagem inicial (abertura)
    if imagem is not None:
        nid, no = _por_titulo(wf, "ROBO_IMAGEM")
        if no is None:
            cands = _por_classe(wf, ["LoadImage"])
            if len(cands) == 1:
                no = cands[0][1]
            else:
                raise SystemExit("ERRO: nao achei o LoadImage. Renomeie o titulo dele para ROBO_IMAGEM.")
        if not _set_input(no, ["image", "file", "filename"], imagem):
            raise SystemExit("ERRO: no ROBO_IMAGEM sem campo image/file.")

    # video de cauda (continuacao)
    if video is not None:
        nid, no = _por_titulo(wf, "ROBO_VIDEO")
        if no is None:
            cands = _por_classe(wf, ["LoadVideo"])
            if len(cands) == 1:
                no = cands[0][1]
            else:
                raise SystemExit("ERRO: nao achei o LoadVideo. Renomeie o titulo dele para ROBO_VIDEO.")
        if not _set_input(no, ["file", "video", "filename", "image"], video):
            raise SystemExit("ERRO: no ROBO_VIDEO sem campo file/video.")

    # prefixo de saida (todos os SaveVideo)
    achou_save = False
    for nid, no in wf.items():
        if "filename_prefix" in no.get("inputs", {}):
            no["inputs"]["filename_prefix"] = prefixo
            achou_save = True
    if not achou_save:
        raise SystemExit("ERRO: nenhum no com filename_prefix (SaveVideo) no workflow.")

    # seeds: randomiza tudo que parecer seed
    for nid, no in wf.items():
        for chave in ("seed", "noise_seed"):
            if chave in no.get("inputs", {}) and isinstance(no["inputs"][chave], (int, float)):
                no["inputs"][chave] = random.randint(0, 2**48)
    return wf

def _frames_grade(segundos, fps=24):
    # numero de frames na grade 17k+5 do H3 (arredonda pra cima)
    alvo = max(5, round(segundos * fps))
    return alvo + (5 - (alvo % 17)) % 17

def encoder_texto(comfy):
    hits = sorted(glob.glob(os.path.join(comfy, "models", "text_encoders", "qwen3vl*")))
    return os.path.basename(hits[0]) if hits else None

def patch_ref(wf_base, prompt, dur, prefixo, imagem, audios, clip_name=None):
    """Abertura com voz unica (wf_abertura_ref.json): imagem = <Picture 1>,
    audios[i] = <Audio i+1>. Slots de voz nao usados sao removidos do grafo."""
    wf = json.loads(json.dumps(wf_base))
    nid_gen, gen = _por_titulo(wf, "ROBO_PROMPT")
    if gen is None:
        cands = _por_classe(wf, ["MiniMaxH3ReferenceToVideo"])
        if len(cands) != 1:
            raise SystemExit("ERRO: wf_abertura_ref sem no ROBO_PROMPT (MiniMaxH3ReferenceToVideo).")
        nid_gen, gen = cands[0]
    if not _set_input(gen, ["prompt", "text"], prompt):
        raise SystemExit("ERRO: no ROBO_PROMPT do wf_abertura_ref sem campo prompt.")
    if "length" in gen["inputs"] and not isinstance(gen["inputs"]["length"], list):
        gen["inputs"]["length"] = _frames_grade(dur)
    # imagem
    nid, no = _por_titulo(wf, "ROBO_IMAGEM")
    if no is None:
        cands = _por_classe(wf, ["LoadImage"])
        if len(cands) != 1:
            raise SystemExit("ERRO: wf_abertura_ref sem no ROBO_IMAGEM.")
        no = cands[0][1]
    if not _set_input(no, ["image", "file", "filename"], imagem):
        raise SystemExit("ERRO: no ROBO_IMAGEM do wf_abertura_ref sem campo image.")
    # vozes: ROBO_VOZ_1..3
    for i in range(1, 4):
        nid, no = _por_titulo(wf, f"ROBO_VOZ_{i}")
        if i <= len(audios):
            if no is None:
                raise SystemExit(f"ERRO: wf_abertura_ref sem no ROBO_VOZ_{i} (precisa de {len(audios)} vozes).")
            if not _set_input(no, ["audio", "file", "filename"], audios[i - 1]):
                raise SystemExit(f"ERRO: no ROBO_VOZ_{i} sem campo audio.")
        elif no is not None:
            # slot vazio: tira o no e a ligacao no gerador
            del wf[nid]
            for k in list(gen["inputs"].keys()):
                v = gen["inputs"][k]
                if isinstance(v, list) and str(v[0]) == str(nid):
                    del gen["inputs"][k]
    # text encoder do pod (o wf vem com placeholder)
    if clip_name:
        for nid, no in _por_classe(wf, ["CLIPLoader"]):
            if "clip_name" in no["inputs"] and "TROCADO" in str(no["inputs"]["clip_name"]):
                no["inputs"]["clip_name"] = clip_name
    # prefixo + seeds
    achou_save = False
    for nid, no in wf.items():
        if "filename_prefix" in no.get("inputs", {}):
            no["inputs"]["filename_prefix"] = prefixo; achou_save = True
        for chave in ("seed", "noise_seed"):
            if chave in no.get("inputs", {}) and isinstance(no["inputs"][chave], (int, float)):
                no["inputs"][chave] = random.randint(0, 2**48)
    if not achou_save:
        raise SystemExit("ERRO: wf_abertura_ref sem SaveVideo (filename_prefix).")
    return wf

# ----------------------------------------------------------------------------- execucao no ComfyUI

def rodar_job(wf, timeout_s, log):
    envio = api("/prompt", {"prompt": wf})
    if "prompt_id" not in envio:
        raise RuntimeError(f"ComfyUI recusou o job: {json.dumps(envio)[:500]}")
    pid = envio["prompt_id"]
    t0 = time.time()
    while True:
        time.sleep(5)
        if time.time() - t0 > timeout_s:
            try: api("/interrupt", {})
            except Exception: pass
            raise RuntimeError(f"timeout ({int(timeout_s/60)} min) — job interrompido")
        try:
            hist = api(f"/history/{pid}")
        except Exception:
            continue
        if pid not in hist:
            continue
        item = hist[pid]
        status = item.get("status", {})
        if status.get("status_str") == "error":
            msgs = json.dumps(status.get("messages", []))[:600]
            raise RuntimeError(f"erro na execucao: {msgs}")
        if status.get("completed") or item.get("outputs"):
            return item.get("outputs", {})

def achar_saida(outputs, comfy_out, prefixo):
    # 1) pelo history
    for no in outputs.values():
        for lista in no.values():
            if isinstance(lista, list):
                for it in lista:
                    if isinstance(it, dict) and str(it.get("filename", "")).lower().endswith(VID_EXT):
                        p = os.path.join(comfy_out, it.get("subfolder", ""), it["filename"])
                        if os.path.isfile(p):
                            return p
    # 2) pelo disco (mais recente com o prefixo)
    hits = sorted(glob.glob(os.path.join(comfy_out, prefixo + "*")), key=os.path.getmtime)
    hits = [h for h in hits if h.lower().endswith(VID_EXT)]
    return hits[-1] if hits else None

def _frames_ancora(segundos, fps=24):
    # o AddGuide arredonda a cauda pra baixo na grade 17k+5 (5, 22, 39...)
    alvo = round(segundos * fps)
    if alvo < 5:
        return 1
    n = 5
    while n + 17 <= alvo:
        n += 17
    return n

def aparar_inicio(caminho, segundos):
    # remove do clipe de continuacao o trecho da cauda que ele repete no comeco
    tmp = caminho + ".tmp.mp4"
    cmd = ["ffmpeg", "-y", "-ss", f"{segundos:.4f}", "-i", caminho,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(tmp):
        raise RuntimeError("ffmpeg falhou ao aparar o inicio: " + r.stderr[-300:])
    os.replace(tmp, caminho)

def extrair_cauda(clipe, destino, segundos):
    # corte EXATO: mesmo tamanho da apara (grade de frames), video e audio juntos
    seg = _frames_ancora(segundos) / 24.0
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","csv=p=0",clipe], capture_output=True, text=True)
    dur = float(r.stdout.strip())
    ini = max(0.0, dur - seg)
    cmd = ["ffmpeg", "-y", "-i", clipe, "-ss", f"{ini:.4f}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", destino]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(destino):
        raise RuntimeError("ffmpeg falhou ao extrair a cauda: " + r.stderr[-300:])

def liberar_memoria(log):
    try:
        api("/free", {"unload_models": True, "free_memory": True})
        log("  memoria do ComfyUI liberada (unload models)")
        time.sleep(5)
    except Exception:
        pass

# ----------------------------------------------------------------------------- principal

def main():
    ap = argparse.ArgumentParser(description="Robo de blocos encadeados MiniMax H3")
    ap.add_argument("--prompts", default="/workspace/projeto/prompts_videos.txt")
    ap.add_argument("--imagens", default="/workspace/projeto/imagens")
    ap.add_argument("--wf-abertura", default="/workspace/projeto/wf_abertura_turbo.json")
    ap.add_argument("--wf-continuacao", default="/workspace/projeto/wf_continuacao_turbo.json")
    ap.add_argument("--wf-abertura-ref", default="/workspace/projeto/wf_abertura_ref.json",
                    help="workflow de abertura com voz de referencia (receita E1D)")
    ap.add_argument("--vozes", default=None, help="vozes.json (padrao: ao lado do prompts_videos.txt)")
    ap.add_argument("--saida", default="/workspace/filme")
    ap.add_argument("--comfy", default="/workspace/ComfyUI")
    ap.add_argument("--dur", type=float, default=10.0, help="duracao padrao por clipe (s)")
    ap.add_argument("--cauda", type=float, default=1.0, help="segundos do fim do clipe anterior usados como semente")
    ap.add_argument("--timeout-min", type=float, default=45.0)
    ap.add_argument("--tentativas", type=int, default=2, help="tentativas por clipe (1 = sem retry)")
    ap.add_argument("--ordem", choices=["rodadas", "filme"], default="rodadas",
                    help="rodadas = todos os elos 1, depois todos os 2... (menos troca de modelo)")
    ap.add_argument("--refazer", default=None, help="txt: 35 = bloco inteiro | 35:3 = refaz do clipe 3 ate o fim")
    ap.add_argument("--apenas", default=None, help="lista de blocos, ex: 35,78")
    ap.add_argument("--validar", action="store_true")
    ap.add_argument("--sem-tar", action="store_true")
    a = ap.parse_args()

    blocos = parse_txt(a.prompts, a.dur)

    caudas_dir = os.path.join(a.saida, "caudas")

    # filtro (refazer / apenas) — "35" refaz o bloco inteiro; "35:3" refaz do clipe 3 ate o fim
    filtro = None
    partir = {}
    if a.refazer:
        filtro = set()
        with open(a.refazer, encoding="utf-8-sig") as f:
            for l in f:
                l = l.strip()
                if not l or l.startswith("#"):
                    continue
                if ":" in l:
                    bn, en = l.split(":", 1)
                    num = int(bn.strip().lstrip("B").lstrip("0") or "0")
                    partir[num] = max(1, int(en.strip()))
                else:
                    num = int(l.lstrip("B").lstrip("0") or "0")
                    partir[num] = 1
                filtro.add(num)
    if a.apenas:
        filtro = {int(x) for x in a.apenas.replace(" ", "").split(",") if x}
        partir = {n: 1 for n in filtro}
    if filtro is not None:
        faltando = filtro - {b["num"] for b in blocos}
        if faltando:
            raise SystemExit(f"ERRO: blocos pedidos que nao existem no txt: {sorted(faltando)}")
        blocos = [b for b in blocos if b["num"] in filtro]

    # vozes unicas por personagem (opcional)
    projeto_dir = os.path.dirname(os.path.abspath(a.prompts))
    vozes_json = a.vozes or os.path.join(projeto_dir, "vozes.json")
    vozes_map = {}
    if os.path.isfile(vozes_json):
        try:
            bruto = json.load(open(vozes_json, encoding="utf-8-sig"))
            vozes_map = {str(k).strip().lower(): str(v).strip() for k, v in bruto.items()}
        except Exception as e:
            raise SystemExit(f"ERRO: vozes.json invalido: {e}")
    def voz_arquivo(tag):
        p = vozes_map.get(tag)
        if not p:
            return None
        return p if os.path.isabs(p) else os.path.join(projeto_dir, p)
    usa_vozes = any(b["clipes"][0]["vozes"] for b in blocos)

    # validacao
    total_clipes = sum(len(b["clipes"]) for b in blocos)
    total_seg = sum(c["dur"] for b in blocos for c in b["clipes"])
    problemas = []
    if usa_vozes and not os.path.isfile(a.wf_abertura_ref):
        problemas.append(f"ha blocos com vozes= mas nao existe {a.wf_abertura_ref} (o boot baixa do GitHub)")
    if usa_vozes and not vozes_map:
        problemas.append(f"ha blocos com vozes= mas nao existe {vozes_json}")
    for b in blocos:
        if len(b["clipes"]) > 4:
            problemas.append(f"bloco {b['num']:03d}: {len(b['clipes'])} elos (maximo combinado: 4)")
        if achar_imagem(a.imagens, b["num"]) is None:
            problemas.append(f"bloco {b['num']:03d}: imagem nao encontrada em {a.imagens}")
        ab = b["clipes"][0]
        for i, tag in enumerate(ab["vozes"], 1):
            arq = voz_arquivo(tag)
            if arq is None:
                problemas.append(f"bloco {b['num']:03d}: voz '{tag}' nao esta no vozes.json")
            elif not os.path.isfile(arq):
                problemas.append(f"bloco {b['num']:03d}: arquivo da voz '{tag}' nao existe: {arq}")
            if f"<Audio {i}>" not in ab["prompt"]:
                problemas.append(f"bloco {b['num']:03d}: vozes= tem '{tag}' na posicao {i} mas o prompt nao cita <Audio {i}>")
        if ab["vozes"] and "subject_definitions:" not in ab["prompt"]:
            problemas.append(f"bloco {b['num']:03d}: abertura com vozes= precisa do formato de referencia (subject_definitions: ...)")
        if not ab["vozes"] and "<Audio 1>" in ab["prompt"]:
            problemas.append(f"bloco {b['num']:03d}: prompt cita <Audio 1> mas a linha [N] nao tem vozes=")
    n_voz = sum(1 for b in blocos if b["clipes"][0]["vozes"])
    print(f"== {len(blocos)} blocos | {total_clipes} clipes | ~{total_seg/60:.1f} min de filme | {n_voz} aberturas com voz unica ==")
    if problemas:
        print("PROBLEMAS:"); [print("  - " + p) for p in problemas]
        raise SystemExit("Corrija o txt/imagens antes de rodar.")
    if a.validar:
        print("Validacao OK. Nada foi gerado (--validar).")
        return

    # pastas e logs
    os.makedirs(a.saida, exist_ok=True)
    meta = os.path.join(a.saida, "_robo"); os.makedirs(meta, exist_ok=True)
    log = Log(os.path.join(meta, "robo.log"))
    falhas_log = os.path.join(meta, "falhas.log")
    estado_path = os.path.join(meta, "estado.json")
    comfy_in = os.path.join(a.comfy, "input")
    comfy_out = os.path.join(a.comfy, "output")
    for p in (comfy_in, comfy_out):
        if not os.path.isdir(p):
            raise SystemExit(f"ERRO: pasta do ComfyUI nao existe: {p}")

    try:
        api("/system_stats")
    except Exception:
        raise SystemExit("ERRO: ComfyUI nao responde em 127.0.0.1:8188. Suba ele antes do robo.")

    wf_ab = json.load(open(a.wf_abertura, encoding="utf-8"))
    wf_co = json.load(open(a.wf_continuacao, encoding="utf-8"))
    wf_ref = json.load(open(a.wf_abertura_ref, encoding="utf-8")) if usa_vozes else None
    clip_name = encoder_texto(a.comfy)

    # checkpoint
    estado = {"blocos": {}}
    if os.path.isfile(estado_path):
        estado = json.load(open(estado_path, encoding="utf-8"))
    def salvar_estado():
        json.dump(estado, open(estado_path, "w", encoding="utf-8"), indent=1)

    def st(num):
        return estado["blocos"].setdefault(f"{num:03d}", {"status": "pendente", "feitos": 0})

    # em modo refazer/apenas: refaz do clipe pedido ate o fim, aproveitando os clipes bons
    if filtro is not None:
        for b in blocos:
            ini = partir.get(b["num"], 1)
            ini = min(max(1, ini), len(b["clipes"]))
            tem_clipe = os.path.isfile(os.path.join(a.saida, f"B{b['num']:03d}_{ini-1}.mp4"))
            tem_cauda = os.path.isfile(os.path.join(caudas_dir, f"cauda_B{b['num']:03d}_{ini}.mp4"))
            if ini > 1 and not tem_clipe and not tem_cauda:
                print(f"[B{b['num']:03d}] aviso: nao achei B{b['num']:03d}_{ini-1}.mp4 nem caudas/cauda_B{b['num']:03d}_{ini}.mp4 — refazendo o bloco inteiro")
                ini = 1
            for f2 in glob.glob(os.path.join(a.saida, f"B{b['num']:03d}_*.mp4")):
                m2 = re.search(r"_(\d+)\.mp4$", f2)
                if m2 and int(m2.group(1)) >= ini:
                    os.remove(f2)
            estado["blocos"][f"{b['num']:03d}"] = {"status": "pendente", "feitos": ini - 1}
        salvar_estado()

# retomada: aproveita os clipes ja feitos se os arquivos existem; senao recomeca o bloco
    if filtro is None:
        for b in blocos:
            s = st(b["num"])
            if s["status"] == "pendente" and s["feitos"] > 0:
                if os.path.isfile(os.path.join(a.saida, f"B{b['num']:03d}_{s['feitos']}.mp4")) or \
                   os.path.isfile(os.path.join(caudas_dir, f"cauda_B{b['num']:03d}_{s['feitos']+1}.mp4")):
                    for f2 in glob.glob(os.path.join(a.saida, f"B{b['num']:03d}_*.mp4")):
                        m2 = re.search(r"_(\d+)\.mp4$", f2)
                        if m2 and int(m2.group(1)) > s["feitos"]:
                            os.remove(f2)
                else:
                    s["feitos"] = 0
                    for f2 in glob.glob(os.path.join(a.saida, f"B{b['num']:03d}_*")):
                        os.remove(f2)
    salvar_estado()

    gerados = []

    def caminho_clipe(num, elo):
        return os.path.join(a.saida, f"B{num:03d}_{elo}.mp4")

    def executar(bloco, elo):
        num = bloco["num"]
        clipe = bloco["clipes"][elo - 1]
        nome = f"B{num:03d}_{elo}"
        prefixo = f"robo/{nome}"
        if elo == 1:
            img_src = achar_imagem(a.imagens, num)
            img_nome = f"robo_img_{num:03d}" + os.path.splitext(img_src)[1]
            shutil.copy2(img_src, os.path.join(comfy_in, img_nome))
            if clipe["vozes"]:
                audios = []
                for i, tag in enumerate(clipe["vozes"], 1):
                    src = voz_arquivo(tag)
                    nome_voz = f"robo_voz_{num:03d}_{i}_{tag}" + os.path.splitext(src)[1]
                    shutil.copy2(src, os.path.join(comfy_in, nome_voz))
                    audios.append(nome_voz)
                wf = patch_ref(wf_ref, clipe["prompt"], clipe["dur"], prefixo, img_nome, audios, clip_name)
            else:
                wf = patch(wf_ab, clipe["prompt"], clipe["dur"], prefixo, imagem=img_nome)
        else:
            # semente = cauda numerada guardada em filme/caudas/ (cauda_B035_3 = semente do clipe 3)
            os.makedirs(caudas_dir, exist_ok=True)
            cauda_guardada = os.path.join(caudas_dir, f"cauda_{nome}.mp4")
            anterior = caminho_clipe(num, elo - 1)
            if not os.path.isfile(cauda_guardada):
                if not os.path.isfile(anterior):
                    raise RuntimeError(f"sem semente: nao existe {anterior} nem {cauda_guardada}")
                extrair_cauda(anterior, cauda_guardada, a.cauda)
            cauda_nome = f"robo_cauda_{nome}.mp4"
            shutil.copy2(cauda_guardada, os.path.join(comfy_in, cauda_nome))
            wf = patch(wf_co, clipe["prompt"], clipe["dur"], prefixo, video=cauda_nome)
        log(f"[{nome}] gerando ({clipe['dur']:g}s{', voz unica: ' + ','.join(clipe['vozes']) if clipe['vozes'] else ''})...")
        t0 = time.time()
        outputs = rodar_job(wf, a.timeout_min * 60, log)
        saida = achar_saida(outputs, comfy_out, os.path.join("robo", nome))
        if not saida:
            raise RuntimeError("job terminou mas nenhum video foi encontrado na saida")
        destino = caminho_clipe(num, elo)
        shutil.copy2(saida, destino)
        if elo > 1:
            # a continuacao repete a cauda no comeco; apara pra emenda ficar exata
            aparar_inicio(destino, _frames_ancora(a.cauda) / 24.0)
        gerados.append(destino)
        if elo < len(bloco["clipes"]):
            # ja guarda a semente do proximo clipe (vai no tar; serve pra refazer em outra instancia)
            os.makedirs(caudas_dir, exist_ok=True)
            prox = os.path.join(caudas_dir, f"cauda_B{num:03d}_{elo+1}.mp4")
            extrair_cauda(destino, prox, a.cauda)
            gerados.append(prox)
        log(f"[{nome}] ok em {int(time.time()-t0)}s -> {destino}")

    def tentar(bloco, elo):
        num = bloco["num"]
        for tent in range(1, a.tentativas + 1):
            try:
                executar(bloco, elo)
                return True
            except Exception as e:
                log(f"[B{num:03d}_{elo}] FALHA tentativa {tent}/{a.tentativas}: {e}")
                liberar_memoria(log)
        with open(falhas_log, "a", encoding="utf-8") as f:
            f.write(f"B{num:03d}_{elo}\n")
        return False

    # fila de trabalho
    if a.ordem == "rodadas":
        max_elos = max(len(b["clipes"]) for b in blocos)
        fila = [(b, e) for e in range(1, max_elos + 1) for b in blocos if len(b["clipes"]) >= e]
    else:
        fila = [(b, e) for b in blocos for e in range(1, len(b["clipes"]) + 1)]

    inicio = time.time()
    for bloco, elo in fila:
        s = st(bloco["num"])
        if s["status"] in ("ok", "falhou"):
            continue
        if s["feitos"] != elo - 1:
            continue  # elo anterior falhou ou ainda nao chegou a vez (ordem rodadas)
        if tentar(bloco, elo):
            s["feitos"] = elo
            if s["feitos"] == len(bloco["clipes"]):
                s["status"] = "ok"
        else:
            s["status"] = "falhou"
            log(f"[B{bloco['num']:03d}] bloco marcado como FALHOU — elos restantes pulados")
        salvar_estado()

    # resumo + refazer sugerido
    oks = [n for n, s in estado["blocos"].items() if s["status"] == "ok"]
    ruins = [n for n, s in estado["blocos"].items() if s["status"] == "falhou"]
    resumo = os.path.join(meta, "resumo.txt")
    with open(resumo, "w", encoding="utf-8") as f:
        f.write(f"Rodada de {datetime.datetime.now():%d/%m/%Y %H:%M}\n")
        f.write(f"Blocos OK: {len(oks)}  |  Blocos com falha: {len(ruins)}\n")
        f.write(f"Clipes gerados nesta rodada: {len(gerados)}\n")
        f.write(f"Tempo total: {(time.time()-inicio)/3600:.1f} h\n")
        if ruins:
            f.write("Falharam: " + ", ".join(ruins) + "\n")
    if ruins:
        with open(os.path.join(meta, "refazer_sugerido.txt"), "w", encoding="utf-8") as f:
            for n in sorted(ruins):
                fe = estado["blocos"][n]["feitos"]
                f.write((f"{int(n)}:{fe+1}" if fe > 0 else str(int(n))) + "\n")
    log(f"FIM: {len(oks)} blocos ok, {len(ruins)} com falha. Resumo em {resumo}")

    # limpeza das caudas temporarias
    for f in glob.glob(os.path.join(comfy_in, "robo_cauda_*")) + glob.glob(os.path.join(comfy_in, "robo_img_*")) + glob.glob(os.path.join(comfy_in, "robo_voz_*")):
        try: os.remove(f)
        except OSError: pass

    # pacote pra download (so o que foi gerado nesta rodada + logs)
    if not a.sem_tar and gerados:
        tar_path = f"/workspace/filme_{datetime.datetime.now():%Y%m%d_%H%M}.tar"
        rel = [os.path.relpath(p, "/workspace") for p in gerados]
        rel += [os.path.relpath(os.path.join(meta, x), "/workspace")
                for x in ("resumo.txt", "robo.log") if os.path.isfile(os.path.join(meta, x))]
        if os.path.isfile(falhas_log):
            rel.append(os.path.relpath(falhas_log, "/workspace"))
        subprocess.run(["tar", "-cf", tar_path, "-C", "/workspace"] + rel, check=False)
        log(f"Pacote pronto pra baixar: {tar_path}")

if __name__ == "__main__":
    main()
EOF_ROBO
chmod +x /workspace/scripts/robo_h3.py

REPO=https://raw.githubusercontent.com/jhox2000/runpod-comfyui-chatterbox-extended/refs/heads/main
for wf in wf_abertura.json wf_continuacao.json wf_abertura_turbo.json wf_continuacao_turbo.json wf_abertura_ref.json; do
  if curl -fsSL "$REPO/$wf" -o "/workspace/projeto/$wf" 2>/dev/null && [ -s "/workspace/projeto/$wf" ]; then
    echo "[boot]   [ok] $wf baixado do seu GitHub"
  else
    rm -f "/workspace/projeto/$wf"
    echo "[boot]   [FALTA] $wf — exporte da interface (Export API) e commite no repo pra virar automatico"
  fi
done

cat > /workspace/LEIA-ME-ROBO.txt << 'EOF_LEIA'
========================= ROBO H3 — COLA RAPIDA =========================
Antes de rodar, arraste para as pastas:
  /workspace/projeto/prompts_videos.txt   (formato [N] / +)
  /workspace/projeto/vozes.json + vozes/  (SO se o filme usa voz unica: os .wav e o
                                           mapa nome -> arquivo; tambem na instancia de refazer)
  /workspace/projeto/imagens/             (001.png, 002.png ...)
  /workspace/projeto/wf_abertura.json     (se o boot avisou que falta)
  /workspace/projeto/wf_continuacao.json  (se o boot avisou que falta)
  /workspace/projeto/rclone.conf          (token do Google Drive; sem ele o backup fica desligado)
  /workspace/projeto/vast_api_key.txt     (chave de API da Vast: com ele o vigia REINICIA a instancia
                                           se a GPU falhar e PARA ela quando o filme termina)

Conferir tudo (nao gera nada):
  cd /workspace && python3 scripts/robo_h3.py --validar

Teste de 1 bloco, acompanhando ao vivo:
  cd /workspace && python3 scripts/robo_h3.py --apenas 1

Rodada completa (pode fechar o navegador / desligar o PC):
  cd /workspace && nohup python3 scripts/robo_h3.py > logs/robo.out 2>&1 & sleep 2; tail -f logs/robo.out

Rodada de correcao (depois da revisao):
  suba o refazer.txt em /workspace/projeto/ e rode:
  cd /workspace && nohup python3 scripts/robo_h3.py --refazer /workspace/projeto/refazer.txt > logs/robo.out 2>&1 & sleep 2; tail -f logs/robo.out

No final: baixe o /workspace/filme_DATA.tar (clipes + resumo + falhas).
Clipes ficam em /workspace/filme/ como B001_1.mp4, B001_2.mp4 ...

VIGIA (ja sobe sozinho no boot; log em /workspace/logs/vigia.log):
  REGRA: a PRIMEIRA largada do robo e sua (comando acima). Toda RETOMADA e do vigia.
  - a cada 5 min: se o robo caiu e a rodada nao chegou no FIM, relanca (retoma de onde parou)
  - se o ComfyUI parou de responder (com ou sem robo vivo), espera 5 min, reinicia ele e relanca o robo
  - se o robo ficar 60 min sem progresso, mata e reinicia tudo
  - backup no Drive: na hora em que o rclone.conf aparece e depois a cada 60 min
      /workspace/filme   -> gdrive:H3/filme    (clipes, caudas, estado do robo)
      /workspace/projeto -> gdrive:H3/projeto  (prompts, imagens, vozes; chaves NAO vao)
  - no FIM de uma rodada completa: backup final e, se existir vast_api_key.txt, PARA a instancia
    (rodada parcial - teste --apenas ou --refazer - nao para)
  - GPU em falha (ComfyUI nao sobe, erro de CUDA): pede REBOOT da instancia a Vast (precisa do
    vast_api_key.txt). Max 3 reboots em 6h; depois faz backup final e PARA a instancia = placa
    ruim, alugue outra maquina e suba o rclone.conf que o vigia restaura tudo do Drive.
  POD NOVO DEPOIS DE PERDER A INSTANCIA: suba SO o rclone.conf em /workspace/projeto/ e espere.
    Em ate 5 min o vigia restaura tudo do Drive e relanca o robo sozinho, de onde parou.
  ATENCAO: ao comecar um FILME NOVO, renomeie antes as pastas H3/filme e H3/projeto no Drive
  (ex: H3/filme_nadia, H3/projeto_nadia), senao o vigia restaura o filme antigo no pod novo.
  NAO rode o boot.sh na mao com o robo trabalhando (ele reinicia o ComfyUI e derruba o clipe).
  Ver o que o vigia fez:  cat /workspace/logs/vigia.log
  Parar o vigia:          pkill -f scripts/vigia.sh
=========================================================================
EOF_LEIA

cat > /workspace/scripts/vigia.sh << 'EOF_VIGIA'
#!/bin/bash
# vigia.sh (v4) - vigia do robo H3. Fica ligado desde o boot e, a cada 5 min:
#   - se o robo caiu e a rodada nao terminou, relanca (retoma de onde parou)
#   - se o ComfyUI parou de responder (com ou sem robo vivo), reinicia ele e relanca o robo
#   - se o robo ficar 60 min sem progresso, mata e reinicia tudo
#   - restaura projeto e clipes do Drive quando o disco esta vazio (pod novo)
#   - backup de hora em hora no Drive (e na hora, assim que o rclone.conf aparece)
#   - no FIM do filme: ultimo backup e, se existir vast_api_key.txt, para a instancia
#   - GPU em falha (ComfyUI nao sobe 2x seguidas com erro de CUDA): pede REBOOT da instancia
#     pela API da Vast (precisa do vast_api_key.txt); max 3 reboots em 6h, depois faz o backup
#     final e PARA a instancia (placa ruim - alugue outra maquina e retome pelo Drive)
# Regra: a PRIMEIRA largada do robo e sua (cd /workspace && nohup python3 scripts/robo_h3.py ...).
# O vigia so relanca sozinho quando ja existe um robo.log, ou seja, uma rodada ja comecou.
LOG=/workspace/logs/vigia.log
ROBO_OUT=/workspace/logs/robo.out
ROBO_LOG=/workspace/filme/_robo/robo.log
PROJ=/workspace/projeto/prompts_videos.txt
CONF=/workspace/projeto/rclone.conf
VAST_KEY=/workspace/projeto/vast_api_key.txt
REMOTO="gdrive:H3/filme"          # clipes, caudas e estado do robo
REMOTO_PROJ="gdrive:H3/projeto"   # prompts, imagens, vozes (rclone.conf e vast_api_key.txt ficam de fora)
CHECA=300        # checa a cada 5 min
BACKUP=3600      # backup a cada 60 min
TRAVADO=3600     # 60 min sem progresso = travado
ESPERA_COMFY=300 # espera ate 5 min o ComfyUI responder antes de reinicia-lo
MAX_RELANCA=3    # relancadas seguidas sem progresso antes de desistir
MAX_REBOOTS=3    # reboots pedidos a Vast dentro de JANELA_REBOOT antes de desistir da placa
JANELA_REBOOT=21600
REBOOTS_ARQ=/workspace/logs/reboots.txt   # sobrevive ao reboot: 1 timestamp por linha

log(){ echo "[$(date '+%d/%m %H:%M')] $*" >> "$LOG"; }
robo_vivo(){ pgrep -f "python3 scripts/robo_h3.py" >/dev/null; }
comfy_ok(){ [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8188/)" = "200" ]; }
terminou(){ [ -f "$ROBO_LOG" ] && tail -n 3 "$ROBO_LOG" | grep -q "FIM:"; }
n_clipes(){ ls /workspace/filme/*.mp4 2>/dev/null | wc -l; }
idade_log(){ if [ -f "$ROBO_LOG" ]; then echo $(( $(date +%s) - $(stat -c %Y "$ROBO_LOG") )); else echo 0; fi; }
tem_drive(){ [ -f "$CONF" ] && command -v rclone >/dev/null; }
rc(){ rclone --config "$CONF" "$@"; }

espera_comfy(){
  local t=0
  while [ $t -lt $ESPERA_COMFY ]; do comfy_ok && return 0; sleep 10; t=$((t+10)); done
  return 1
}
sobe_comfy(){
  log "ComfyUI nao responde - reiniciando"
  pkill -f "main.py --listen"; sleep 5
  cd /workspace/ComfyUI && nohup python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header > /workspace/logs/comfy.out 2>&1 &
  if espera_comfy; then log "ComfyUI de volta"; return 0; fi
  log "ERRO: ComfyUI nao voltou em 5 min"; return 1
}
garante_comfy(){ comfy_ok && return 0; espera_comfy && return 0; sobe_comfy; }
gpu_com_falha(){ grep -qiE "CUDA unknown error|CUDA not available|no CUDA-capable|CUDA error|CUDA driver" /workspace/logs/comfy.out 2>/dev/null; }
vast_id(){ local id=$(hostname | sed -n 's/^C\.\([0-9]\+\)$/\1/p'); [ -z "$id" ] && id="$CONTAINER_ID"; echo "$id"; }
vast_api(){  # vast_api METODO CAMINHO [JSON]
  local key=$(tr -d ' \r\n' < "$VAST_KEY")
  curl -s -m 30 -X "$1" "https://console.vast.ai/api/v0$2" -H "Authorization: Bearer $key" -H "Content-Type: application/json" ${3:+-d "$3"}
}
reboots_recentes(){ local agora=$(date +%s); [ -f "$REBOOTS_ARQ" ] && awk -v a="$agora" -v j="$JANELA_REBOOT" 'a-$1<j' "$REBOOTS_ARQ" | wc -l || echo 0; }
pede_reboot(){
  [ -f "$VAST_KEY" ] || { log "GPU em falha e sem $VAST_KEY: nao consigo pedir reboot - reinicie pelo painel da Vast"; return 1; }
  local n=$(reboots_recentes)
  if [ "$n" -ge "$MAX_REBOOTS" ]; then
    log "GPU em falha pela ${n}a vez em 6h: placa ruim. Backup final e PARANDO a instancia - alugue outra e suba o rclone.conf"
    backup; para_instancia; sleep 600; return 1
  fi
  local id=$(vast_id); [ -z "$id" ] && { log "ERRO: nao descobri o ID da instancia; reinicie pelo painel"; return 1; }
  date +%s >> "$REBOOTS_ARQ"
  log "GPU em falha (ComfyUI nao sobe, erro de CUDA) - pedindo REBOOT da instancia $id a Vast (reboot $((n+1))/$MAX_REBOOTS em 6h)"
  local resp=$(vast_api PUT "/instances/reboot/$id/" '{}')
  log "resposta da Vast: ${resp:0:200}"
  sleep 600   # o reboot derruba este processo; se nao derrubar, o loop continua e tenta de novo
}
mata_robo(){ pkill -f "python3 scripts/robo_h3.py"; sleep 3; }
lanca_robo(){
  cd /workspace && nohup python3 scripts/robo_h3.py >> "$ROBO_OUT" 2>&1 &
  relancadas=$((relancadas+1)); log "robo relancado (tentativa $relancadas/$MAX_RELANCA) - retoma de onde parou"
}
backup(){
  tem_drive || { log "backup pulado: falta $CONF ou o rclone"; return 1; }
  local ok=0
  if rc copy /workspace/filme "$REMOTO" -q 2>>"$LOG"; then
    log "backup ok: $(rc lsf "$REMOTO" 2>/dev/null | grep -c '\.mp4$') clipes no Drive"
  else
    log "ERRO no backup dos clipes (ver linhas acima)"; ok=1
  fi
  if [ -f "$PROJ" ]; then
    if rc copy /workspace/projeto "$REMOTO_PROJ" --exclude rclone.conf --exclude vast_api_key.txt -q 2>>"$LOG"; then
      log "backup do projeto ok (prompts, imagens, vozes)"
    else
      log "ERRO no backup do projeto"; ok=1
    fi
  fi
  return $ok
}
restaura(){
  # so faz sentido com Drive; roda a cada passada ate conseguir (o rclone.conf pode chegar depois do boot)
  tem_drive || return 1
  if [ ! -f "$PROJ" ]; then
    if rc lsf "$REMOTO_PROJ" 2>/dev/null | grep -q prompts_videos.txt; then
      log "restauracao: projeto ausente no disco - baixando do Drive (prompts, imagens, vozes)"
      rc copy "$REMOTO_PROJ" /workspace/projeto -q 2>>"$LOG" \
        && log "restauracao do projeto ok" || { log "ERRO na restauracao do projeto"; return 1; }
    fi
  fi
  if [ "$(n_clipes)" -eq 0 ]; then
    local r=$(rc lsf "$REMOTO" 2>/dev/null | grep -c '\.mp4$')
    if [ "$r" -gt 0 ]; then
      log "restauracao: disco vazio e Drive com $r clipes - baixando (clipes, caudas, estado do robo)"
      rc copy "$REMOTO" /workspace/filme -q 2>>"$LOG" \
        && log "restauracao ok: $(n_clipes) clipes no disco - o robo vai retomar de onde parou" \
        || { log "ERRO na restauracao dos clipes"; return 1; }
    fi
  fi
  return 0
}
rodada_completa(){
  # a rodada que acabou cobre todos os blocos do txt? (teste --apenas ou --refazer parcial = nao)
  local fim=$(grep "FIM:" "$ROBO_LOG" | tail -n 1)
  local oks=$(echo "$fim" | sed -n 's/.*FIM: \([0-9]\+\) blocos ok, \([0-9]\+\) com falha.*/\1/p')
  local fal=$(echo "$fim" | sed -n 's/.*FIM: \([0-9]\+\) blocos ok, \([0-9]\+\) com falha.*/\2/p')
  local tot=$(grep -cE '^\[[0-9]+\]' "$PROJ" 2>/dev/null)
  [ -n "$oks" ] && [ -n "$fal" ] && [ -n "$tot" ] && [ $((oks+fal)) -ge "$tot" ]
}
para_instancia(){
  [ -f "$VAST_KEY" ] || { log "sem $VAST_KEY, a instancia fica ligada (pare pelo painel)"; return; }
  local id=$(vast_id); [ -z "$id" ] && { log "ERRO: nao descobri o ID da instancia; pare pelo painel"; return; }
  log "pedindo a Vast pra PARAR a instancia $id"
  local resp=$(vast_api PUT "/instances/$id/" '{"state":"stopped"}')
  log "resposta da Vast: ${resp:0:200}"
}

relancadas=0; ultimo_backup=0; restaurado=0; tinha_conf=0; fim_tratado=0; falhas_comfy=0
log "vigia iniciado - $(n_clipes) clipes no disco"
ultimo_n=$(n_clipes)
# um FIM que ja existia quando o vigia subiu nao e novidade: nao para a instancia por causa dele
terminou && fim_tratado=1
while true; do
  # restauracao (pod novo): tenta a cada passada ate conseguir
  if [ "$restaurado" -eq 0 ] && restaura; then restaurado=1; terminou && fim_tratado=1; fi

  n=$(n_clipes)
  if [ "$n" -gt "$ultimo_n" ]; then relancadas=0; ultimo_n=$n; fi
  uso=$(df /workspace | awk 'NR==2{print $5+0}'); [ "$uso" -ge 90 ] && log "AVISO: disco em ${uso}%"

  if [ -f "$PROJ" ] && [ -f "$ROBO_LOG" ] && ! terminou; then
    if robo_vivo; then
      if ! comfy_ok; then
        log "ComfyUI parou de responder com o robo vivo"
        if ! espera_comfy; then
          mata_robo
          if sobe_comfy; then falhas_comfy=0; lanca_robo; else falhas_comfy=$((falhas_comfy+1)); fi
        fi
      elif [ "$(idade_log)" -gt "$TRAVADO" ]; then
        log "robo sem progresso ha $(( $(idade_log)/60 )) min - matando e reiniciando tudo"
        mata_robo
        if sobe_comfy; then falhas_comfy=0; lanca_robo; else falhas_comfy=$((falhas_comfy+1)); fi
      fi
    else
      if [ "$relancadas" -lt "$MAX_RELANCA" ]; then
        log "robo nao esta rodando e o filme nao terminou"
        if garante_comfy; then falhas_comfy=0; lanca_robo; else falhas_comfy=$((falhas_comfy+1)); fi
      elif [ "$relancadas" -eq "$MAX_RELANCA" ]; then
        log "DESISTI: $MAX_RELANCA relancadas sem progresso - olhe $ROBO_OUT e relance na mao"; relancadas=$((relancadas+1))
      fi
    fi
  fi

  # GPU em falha: ComfyUI nao sobe 2x seguidas (ou 1x com erro de CUDA explicito) -> reboot pela Vast
  if [ "$falhas_comfy" -ge 2 ] || { [ "$falhas_comfy" -ge 1 ] && gpu_com_falha; }; then
    pede_reboot && falhas_comfy=0
  fi

  # backup: de hora em hora, e na hora em que o rclone.conf aparecer
  agora=$(date +%s)
  if [ -f "$CONF" ] && [ "$tinha_conf" -eq 0 ]; then tinha_conf=1; ultimo_backup=0; log "rclone.conf encontrado - backup no Drive ligado"; fi
  if [ $((agora-ultimo_backup)) -ge $BACKUP ]; then backup; ultimo_backup=$agora; fi

  # fim do filme: ultimo backup e parada da instancia (uma vez so)
  if [ "$fim_tratado" -eq 0 ] && [ -f "$ROBO_LOG" ] && terminou && ! robo_vivo; then
    fim_tratado=1
    log "FIM detectado - fazendo o backup final"
    if ! tem_drive; then
      log "sem Drive configurado: nada a salvar, instancia continua ligada (pare pelo painel)"
    elif ! backup; then
      log "backup final com erro - NAO vou parar a instancia; confira o Drive"
    elif rodada_completa; then
      para_instancia
    else
      log "rodada parcial (teste --apenas ou --refazer): instancia continua ligada"
    fi
  fi

  sleep $CHECA
done
EOF_VIGIA
chmod +x /workspace/scripts/vigia.sh

echo "[boot] (8/9) Subindo o ComfyUI..."
cd /workspace/ComfyUI
pkill -f "main.py --listen" || true; sleep 2
nohup python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header > /workspace/logs/comfyui.log 2>&1 &

echo "[boot] (9/9) Subindo o VIGIA (relanca o robo, reinicia o ComfyUI, backup no Drive)..."
pkill -f "scripts/vigia.sh" || true; sleep 1
nohup setsid bash /workspace/scripts/vigia.sh > /workspace/logs/vigia.out 2>&1 &
[ -f /workspace/projeto/rclone.conf ] && echo "[boot]   rclone.conf encontrado: backup no Drive ligado" || echo "[boot]   sem /workspace/projeto/rclone.conf: backup no Drive DESLIGADO (suba o arquivo em projeto/)"
df -h /workspace | tail -1
echo "[boot] PRONTO. Leia o /workspace/LEIA-ME-ROBO.txt e suba os arquivos do projeto."
