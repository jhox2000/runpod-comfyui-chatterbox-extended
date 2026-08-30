#!/bin/bash
# =============================================================================
# start-vast-h3-robo.sh — boot 1-click na Vast.ai
# ComfyUI + MiniMax H3 + ROBO de blocos encadeados, tudo pronto ao ligar.
#
# On-start do template Vast (uma linha):
# bash -c "mkdir -p /workspace && curl -fsSL https://raw.githubusercontent.com/jhox2000/runpod-comfyui-chatterbox-extended/refs/heads/main/start-vast-h3-robo.sh -o /workspace/boot.sh && bash /workspace/boot.sh"
#
# Variaveis opcionais (definir no template se precisar):
#   H3_PRECISION=pruned_int8_convrot | pruned_bf16    (padrao: pruned_int8_convrot)
#   H3_TEXT_ENCODER=nvfp4 | int8                      (padrao: nvfp4 na 5090, int8 nas demais)
#   H3_SKIP_R2V=1        pula o checkpoint REF2VA     (NAO use: a continuacao precisa dele)
# =============================================================================
mkdir -p /workspace/logs /workspace/scripts /workspace/projeto/imagens /workspace/filme
exec >> /workspace/logs/boot-h3.log 2>&1
set -e
echo "[boot] ====== $(date) ======"

echo "[boot] (1/8) Conectividade..."
curl -fsS -o /dev/null https://github.com || { echo "[boot] ERRO: sem acesso ao GitHub"; exit 1; }
curl -fsS -o /dev/null https://huggingface.co || { echo "[boot] ERRO: sem acesso ao HuggingFace"; exit 1; }
[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate || true
unset HF_HUB_ENABLE_HF_TRANSFER; export HF_XET_HIGH_PERFORMANCE=1

echo "[boot] (2/8) GPU e dependencias base..."
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo desconhecida)
echo "[boot]   GPU: $GPU"
case "$GPU" in *5090*|*B200*|*B100*|*RTX\ PRO*) BLACKWELL=1;; *) BLACKWELL=0;; esac
TE="${H3_TEXT_ENCODER:-}"
if [ -z "$TE" ]; then [ "$BLACKWELL" = "1" ] && TE=nvfp4 || TE=int8; fi
[ "$BLACKWELL" = "0" ] && echo "[boot]   AVISO: GPU nao e Blackwell -> text encoder $TE"
python3 -c "import torch" 2>/dev/null || pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) || echo "[boot] AVISO: ffmpeg nao instalou (o robo precisa dele)"

echo "[boot] (3/8) ComfyUI (o H3 recebe correcoes semanais, sempre atualiza)..."
cd /workspace
[ -d ComfyUI ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
cd /workspace/ComfyUI
git fetch --depth=1 origin master && git reset --hard FETCH_HEAD
pip install -q -r requirements.txt
pip install -q -U huggingface_hub hf_xet
grep -oE '__version__ = "[^"]+"' comfyui_version.py || true

echo "[boot] (4/8) Modelos do H3 (~68GB, pula o que ja existe)..."
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

echo "[boot] (5/8) Conferindo tamanhos..."
chk() { local g=$(du -BG "$1" | cut -f1 | tr -d G); [ "$g" -ge "$2" ] && echo "[boot]   [ok] $(basename $1) ${g}GB" || { echo "[boot] ERRO: $1 truncado (${g}GB)"; exit 1; }; }
chk "$M/diffusion_models/minimax_h3_fl2va_${PREC}.safetensors" 19
[ "${H3_SKIP_R2V:-0}" != "1" ] && chk "$M/diffusion_models/minimax_h3_ref2va_${PREC}.safetensors" 19
chk "$M/vae/minimax_h3_video_vae_fp16.safetensors" 4

echo "[boot] (6/8) Workflows oficiais..."
W=/workspace/ComfyUI/user/default/workflows
mkdir -p "$W" /workspace/ComfyUI/output/robo
T=https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates
curl -fsSL $T/video_minimax_h3_i2v.json | sed 's#"video/MiniMax_H3"#"robo/H3"#g' > "$W/1 - H3 Imagem para Video (I2V).json" || true
curl -fsSL $T/video_minimax_h3_r2v.json | sed 's#"video/MiniMax_H3"#"robo/H3"#g' > "$W/2 - H3 Referencia para Video (R2V).json" || true
curl -fsSL $T/video_minimax_h3_t2v.json | sed 's#"video/MiniMax_H3"#"robo/H3"#g' > "$W/3 - H3 Texto para Video (T2V).json" || true

echo "[boot] (7/8) Instalando o ROBO e buscando os workflows do robo no seu repo..."
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
                atual["clipes"].append(_clipe(linha[1:].strip(), dur_padrao))
            else:
                raise SystemExit(f"ERRO linha {n}: linha nao comeca com [N] nem com + :\n  {linha[:80]}")
    nums = [b["num"] for b in blocos]
    dup = {x for x in nums if nums.count(x) > 1}
    if dup:
        raise SystemExit(f"ERRO: numero de bloco repetido no txt: {sorted(dup)}")
    return blocos

def _clipe(texto, dur_padrao):
    dur = dur_padrao
    m = RE_DUR.match(texto)
    if m:
        dur = float(m.group(1))
        texto = texto[m.end():]
    if not texto:
        raise SystemExit("ERRO: prompt vazio depois do marcador de duracao.")
    return {"prompt": texto, "dur": dur}

def achar_imagem(pasta, num):
    for padrao in (f"{num:03d}", f"{num:02d}", f"{num}"):
        for ext in IMG_EXT:
            hits = sorted(glob.glob(os.path.join(pasta, padrao + "*" + ext)))
            for h in hits:
                base = os.path.basename(h)
                pref = base.split(".")[0]
                if pref in (f"{num:03d}", f"{num:02d}", str(num)):
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
    ap.add_argument("--saida", default="/workspace/filme")
    ap.add_argument("--comfy", default="/workspace/ComfyUI")
    ap.add_argument("--dur", type=float, default=10.0, help="duracao padrao por clipe (s)")
    ap.add_argument("--cauda", type=float, default=1.0, help="segundos do fim do clipe anterior usados como semente")
    ap.add_argument("--timeout-min", type=float, default=45.0)
    ap.add_argument("--tentativas", type=int, default=2, help="tentativas por clipe (1 = sem retry)")
    ap.add_argument("--ordem", choices=["rodadas", "filme"], default="rodadas",
                    help="rodadas = todos os elos 1, depois todos os 2... (menos troca de modelo)")
    ap.add_argument("--refazer", default=None, help="txt com numeros de bloco, um por linha")
    ap.add_argument("--apenas", default=None, help="lista de blocos, ex: 35,78")
    ap.add_argument("--validar", action="store_true")
    ap.add_argument("--sem-tar", action="store_true")
    a = ap.parse_args()

    blocos = parse_txt(a.prompts, a.dur)

    # filtro (refazer / apenas)
    filtro = None
    if a.refazer:
        with open(a.refazer, encoding="utf-8-sig") as f:
            filtro = {int(l.strip().lstrip("B").lstrip("0") or "0")
                      for l in f if l.strip() and not l.strip().startswith("#")}
    if a.apenas:
        filtro = {int(x) for x in a.apenas.replace(" ", "").split(",") if x}
    if filtro is not None:
        faltando = filtro - {b["num"] for b in blocos}
        if faltando:
            raise SystemExit(f"ERRO: blocos pedidos que nao existem no txt: {sorted(faltando)}")
        blocos = [b for b in blocos if b["num"] in filtro]

    # validacao
    total_clipes = sum(len(b["clipes"]) for b in blocos)
    total_seg = sum(c["dur"] for b in blocos for c in b["clipes"])
    problemas = []
    for b in blocos:
        if len(b["clipes"]) > 4:
            problemas.append(f"bloco {b['num']:03d}: {len(b['clipes'])} elos (maximo combinado: 4)")
        if achar_imagem(a.imagens, b["num"]) is None:
            problemas.append(f"bloco {b['num']:03d}: imagem nao encontrada em {a.imagens}")
    print(f"== {len(blocos)} blocos | {total_clipes} clipes | ~{total_seg/60:.1f} min de filme ==")
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

    # checkpoint
    estado = {"blocos": {}}
    if os.path.isfile(estado_path):
        estado = json.load(open(estado_path, encoding="utf-8"))
    def salvar_estado():
        json.dump(estado, open(estado_path, "w", encoding="utf-8"), indent=1)

    def st(num):
        return estado["blocos"].setdefault(f"{num:03d}", {"status": "pendente", "feitos": 0})

    # em modo refazer/apenas, forca os blocos filtrados a recomecarem do zero
    if filtro is not None:
        for b in blocos:
            estado["blocos"][f"{b['num']:03d}"] = {"status": "pendente", "feitos": 0}
        salvar_estado()

    # bloco parcialmente feito em rodada anterior recomeca do zero (bloco e atomico)
    for b in blocos:
        s = st(b["num"])
        if s["status"] == "pendente" and s["feitos"] > 0:
            s["feitos"] = 0
            for f in glob.glob(os.path.join(a.saida, f"B{b['num']:03d}_*")):
                os.remove(f)
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
            wf = patch(wf_ab, clipe["prompt"], clipe["dur"], prefixo, imagem=img_nome)
        else:
            anterior = caminho_clipe(num, elo - 1)
            cauda_nome = f"robo_cauda_{nome}.mp4"
            extrair_cauda(anterior, os.path.join(comfy_in, cauda_nome), a.cauda)
            wf = patch(wf_co, clipe["prompt"], clipe["dur"], prefixo, video=cauda_nome)
        log(f"[{nome}] gerando ({clipe['dur']:g}s)...")
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
            f.write("\n".join(str(int(n)) for n in sorted(ruins)) + "\n")
    log(f"FIM: {len(oks)} blocos ok, {len(ruins)} com falha. Resumo em {resumo}")

    # limpeza das caudas temporarias
    for f in glob.glob(os.path.join(comfy_in, "robo_cauda_*")) + glob.glob(os.path.join(comfy_in, "robo_img_*")):
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
for wf in wf_abertura.json wf_continuacao.json wf_abertura_turbo.json wf_continuacao_turbo.json; do
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
  /workspace/projeto/imagens/             (001.png, 002.png ...)
  /workspace/projeto/wf_abertura.json     (se o boot avisou que falta)
  /workspace/projeto/wf_continuacao.json  (se o boot avisou que falta)

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
=========================================================================
EOF_LEIA

echo "[boot] (8/8) Subindo o ComfyUI..."
cd /workspace/ComfyUI
pkill -f "main.py --listen" || true; sleep 2
nohup python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header > /workspace/logs/comfyui.log 2>&1 &
df -h /workspace | tail -1
echo "[boot] PRONTO. Leia o /workspace/LEIA-ME-ROBO.txt e suba os arquivos do projeto."
