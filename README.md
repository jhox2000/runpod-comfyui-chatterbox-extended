# RunPod 1-Click Template: ComfyUI (Wan 2.2 + Qwen-Image) + Chatterbox Multilingual TTS

A self-contained RunPod template that boots a ready-to-use content-production
environment. One deploy gives you, with **no manual setup**, a full pipeline for:

- **Image generation** (text → image) with Qwen-Image 2512 + Lightning 4-step LoRA.
- **Image editing** with Qwen-Image-Edit 2511 — keep the *same face* across scenes,
  or combine *two people* into one image.
- **Image-to-video** with Wan 2.2 14B I2V (fp8 + lightx2v 4-step LoRAs, ~5x faster).
- **Batch automation:** drop a list of prompts and the pod generates **all** images
  in order, then animates **all** of them into videos in order — hands-free.
- **Multilingual text-to-speech** with voice cloning, via
  [petermg/Chatterbox-TTS-Extended](https://github.com/petermg/Chatterbox-TTS-Extended)
  patched to use [Resemble AI's Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox)
  (23 languages), pre-set to a production configuration (Spanish default).

Deploy the pod, wait for the first boot, and use it.

---

## What's included

| Service | Port | Purpose |
|---|---|---|
| ComfyUI | 8188 | Image generation, image editing, and image-to-video |
| Chatterbox-TTS-Extended | 7860 | Multilingual TTS with voice cloning |
| JupyterLab | 8888 | File access / terminal (prompt files, downloads, debugging) |

### Pre-installed ComfyUI workflows (sidebar → Workflows)

Five ready-to-run workflows appear in the ComfyUI sidebar on first open — one click each:

1. **1 - Video (Wan 2.2)** — animate a single image into a short video.
2. **2 - Gerar Imagem (Qwen)** — single image from a text prompt (4-step Lightning).
3. **3 - Editar Rosto 1 e 2 pessoas (Qwen)** — edit keeping the same face / merge two people.
4. **4 - Gerar Imagens em Lote (Qwen)** — batch: read `prompts.txt`, generate all images.
5. **5 - Gerar Videos em Lote (Wan)** — batch: animate a folder of images into videos.

Each workflow includes on-canvas notes with aspect-ratio presets and (for the batch
ones) step-by-step usage instructions.

### Custom nodes installed on boot

- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- [Comfyui-QwenEditUtils](https://github.com/lrzjason/Comfyui-QwenEditUtils) — required by the image editor.
- [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui) — reads prompts line-by-line and loads images in sequence (batch). **Pinned** to a known-good commit.
- [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) — video helpers. **Pinned** to a known-good commit.

---

## Models (pre-downloaded on first boot, ~70 GB total)

**Wan 2.2 I2V (video):**
- `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`
- `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
- `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors`
- `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors`
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `wan_2.1_vae.safetensors`

**Qwen-Image (image generation + editing):**
- `qwen_image_2512_fp8_e4m3fn.safetensors` (generator)
- `Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors` (generator LoRA)
- `qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors` (editor)
- `qwen_2.5_vl_7b_fp8_scaled.safetensors` (text encoder, shared) + `qwen_image_vae.safetensors` (VAE, shared)

**Chatterbox Multilingual** weights are auto-downloaded by the library on first launch (~3.2 GB VRAM).

---

## Requirements

- **GPU:** RTX 5090 (Blackwell, sm_120) recommended and tested. Any GPU with 24 GB+ VRAM and PyTorch 2.7+/CUDA 12.8 support should work, but only the 5090 is verified.
- **Container image:** `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **Container disk:** **150 GB** (models + dependencies; a smaller disk can fail mid-download).
- **HTTP ports exposed:** 7860, 8188, 8888.
- **Environment variable:** `HF_TOKEN` (see below) — strongly recommended.

---

## Recommended: set `HF_TOKEN` for fast downloads

Hugging Face throttles **anonymous** downloads, which can make the first boot crawl
(or stall). A free read-only token removes the throttle.

1. Create a free token at Hugging Face → Settings → Access Tokens → **Read**.
2. In your RunPod **template settings**, add an **Environment Variable**:
   - **Key:** `HF_TOKEN`
   - **Value:** your `hf_...` token

The download library reads `HF_TOKEN` automatically — **no change to `start.sh` is
needed**, and the token stays out of the (public) script.

---

## How to use

1. Deploy a pod from this template on RunPod.
2. Wait for the first boot (**~15-20 minutes** with `HF_TOKEN` set). Follow progress in the pod's **Logs** tab.
3. When the log shows `[boot] TUDO PRONTO!`, give the services another **1-2 minutes** to load.
4. Click **Connect** and open the service you want:
   - **8188** → ComfyUI · **7860** → Chatterbox · **8888** → JupyterLab.

---

## Batch automation pipeline

The batch is **two sequential phases** (Wan and Qwen can't share VRAM, so images first, then videos).

### Phase 1 — Images from a prompt list
1. In JupyterLab, drop a **`prompts.txt`** into `ComfyUI/input/`
   (plain UTF-8 text, **one prompt per line**, no numbering, no curly quotes — don't use Word/Docs).
2. Open **"4 - Gerar Imagens em Lote (Qwen)"**.
3. Change the `label` field (e.g. `v2`) to reset the line counter; set **batch count = number of prompts**; click **Run** once.
4. Output: `ComfyUI/output/` → `lote_00001_.png`, `lote_00002_.png`, ... (in prompt order).

### Phase 2 — Videos from the images
1. Open **"5 - Gerar Videos em Lote (Wan)"** (reads `ComfyUI/output/`, pattern `lote*`).
2. Change `label` to start at the first image; set **batch count = number of images**; click **Run** once.
3. Output: `ComfyUI/output/video/` → `Wan2.2_..._00001_.mp4`, ... (~1 min per video on a 5090).

### Getting your files out
Videos don't preview well in JupyterLab. Watch them in ComfyUI, or **right-click the
`output` folder → "Download as Archive"** to grab everything as a `.zip`.

### Aspect-ratio presets
**Image** (set `largura`/`altura`): 1:1 `1328x1328` · 16:9 `1664x928` · 9:16 `928x1664` · 4:3 `1472x1104` · 3:4 `1104x1472`.
**Video** (set `width`/`height`, bigger = slower): 16:9 `832x480` or `1280x720` · 9:16 `480x832` or `720x1280` · 1:1 `720x720`. Match the video aspect ratio to your images.

---

## Chatterbox: production defaults

The UI now opens **directly in a production configuration** (no manual setup each time):

- Language: **Spanish (es)** by default (changeable in the dropdown).
- Number of Candidates Per Chunk: **1**
- Max Attempts Per Candidate: **2**
- Whisper checking: **ON**, using faster-whisper — keeps the quality safety net that
  catches artifacts in long audiobooks while still being fast.
- Voice style (exaggeration / CFG / temperature) left at validated values.

This is a "fast but safe" balance. If you want maximum speed at the cost of the quality
check, you can turn **Bypass Whisper Checking** ON and drop candidates/attempts to 1 in the UI.

> **Note:** the multilingual Chatterbox variant always watermarks its audio output
> (Resemble AI license limitation) — `apply_watermark=False` is not supported.

---

## VRAM management

All services start on boot but only use VRAM **during generation**. The 5090's 32 GB
handles either video, image, or TTS — **but not two at once**. Run them sequentially
to avoid out-of-memory crashes.

---

## What the boot script does

`start.sh` is autonomous and idempotent. On a clean pod it:

1. Starts JupyterLab (with cross-origin headers enabled so file/preview operations work behind the RunPod proxy).
2. Installs system dependencies and clones ComfyUI + ComfyUI-Manager (removes xformers, incompatible with Blackwell).
3. Downloads the Wan 2.2 and Qwen-Image models from Hugging Face (uses `HF_TOKEN` if present).
4. Installs the custom nodes (QwenEditUtils, WAS Node Suite, VideoHelperSuite — the latter two pinned to known-good commits).
5. Writes the five ready-to-use workflows into the ComfyUI sidebar.
6. Clones Chatterbox-TTS-Extended into an isolated Python 3.10 venv; installs PyTorch 2.11+cu128, then `chatterbox-tts` 0.1.7, then reinstalls torch for CUDA 12.8 (the package pulls in an incompatible CUDA 12.4 build).
7. Applies the patches:
   - **Multilingual patch** to `Chatter.py` (uses `ChatterboxMultilingualTTS`, adds the 23-language dropdown, switches the API to `language_id=`), plus production defaults baked in.
   - **`t3.py` alignment_stream_analyzer patch** — disables a buggy path in `chatterbox-tts` 0.1.7 that crashes on ~95% of chunks in long multilingual generation. An `assert` guards the patch so the boot fails loudly if upstream changes that line.
8. Launches ComfyUI (`--enable-cors-header`, so the RunPod proxy doesn't return 403) and the Chatterbox Gradio UI.

If the install directories already exist, the script skips reinstallation and just relaunches the services.

---

## Known limitations

- **No persistent storage by default.** Files in `/workspace/ComfyUI/output/` and the Chatterbox folder are lost when the pod is terminated. Download what you want to keep, or attach a network volume.
- **First boot is slow** (~15-20 min) due to the ~70 GB download. Restarts of the *same* pod are fast (downloads are skipped if files exist).
- **TTS audio is always watermarked** (Resemble AI multilingual limitation).
- **Tested only on RTX 5090.** Other GPUs may work but are unverified.

---

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) · [Wan 2.2](https://github.com/Wan-Video/Wan2.2) (Alibaba) · [Comfy-Org repackaged weights](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
- [Qwen-Image](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) · [lightx2v Lightning LoRAs](https://huggingface.co/lightx2v)
- [petermg/Chatterbox-TTS-Extended](https://github.com/petermg/Chatterbox-TTS-Extended) · [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
- [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui) · [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) · [Comfyui-QwenEditUtils](https://github.com/lrzjason/Comfyui-QwenEditUtils)

## License

This repository contains only a boot script and workflow files. All upstream projects keep their own licenses (Apache 2.0, MIT, etc). Generated audio is watermarked per Resemble AI's terms.
