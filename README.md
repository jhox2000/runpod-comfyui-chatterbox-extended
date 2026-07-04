# RunPod 1-Click Template: ComfyUI (Wan 2.2 + Qwen-Image)

A self-contained RunPod template that boots a ready-to-use content-production
environment. One deploy gives you, with **no manual setup**, a full pipeline for:

- **Image generation** (text → image) with Qwen-Image 2512 + Lightning 4-step LoRA.
- **Image editing** with Qwen-Image-Edit 2511 — keep the *same face* across scenes,
  or combine *two people* into one image.
- **Image-to-video** with Wan 2.2 14B I2V (fp8 + lightx2v 4-step LoRAs, ~5x faster).
- **Batch automation:** drop a list of prompts and the pod generates **all** images
  in order, then animates **all** of them into videos in order — hands-free.

Deploy the pod, wait for the first boot, and use it.

---

## What's included

| Service | Port | Purpose |
|---|---|---|
| ComfyUI | 8188 | Image generation, image editing, and image-to-video |
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

> The workflow files are (re)written on every boot, so template updates always reach
> the pod. If you customize a workflow inside ComfyUI, **save it under a different name**.

### Custom nodes installed on boot

- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- [Comfyui-QwenEditUtils](https://github.com/lrzjason/Comfyui-QwenEditUtils) — required by the image editor.
- [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui) — reads prompts line-by-line and loads images in sequence (batch). **Pinned** to a known-good commit.
- [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) — video helpers. **Pinned** to a known-good commit.

---

## Models (pre-downloaded on first boot, ~88 GB total)

**Wan 2.2 I2V (video, ~35 GB):**
- `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`
- `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
- `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors`
- `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors`
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `wan_2.1_vae.safetensors`

**Qwen-Image (image generation + editing, ~53 GB):**
- `qwen_image_2512_fp8_e4m3fn.safetensors` (generator)
- `Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors` (generator LoRA)
- `qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors` (editor)
- `qwen_2.5_vl_7b_fp8_scaled.safetensors` (text encoder, shared) + `qwen_image_vae.safetensors` (VAE, shared)

---

## Requirements

- **GPU:** RTX 5090 (Blackwell, sm_120) recommended and tested. Any GPU with 24 GB+ VRAM and PyTorch 2.7+/CUDA 12.8 support should work, but only the 5090 is verified.
- **Container image:** `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **Container disk:** **150 GB** (models + dependencies; a smaller disk can fail mid-download).
- **HTTP ports exposed:** 8188, 8888.
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
3. When the log shows `[boot] TUDO PRONTO!`, give ComfyUI another **1-2 minutes** to load.
4. Click **Connect** and open the service you want:
   - **8188** → ComfyUI · **8888** → JupyterLab.

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

## VRAM management

Models only occupy VRAM **during generation**. The 5090's 32 GB handles either
video generation or image generation — **but not both at once**. Run the batch
phases sequentially (images first, then videos) to avoid out-of-memory crashes.

---

## What the boot script does

`start.sh` is autonomous and idempotent. On boot it:

1. Starts JupyterLab (with cross-origin headers enabled so file/preview operations work behind the RunPod proxy).
2. Installs system dependencies (ffmpeg + debug tools).
3. Clones ComfyUI + ComfyUI-Manager **only if missing**, then (re)installs their Python dependencies **on every boot** and removes xformers (incompatible with Blackwell).
4. Downloads the Wan 2.2 and Qwen-Image models from Hugging Face (uses `HF_TOKEN` if present) — skipped if the files already exist.
5. Installs the custom nodes (QwenEditUtils, WAS Node Suite, VideoHelperSuite — the latter two pinned to known-good commits). Clones are conditional; their Python dependencies are also (re)installed on every boot.
6. Writes the five ready-to-use workflows into the ComfyUI sidebar.
7. Launches ComfyUI (`--enable-cors-header`, so the RunPod proxy doesn't return 403).

**Why "pip installs on every boot"?** On RunPod, `/workspace` (a volume) survives pod
recreation, but the container — where pip packages live — does not. Reinstalling the
Python dependencies unconditionally means a pod recreated on top of an existing volume
comes back up correctly instead of failing with missing packages. On a clean deploy
the behavior is identical; on a restart it costs ~1-2 extra minutes.

---

## Known limitations

- **No persistent storage by default.** Files in `/workspace/ComfyUI/output/` are lost when the pod is terminated. Download what you want to keep, or attach a network volume.
- **First boot is slow** (~15-20 min) due to the ~88 GB download. Restarts reusing the same volume are much faster (model downloads are skipped; only pip dependencies are reinstalled).
- **Tested only on RTX 5090.** Other GPUs may work but are unverified.

---

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) · [Wan 2.2](https://github.com/Wan-Video/Wan2.2) (Alibaba) · [Comfy-Org repackaged weights](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
- [Qwen-Image](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) · [lightx2v Lightning LoRAs](https://huggingface.co/lightx2v)
- [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui) · [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) · [Comfyui-QwenEditUtils](https://github.com/lrzjason/Comfyui-QwenEditUtils)

## License

This repository contains only a boot script and workflow files. All upstream projects keep their own licenses (Apache 2.0, MIT, etc).
