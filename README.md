# 1-Click Template: ComfyUI (Wan 2.2 + Qwen-Image) — RunPod & Vast.ai

A self-contained template that boots a ready-to-use content-production
environment on **RunPod** or **Vast.ai**. One deploy gives you, with **no manual
setup**, a full pipeline for:

- **Image generation** (text → image) with Qwen-Image 2512.
- **Image editing** with Qwen-Image-Edit 2511 — keep the *same face* across scenes,
  or combine *two people* into one image.
- **Image-to-video** with Wan 2.2 14B I2V (fp8).
- **Batch automation:** drop a list of prompts and the pod generates **all** images
  in order, then animates **all** of them into videos in order — hands-free.

Deploy, wait for the first boot, and use it.

> **Quality default:** all workflows run in **RAW mode** (full steps, no Lightning
> speed-up) for maximum quality and motion dynamics. The Lightning LoRAs are still
> downloaded and sit **bypassed** inside the workflows — select the LoRA nodes and
> press **Ctrl+B** to re-enable fast mode. See [Raw mode vs. Lightning](#raw-mode-vs-lightning).

---

## Two boot scripts, one repo

| Script | Platform | Base image |
|---|---|---|
| `start.sh` | RunPod | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` |
| `start-vast.sh` | Vast.ai | `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311` |

Both install the same ComfyUI + models + workflows. The differences are purely
platform plumbing:

- **Jupyter:** on RunPod the script launches JupyterLab itself (port 8888). On
  Vast, JupyterLab is provided by the platform (the instance's **Open** button),
  so `start-vast.sh` skips it.
- **Python environment:** the Vast base image ships a venv at `/venv/main`, which
  the script activates; it also installs PyTorch cu128 explicitly (the Vast base
  image doesn't bundle torch). The RunPod image already includes torch.
- **`/workspace`:** RunPod creates it automatically; on Vast it only exists if you
  mount a volume, so the on-start command creates it first (`mkdir -p /workspace`).

---

## What's included

| Service | Port | Purpose |
|---|---|---|
| ComfyUI | 8188 | Image generation, image editing, and image-to-video |
| JupyterLab | 8888 (RunPod) / platform-provided (Vast) | File access / terminal (prompt files, downloads, debugging) |

### Pre-installed ComfyUI workflows (sidebar → Workflows)

Five ready-to-run workflows appear in the ComfyUI sidebar on first open — one click each:

1. **1 - Video (Wan 2.2)** — animate a single image into a short video. *(raw: 20 steps, CFG 3.5, shift 8)*
2. **2 - Gerar Imagem (Qwen)** — single image from a text prompt. *(raw: 20 steps, CFG 2.5)*
3. **3 - Editar Rosto 1 e 2 pessoas (Qwen)** — edit keeping the same face / merge two people. *(4 steps / CFG 1 — Lightning is **embedded** in the 2511 model; there is no raw variant)*
4. **4 - Gerar Imagens em Lote (Qwen)** — batch: read `prompts.txt`, generate all images. *(raw)*
5. **5 - Gerar Videos em Lote (Wan)** — batch: animate a folder of images into videos. *(raw)*

Each workflow includes on-canvas notes with aspect-ratio presets and (for the batch
ones) step-by-step usage instructions.

> The workflow files are (re)written on every boot, so template updates always reach
> the pod. If you customize a workflow inside ComfyUI, **save it under a different name**.

### Raw mode vs. Lightning

| Mode | Wan 2.2 video (5 s clip, RTX 5090) | Quality |
|---|---|---|
| **Raw (default)** | ~5–8 min per clip | Best motion dynamics and detail |
| Lightning (LoRA re-enabled) | ~1 min per clip | Faster, but noticeably less motion |

To switch a Wan workflow to fast mode: un-bypass the two lightx2v LoRA nodes
(**Ctrl+B**), set both KSamplers to **4 steps / CFG 1** with the step split
**0→2 / 2→4**, and set both `ModelSamplingSD3` nodes to shift **5**.
The Qwen generator has a simpler toggle: flip **"Enable 8 Steps LoRA?"** to `true`.

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
- `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` *(bypassed by default)*
- `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` *(bypassed by default)*
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `wan_2.1_vae.safetensors`

**Qwen-Image (image generation + editing, ~53 GB):**
- `qwen_image_2512_fp8_e4m3fn.safetensors` (generator)
- `Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors` (generator LoRA, **off by default**)
- `qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors` (editor, Lightning embedded)
- `qwen_2.5_vl_7b_fp8_scaled.safetensors` (text encoder, shared) + `qwen_image_vae.safetensors` (VAE, shared)

---

## Deploying on RunPod (`start.sh`)

**Requirements**

- **GPU:** RTX 5090 (Blackwell, sm_120) recommended and tested. Any GPU with 24 GB+ VRAM and PyTorch 2.7+/CUDA 12.8 support should work, but only the 5090 is verified.
- **Container image:** `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **Container disk:** **150 GB** (models + dependencies; a smaller disk can fail mid-download).
- **HTTP ports exposed:** 8188, 8888.
- **Container Start Command:** point it at the raw GitHub URL of `start.sh`.
- **Environment variable:** `HF_TOKEN` (see [below](#recommended-set-hf_token-for-fast-downloads)).

**Usage**

1. Deploy a pod from the template.
2. Wait for the first boot (**~15–20 minutes** with `HF_TOKEN` set). Follow progress in the pod's **Logs** tab.
3. When the log shows `[boot] TUDO PRONTO!`, give ComfyUI another **1–2 minutes** to load.
4. Click **Connect**: **8188** → ComfyUI · **8888** → JupyterLab.

---

## Deploying on Vast.ai (`start-vast.sh`)

**Template configuration**

- **Image Path / Version Tag:** `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311`
  (Vast's own image — its layers are cached on most hosts, so the docker pull is
  usually instant).
- **Launch Mode:** *Jupyter-python notebook + SSH*, with **Use Jupyter Lab interface** checked.
  Also check **Jupyter direct HTTPS** — without it, Jupyter traffic goes through
  Vast's proxy, which frequently returns **504 Gateway Time-out**. Direct HTTPS
  requires a one-time install of Vast's TLS certificate in your browser (Vast links
  the instructions when you first connect).
- **On-start Script:**

  ```
  bash -c "mkdir -p /workspace && curl -fsSL https://raw.githubusercontent.com/jhox2000/runpod-comfyui-chatterbox-extended/refs/heads/main/start-vast.sh -o /workspace/start-vast.sh && bash /workspace/start-vast.sh"
  ```

  The `mkdir -p /workspace` matters: unlike RunPod, Vast does **not** create
  `/workspace` unless a volume is mounted. Without it, the curl fails with
  `curl: (23) Failure writing output to destination` and nothing installs.
- **Docker Options:** `-p 8188:8188 -e HF_TOKEN=hf_...` (and declare port **8188 TCP** in the Ports section).
- **Disk:** **200 GB** container disk.
- **Extra Filters (recommended):**

  ```
  cuda_max_good>=12.8 inet_down>500 inet_up>100 verified=true
  ```

  Vast machines are individually owned and wildly uneven. This template downloads
  **~88 GB of models on first boot**, so host **download speed is the single most
  important factor** — at 100 Mbps that's ~2 hours; at 1 Gbps, ~12 minutes. The
  filter hides slow and unverified machines (the usual cause of "stuck" boots and
  Jupyter that never opens). When picking an offer, prefer hosts showing ≥1 Gbps down.

**Accessing the services**

- **ComfyUI:** click the **IP address button** on the instance card (not the
  `>_Connect` button — that one only shows terminal options). The "Open Ports"
  panel shows the mapping, e.g. `68.193.196.85:41234 -> 8188/tcp`. Open that
  `IP:port` in your browser. The external port is random per instance.
- **JupyterLab:** the instance's **Open** button (provided by Vast — the script
  does not start its own Jupyter).

**Watching progress:** open a Jupyter terminal and use
`tail -f /workspace/logs/comfyui.log` (ComfyUI live log) or
`watch -n 5 du -sh /workspace/ComfyUI/models/*` (model download progress).

---

## Recommended: set `HF_TOKEN` for fast downloads

Hugging Face throttles **anonymous** downloads, which can make the first boot crawl
(or stall). A free read-only token removes the throttle.

1. Create a free token at Hugging Face → Settings → Access Tokens → **Read**.
2. **RunPod:** add it as a template **Environment Variable** (`HF_TOKEN` = `hf_...`).
   **Vast.ai:** add it to the template's **Environment Variables** section or via
   Docker Options (`-e HF_TOKEN=hf_...`).

The download library reads `HF_TOKEN` automatically — no change to the scripts is
needed, and the token stays out of the (public) repository.

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
3. Output: `ComfyUI/output/video/` → `Wan2.2_..._00001_.mp4`, ...
   **Raw mode is ~5–8 min per video on a 5090** (~1 min if you re-enable Lightning) — plan batch sizes accordingly.

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

## What the boot scripts do

Both scripts are autonomous and idempotent. On boot they:

1. *(RunPod only)* Start JupyterLab with cross-origin headers enabled (so file/preview operations work behind the proxy). *(Vast only)* Activate the `/venv/main` environment and install PyTorch cu128 (the Vast base image doesn't bundle torch).
2. Install system dependencies (ffmpeg + debug tools).
3. Clone ComfyUI + ComfyUI-Manager **only if missing**, then (re)install their Python dependencies **on every boot** and remove xformers (incompatible with Blackwell).
4. Download the Wan 2.2 and Qwen-Image models from Hugging Face (uses `HF_TOKEN` if present) — skipped if the files already exist.
5. Install the custom nodes (QwenEditUtils, WAS Node Suite, VideoHelperSuite — the latter two pinned to known-good commits). Clones are conditional; their Python dependencies are also (re)installed on every boot.
6. Write the five ready-to-use workflows into the ComfyUI sidebar (raw mode by default).
7. Launch ComfyUI (`--enable-cors-header`, so the platform proxies don't return 403).

**Why "pip installs on every boot"?** `/workspace` (a volume) can survive pod
recreation, but the container — where pip packages live — does not. Reinstalling the
Python dependencies unconditionally means a pod recreated on top of an existing volume
comes back up correctly instead of failing with missing packages. On a clean deploy
the behavior is identical; on a restart it costs ~1–2 extra minutes.

---

## Known limitations

- **No persistent storage by default.** Files in `/workspace/ComfyUI/output/` are lost when the instance is terminated. Download what you want to keep, or attach a volume.
- **First boot is slow** (~15–20 min on a fast host) due to the ~88 GB download. On Vast, boot time depends heavily on the host's internet speed — use the recommended filters. Restarts reusing the same volume are much faster (model downloads are skipped; only pip dependencies are reinstalled).
- **Vast proxy quirks.** Without "Jupyter direct HTTPS", the Jupyter button often 504s. ComfyUI is reached via the IP:port mapping, not via a service button.
- **Tested only on RTX 5090.** Other GPUs may work but are unverified.

---

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) · [Wan 2.2](https://github.com/Wan-Video/Wan2.2) (Alibaba) · [Comfy-Org repackaged weights](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
- [Qwen-Image](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) · [lightx2v Lightning LoRAs](https://huggingface.co/lightx2v)
- [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui) · [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) · [Comfyui-QwenEditUtils](https://github.com/lrzjason/Comfyui-QwenEditUtils)

## License

This repository contains only boot scripts and workflow files. All upstream projects keep their own licenses (Apache 2.0, MIT, etc).
