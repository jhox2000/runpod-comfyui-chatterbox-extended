# RunPod 1-Click Template: ComfyUI Wan 2.2 + Chatterbox Multilingual TTS

A self-contained RunPod template that boots a ready-to-use environment for:

- **Image-to-video generation** with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) + Wan 2.2 14B I2V (fp8 + lightx2v 4-step LoRAs for ~5x faster inference).
- **Multilingual text-to-speech** with voice cloning, via [petermg/Chatterbox-TTS-Extended](https://github.com/petermg/Chatterbox-TTS-Extended) patched to use [Resemble AI's Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (23 languages).

No manual setup. Deploy the pod, wait for the first boot, and use it.

---

## What's included

| Service | Port | Purpose |
|---|---|---|
| ComfyUI | 8188 | Image-to-video generation (Wan 2.2 I2V 14B) |
| Chatterbox-TTS-Extended | 7860 | Multilingual TTS with voice cloning |
| JupyterLab | 8888 | Optional terminal/file access for debugging |

**Models pre-downloaded on first boot (~35 GB):**
- `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`
- `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
- `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors`
- `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors`
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `wan_2.1_vae.safetensors`
- Chatterbox Multilingual weights (auto-downloaded by the library, ~3.2 GB VRAM)

---

## Requirements

- **GPU:** RTX 5090 (Blackwell, sm_120) recommended. Any compute capability >= 8.0 with PyTorch 2.7+ / CUDA 12.8 support should work, but only the 5090 has been tested.
- **Container image:** `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **Container disk:** 100 GB minimum (models + dependencies ~70 GB).
- **HTTP ports exposed:** 7860, 8188, 8888.

---

## How to use

1. Deploy a pod from this template on RunPod.
2. Wait for the first boot to finish (**~15-20 minutes** — the script installs dependencies and downloads ~35 GB of models). You can follow progress in the pod's **Logs** tab.
3. When the log shows `[boot] TUDO PRONTO!`, the services are starting. Give them another **1-2 minutes** to fully load the models.
4. Click **Connect** on the pod and open the HTTP service you want:
   - **Port 8188** → ComfyUI (image-to-video).
   - **Port 7860** → Chatterbox (TTS).
   - **Port 8888** → JupyterLab (terminal, optional).

### ComfyUI: first-time Wan 2.2 workflow

After ComfyUI opens, load the **"Wan 2.2 14B Image to Video"** workflow from the built-in template browser (top-left menu → Workflow → Browse Templates → Video → Wan 2.2 I2V). The required models are already in place under `/workspace/ComfyUI/models/`.

For speed, generate at **832x480** or **1280x720** first; upscale afterward if needed. The lightx2v LoRA brings generation down to ~1 minute on a 5090.

### Chatterbox: TTS configuration tips

For long-form generation (audiobooks):
- Number of Candidates Per Chunk: **1**
- Max Attempts Per Candidate: **1**
- Bypass Whisper Checking: **ON** (much faster)
- Parallel Workers: **6**
- Enable Sentence Batching: **ON**
- CFG Weight: **0.5** (default of 1.0 sounds monotone)

---

## Important: VRAM management

Both services start on boot but only consume VRAM **during generation**. The RTX 5090's 32 GB is enough for either one, but **not both at the same time**. Always run video generation and TTS generation sequentially, not in parallel, to avoid OOM crashes.

---

## What the boot script does

The `start.sh` script in this repo is autonomous. On a clean pod it:

1. Installs system dependencies (ffmpeg, git-lfs, Python 3.10 venv, etc).
2. Clones ComfyUI + ComfyUI-Manager, removes xformers (incompatible with Blackwell).
3. Downloads the Wan 2.2 I2V fp8 models and lightx2v LoRAs from Hugging Face.
4. Clones Chatterbox-TTS-Extended into a Python 3.10 venv.
5. Installs PyTorch 2.11+cu128 (Blackwell-compatible), then `chatterbox-tts` 0.1.7.
6. Cleans up the CUDA 12.4 libraries that `chatterbox-tts` pulls in and reinstalls torch for CUDA 12.8.
7. Applies two patches:
   - **Chatter.py multilingual patch:** swaps imports and model loading to use `ChatterboxMultilingualTTS`, adds a 23-language dropdown to the UI, and switches the API to `language_id=...` (the multilingual variant doesn't accept `apply_watermark=`).
   - **`t3.py` alignment_stream_analyzer patch:** disables a buggy code path in `chatterbox-tts` 0.1.7 that crashes on ~95% of chunks when generating long texts with `ChatterboxMultilingualTTS`. The patch flips a single `if` to force the existing "analyzer disabled" branch. An `assert` guards the patch so the boot fails loudly if Resemble AI changes that line in a future version.
8. Launches ComfyUI (with `--enable-cors-header` so the RunPod proxy doesn't return 403) and the Chatterbox Gradio UI.

The script is idempotent — if the pod restarts and the install directories already exist, it skips reinstallation and just relaunches the services.

---

## Known limitations

- **No persistent storage by default.** Generated files in `/workspace/ComfyUI/output/` and `/workspace/Chatterbox-TTS-Extended/` are lost when the pod is terminated. Download anything you want to keep, or attach a network volume.
- **First boot is slow** (~15-20 min) because of the ~35 GB download. Subsequent restarts of the same pod are fast (script skips downloads if files exist).
- **Watermark is always applied to TTS output** — the multilingual variant of Chatterbox does not accept `apply_watermark=False` (Resemble AI license limitation).
- **Tested only on RTX 5090.** Other GPUs may work but are unverified.

---

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — the node-based diffusion UI.
- [Wan 2.2](https://github.com/Wan-Video/Wan2.2) — the video diffusion model (Alibaba).
- [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) — the model weights repackaged for ComfyUI.
- [petermg/Chatterbox-TTS-Extended](https://github.com/petermg/Chatterbox-TTS-Extended) — the Gradio UI around Chatterbox.
- [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) — the underlying TTS model.

---

## License

This repository only contains a boot script. All upstream projects keep their own licenses (Apache 2.0, MIT, etc). Generated audio is watermarked per Resemble AI's terms.
