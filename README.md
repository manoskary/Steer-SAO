# Steer-SAO

Steer-SAO adds a MuseControlLite-style adapter layer to Stable Audio 3 Small Music. It uses
the current `stable-audio-3` codebase and trains new SA3-shaped adapters; the released
MuseControlLite checkpoints target Stable Audio Open 1.0 and are not shape-compatible.

## What Is Included

- SA3-native decoupled cross-attention adapters for musical attributes and reference audio.
- Dynamic latent-length handling for SA3/SAME instead of MuseControlLite's fixed 1024 tokens.
- Control extraction for melody, rhythm, dynamics, and audio-reference conditioning.
- CLI entrypoints for manifest validation/precompute, adapter training, and controlled generation.
- CPU-safe unit tests that do not require gated Hugging Face downloads.

## Setup

Use Python 3.10 or newer. The pinned upstream SA3 dependency is:

```bash
uv sync --extra dev --extra data --extra train
```

If `uv` is not available:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,train]"
```

The target model is gated. Authenticate outside the repo:

```bash
huggingface-cli login
```

or set `HF_TOKEN` in your shell. If the pinned upstream SA3 dependency is private, make
sure local Git can read it via your credential helper. Do not put real tokens in tracked files.

## Usage

For Linux GPU setup, dataset options, preprocessing, and first training runs, see
[docs/linux_gpu_start.md](docs/linux_gpu_start.md).

Fetch a Hugging Face audio-text dataset into a local manifest:

```bash
steer-sao hf-manifest \
  --dataset mrfakename/cc0-music-captioned \
  --split train \
  --out data/cc0_music/train_50.jsonl \
  --audio-dir data/cc0_music/audio \
  --limit 50
```

Prepare control features:

```bash
steer-sao prepare-data --manifest data/train.jsonl --out cache/controls
```

Train adapters:

```bash
steer-sao train --config configs/train_all_controls.yaml
```

Generate:

```bash
steer-sao generate \
  --prompt "lo-fi house loop with warm Rhodes chords, 120 BPM" \
  --duration 30 \
  --adapter checkpoints/mcl_sa3.safetensors \
  --control-audio reference.wav \
  --controls melody_stereo,rhythm,dynamics \
  --out generated_audio/out.wav
```

Launch the Gradio app on GPU 9:

```bash
python scripts/gradio_app.py \
  --checkpoint checkpoints/mcl_sa3_all_controls_cc0_full_lr2e6_step_020000.safetensors \
  --gpu-index 9
```

Launch with the default local checkpoint and automatic device selection:

```bash
python scripts/gradio_app.py
```

For a Hugging Face Space, `app.py` exposes the Gradio `demo`. Keep the adapter checkpoint in
a private model repo and add both secrets:

- `GITHUB_TOKEN`: read access to private GitHub dependencies such as `stable-audio-3`.
- `HF_TOKEN`: read access to the private Hugging Face checkpoint repo and gated base model.

Then set:

```bash
STEER_SAO_CHECKPOINT_REPO=manoskary/sao3-small-control
STEER_SAO_CHECKPOINT_FILENAME=mcl_sa3_all_controls_cc0_full_lr2e6_step_020000.safetensors
```

The Space `requirements.txt` installs only public dependencies. On startup, `app.py` adds
the local `src/` tree to Python and installs `stable-audio-3` with `GITHUB_TOKEN` if it is
not already available. Override the source with `STEER_SAO_STABLE_AUDIO_3_REPO` or
`STEER_SAO_STABLE_AUDIO_3_REF` only when changing the pinned upstream dependency.
Keep the PyTorch requirements on a ZeroGPU-supported CUDA 12.8-backed release for Blackwell
hardware; older PyTorch 2.7.x wheels do not include `sm_120` kernels.
If the Gradio Space is a separate repository that installs Steer-SAO from GitHub, its
`requirements.txt` must preinstall the matching CUDA wheels before `steer-sao`, for example:

```text
torch==2.8.0
torchaudio==2.8.0
gradio==6.15.0
huggingface_hub>=0.36.0
steer-sao @ git+https://github.com/manoskary/Steer-SAO.git@main
```

## Manifest Schema

Each JSONL row needs:

```json
{"audio_path": "audio/example.wav", "prompt": "caption", "duration": 30.0, "split": "train"}
```

Optional precomputed paths may be stored under `controls`, for example:

```json
{"controls": {"melody_stereo": "cache/controls/000001.melody_stereo.pt"}}
```

## Important Notes

- Existing MuseControlLite adapter weights are intentionally not loaded.
- Adapter checkpoints are saved as `safetensors` with metadata describing the base SA3 model,
  model revision, control set, and adapter config.
- If a Hugging Face token was pasted into chat or logs, rotate it before real runs.
