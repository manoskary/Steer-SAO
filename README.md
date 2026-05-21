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

or set `HF_TOKEN` in your shell. Do not put real tokens in tracked files.

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
