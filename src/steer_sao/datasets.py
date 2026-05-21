from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def _require_datasets():
    try:
        from datasets import Audio, load_dataset
    except Exception as exc:
        raise RuntimeError(
            "The Hugging Face datasets package is required. Install with `uv sync --extra data`."
        ) from exc
    return Audio, load_dataset


def _write_audio(path: Path, array, sample_rate: int) -> float:
    import soundfile as sf

    audio = np.asarray(array, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[0] <= 8:
        audio = audio.T
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
    return float(audio.shape[0]) / float(sample_rate)


def hf_audio_dataset_to_manifest(
    dataset: str,
    split: str,
    out_manifest: str,
    audio_dir: str,
    audio_column: str = "audio",
    text_column: str = "text",
    limit: Optional[int] = None,
    trust_remote_code: bool = False,
) -> int:
    """Download an HF audio-text dataset into local WAVs plus Steer-SAO JSONL."""

    Audio, load_dataset = _require_datasets()
    ds = load_dataset(dataset, split=split, trust_remote_code=trust_remote_code)
    ds = ds.cast_column(audio_column, Audio(decode=True))

    out_path = Path(out_manifest)
    audio_root = Path(audio_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(ds):
            if limit is not None and count >= limit:
                break
            audio = row[audio_column]
            prompt = row.get(text_column) or row.get("caption") or row.get("prompt")
            if not prompt:
                continue
            sample_rate = int(audio["sampling_rate"])
            wav_path = audio_root / ("%08d.wav" % index)
            duration = _write_audio(wav_path, audio["array"], sample_rate)
            payload = {
                "audio_path": str(wav_path),
                "prompt": str(prompt),
                "duration": round(duration, 4),
                "split": split,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            count += 1
    return count

