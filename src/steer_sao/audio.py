from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _load_with_torchaudio(path: str) -> Tuple[torch.Tensor, int]:
    import torchaudio

    waveform, sample_rate = torchaudio.load(path)
    return waveform, int(sample_rate)


def _load_with_soundfile(path: str) -> Tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data).transpose(0, 1).contiguous()
    return waveform, int(sample_rate)


def load_audio(
    path: str,
    target_sr: int = 44100,
    target_channels: int = 2,
    target_length: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    """Load audio as `[channels, samples]` float tensor."""

    if not Path(path).exists():
        raise FileNotFoundError(path)

    try:
        waveform, sample_rate = _load_with_torchaudio(path)
    except Exception:
        waveform, sample_rate = _load_with_soundfile(path)

    waveform = waveform.to(torch.float32)
    waveform = prepare_audio_tensor(
        waveform,
        in_sr=sample_rate,
        target_sr=target_sr,
        target_channels=target_channels,
        target_length=target_length,
    )
    if device is not None:
        waveform = waveform.to(device)
    return waveform, target_sr


def prepare_audio_tensor(
    waveform: torch.Tensor,
    in_sr: int,
    target_sr: int = 44100,
    target_channels: int = 2,
    target_length: Optional[int] = None,
) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [channels, samples]")

    if in_sr != target_sr:
        try:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform, in_sr, target_sr)
        except Exception:
            new_len = int(round(waveform.shape[-1] * float(target_sr) / float(in_sr)))
            waveform = F.interpolate(
                waveform.unsqueeze(0),
                size=new_len,
                mode="linear",
                align_corners=False,
            ).squeeze(0)

    if waveform.shape[0] == 1 and target_channels == 2:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > target_channels:
        waveform = waveform[:target_channels]
    elif waveform.shape[0] < target_channels:
        repeats = target_channels // waveform.shape[0] + 1
        waveform = waveform.repeat(repeats, 1)[:target_channels]

    if target_length is not None:
        if waveform.shape[-1] < target_length:
            waveform = F.pad(waveform, (0, target_length - waveform.shape[-1]))
        else:
            waveform = waveform[..., :target_length]

    return waveform.contiguous()


def peak_normalize(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = waveform.abs().amax().clamp_min(eps)
    return waveform / peak


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int = 44100) -> None:
    output = waveform.detach().to(torch.float32).cpu()
    if output.ndim == 3:
        output = output[0]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import torchaudio

        torchaudio.save(path, output, sample_rate)
    except Exception:
        import soundfile as sf

        sf.write(path, output.transpose(0, 1).numpy(), sample_rate)

