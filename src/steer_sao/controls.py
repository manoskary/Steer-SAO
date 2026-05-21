from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from steer_sao.audio import load_audio
from steer_sao.types import ATTRIBUTE_CONTROLS, normalize_control_names


RAW_CONTROL_DIMS = {
    "dynamics": 1,
    "rhythm": 2,
    "melody_mono": 12,
    "melody_stereo": 24,
    "audio": 256,
}


@dataclass
class ControlFeatureSet:
    features: Dict[str, torch.Tensor]
    sample_rate: int = 44100

    def attribute_features(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.features.items() if k in ATTRIBUTE_CONTROLS}


def _frames(waveform: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [channels, samples]")
    if waveform.shape[-1] < frame_length:
        waveform = F.pad(waveform, (0, frame_length - waveform.shape[-1]))
    return waveform.unfold(-1, frame_length, hop_length)


def extract_dynamics(waveform: torch.Tensor, frame_length: int = 2048, hop_length: int = 512) -> torch.Tensor:
    mono = waveform.mean(dim=0, keepdim=True)
    framed = _frames(mono, frame_length, hop_length)
    rms = framed.square().mean(dim=-1).sqrt().clamp_min(1e-7)
    db = 20.0 * torch.log10(rms)
    db = (db - db.mean()) / db.std().clamp_min(1e-5)
    return db.unsqueeze(0)


def extract_rhythm(waveform: torch.Tensor, frame_length: int = 2048, hop_length: int = 512) -> torch.Tensor:
    dyn = extract_dynamics(waveform, frame_length, hop_length).squeeze(0)
    onset = F.pad((dyn[:, 1:] - dyn[:, :-1]).clamp_min(0), (1, 0))
    onset = onset / onset.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    energy = (dyn - dyn.amin(dim=-1, keepdim=True))
    energy = energy / energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat([onset, energy], dim=0).unsqueeze(0)


def _chroma_for_channel(
    channel: torch.Tensor,
    sample_rate: int,
    n_fft: int = 4096,
    hop_length: int = 1024,
) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=channel.device, dtype=channel.dtype)
    stft = torch.stft(
        channel,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    power = stft.abs().square()
    freqs = torch.arange(power.shape[0], device=channel.device, dtype=channel.dtype)
    freqs = freqs * float(sample_rate) / float(n_fft)
    valid = freqs >= 20.0
    safe_freqs = freqs[valid].clamp_min(1e-6)
    midi = torch.round(69.0 + 12.0 * torch.log2(safe_freqs / 440.0)).to(torch.long)
    chroma_bins = torch.remainder(midi, 12)
    chroma = channel.new_zeros(12, power.shape[-1])
    valid_power = power[valid]
    for pitch_class in range(12):
        mask = chroma_bins == pitch_class
        if mask.any():
            chroma[pitch_class] = valid_power[mask].sum(dim=0)
    chroma = torch.log1p(chroma)
    chroma = chroma / chroma.amax(dim=0, keepdim=True).clamp_min(1e-6)
    return chroma


def extract_melody(
    waveform: torch.Tensor,
    sample_rate: int = 44100,
    stereo: bool = True,
) -> torch.Tensor:
    if stereo:
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        left = _chroma_for_channel(waveform[0], sample_rate)
        right = _chroma_for_channel(waveform[1], sample_rate)
        return torch.cat([left, right], dim=0).unsqueeze(0)
    mono = waveform.mean(dim=0)
    return _chroma_for_channel(mono, sample_rate).unsqueeze(0)


def resize_feature(feature: torch.Tensor, target_len: int) -> torch.Tensor:
    if feature.shape[-1] == target_len:
        return feature
    return F.interpolate(feature, size=target_len, mode="linear", align_corners=False)


def extract_control_features_from_audio(
    audio_path: str,
    controls: Iterable[str],
    target_sr: int = 44100,
    device: Optional[torch.device] = None,
) -> ControlFeatureSet:
    requested = normalize_control_names(controls)
    waveform, sr = load_audio(audio_path, target_sr=target_sr, device=device)
    features: Dict[str, torch.Tensor] = {}
    if "dynamics" in requested:
        features["dynamics"] = extract_dynamics(waveform)
    if "rhythm" in requested:
        features["rhythm"] = extract_rhythm(waveform)
    if "melody_mono" in requested:
        features["melody_mono"] = extract_melody(waveform, sr, stereo=False)
    if "melody_stereo" in requested:
        features["melody_stereo"] = extract_melody(waveform, sr, stereo=True)
    return ControlFeatureSet(features=features, sample_rate=sr)


class ConvControlEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor, target_len: int) -> torch.Tensor:
        feature = resize_feature(feature, target_len)
        tokens = self.net(feature)
        return tokens.transpose(1, 2).contiguous()


class MuseControlConditioner(nn.Module):
    """Encodes raw time-varying controls to SA3 cross-attention token dimension."""

    def __init__(self, output_dim: int = 1024, hidden_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.attribute_encoders = nn.ModuleDict(
            {
                name: ConvControlEncoder(RAW_CONTROL_DIMS[name], output_dim, hidden_dim)
                for name in ATTRIBUTE_CONTROLS
            }
        )
        self.audio_encoder = ConvControlEncoder(RAW_CONTROL_DIMS["audio"], output_dim, hidden_dim)

    def encode_attributes(
        self,
        features: Dict[str, torch.Tensor],
        target_len: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        tokens = None
        for name, feature in features.items():
            if name not in self.attribute_encoders:
                continue
            feature = feature.to(device=device, dtype=torch.float32)
            encoded = self.attribute_encoders[name](feature, target_len)
            tokens = encoded if tokens is None else tokens + encoded
        if tokens is not None and dtype is not None:
            tokens = tokens.to(dtype=dtype)
        return tokens

    def encode_audio_latents(
        self,
        audio_latents: torch.Tensor,
        target_len: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        feature = resize_feature(audio_latents.to(device=device, dtype=torch.float32), target_len)
        tokens = self.audio_encoder(feature, target_len)
        if dtype is not None:
            tokens = tokens.to(dtype=dtype)
        return tokens
