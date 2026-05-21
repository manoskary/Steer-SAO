from __future__ import annotations

import math
from typing import Optional


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DOWNSAMPLING_RATIO = 4096
DEFAULT_SMALL_MAX_SAMPLES = 5324800


def round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return ((int(value) + multiple - 1) // multiple) * multiple


def audio_samples_for_duration(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    if duration <= 0:
        raise ValueError("duration must be positive")
    return int(math.ceil(duration * sample_rate))


def latent_length_from_samples(
    audio_samples: int,
    downsampling_ratio: int = DEFAULT_DOWNSAMPLING_RATIO,
) -> int:
    if audio_samples <= 0:
        raise ValueError("audio_samples must be positive")
    if downsampling_ratio <= 0:
        raise ValueError("downsampling_ratio must be positive")
    return int(math.ceil(audio_samples / float(downsampling_ratio)))


def adapt_audio_sample_size(
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    downsampling_ratio: int = DEFAULT_DOWNSAMPLING_RATIO,
    duration_padding_sec: float = 6.0,
    max_samples: int = DEFAULT_SMALL_MAX_SAMPLES,
    latent_align: int = 2,
) -> int:
    """Match SA3's variable-length habit while keeping SAME alignment.

    SA3 adds headroom during generation, rounds to the pretransform stride, and
    clamps to the model maximum. `latent_align=2` mirrors the public SA3 config
    for SAME-S where chunk_size // stride is 32 // 16.
    """

    padded = audio_samples_for_duration(duration + duration_padding_sec, sample_rate)
    aligned = round_up(padded, downsampling_ratio)
    if latent_align > 1:
        aligned = round_up(aligned, downsampling_ratio * latent_align)
    return min(aligned, max_samples)


def latent_length_for_duration(
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    downsampling_ratio: int = DEFAULT_DOWNSAMPLING_RATIO,
    duration_padding_sec: float = 6.0,
    max_samples: Optional[int] = DEFAULT_SMALL_MAX_SAMPLES,
    latent_align: int = 2,
) -> int:
    audio_samples = adapt_audio_sample_size(
        duration=duration,
        sample_rate=sample_rate,
        downsampling_ratio=downsampling_ratio,
        duration_padding_sec=duration_padding_sec,
        max_samples=max_samples if max_samples is not None else 10**18,
        latent_align=latent_align,
    )
    return audio_samples // downsampling_ratio

