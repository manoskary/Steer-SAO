from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ATTRIBUTE_CONTROLS = ("melody_mono", "melody_stereo", "rhythm", "dynamics")
ALL_CONTROLS = ATTRIBUTE_CONTROLS + ("audio",)


def normalize_control_names(controls: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if controls is None:
        return ()

    normalized: List[str] = []
    for raw in controls:
        name = raw.strip().lower().replace("-", "_")
        if name == "melody":
            name = "melody_stereo"
        if name not in ALL_CONTROLS:
            raise ValueError(
                "Unknown control %r. Expected one of: %s"
                % (raw, ", ".join(ALL_CONTROLS))
            )
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


@dataclass(frozen=True)
class GuidanceScales:
    """MuseControlLite-style guidance scales.

    text applies to the text-vs-unconditional branch, attribute applies to
    text+musical-attribute controls, and audio applies to the additional
    reference-audio branch.
    """

    text: float = 7.0
    attribute: float = 1.5
    audio: float = 1.0

    @classmethod
    def from_mapping(cls, values: Optional[Dict[str, float]]) -> "GuidanceScales":
        if values is None:
            return cls()
        defaults = cls()
        return cls(
            text=float(values.get("text", values.get("cfg_scale_text", defaults.text))),
            attribute=float(
                values.get("attribute", values.get("cfg_scale_attr", defaults.attribute))
            ),
            audio=float(values.get("audio", values.get("cfg_scale_audio", defaults.audio))),
        )


@dataclass(frozen=True)
class ControlInputs:
    """Control inputs for generation or training."""

    control_audio: Optional[str] = None
    controls: Tuple[str, ...] = field(default_factory=tuple)
    melody_audio: Optional[str] = None
    rhythm_audio: Optional[str] = None
    dynamics_audio: Optional[str] = None
    audio_reference: Optional[str] = None
    precomputed: Dict[str, str] = field(default_factory=dict)

    def enabled(self) -> Tuple[str, ...]:
        return normalize_control_names(self.controls)

    def path_for(self, control: str) -> Optional[str]:
        if control in self.precomputed:
            return self.precomputed[control]
        if control in ("melody_mono", "melody_stereo") and self.melody_audio:
            return self.melody_audio
        if control == "rhythm" and self.rhythm_audio:
            return self.rhythm_audio
        if control == "dynamics" and self.dynamics_audio:
            return self.dynamics_audio
        if control == "audio" and self.audio_reference:
            return self.audio_reference
        return self.control_audio


@dataclass(frozen=True)
class AdapterConfig:
    embed_dim: int = 1024
    control_dim: int = 1024
    hidden_dim: int = 256
    position_encoding: bool = True
    train_attribute_branch: bool = True
    train_audio_branch: bool = True
    control_types: Tuple[str, ...] = ALL_CONTROLS
    base_model_id: str = "stabilityai/stable-audio-3-small-music"
    base_model_revision: str = "0fef1392cd842149a2b6d445e181c97608faac06"

    def to_metadata(self) -> Dict[str, str]:
        return {
            "adapter_config": (
                "{"
                f'"embed_dim":{self.embed_dim},'
                f'"control_dim":{self.control_dim},'
                f'"hidden_dim":{self.hidden_dim},'
                f'"position_encoding":{str(self.position_encoding).lower()},'
                f'"train_attribute_branch":{str(self.train_attribute_branch).lower()},'
                f'"train_audio_branch":{str(self.train_audio_branch).lower()},'
                f'"control_types":"{",".join(self.control_types)}"'
                "}"
            ),
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
        }


@dataclass(frozen=True)
class ManifestRow:
    audio_path: str
    prompt: str
    duration: Optional[float] = None
    split: Optional[str] = None
    controls: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    model: str
    manifest: str
    output_dir: str
    adapter_name: str
    controls: Tuple[str, ...]
    max_steps: int = 10000
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    seed: int = 42
    save_every: int = 1000
    log_every: int = 25
    duration_padding_sec: float = 6.0
    guidance: GuidanceScales = field(default_factory=GuidanceScales)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
