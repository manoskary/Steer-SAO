from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch

from steer_sao.adapters import MuseControlAdapterManager
from steer_sao.alignment import adapt_audio_sample_size
from steer_sao.audio import load_audio
from steer_sao.checkpoints import (
    load_adapter_checkpoint,
    merge_checkpoint_state,
    save_adapter_checkpoint,
    split_checkpoint_state,
)
from steer_sao.controls import (
    MuseControlConditioner,
    extract_control_features_from_audio,
)
from steer_sao.guidance import ControlledDiTModel
from steer_sao.types import (
    ATTRIBUTE_CONTROLS,
    AdapterConfig,
    ControlInputs,
    GuidanceScales,
    normalize_control_names,
)


class SteerSAO:
    def __init__(
        self,
        sa3_model,
        adapter_config: Optional[AdapterConfig] = None,
    ) -> None:
        self.sa3 = sa3_model
        self.adapter_config = adapter_config or AdapterConfig()
        self.adapter_manager = MuseControlAdapterManager(self.adapter_config)
        self.adapter_manager.install(self.sa3)
        self.conditioner = MuseControlConditioner(
            output_dim=self.adapter_config.control_dim,
            hidden_dim=self.adapter_config.hidden_dim,
        )
        device = torch.device(str(self.sa3.device))
        self.conditioner.to(device)
        self.adapter_manager.freeze_base_train_adapters(self.sa3.model)

    @classmethod
    def from_pretrained(
        cls,
        model: str = "small-music",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        model_half: bool = True,
    ) -> "SteerSAO":
        try:
            from stable_audio_3 import StableAudioModel
        except Exception as exc:
            raise RuntimeError(
                "stable-audio-3 is required. Install this package with its dependencies first."
            ) from exc

        sa3_model = StableAudioModel.from_pretrained(
            model,
            device=device,
            model_half=model_half,
        )
        instance = cls(sa3_model)
        if adapter_path:
            instance.load_adapter(adapter_path)
        return instance

    @property
    def device(self) -> torch.device:
        return torch.device(str(self.sa3.device))

    @property
    def sample_rate(self) -> int:
        return int(self.sa3.model.sample_rate)

    def trainable_parameters(self):
        for parameter in self.sa3.model.parameters():
            if parameter.requires_grad:
                yield parameter
        for parameter in self.conditioner.parameters():
            yield parameter

    def _adapter_state(self) -> Dict[str, torch.Tensor]:
        adapter_state = self.adapter_manager.adapter_state_dict(self.sa3.model)
        conditioner_state = self.conditioner.state_dict()
        return merge_checkpoint_state(adapter_state, conditioner_state)

    def save_adapter(self, path: str, extra_metadata: Optional[Dict[str, str]] = None) -> None:
        save_adapter_checkpoint(
            path,
            self._adapter_state(),
            self.adapter_config,
            extra_metadata=extra_metadata,
        )

    def load_adapter(self, path: str) -> None:
        state, _metadata = load_adapter_checkpoint(path)
        adapter_state, conditioner_state = split_checkpoint_state(state)
        self.adapter_manager.load_adapter_state_dict(self.sa3.model, adapter_state)
        if conditioner_state:
            self.conditioner.load_state_dict(conditioner_state, strict=False)

    @contextmanager
    def _controlled_dit(self) -> Iterator[None]:
        original = self.sa3.model.model
        if isinstance(original, ControlledDiTModel):
            yield
            return
        self.sa3.model.model = ControlledDiTModel(original)
        try:
            yield
        finally:
            self.sa3.model.model = original

    def _latent_target_len(
        self,
        duration: float,
        duration_padding_sec: float,
        sample_size: Optional[int],
    ) -> int:
        max_samples = sample_size or int(self.sa3.model_config["sample_size"])
        downsampling_ratio = int(self.sa3.model.pretransform.downsampling_ratio)
        audio_samples = adapt_audio_sample_size(
            duration=duration,
            sample_rate=self.sample_rate,
            downsampling_ratio=downsampling_ratio,
            duration_padding_sec=duration_padding_sec,
            max_samples=max_samples,
        )
        return audio_samples // downsampling_ratio

    def _attribute_tokens(
        self,
        controls: ControlInputs,
        target_len: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        enabled = controls.enabled()
        attr_controls = tuple(c for c in enabled if c in ATTRIBUTE_CONTROLS)
        if not attr_controls:
            return None

        merged_features: Dict[str, torch.Tensor] = {}
        for control in attr_controls:
            path = controls.path_for(control)
            if path is None:
                raise ValueError("Control %s was requested but no control audio was provided" % control)
            features = extract_control_features_from_audio(path, [control], device=self.device)
            merged_features.update(features.features)

        return self.conditioner.encode_attributes(
            merged_features,
            target_len=target_len,
            dtype=dtype,
            device=self.device,
        )

    def _audio_tokens(
        self,
        controls: ControlInputs,
        target_len: int,
        duration: float,
        duration_padding_sec: float,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if "audio" not in controls.enabled():
            return None
        path = controls.path_for("audio")
        if path is None:
            raise ValueError("Audio control requested but no control audio was provided")

        audio_sample_size = adapt_audio_sample_size(
            duration=duration,
            sample_rate=self.sample_rate,
            downsampling_ratio=int(self.sa3.model.pretransform.downsampling_ratio),
            duration_padding_sec=duration_padding_sec,
            max_samples=int(self.sa3.model_config["sample_size"]),
        )
        waveform, sr = load_audio(
            path,
            target_sr=self.sample_rate,
            target_channels=self.sa3.model.pretransform.io_channels,
            target_length=audio_sample_size,
            device=self.device,
        )
        encoded, _ = self.sa3._encode_audio_input((sr, waveform), audio_sample_size)
        return self.conditioner.encode_audio_latents(
            encoded,
            target_len=target_len,
            dtype=dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        duration: float,
        controls: Optional[ControlInputs] = None,
        guidance: Optional[GuidanceScales] = None,
        seed: Optional[int] = None,
        steps: int = 50,
        negative_prompt: Optional[str] = None,
        sample_size: Optional[int] = None,
        duration_padding_sec: float = 6.0,
        **kwargs,
    ) -> torch.Tensor:
        controls = controls or ControlInputs()
        guidance = guidance or GuidanceScales()
        normalize_control_names(controls.enabled())
        target_len = self._latent_target_len(duration, duration_padding_sec, sample_size)
        model_dtype = next(self.sa3.model.model.parameters()).dtype
        attr_tokens = self._attribute_tokens(controls, target_len, model_dtype)
        audio_tokens = self._audio_tokens(
            controls,
            target_len,
            duration,
            duration_padding_sec,
            model_dtype,
        )

        with self._controlled_dit():
            return self.sa3.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                duration=duration,
                steps=steps,
                cfg_scale=guidance.text,
                seed=-1 if seed is None else int(seed),
                sample_size=sample_size or int(self.sa3.model_config["sample_size"]),
                duration_padding_sec=duration_padding_sec,
                control_guidance=guidance,
                attribute_control_tokens=attr_tokens,
                audio_control_tokens=audio_tokens,
                **kwargs,
            )

    def train_adapter(self, train_config_path: str) -> None:
        from steer_sao.training.trainer import train_adapter_from_config

        train_adapter_from_config(train_config_path, model=self)
