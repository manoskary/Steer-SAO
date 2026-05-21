from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from steer_sao.alignment import adapt_audio_sample_size
from steer_sao.audio import load_audio
from steer_sao.controls import extract_control_features_from_audio
from steer_sao.guidance import ControlledDiTModel
from steer_sao.manifest import load_manifest
from steer_sao.model import SteerSAO
from steer_sao.types import (
    ATTRIBUTE_CONTROLS,
    AdapterConfig,
    ControlInputs,
    GuidanceScales,
    TrainConfig,
    normalize_control_names,
)


def _load_config_file(path: str) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    text = Path(path).read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    try:
        import yaml
    except Exception as exc:
        return _minimal_yaml_load(text)
    return yaml.safe_load(text)


def _parse_scalar(value: str):
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _minimal_yaml_load(text: str) -> Dict[str, Any]:
    """Small fallback for the simple config files shipped with this repo."""

    lines = text.splitlines()
    root: Dict[str, Any] = {}
    stack = [(-1, root)]

    def next_content_is_list(start: int, indent: int) -> bool:
        for later in lines[start + 1 :]:
            if not later.strip() or later.lstrip().startswith("#"):
                continue
            later_indent = len(later) - len(later.lstrip(" "))
            if later_indent <= indent:
                return False
            return later.strip().startswith("- ")
        return False

    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise RuntimeError("Invalid fallback YAML list near: %s" % stripped)
            parent.append(_parse_scalar(stripped[2:]))
            continue

        key, sep, value = stripped.partition(":")
        if not sep:
            raise RuntimeError("Invalid fallback YAML line: %s" % stripped)
        key = key.strip()
        if value.strip():
            parent[key] = _parse_scalar(value)
        else:
            child = [] if next_content_is_list(index, indent) else {}
            parent[key] = child
            stack.append((indent, child))
    return root


def load_train_config(path: str) -> TrainConfig:
    raw = _load_config_file(path)
    training = raw.get("training", {})
    adapter_raw = raw.get("adapter", {})
    controls = normalize_control_names(raw.get("controls", []))
    adapter = AdapterConfig(
        embed_dim=int(adapter_raw.get("embed_dim", 1024)),
        control_dim=int(adapter_raw.get("control_dim", 1024)),
        hidden_dim=int(adapter_raw.get("hidden_dim", 256)),
        position_encoding=bool(adapter_raw.get("position_encoding", True)),
        train_attribute_branch=bool(adapter_raw.get("train_attribute_branch", True)),
        train_audio_branch=bool(adapter_raw.get("train_audio_branch", True)),
        control_types=controls or AdapterConfig().control_types,
    )
    return TrainConfig(
        model=str(raw.get("model", "small-music")),
        manifest=str(raw["manifest"]),
        output_dir=str(raw.get("output_dir", "checkpoints")),
        adapter_name=str(raw.get("adapter_name", "mcl_sa3")),
        controls=controls,
        max_steps=int(training.get("max_steps", 10000)),
        batch_size=int(training.get("batch_size", 1)),
        learning_rate=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        seed=int(training.get("seed", 42)),
        save_every=int(training.get("save_every", 1000)),
        log_every=int(training.get("log_every", 25)),
        duration_padding_sec=float(training.get("duration_padding_sec", 6.0)),
        guidance=GuidanceScales.from_mapping(raw.get("guidance")),
        adapter=adapter,
    )


def _row_duration(row, sample_rate: int) -> float:
    if row.duration is not None:
        return float(row.duration)
    try:
        waveform, _ = load_audio(row.audio_path, target_sr=sample_rate)
    except Exception:
        return 30.0
    return float(waveform.shape[-1]) / float(sample_rate)


def _conditioning_inputs(model: SteerSAO, prompt: str, duration: float, latent_len: int):
    cond = [{"prompt": prompt, "seconds_total": duration}]
    tensors = model.sa3.model.conditioner(cond, model.device)
    mask = torch.zeros((1, 1, latent_len), device=model.device)
    inpaint_input = torch.zeros((1, model.sa3.model.io_channels, latent_len), device=model.device)
    tensors["inpaint_mask"] = [mask]
    tensors["inpaint_masked_input"] = [inpaint_input]
    return model.sa3.model.get_conditioning_inputs(tensors), cond


def _control_tokens(
    model: SteerSAO,
    row,
    controls,
    target_len: int,
    audio_sample_size: int,
    dtype: torch.dtype,
):
    control_inputs = ControlInputs(
        control_audio=row.audio_path,
        controls=controls,
        precomputed=row.controls,
    )
    attr_controls = tuple(c for c in controls if c in ATTRIBUTE_CONTROLS)
    attr_tokens = None
    if attr_controls:
        features = {}
        for control in attr_controls:
            feature_path = row.controls.get(control)
            if feature_path:
                features[control] = torch.load(feature_path, map_location=model.device)
            else:
                extracted = extract_control_features_from_audio(
                    control_inputs.path_for(control),
                    [control],
                    device=model.device,
                )
                features.update(extracted.features)
        attr_tokens = model.conditioner.encode_attributes(
            features,
            target_len=target_len,
            dtype=dtype,
            device=model.device,
        )

    audio_tokens = None
    if "audio" in controls:
        waveform, sr = load_audio(
            control_inputs.path_for("audio"),
            target_sr=model.sample_rate,
            target_channels=model.sa3.model.pretransform.io_channels,
            target_length=audio_sample_size,
            device=model.device,
        )
        encoded, _ = model.sa3._encode_audio_input((sr, waveform), audio_sample_size)
        audio_tokens = model.conditioner.encode_audio_latents(
            encoded,
            target_len=target_len,
            dtype=dtype,
            device=model.device,
        )
    return attr_tokens, audio_tokens


def train_adapter_from_config(path: str, model: Optional[SteerSAO] = None) -> None:
    config = load_train_config(path)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    rows = load_manifest(config.manifest)
    if model is None:
        model = SteerSAO.from_pretrained(config.model)

    model.conditioner.train()
    for parameter in model.conditioner.parameters():
        parameter.requires_grad_(True)

    original_dit = model.sa3.model.model
    controlled_dit = ControlledDiTModel(original_dit)
    model.sa3.model.model = controlled_dit

    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for step in range(1, config.max_steps + 1):
            row = rows[(step - 1) % len(rows)]
            duration = _row_duration(row, model.sample_rate)
            audio_sample_size = adapt_audio_sample_size(
                duration,
                sample_rate=model.sample_rate,
                downsampling_ratio=int(model.sa3.model.pretransform.downsampling_ratio),
                duration_padding_sec=config.duration_padding_sec,
                max_samples=int(model.sa3.model_config["sample_size"]),
            )
            waveform, sr = load_audio(
                row.audio_path,
                target_sr=model.sample_rate,
                target_channels=model.sa3.model.pretransform.io_channels,
                target_length=audio_sample_size,
                device=model.device,
            )
            clean, _ = model.sa3._encode_audio_input((sr, waveform), audio_sample_size)
            clean = clean.to(next(original_dit.parameters()).dtype)
            latent_len = clean.shape[-1]

            cond_inputs, conditioning = _conditioning_inputs(model, row.prompt, duration, latent_len)
            cond_inputs = {
                key: value.to(clean.dtype) if torch.is_tensor(value) else value
                for key, value in cond_inputs.items()
            }
            attr_tokens, audio_tokens = _control_tokens(
                model,
                row,
                config.controls,
                latent_len,
                audio_sample_size,
                clean.dtype,
            )

            t = torch.rand(clean.shape[0], device=model.device, dtype=clean.dtype).clamp(1e-5, 1.0)
            noise = torch.randn_like(clean)
            noised = clean * (1.0 - t[:, None, None]) + noise * t[:, None, None]
            targets = noise - clean

            output = controlled_dit(
                noised,
                t,
                **cond_inputs,
                cfg_scale=1.0,
                control_guidance=GuidanceScales(text=1.0, attribute=1.0, audio=1.0),
                attribute_control_tokens=attr_tokens,
                audio_control_tokens=audio_tokens,
            )
            loss = F.mse_loss(output, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
            optimizer.step()

            if step == 1 or step % config.log_every == 0:
                print("step=%s loss=%.6f duration=%.2fs" % (step, float(loss), duration))

            if step % config.save_every == 0 or step == config.max_steps:
                ckpt = output_dir / ("%s_step_%06d.safetensors" % (config.adapter_name, step))
                model.save_adapter(
                    str(ckpt),
                    extra_metadata={
                        "step": str(step),
                        "manifest": config.manifest,
                        "controls": ",".join(config.controls),
                    },
                )
    finally:
        model.sa3.model.model = original_dit
