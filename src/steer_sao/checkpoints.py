from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from steer_sao.types import AdapterConfig


def _require_safetensors():
    try:
        from safetensors.torch import load_file, save_file
    except Exception as exc:
        raise RuntimeError(
            "safetensors is required for adapter checkpoints. Install with `pip install safetensors`."
        ) from exc
    return load_file, save_file


def save_adapter_checkpoint(
    path: str,
    state_dict: Dict[str, torch.Tensor],
    config: AdapterConfig,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> None:
    _, save_file = _require_safetensors()
    metadata = {
        "format": "steer-sao-adapter-v1",
        "base_model_id": config.base_model_id,
        "base_model_revision": config.base_model_revision,
        "control_types": ",".join(config.control_types),
        "adapter_config_json": json.dumps(
            {
                "embed_dim": config.embed_dim,
                "control_dim": config.control_dim,
                "hidden_dim": config.hidden_dim,
                "position_encoding": config.position_encoding,
                "train_attribute_branch": config.train_attribute_branch,
                "train_audio_branch": config.train_audio_branch,
                "control_types": list(config.control_types),
            },
            sort_keys=True,
        ),
    }
    if extra_metadata:
        metadata.update({str(k): str(v) for k, v in extra_metadata.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, path, metadata=metadata)


def load_adapter_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    load_file, _ = _require_safetensors()
    from safetensors import safe_open

    state = load_file(path, device="cpu")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    if metadata.get("format") != "steer-sao-adapter-v1":
        raise ValueError("Unsupported adapter checkpoint format in %s" % path)
    return state, metadata


def split_checkpoint_state(
    state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    adapter_state = {k: v for k, v in state.items() if k.startswith("adapters.")}
    conditioner_state = {
        k[len("conditioner.") :]: v
        for k, v in state.items()
        if k.startswith("conditioner.")
    }
    return adapter_state, conditioner_state


def merge_checkpoint_state(
    adapter_state: Dict[str, torch.Tensor],
    conditioner_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    merged = dict(adapter_state)
    for key, value in conditioner_state.items():
        merged["conditioner." + key] = value.detach().cpu()
    return merged
