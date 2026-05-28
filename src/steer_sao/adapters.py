from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from steer_sao.types import AdapterConfig


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


@dataclass
class ActiveControlContext:
    attribute_tokens: Optional[torch.Tensor] = None
    audio_tokens: Optional[torch.Tensor] = None


_CONTROL_CONTEXT: ContextVar[Optional[ActiveControlContext]] = ContextVar(
    "steer_sao_control_context", default=None
)


def current_control_context() -> Optional[ActiveControlContext]:
    return _CONTROL_CONTEXT.get()


@contextmanager
def use_control_context(context: Optional[ActiveControlContext]) -> Iterator[None]:
    token = _CONTROL_CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTROL_CONTEXT.reset(token)


def _shape_as_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, seq, width = x.shape
    if width % heads != 0:
        raise ValueError("Projected width %s is not divisible by %s heads" % (width, heads))
    dim = width // heads
    return x.view(batch, seq, heads, dim).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(batch, seq, heads * dim)


def add_fractional_positions(tokens: torch.Tensor) -> torch.Tensor:
    """Add deterministic time positions to control tokens.

    This keeps control features aligned to the latent timeline without changing
    SA3's own text-conditioning path.
    """

    if tokens.shape[1] <= 1:
        return tokens

    batch, seq, dim = tokens.shape
    half = dim // 2
    if half == 0:
        return tokens
    pos = torch.linspace(0.0, 1.0, seq, device=tokens.device, dtype=tokens.dtype)
    freq = torch.exp(
        torch.linspace(0.0, 8.0, half, device=tokens.device, dtype=tokens.dtype)
    )
    angles = pos[:, None] * freq[None, :] * torch.pi
    pe = tokens.new_zeros(seq, dim)
    pe[:, :half] = torch.sin(angles)
    pe[:, half : half + half] = torch.cos(angles)
    return tokens + pe.unsqueeze(0).expand(batch, -1, -1)


class DecoupledCrossAttentionAdapter(nn.Module):
    """One MuseControlLite-style decoupled attention branch."""

    def __init__(
        self,
        base_attention: nn.Module,
        control_dim: int,
        position_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.control_dim = int(control_dim)
        self.position_encoding = bool(position_encoding)
        self.inner_dim = int(base_attention.dim)
        self.dim_heads = int(base_attention.dim_heads)
        self.num_heads = int(base_attention.num_heads)
        self.kv_heads = int(base_attention.kv_heads)
        self.to_k = nn.Linear(self.control_dim, self.kv_heads * self.dim_heads, bias=False)
        self.to_v = nn.Linear(self.control_dim, self.kv_heads * self.dim_heads, bias=False)
        self.to_out = zero_module(nn.Linear(self.inner_dim, self.inner_dim, bias=False))

    def forward(
        self,
        query_input: torch.Tensor,
        base_attention: nn.Module,
        control_tokens: torch.Tensor,
        flex_attention_block_mask=None,
        flex_attention_score_mod=None,
        flash_attn_sliding_window=None,
        padding_mask=None,
    ) -> torch.Tensor:
        if control_tokens is None:
            return torch.zeros_like(query_input)
        if control_tokens.ndim != 3:
            raise ValueError("control_tokens must have shape [batch, tokens, channels]")

        adapter_dtype = self.to_k.weight.dtype
        control_tokens = control_tokens.to(device=query_input.device, dtype=adapter_dtype)
        if self.position_encoding:
            control_tokens = add_fractional_positions(control_tokens)

        q_projected = base_attention.to_q(query_input)
        if getattr(base_attention, "differential", False):
            q_projected = q_projected.chunk(2, dim=-1)[0]
        q = _shape_as_heads(q_projected, self.num_heads).to(dtype=adapter_dtype)
        k = _shape_as_heads(self.to_k(control_tokens), self.kv_heads)
        v = _shape_as_heads(self.to_v(control_tokens), self.kv_heads)

        qk_norm = getattr(base_attention, "qk_norm", "none")
        if qk_norm == "l2":
            q = F.normalize(q, dim=-1, eps=getattr(base_attention, "qk_norm_eps", 1e-6))
            k = F.normalize(k, dim=-1, eps=getattr(base_attention, "qk_norm_eps", 1e-6))
        elif qk_norm != "none" and hasattr(base_attention, "apply_qk_layernorm"):
            q, k = base_attention.apply_qk_layernorm(q, k)

        attended = base_attention.apply_attn(
            q,
            k,
            v,
            causal=False,
            flex_attention_block_mask=flex_attention_block_mask,
            flex_attention_score_mod=flex_attention_score_mod,
            flash_attn_sliding_window=flash_attn_sliding_window,
            padding_mask=padding_mask,
            varlen_metadata=None,
        )
        return self.to_out(_merge_heads(attended)).to(dtype=query_input.dtype)


class MuseControlCrossAttention(nn.Module):
    """Drop-in wrapper around an SA3 cross-attention module."""

    def __init__(
        self,
        base_attention: nn.Module,
        config: AdapterConfig,
        layer_name: str,
    ) -> None:
        super().__init__()
        self.base_attention = base_attention
        self.layer_name = layer_name
        self.attribute_adapter = (
            DecoupledCrossAttentionAdapter(
                base_attention,
                config.control_dim,
                position_encoding=config.position_encoding,
            )
            if config.train_attribute_branch
            else None
        )
        self.audio_adapter = (
            DecoupledCrossAttentionAdapter(
                base_attention,
                config.control_dim,
                position_encoding=config.position_encoding,
            )
            if config.train_audio_branch
            else None
        )

    def forward(
        self,
        x,
        context=None,
        rotary_pos_emb=None,
        rotary_pos_emb_k=None,
        causal=None,
        flex_attention_block_mask=None,
        flex_attention_score_mod=None,
        flash_attn_sliding_window=None,
        padding_mask=None,
        varlen_metadata=None,
    ):
        base = self.base_attention(
            x,
            context=context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_emb_k=rotary_pos_emb_k,
            causal=causal,
            flex_attention_block_mask=flex_attention_block_mask,
            flex_attention_score_mod=flex_attention_score_mod,
            flash_attn_sliding_window=flash_attn_sliding_window,
            padding_mask=padding_mask,
            varlen_metadata=varlen_metadata,
        )

        control_context = current_control_context()
        if control_context is None:
            return base

        output = base
        if self.attribute_adapter is not None and control_context.attribute_tokens is not None:
            output = output + self.attribute_adapter(
                x,
                self.base_attention,
                control_context.attribute_tokens,
                flex_attention_block_mask=flex_attention_block_mask,
                flex_attention_score_mod=flex_attention_score_mod,
                flash_attn_sliding_window=flash_attn_sliding_window,
                padding_mask=None,
            )
        if self.audio_adapter is not None and control_context.audio_tokens is not None:
            output = output + self.audio_adapter(
                x,
                self.base_attention,
                control_context.audio_tokens,
                flex_attention_block_mask=flex_attention_block_mask,
                flex_attention_score_mod=flex_attention_score_mod,
                flash_attn_sliding_window=flash_attn_sliding_window,
                padding_mask=None,
            )
        return output


def _iter_cross_attention_blocks(root: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    for name, module in root.named_modules():
        cross_attn = getattr(module, "cross_attn", None)
        if cross_attn is None:
            continue
        if isinstance(cross_attn, MuseControlCrossAttention):
            yield name, module
            continue
        if hasattr(cross_attn, "to_q") and hasattr(cross_attn, "to_kv"):
            yield name, module


def resolve_dit_root(model) -> nn.Module:
    """Resolve StableAudioModel, ConditionedDiffusionModelWrapper, or DiTWrapper roots."""

    if hasattr(model, "dit"):
        return model.dit.model
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        return model.model
    if hasattr(model, "transformer"):
        return model
    raise TypeError("Could not resolve an SA3 DiffusionTransformer root from %r" % type(model))


def _canonical_adapter_module_name(name: str) -> str:
    name = name.replace(".inner.", ".")
    if name.startswith("inner."):
        name = name[len("inner.") :]
    return name


class MuseControlAdapterManager(nn.Module):
    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.installed_layers: List[str] = []

    def install(self, model) -> int:
        root = resolve_dit_root(model)
        count = 0
        for name, block in _iter_cross_attention_blocks(root):
            if isinstance(block.cross_attn, MuseControlCrossAttention):
                count += 1
                continue
            reference_parameter = next(block.cross_attn.parameters(), None)
            wrapped = MuseControlCrossAttention(block.cross_attn, self.config, name)
            if reference_parameter is not None:
                wrapped.to(device=reference_parameter.device)
            block.cross_attn = wrapped
            self.installed_layers.append(name)
            count += 1
        if count == 0:
            raise RuntimeError("No SA3 cross-attention layers were found to wrap")
        return count

    @staticmethod
    def freeze_base_train_adapters(model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, MuseControlCrossAttention):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
                for parameter in module.base_attention.parameters():
                    parameter.requires_grad_(False)

    @staticmethod
    def adapter_state_dict(model, prefix: str = "adapters.") -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if isinstance(module, MuseControlCrossAttention):
                module_name = _canonical_adapter_module_name(name)
                module_state = module.state_dict()
                for key, value in module_state.items():
                    if key.startswith("base_attention."):
                        continue
                    state[prefix + module_name + "." + key] = value.detach().cpu()
        return state

    @staticmethod
    def load_adapter_state_dict(model, state: Dict[str, torch.Tensor], prefix: str = "adapters.") -> None:
        own = dict(model.named_modules())
        grouped: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, value in state.items():
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            parts = rest.split(".")
            module_name = ".".join(parts[:-3])
            param_name = ".".join(parts[-3:])
            grouped.setdefault(module_name, {})[param_name] = value

        missing_layers = []
        for module_name, module_state in grouped.items():
            module = own.get(module_name)
            if module is None and "inner." in module_name:
                module = own.get(_canonical_adapter_module_name(module_name))
            if module is None:
                missing_layers.append(module_name)
                continue
            module.load_state_dict(module_state, strict=False)
        if missing_layers:
            raise RuntimeError("Adapter checkpoint referenced missing layers: %s" % missing_layers)


def trainable_adapter_parameters(model) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
