from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from steer_sao.adapters import ActiveControlContext, use_control_context
from steer_sao.types import GuidanceScales


def _repeat_or_none(value: Optional[torch.Tensor], count: int) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return torch.cat([value] * count, dim=0)


def _zero_like_or_none(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return torch.zeros_like(value)


def _cat(values: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(values, dim=0)


def _split(output: torch.Tensor, names: List[str], batch_size: int) -> Dict[str, torch.Tensor]:
    chunks = output.split(batch_size, dim=0)
    return {name: chunk for name, chunk in zip(names, chunks)}


class ControlledDiTModel(nn.Module):
    """Wrap an SA3 DiTWrapper and perform MCL multi-branch guidance.

    The wrapped model is still called once per sampling step. Branches are
    concatenated along batch dimension, matching SA3's batched CFG style while
    adding attribute and audio-control branches.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_attn_cond: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        negative_cross_attn_cond: Optional[torch.Tensor] = None,
        negative_cross_attn_mask: Optional[torch.Tensor] = None,
        input_concat_cond: Optional[torch.Tensor] = None,
        local_add_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        negative_global_cond: Optional[torch.Tensor] = None,
        prepend_cond: Optional[torch.Tensor] = None,
        prepend_cond_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        control_guidance: Optional[GuidanceScales] = None,
        attribute_control_tokens: Optional[torch.Tensor] = None,
        audio_control_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        guidance = control_guidance or GuidanceScales(text=float(cfg_scale))

        has_text_guidance = cross_attn_cond is not None and guidance.text != 1.0
        has_attr = attribute_control_tokens is not None
        has_audio = audio_control_tokens is not None

        if not has_text_guidance and not has_attr and not has_audio:
            return self.inner(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_mask=cross_attn_mask,
                input_concat_cond=input_concat_cond,
                local_add_cond=local_add_cond,
                global_cond=global_cond,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                cfg_scale=1.0,
                **kwargs,
            )

        batch_size = x.shape[0]
        names: List[str] = []
        xs: List[torch.Tensor] = []
        ts: List[torch.Tensor] = []
        cross: List[torch.Tensor] = []
        cross_masks: List[torch.Tensor] = []
        attr_tokens: List[torch.Tensor] = []
        audio_tokens: List[torch.Tensor] = []

        zero_cross = (
            negative_cross_attn_cond
            if negative_cross_attn_cond is not None
            else _zero_like_or_none(cross_attn_cond)
        )
        zero_cross_mask = (
            negative_cross_attn_mask
            if negative_cross_attn_mask is not None
            else cross_attn_mask
        )

        def add_branch(
            name: str,
            branch_cross: Optional[torch.Tensor],
            branch_mask: Optional[torch.Tensor],
            branch_attr: Optional[torch.Tensor],
            branch_audio: Optional[torch.Tensor],
        ) -> None:
            names.append(name)
            xs.append(x)
            ts.append(t)
            if branch_cross is not None:
                cross.append(branch_cross)
            if branch_mask is not None:
                cross_masks.append(branch_mask)
            if attribute_control_tokens is not None:
                attr_tokens.append(
                    branch_attr
                    if branch_attr is not None
                    else torch.zeros_like(attribute_control_tokens)
                )
            if audio_control_tokens is not None:
                audio_tokens.append(
                    branch_audio
                    if branch_audio is not None
                    else torch.zeros_like(audio_control_tokens)
                )

        if has_text_guidance:
            add_branch("uncond", zero_cross, zero_cross_mask, None, None)
            add_branch("text", cross_attn_cond, cross_attn_mask, None, None)
        else:
            add_branch("text", cross_attn_cond, cross_attn_mask, None, None)

        if has_attr:
            add_branch("attr", cross_attn_cond, cross_attn_mask, attribute_control_tokens, None)
        if has_audio:
            add_branch(
                "audio",
                cross_attn_cond,
                cross_attn_mask,
                attribute_control_tokens if has_attr else None,
                audio_control_tokens,
            )

        branch_count = len(names)
        x_cat = _cat(xs)
        t_cat = _cat(ts)
        cross_cat = _cat(cross) if cross else None
        cross_mask_cat = _cat(cross_masks) if cross_masks else None
        attr_cat = _cat(attr_tokens) if attr_tokens else None
        audio_cat = _cat(audio_tokens) if audio_tokens else None

        with use_control_context(ActiveControlContext(attr_cat, audio_cat)):
            output = self.inner(
                x_cat,
                t_cat,
                cross_attn_cond=cross_cat,
                cross_attn_mask=cross_mask_cat,
                input_concat_cond=_repeat_or_none(input_concat_cond, branch_count),
                local_add_cond=_repeat_or_none(local_add_cond, branch_count),
                global_cond=_repeat_or_none(global_cond, branch_count),
                prepend_cond=_repeat_or_none(prepend_cond, branch_count),
                prepend_cond_mask=_repeat_or_none(prepend_cond_mask, branch_count),
                cfg_scale=1.0,
                batch_cfg=True,
                **kwargs,
            )

        outputs = _split(output, names, batch_size)

        uncond = outputs.get("uncond", outputs["text"])
        text = outputs["text"]
        result = uncond + guidance.text * (text - uncond) if has_text_guidance else text

        attr_base = text
        if has_attr:
            attr = outputs["attr"]
            result = result + guidance.attribute * (attr - attr_base)
            attr_base = attr

        if has_audio:
            audio = outputs["audio"]
            result = result + guidance.audio * (audio - attr_base)

        return result

