import unittest

import torch
from torch import nn

from steer_sao.adapters import current_control_context
from steer_sao.guidance import ControlledDiTModel
from steer_sao.types import GuidanceScales


class RecordingInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_batch_cfg = None
        self.last_padding_batch = None

    def forward(self, x, t, cross_attn_cond=None, padding_mask=None, batch_cfg=False, **_kwargs):
        self.last_batch_cfg = batch_cfg
        self.last_padding_batch = None if padding_mask is None else padding_mask.shape[0]
        if padding_mask is not None and padding_mask.shape[0] != x.shape[0]:
            raise AssertionError("padding_mask batch size must match expanded guidance batch")

        value = torch.zeros(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        if cross_attn_cond is not None:
            value = value + cross_attn_cond[:, :1, :1]
        ctx = current_control_context()
        if ctx is not None and ctx.attribute_tokens is not None:
            value = value + ctx.attribute_tokens[:, :1, :1]
        if ctx is not None and ctx.audio_tokens is not None:
            value = value + ctx.audio_tokens[:, :1, :1]
        return x + value


class GuidanceKwargTests(unittest.TestCase):
    def test_sampler_batch_cfg_kwarg_is_consumed_before_inner_call(self):
        inner = RecordingInner()
        wrapper = ControlledDiTModel(inner)

        out = wrapper(
            torch.zeros(1, 1, 1),
            torch.ones(1),
            cross_attn_cond=torch.full((1, 1, 1), 2.0),
            negative_cross_attn_cond=torch.zeros(1, 1, 1),
            batch_cfg=True,
            padding_mask=torch.ones(1, 4, dtype=torch.bool),
            cfg_scale=7.0,
            control_guidance=GuidanceScales(text=7.0, attribute=1.5, audio=1.0),
            attribute_control_tokens=torch.full((1, 1, 1), 3.0),
            audio_control_tokens=torch.full((1, 1, 1), 5.0),
        )

        self.assertTrue(torch.allclose(out, torch.tensor([[[23.5]]])))
        self.assertTrue(inner.last_batch_cfg)
        self.assertEqual(inner.last_padding_batch, 4)


if __name__ == "__main__":
    unittest.main()
