import unittest

import torch
from torch import nn
import torch.nn.functional as F

from steer_sao.adapters import (
    ActiveControlContext,
    MuseControlCrossAttention,
    current_control_context,
    use_control_context,
)
from steer_sao.guidance import ControlledDiTModel
from steer_sao.types import AdapterConfig, GuidanceScales


class FakeAttention(nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim // heads
        self.num_heads = heads
        self.kv_heads = heads
        self.differential = False
        self.qk_norm = "none"
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def apply_attn(self, q, k, v, causal=False, **_kwargs):
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        if causal:
            mask = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def forward(self, x, context=None, **_kwargs):
        context = x if context is None else context
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, self.num_heads, self.dim_heads).transpose(1, 2)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.view(b, context.shape[1], self.kv_heads, self.dim_heads).transpose(1, 2)
        v = v.view(b, context.shape[1], self.kv_heads, self.dim_heads).transpose(1, 2)
        out = self.apply_attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.dim)
        return self.to_out(out)


class AdapterTests(unittest.TestCase):
    def test_zero_initialized_adapter_preserves_base_output(self):
        torch.manual_seed(0)
        base = FakeAttention()
        wrapped = MuseControlCrossAttention(base, AdapterConfig(embed_dim=8, control_dim=8), "x")
        x = torch.randn(1, 3, 8)
        context = torch.randn(1, 4, 8)
        tokens = torch.randn(1, 5, 8)
        expected = base(x, context=context)
        with use_control_context(ActiveControlContext(attribute_tokens=tokens)):
            actual = wrapped(x, context=context)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


class EchoInner(nn.Module):
    def forward(self, x, t, cross_attn_cond=None, **_kwargs):
        value = torch.zeros(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        if cross_attn_cond is not None:
            value = value + cross_attn_cond[:, :1, :1]
        ctx = current_control_context()
        if ctx is not None and ctx.attribute_tokens is not None:
            value = value + ctx.attribute_tokens[:, :1, :1]
        if ctx is not None and ctx.audio_tokens is not None:
            value = value + ctx.audio_tokens[:, :1, :1]
        return x + value


class GuidanceTests(unittest.TestCase):
    def test_four_branch_guidance_formula(self):
        wrapper = ControlledDiTModel(EchoInner())
        x = torch.zeros(1, 1, 1)
        t = torch.ones(1)
        out = wrapper(
            x,
            t,
            cross_attn_cond=torch.full((1, 1, 1), 2.0),
            negative_cross_attn_cond=torch.zeros(1, 1, 1),
            cfg_scale=7.0,
            control_guidance=GuidanceScales(text=7.0, attribute=1.5, audio=1.0),
            attribute_control_tokens=torch.full((1, 1, 1), 3.0),
            audio_control_tokens=torch.full((1, 1, 1), 5.0),
        )
        self.assertTrue(torch.allclose(out, torch.tensor([[[23.5]]])))

    def test_three_branch_attribute_only(self):
        wrapper = ControlledDiTModel(EchoInner())
        out = wrapper(
            torch.zeros(1, 1, 1),
            torch.ones(1),
            cross_attn_cond=torch.full((1, 1, 1), 2.0),
            negative_cross_attn_cond=torch.zeros(1, 1, 1),
            cfg_scale=2.0,
            control_guidance=GuidanceScales(text=2.0, attribute=3.0, audio=1.0),
            attribute_control_tokens=torch.full((1, 1, 1), 4.0),
        )
        self.assertTrue(torch.allclose(out, torch.tensor([[[16.0]]])))


if __name__ == "__main__":
    unittest.main()
