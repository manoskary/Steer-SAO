import unittest

import torch
from torch import nn

from steer_sao.adapters import MuseControlAdapterManager
from steer_sao.types import AdapterConfig


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


class AdapterInstallTests(unittest.TestCase):
    def test_installed_adapters_follow_base_device_without_adopting_base_dtype(self):
        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.cross_attn = FakeAttention().to(dtype=torch.float64)

        class FakeRoot(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Identity()
                self.block = FakeBlock()

        root = FakeRoot()
        manager = MuseControlAdapterManager(AdapterConfig(embed_dim=8, control_dim=8))

        self.assertEqual(manager.install(root), 1)

        wrapped = root.block.cross_attn
        reference = next(wrapped.base_attention.parameters())
        adapter_parameter = next(wrapped.attribute_adapter.parameters())
        self.assertEqual(adapter_parameter.device, reference.device)
        self.assertEqual(adapter_parameter.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
