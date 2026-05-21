import tempfile
import unittest
from pathlib import Path

import torch

from steer_sao.checkpoints import (
    load_adapter_checkpoint,
    merge_checkpoint_state,
    save_adapter_checkpoint,
    split_checkpoint_state,
)
from steer_sao.types import AdapterConfig


class CheckpointTests(unittest.TestCase):
    def test_split_and_merge_state(self):
        merged = merge_checkpoint_state(
            {"adapters.layer.weight": torch.ones(1)},
            {"x.weight": torch.zeros(1)},
        )
        adapter, conditioner = split_checkpoint_state(merged)
        self.assertIn("adapters.layer.weight", adapter)
        self.assertIn("x.weight", conditioner)

    def test_safetensors_round_trip_when_available(self):
        try:
            import safetensors  # noqa: F401
        except Exception:
            self.skipTest("safetensors is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.safetensors"
            save_adapter_checkpoint(
                str(path),
                {"adapters.x.attribute_adapter.to_k.weight": torch.ones(1, 1)},
                AdapterConfig(control_types=("melody_stereo",)),
            )
            state, metadata = load_adapter_checkpoint(str(path))
            self.assertIn("adapters.x.attribute_adapter.to_k.weight", state)
            self.assertEqual(metadata["format"], "steer-sao-adapter-v1")


if __name__ == "__main__":
    unittest.main()

