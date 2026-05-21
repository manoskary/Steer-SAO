import unittest

import torch

from steer_sao.controls import (
    MuseControlConditioner,
    extract_dynamics,
    extract_melody,
    extract_rhythm,
)


class ControlTests(unittest.TestCase):
    def test_extractors_return_batched_time_series(self):
        waveform = torch.randn(2, 8192)
        self.assertEqual(extract_dynamics(waveform).ndim, 3)
        self.assertEqual(extract_rhythm(waveform).shape[1], 2)
        self.assertEqual(extract_melody(waveform, stereo=False).shape[1], 12)
        self.assertEqual(extract_melody(waveform, stereo=True).shape[1], 24)

    def test_conditioner_shapes(self):
        conditioner = MuseControlConditioner(output_dim=16, hidden_dim=8)
        features = {
            "dynamics": torch.randn(1, 1, 5),
            "rhythm": torch.randn(1, 2, 5),
        }
        tokens = conditioner.encode_attributes(features, target_len=7)
        self.assertEqual(tuple(tokens.shape), (1, 7, 16))
        audio = conditioner.encode_audio_latents(torch.randn(1, 256, 4), target_len=7)
        self.assertEqual(tuple(audio.shape), (1, 7, 16))


if __name__ == "__main__":
    unittest.main()

