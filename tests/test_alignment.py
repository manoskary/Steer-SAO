import unittest

from steer_sao.alignment import (
    adapt_audio_sample_size,
    latent_length_for_duration,
    latent_length_from_samples,
)


class AlignmentTests(unittest.TestCase):
    def test_latent_length_rounds_up(self):
        self.assertEqual(latent_length_from_samples(4096), 1)
        self.assertEqual(latent_length_from_samples(4097), 2)

    def test_adapt_audio_sample_size_aligns(self):
        samples = adapt_audio_sample_size(1.0, duration_padding_sec=0.0, latent_align=2)
        self.assertEqual(samples % (4096 * 2), 0)

    def test_duration_clamps_to_small_max(self):
        self.assertEqual(latent_length_for_duration(999.0), 5324800 // 4096)


if __name__ == "__main__":
    unittest.main()

