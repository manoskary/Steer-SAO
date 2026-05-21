import json
import tempfile
import unittest
from pathlib import Path

from steer_sao.manifest import load_manifest, parse_manifest_row, rows_to_jsonl


class ManifestTests(unittest.TestCase):
    def test_parse_valid_row(self):
        row = parse_manifest_row(
            {
                "audio_path": "a.wav",
                "prompt": "piano",
                "duration": 2,
                "controls": {"melody_stereo": "m.pt"},
            },
            1,
        )
        self.assertEqual(row.audio_path, "a.wav")
        self.assertEqual(row.duration, 2.0)

    def test_rejects_unknown_control(self):
        with self.assertRaises(ValueError):
            parse_manifest_row(
                {"audio_path": "a.wav", "prompt": "piano", "controls": {"bad": "x.pt"}},
                1,
            )

    def test_load_and_dump_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            path.write_text(json.dumps({"audio_path": "a.wav", "prompt": "p"}) + "\n")
            rows = load_manifest(str(path))
            self.assertEqual(len(rows), 1)
            self.assertIn('"audio_path": "a.wav"', rows_to_jsonl(rows))


if __name__ == "__main__":
    unittest.main()

