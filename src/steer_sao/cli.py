from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from steer_sao.audio import save_audio
from steer_sao.controls import extract_control_features_from_audio
from steer_sao.manifest import load_manifest
from steer_sao.model import SteerSAO
from steer_sao.training.trainer import train_adapter_from_config
from steer_sao.types import ControlInputs, GuidanceScales, normalize_control_names


def _split_controls(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def prepare_data(args) -> None:
    rows = load_manifest(args.manifest)
    controls = normalize_control_names(_split_controls(args.controls))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "manifest.controls.jsonl"
    with index_path.open("w", encoding="utf-8") as index:
        for i, row in enumerate(rows):
            features = extract_control_features_from_audio(row.audio_path, controls)
            control_paths = dict(row.controls)
            stem = "%06d" % i
            for name, tensor in features.features.items():
                path = out_dir / ("%s.%s.pt" % (stem, name))
                torch.save(tensor.cpu(), path)
                control_paths[name] = str(path)
            payload = {
                "audio_path": row.audio_path,
                "prompt": row.prompt,
                "controls": control_paths,
            }
            if row.duration is not None:
                payload["duration"] = row.duration
            if row.split is not None:
                payload["split"] = row.split
            index.write(json.dumps(payload, sort_keys=True) + "\n")
    print("Wrote %s" % index_path)


def train(args) -> None:
    train_adapter_from_config(args.config)


def generate(args) -> None:
    controls = normalize_control_names(_split_controls(args.controls))
    model = SteerSAO.from_pretrained(
        model=args.model,
        adapter_path=args.adapter,
        device=args.device,
        model_half=not args.no_half,
    )
    audio = model.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        duration=args.duration,
        controls=ControlInputs(control_audio=args.control_audio, controls=controls),
        guidance=GuidanceScales(
            text=args.cfg_scale_text,
            attribute=args.cfg_scale_attr,
            audio=args.cfg_scale_audio,
        ),
        seed=args.seed,
        steps=args.steps,
        sampler_type=args.sampler_type,
    )
    save_audio(args.out, audio[0], model.sample_rate)
    print("Wrote %s" % args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="steer-sao")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare-data", help="Validate manifest and precompute controls")
    p_prepare.add_argument("--manifest", required=True)
    p_prepare.add_argument("--out", required=True)
    p_prepare.add_argument(
        "--controls",
        default="melody_stereo,rhythm,dynamics",
        help="Comma-separated controls to precompute",
    )
    p_prepare.set_defaults(func=prepare_data)

    p_train = sub.add_parser("train", help="Train SA3 MuseControlLite adapters")
    p_train.add_argument("--config", required=True)
    p_train.set_defaults(func=train)

    p_gen = sub.add_parser("generate", help="Generate controlled music")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--duration", type=float, required=True)
    p_gen.add_argument("--adapter")
    p_gen.add_argument("--control-audio")
    p_gen.add_argument("--controls", default="")
    p_gen.add_argument("--out", default="generated_audio/out.wav")
    p_gen.add_argument("--model", default="small-music")
    p_gen.add_argument("--device")
    p_gen.add_argument("--negative-prompt")
    p_gen.add_argument("--steps", type=int, default=50)
    p_gen.add_argument("--seed", type=int)
    p_gen.add_argument("--cfg-scale-text", type=float, default=7.0)
    p_gen.add_argument("--cfg-scale-attr", type=float, default=1.5)
    p_gen.add_argument("--cfg-scale-audio", type=float, default=1.0)
    p_gen.add_argument("--sampler-type", default="euler")
    p_gen.add_argument("--no-half", action="store_true")
    p_gen.set_defaults(func=generate)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

