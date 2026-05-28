from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_CHECKPOINT = "checkpoints/mcl_sa3_all_controls_cc0_full_lr2e6_step_020000.safetensors"
DEFAULT_MODEL = "small-music"
DEFAULT_OUTPUT_DIR = "generated_audio"
CONTROL_CHOICES = [
    ("Melody", "melody_stereo"),
    ("Rhythm", "rhythm"),
    ("Dynamics", "dynamics"),
    ("Audio reference", "audio"),
]
PROMPT_EXAMPLES = [
    ["lo-fi house loop with warm Rhodes chords, 120 BPM", ""],
    ["ambient electronic instrumental with spacious pads and soft tape noise", ""],
    ["cinematic downtempo groove with deep bass, brushed percussion, and evolving synths", ""],
    ["bright synthpop instrumental, punchy drums, nostalgic chords, 118 BPM", ""],
]
MODE_PRESETS = {
    "text": {
        "controls": [],
        "text": 7.0,
        "attribute": 0.0,
        "audio": 0.0,
    },
    "attributes": {
        "controls": ["melody_stereo", "rhythm", "dynamics"],
        "text": 7.0,
        "attribute": 1.5,
        "audio": 0.0,
    },
    "full": {
        "controls": ["melody_stereo", "rhythm", "dynamics", "audio"],
        "text": 7.0,
        "attribute": 1.5,
        "audio": 1.0,
    },
}
APP_CSS = """
.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
}
#steer-title h1 {
  margin-bottom: 0.25rem;
}
#steer-title p {
  color: var(--body-text-color-subdued);
  margin-top: 0;
}
.steer-panel {
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  padding: 14px;
  background: var(--background-fill-primary);
}
.steer-status {
  font-size: 0.92rem;
}
.steer-status table {
  width: 100%;
}
.steer-status td,
.steer-status th {
  padding: 4px 8px;
}
"""


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    negative_prompt: Optional[str]
    duration: float
    steps: int
    seed: int
    sampler_type: str
    controls: tuple[str, ...]
    control_audio_path: Optional[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Steer-SAO Gradio app")
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("STEER_SAO_CHECKPOINT", DEFAULT_CHECKPOINT),
        help="Local adapter checkpoint path. Ignored when --checkpoint-repo is used.",
    )
    parser.add_argument(
        "--checkpoint-repo",
        default=os.getenv("STEER_SAO_CHECKPOINT_REPO"),
        help="Optional private Hugging Face model repo containing the adapter checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default=os.getenv("STEER_SAO_CHECKPOINT_FILENAME"),
        help="Filename inside --checkpoint-repo. Defaults to the basename of --checkpoint.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("STEER_SAO_MODEL", DEFAULT_MODEL),
        help="Stable Audio 3 base model identifier",
    )
    parser.add_argument(
        "--gpu-index",
        default=os.getenv("STEER_SAO_GPU_INDEX"),
        help="Physical GPU index to expose. Leave unset to use the process environment.",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("STEER_SAO_DEVICE", "auto"),
        help="Torch device after any GPU masking. Use 'auto' for cuda:0 when available.",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        default=_truthy_env("STEER_SAO_NO_HALF"),
        help="Disable half precision model loading.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        help="Gradio server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        help="Gradio server port",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=_truthy_env("GRADIO_SHARE"),
        help="Enable Gradio share link",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("STEER_SAO_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Directory for written audio files",
    )
    return parser


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _set_gpu_environment(gpu_index: Optional[str]) -> None:
    if gpu_index:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _timestamped_filename(prompt: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in prompt).strip("_")
    slug = "_".join(part for part in slug.split("_") if part)
    slug = slug[:56] if slug else "sample"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{slug}.wav"


def _filepath_value(value):
    if isinstance(value, dict):
        return value.get("name") or value.get("path")
    return value


def _checkpoint_filename(checkpoint: str, checkpoint_filename: Optional[str]) -> str:
    if checkpoint_filename:
        return checkpoint_filename
    return Path(checkpoint).name


def _hf_token() -> Optional[str]:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


def _resolve_checkpoint(
    checkpoint: str,
    checkpoint_repo: Optional[str],
    checkpoint_filename: Optional[str],
) -> str:
    if checkpoint_repo:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError("Install huggingface_hub to download a remote checkpoint.") from exc

        return hf_hub_download(
            repo_id=checkpoint_repo,
            filename=_checkpoint_filename(checkpoint, checkpoint_filename),
            token=_hf_token(),
        )

    local_path = Path(checkpoint).expanduser()
    if local_path.exists():
        return str(local_path)

    raise FileNotFoundError(
        "Adapter checkpoint not found at %s. Set STEER_SAO_CHECKPOINT or "
        "STEER_SAO_CHECKPOINT_REPO." % checkpoint
    )


def _checkpoint_status_markdown(
    checkpoint: str,
    checkpoint_repo: Optional[str],
    checkpoint_filename: Optional[str],
) -> str:
    if checkpoint_repo:
        token_state = "set" if _hf_token() else "missing"
        filename = _checkpoint_filename(checkpoint, checkpoint_filename)
        return (
            "| Setting | Value |\n"
            "|---|---|\n"
            f"| Checkpoint repo | `{checkpoint_repo}` |\n"
            f"| Filename | `{filename}` |\n"
            f"| HF token | `{token_state}` |\n"
            "| Load mode | first generation |\n"
        )

    local_path = Path(checkpoint).expanduser()
    if local_path.exists():
        return _local_checkpoint_status(local_path)

    return (
        "| Setting | Value |\n"
        "|---|---|\n"
        f"| Checkpoint | `{checkpoint}` |\n"
        "| Status | missing |\n"
    )


def _local_checkpoint_status(path: Path) -> str:
    try:
        from safetensors import safe_open
    except Exception:
        return (
            "| Setting | Value |\n"
            "|---|---|\n"
            f"| Checkpoint | `{path}` |\n"
            "| Metadata | safetensors unavailable |\n"
        )

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensor_count = len(handle.keys())

    rows = [
        ("Checkpoint", f"`{path}`"),
        ("Status", "ready"),
        ("Tensors", str(tensor_count)),
        ("Step", metadata.get("step", "unknown")),
        ("Controls", metadata.get("controls") or metadata.get("control_types", "unknown")),
        ("Base", metadata.get("base_model_id", "unknown")),
    ]
    rendered = ["| Setting | Value |", "|---|---|"]
    rendered.extend(f"| {key} | {value} |" for key, value in rows)
    return "\n".join(rendered)


def _resolve_device(device: Optional[str]) -> str:
    if device and device != "auto":
        return device

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_model(
    checkpoint: str,
    checkpoint_repo: Optional[str],
    checkpoint_filename: Optional[str],
    model_name: str,
    device: str,
    model_half: bool,
):
    from steer_sao.model import SteerSAO

    resolved_device = _resolve_device(device)
    resolved_checkpoint = _resolve_checkpoint(checkpoint, checkpoint_repo, checkpoint_filename)
    use_half = bool(model_half and resolved_device.startswith("cuda"))
    model = SteerSAO.from_pretrained(
        model=model_name,
        adapter_path=resolved_checkpoint,
        device=resolved_device,
        model_half=use_half,
    )
    return model, resolved_checkpoint, resolved_device


def _normalize_seed(seed) -> int:
    if seed is None or seed == "":
        return random.randint(0, 2**31 - 1)
    seed_value = int(seed)
    if seed_value < 0:
        return random.randint(0, 2**31 - 1)
    return seed_value


def _prepare_generation_request(
    prompt: str,
    negative_prompt: str,
    duration: float,
    steps: int,
    seed,
    sampler_type: str,
    selected_controls: Optional[Iterable[str]],
    control_audio,
) -> GenerationRequest:
    from steer_sao.types import normalize_control_names

    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        raise ValueError("Add a prompt before generating.")

    controls = normalize_control_names(selected_controls or ())
    control_audio_path = _filepath_value(control_audio)
    if controls and not control_audio_path:
        raise ValueError("Upload reference audio or clear the selected controls.")

    clean_negative = (negative_prompt or "").strip() or None
    return GenerationRequest(
        prompt=clean_prompt,
        negative_prompt=clean_negative,
        duration=float(duration),
        steps=int(steps),
        seed=_normalize_seed(seed),
        sampler_type=sampler_type or "euler",
        controls=controls,
        control_audio_path=control_audio_path,
    )


def _format_result_status(
    request: GenerationRequest,
    output_path: Path,
    elapsed: float,
    resolved_checkpoint: str,
    resolved_device: str,
) -> str:
    controls = ", ".join(request.controls) if request.controls else "text only"
    return (
        "| Result | Value |\n"
        "|---|---|\n"
        f"| File | `{output_path}` |\n"
        f"| Seed | `{request.seed}` |\n"
        f"| Controls | `{controls}` |\n"
        f"| Steps | `{request.steps}` |\n"
        f"| Duration | `{request.duration:g}s` |\n"
        f"| Device | `{resolved_device}` |\n"
        f"| Checkpoint | `{resolved_checkpoint}` |\n"
        f"| Time | `{elapsed:.1f}s` |\n"
    )


def _apply_mode_preset(mode: str):
    preset = MODE_PRESETS.get(mode, MODE_PRESETS["attributes"])
    return (
        preset["controls"],
        preset["text"],
        preset["attribute"],
        preset["audio"],
    )


def _random_seed():
    return random.randint(0, 2**31 - 1)


def launch_kwargs():
    import gradio as gr

    return {
        "theme": gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="amber",
            neutral_hue="zinc",
            radius_size="sm",
        ),
        "css": APP_CSS,
    }


def _build_interface(
    checkpoint: str,
    checkpoint_repo: Optional[str],
    checkpoint_filename: Optional[str],
    model_name: str,
    device: str,
    model_half: bool,
    output_dir: Path,
):
    import gradio as gr

    def generate_audio(
        prompt: str,
        negative_prompt: str,
        duration: float,
        steps: int,
        seed,
        sampler_type: str,
        text_scale: float,
        attribute_scale: float,
        audio_scale: float,
        selected_controls,
        control_audio,
    ):
        from steer_sao.audio import save_audio
        from steer_sao.types import ControlInputs, GuidanceScales

        try:
            request = _prepare_generation_request(
                prompt=prompt,
                negative_prompt=negative_prompt,
                duration=duration,
                steps=steps,
                seed=seed,
                sampler_type=sampler_type,
                selected_controls=selected_controls,
                control_audio=control_audio,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        start_time = time.perf_counter()
        model, resolved_checkpoint, resolved_device = _load_model(
            checkpoint,
            checkpoint_repo,
            checkpoint_filename,
            model_name,
            device,
            model_half,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / _timestamped_filename(request.prompt)

        waveform = model.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            duration=request.duration,
            controls=ControlInputs(
                control_audio=request.control_audio_path,
                controls=request.controls,
            ),
            guidance=GuidanceScales(
                text=float(text_scale),
                attribute=float(attribute_scale),
                audio=float(audio_scale),
            ),
            seed=request.seed,
            steps=request.steps,
            sampler_type=request.sampler_type,
        )
        save_audio(str(output_path), waveform[0], model.sample_rate)
        status = _format_result_status(
            request=request,
            output_path=output_path,
            elapsed=time.perf_counter() - start_time,
            resolved_checkpoint=resolved_checkpoint,
            resolved_device=resolved_device,
        )
        return str(output_path), str(output_path), status

    with gr.Blocks(title="Steer-SAO") as demo:
        gr.Markdown(
            """
            # Steer-SAO
            Controlled music generation for Stable Audio 3 adapter checkpoints.
            """,
            elem_id="steer-title",
        )

        with gr.Row():
            with gr.Column(scale=7, min_width=520):
                with gr.Group(elem_classes="steer-panel"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=PROMPT_EXAMPLES[0][0],
                        lines=4,
                        max_lines=6,
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative prompt",
                        value="",
                        lines=2,
                        max_lines=3,
                    )
                    gr.Examples(
                        examples=PROMPT_EXAMPLES,
                        inputs=[prompt, negative_prompt],
                        cache_examples=False,
                    )

                with gr.Group(elem_classes="steer-panel"):
                    control_audio = gr.Audio(
                        label="Reference audio",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    mode = gr.Radio(
                        label="Mode",
                        choices=[
                            ("Text", "text"),
                            ("Attributes", "attributes"),
                            ("Full reference", "full"),
                        ],
                        value="attributes",
                    )
                    controls = gr.CheckboxGroup(
                        label="Controls",
                        choices=CONTROL_CHOICES,
                        value=MODE_PRESETS["attributes"]["controls"],
                    )

            with gr.Column(scale=5, min_width=360):
                with gr.Group(elem_classes="steer-panel"):
                    duration = gr.Slider(
                        label="Duration",
                        minimum=5,
                        maximum=120,
                        step=1,
                        value=30,
                    )
                    steps = gr.Slider(
                        label="Steps",
                        minimum=5,
                        maximum=30,
                        step=1,
                        value=10,
                    )
                    sampler_type = gr.Dropdown(
                        label="Sampler",
                        choices=["euler", "dpmpp-3m-sde"],
                        value="euler",
                        allow_custom_value=True,
                    )
                    with gr.Row():
                        seed = gr.Number(
                            label="Seed",
                            value=0,
                            precision=0,
                            minimum=-1,
                        )
                        random_seed = gr.Button("Random", size="sm")

                    with gr.Accordion("Guidance", open=True):
                        text_scale = gr.Slider(
                            label="Text",
                            minimum=0,
                            maximum=15,
                            step=0.1,
                            value=MODE_PRESETS["attributes"]["text"],
                        )
                        attribute_scale = gr.Slider(
                            label="Attributes",
                            minimum=0,
                            maximum=5,
                            step=0.1,
                            value=MODE_PRESETS["attributes"]["attribute"],
                        )
                        audio_scale = gr.Slider(
                            label="Audio",
                            minimum=0,
                            maximum=5,
                            step=0.1,
                            value=MODE_PRESETS["attributes"]["audio"],
                        )

                    generate = gr.Button("Generate Audio", variant="primary")

        with gr.Row():
            with gr.Column(scale=7, min_width=520):
                output_audio = gr.Audio(
                    label="Generated audio",
                    type="filepath",
                    autoplay=False,
                )
            with gr.Column(scale=5, min_width=360):
                output_file = gr.File(label="WAV file")
                run_status = gr.Markdown(
                    label="Run status",
                    value="",
                    elem_classes="steer-status",
                )

        with gr.Accordion("Model", open=False):
            gr.Markdown(
                _checkpoint_status_markdown(checkpoint, checkpoint_repo, checkpoint_filename),
                elem_classes="steer-status",
            )

        mode.change(
            fn=_apply_mode_preset,
            inputs=mode,
            outputs=[controls, text_scale, attribute_scale, audio_scale],
            show_progress="hidden",
        )
        random_seed.click(
            fn=_random_seed,
            inputs=None,
            outputs=seed,
            show_progress="hidden",
        )
        generate.click(
            fn=generate_audio,
            inputs=[
                prompt,
                negative_prompt,
                duration,
                steps,
                seed,
                sampler_type,
                text_scale,
                attribute_scale,
                audio_scale,
                controls,
                control_audio,
            ],
            outputs=[output_audio, output_file, run_status],
            show_progress="full",
        )

    demo.queue(max_size=8, default_concurrency_limit=1)
    return demo


def create_demo(
    checkpoint: Optional[str] = None,
    checkpoint_repo: Optional[str] = None,
    checkpoint_filename: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    model_half: Optional[bool] = None,
    output_dir: Optional[str] = None,
):
    return _build_interface(
        checkpoint=checkpoint or os.getenv("STEER_SAO_CHECKPOINT", DEFAULT_CHECKPOINT),
        checkpoint_repo=checkpoint_repo or os.getenv("STEER_SAO_CHECKPOINT_REPO"),
        checkpoint_filename=checkpoint_filename or os.getenv("STEER_SAO_CHECKPOINT_FILENAME"),
        model_name=model_name or os.getenv("STEER_SAO_MODEL", DEFAULT_MODEL),
        device=device or os.getenv("STEER_SAO_DEVICE", "auto"),
        model_half=(
            not _truthy_env("STEER_SAO_NO_HALF") if model_half is None else model_half
        ),
        output_dir=Path(output_dir or os.getenv("STEER_SAO_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _set_gpu_environment(args.gpu_index)

    demo = _build_interface(
        checkpoint=args.checkpoint,
        checkpoint_repo=args.checkpoint_repo,
        checkpoint_filename=args.checkpoint_filename,
        model_name=args.model,
        device=args.device,
        model_half=not args.no_half,
        output_dir=Path(args.output_dir),
    )
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        **launch_kwargs(),
    )


if __name__ == "__main__":
    main()
