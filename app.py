from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
STABLE_AUDIO_3_REPO = "https://github.com/Stability-AI/stable-audio-3.git"
STABLE_AUDIO_3_REF = "fef6d875fbc7166d2e9daae035279a1c70ee3d61"
HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _add_local_src_to_path() -> None:
    for path in (ROOT, SRC):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _sync_hf_token_aliases() -> None:
    token = next((os.getenv(name) for name in HF_TOKEN_ENV_NAMES if os.getenv(name)), None)
    if not token:
        return
    for name in HF_TOKEN_ENV_NAMES:
        os.environ.setdefault(name, token)


def _is_hf_space() -> bool:
    return bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or os.getenv("SPACE_HOST"))


def _should_bootstrap_private_deps() -> bool:
    configured = os.getenv("STEER_SAO_BOOTSTRAP_PRIVATE_DEPS")
    if configured is not None:
        return _truthy(configured)
    return _is_hf_space() or bool(os.getenv("GITHUB_TOKEN"))


def _github_auth_env(token: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PIP_NO_INPUT"] = "1"
    try:
        count = int(env.get("GIT_CONFIG_COUNT", "0"))
    except ValueError:
        count = 0

    auth_base = f"https://x-access-token:{quote(token, safe='')}@github.com/"
    rewrites = (
        ("https://github.com/", auth_base),
        ("git@github.com:", auth_base),
    )
    for instead_of, replacement in rewrites:
        env[f"GIT_CONFIG_KEY_{count}"] = f"url.{replacement}.insteadOf"
        env[f"GIT_CONFIG_VALUE_{count}"] = instead_of
        count += 1
    env["GIT_CONFIG_COUNT"] = str(count)
    return env


def _install_stable_audio_3() -> None:
    if importlib.util.find_spec("stable_audio_3") is not None:
        return

    if not _should_bootstrap_private_deps():
        return

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "stable-audio-3 is not installed and GITHUB_TOKEN is missing. "
            "Add GITHUB_TOKEN as a Hugging Face Space secret or install stable-audio-3 "
            "before starting the app."
        )

    repo = os.getenv("STEER_SAO_STABLE_AUDIO_3_REPO", STABLE_AUDIO_3_REPO)
    ref = os.getenv("STEER_SAO_STABLE_AUDIO_3_REF", STABLE_AUDIO_3_REF)
    git_repo = repo if repo.startswith("git+") else f"git+{repo}"
    package = f"stable-audio-3 @ {git_repo}@{ref}"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--no-deps",
        package,
    ]

    print("Installing stable-audio-3 from GitHub using GITHUB_TOKEN...")
    try:
        subprocess.check_call(command, env=_github_auth_env(token))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to install stable-audio-3 from GitHub. Check that GITHUB_TOKEN "
            "can read the private repository and that STEER_SAO_STABLE_AUDIO_3_REF "
            "points to an accessible commit or branch."
        ) from exc


def _bootstrap_runtime() -> None:
    _add_local_src_to_path()
    _sync_hf_token_aliases()
    _install_stable_audio_3()


_bootstrap_runtime()

from scripts.gradio_app import create_demo, launch_kwargs  # noqa: E402


demo = create_demo()


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        **launch_kwargs(),
    )
