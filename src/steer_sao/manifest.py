from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from steer_sao.types import ManifestRow, normalize_control_names


def parse_manifest_row(raw: dict, line_number: int = 0) -> ManifestRow:
    if not isinstance(raw, dict):
        raise ValueError("Manifest line %s must be a JSON object" % line_number)

    audio_path = raw.get("audio_path")
    prompt = raw.get("prompt")
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError("Manifest line %s is missing audio_path" % line_number)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Manifest line %s is missing prompt" % line_number)

    duration = raw.get("duration")
    if duration is not None:
        duration = float(duration)
        if duration <= 0:
            raise ValueError("Manifest line %s has non-positive duration" % line_number)

    controls = raw.get("controls") or {}
    if not isinstance(controls, dict):
        raise ValueError("Manifest line %s controls must be an object" % line_number)
    normalize_control_names(controls.keys())

    split = raw.get("split")
    if split is not None and not isinstance(split, str):
        raise ValueError("Manifest line %s split must be a string" % line_number)

    return ManifestRow(
        audio_path=audio_path,
        prompt=prompt,
        duration=duration,
        split=split,
        controls={str(k): str(v) for k, v in controls.items()},
    )


def load_manifest(path: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(parse_manifest_row(json.loads(stripped), line_number))
    if not rows:
        raise ValueError("Manifest %s did not contain any rows" % path)
    return rows


def rows_to_jsonl(rows: Iterable[ManifestRow]) -> str:
    lines = []
    for row in rows:
        payload = {
            "audio_path": row.audio_path,
            "prompt": row.prompt,
            "controls": row.controls,
        }
        if row.duration is not None:
            payload["duration"] = row.duration
        if row.split is not None:
            payload["split"] = row.split
        lines.append(json.dumps(payload, sort_keys=True))
    return "\n".join(lines) + "\n"

