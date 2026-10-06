from __future__ import annotations

from pathlib import Path


def scan_video_files(folder: Path, exts: tuple[str, ...]) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
