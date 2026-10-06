#!/usr/bin/env python3
"""
Organize pseudo_labels output for RoboFlow upload.
For each video folder, creates {video_name}_RoboFlow/ with frames/ + pseudo_labels.json,
then collects all _RoboFlow folders into pseudo_labels_清晨_晚間/.
"""

import os
import shutil
from pathlib import Path

PSEUDO_LABELS_DIR = Path("/path/to/data/pseudo_labels")
OUTPUT_DIR = Path("/path/to/data/pseudo_labels_清晨_晚間")

OUTPUT_DIR.mkdir(exist_ok=True)

video_folders = sorted([d for d in PSEUDO_LABELS_DIR.iterdir() if d.is_dir()])
total = len(video_folders)

print(f"Found {total} video folders")
print(f"Output: {OUTPUT_DIR}\n")

for i, video_dir in enumerate(video_folders, 1):
    video_name = video_dir.name
    roboflow_name = f"{video_name}_RoboFlow"
    dest = OUTPUT_DIR / roboflow_name

    frames_src = video_dir / "frames"
    json_src = video_dir / "pseudo_labels.json"

    missing = []
    if not frames_src.exists():
        missing.append("frames/")
    if not json_src.exists():
        missing.append("pseudo_labels.json")

    if missing:
        print(f"[{i}/{total}] SKIP {video_name} — missing: {', '.join(missing)}")
        continue

    if dest.exists():
        print(f"[{i}/{total}] SKIP {roboflow_name} — already exists")
        continue

    dest.mkdir()
    shutil.copytree(frames_src, dest / "frames")
    shutil.copy2(json_src, dest / "pseudo_labels.json")

    frame_count = len(list((dest / "frames").iterdir()))
    print(f"[{i}/{total}] OK  {roboflow_name}  ({frame_count} frames)")

print(f"\nDone. Output: {OUTPUT_DIR}")
print(f"Total folders: {len(list(OUTPUT_DIR.iterdir()))}")
