"""
從 test_data/ 的影片抽幀，輸出至 test_data/frames/
用法：python3 extract_frames.py
"""

import os
import subprocess

VIDEO_PATH = "/path/to/repo/co-detr/test_data/Night_Litter_539_8_loc539_20240812003815_00m51s.avi"
OUTPUT_DIR = "/path/to/repo/co-detr/test_data/539_frames"
FPS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

cmd = [
    "ffmpeg", "-y",
    "-i", VIDEO_PATH,
    "-vf", f"fps={FPS}",
    "-q:v", "2",           # JPEG 品質（2 = 高品質）
    os.path.join(OUTPUT_DIR, "frame_%04d.jpg"),
]

print(f"抽幀中（{FPS} fps）...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("ffmpeg 錯誤：")
    print(result.stderr)
else:
    frames = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg"))
    print(f"完成，共抽出 {len(frames)} 幀 → {OUTPUT_DIR}")
