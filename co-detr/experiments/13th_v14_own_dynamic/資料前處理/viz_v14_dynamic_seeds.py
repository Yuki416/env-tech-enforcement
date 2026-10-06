#!/usr/bin/env python3
"""把 v14 動態種子幀畫成 montage，人工檢查是否真為『丟/落垃圾』動作。

- 綠框 = litter GT。標題 = clip 短名 / frame idx / 該幀最大 litter 位移(px/幀)。
- 每個 clip 取 1 個代表種子幀，最多 24 格，鋪成網格。
"""
from PIL import Image, ImageDraw
from pathlib import Path
import importlib.util, sys

_HERE = "/path/to/repo/co-detr/experiments/13th_v14_own_dynamic/資料前處理"
spec = importlib.util.spec_from_file_location("b", f"{_HERE}/build_v14_dynamic_litter.py")
b = importlib.util.module_from_spec(spec); sys.modules["b"] = b; spec.loader.exec_module(b)

# 註：此為早期『單幀 montage』，已被 verify_v14_dynamic_pairs.py（前後幀配對圖）取代；
#     若要重跑需配合 build 的新 select_frames 6-tuple 回傳值調整。
SPLIT = "train"
OUT = Path(f"{_HERE}/v14_dynamic_seeds_montage.png")


def seed_speed(imgs, litter_by_img, clip_frames, img_id):
    """回傳該種子幀最大 litter 位移 px/幀（用於標註）。"""
    im = imgs[img_id]
    frames = sorted(clip_frames[im["clip"]])
    idx = [k for k, (_, i) in enumerate(frames) if i == img_id][0]
    f_idx = frames[idx][0]
    gate = b.MATCH_GATE_FRAC * (im["width"] ** 2 + im["height"] ** 2) ** 0.5
    nxt = next((frames[j] for j in range(idx + 1, len(frames))
                if 0 < frames[j][0] - f_idx <= b.GAP_CAP), None)
    if not nxt:
        return 0.0
    dframe = nxt[0] - f_idx
    dst_c = [b.center(x) for x in litter_by_img[nxt[1]]]
    best_all = 0.0
    for box in litter_by_img[img_id]:
        cx, cy = b.center(box)
        d = min((((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5) for dx, dy in dst_c)
        if d <= gate:
            best_all = max(best_all, d / dframe)
    return best_all


def main():
    j, imgs, litter_by_img, seed_frames, selected = b.select_frames(SPLIT)
    from collections import defaultdict
    clip_frames = defaultdict(list)
    for iid in litter_by_img:
        clip_frames[imgs[iid]["clip"]].append((imgs[iid]["frame"], iid))

    # 每 clip 取一個種子幀
    by_clip = {}
    for iid in seed_frames:
        by_clip.setdefault(imgs[iid]["clip"], iid)
    picks = list(by_clip.values())[:24]

    cell_w, cell_h, bar = 480, 300, 22  # 每格影像尺寸 + 上方標題列
    cols = 4; n = len(picks); rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + bar)), (20, 20, 20))
    draw_c = ImageDraw.Draw(canvas)

    for k, iid in enumerate(picks):
        im = imgs[iid]
        img = Image.open(b.SRC / SPLIT / im["file_name"]).convert("RGB")
        sx, sy = cell_w / img.width, cell_h / img.height
        img = img.resize((cell_w, cell_h))
        d = ImageDraw.Draw(img)
        for box in litter_by_img[iid]:
            x, y, w, h = box
            d.rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy], outline=(0, 255, 0), width=3)
        r, c = divmod(k, cols)
        ox, oy = c * cell_w, r * (cell_h + bar)
        sp = seed_speed(imgs, litter_by_img, clip_frames, iid)
        short = im["clip"].split("bureau_")[-1] if "bureau_" in im["clip"] else im["clip"]
        short = "".join(ch for ch in short if ord(ch) < 128)[:20]  # PIL 內建字型只吃 ASCII
        draw_c.text((ox + 4, oy + 4), f"{short} f{im['frame']} | {sp:.0f}px/f", fill=(255, 255, 0))
        canvas.paste(img, (ox, oy + bar))

    canvas.save(OUT)
    print(f"saved {OUT}  ({n} 個 clip 的種子幀)")


if __name__ == "__main__":
    main()
