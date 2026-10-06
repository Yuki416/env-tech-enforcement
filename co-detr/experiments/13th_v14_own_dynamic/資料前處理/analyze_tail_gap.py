#!/usr/bin/env python3
"""量測每個 clip 影片『尾巴』有多長：最後一個動態 seed frame 之後，還拖了多少 frame 到影片結束。

對每個 clip 回報：
  N              = 影片總 frame 數（= 目前 v14 連續片段選中的幀）
  vstart..vend   = 影片起訖 frame index
  seed 範圍       = 第一個~最後一個動態 seed 的 frame index
  head_gap       = 第一個 seed - 影片起點（= 前置 padding）
  tail_gap       = 影片結束 - 最後一個 seed（= 動態結束後拖的尾巴，frame 單位）
  tail_frames    = frame index > 最後 seed 的幀數（尾巴實際張數）
  tail_%         = tail_frames / N（尾巴佔影片比例）
不改任何資料，只讀數值。
"""
from collections import defaultdict
import importlib.util, sys, statistics

_HERE = "/path/to/repo/co-detr/experiments/13th_v14_own_dynamic/資料前處理"
spec = importlib.util.spec_from_file_location("b", f"{_HERE}/build_v14_dynamic_litter.py")
b = importlib.util.module_from_spec(spec); sys.modules["b"] = b; spec.loader.exec_module(b)


def analyze(split):
    j, imgs, boxes_by_cat, seed_frames, selected, _ = b.select_frames(split)
    sel_by_clip = defaultdict(list)
    for iid in selected:
        sel_by_clip[imgs[iid]["clip"]].append(imgs[iid]["frame"])
    seed_by_clip = defaultdict(list)
    for iid in seed_frames:
        seed_by_clip[imgs[iid]["clip"]].append(imgs[iid]["frame"])

    rows = []
    for clip, fr in sel_by_clip.items():
        fr = sorted(fr); seeds = sorted(seed_by_clip[clip])
        N = len(fr); vstart, vend = fr[0], fr[-1]
        first_seed, last_seed = seeds[0], seeds[-1]
        tail_gap = vend - last_seed
        tail_frames = sum(1 for f in fr if f > last_seed)
        head_gap = first_seed - vstart
        rows.append((clip, N, vstart, vend, first_seed, last_seed,
                     head_gap, tail_gap, tail_frames, tail_frames / N))

    rows.sort(key=lambda r: -r[-1])  # 尾巴比例大的排前面
    print(f"\n================ [{split}] 每個 clip 的尾巴量測（共 {len(rows)} 部）================")
    print(f"{'clip(末20字)':<22}{'N':>4}{'vend':>6}{'lastSeed':>9}{'tailGap':>8}{'tailFr':>7}{'tail%':>7}")
    for clip, N, vs, ve, fs, ls, hg, tg, tf, tp in rows:
        short = clip[-20:]
        print(f"{short:<22}{N:>4}{ve:>6}{ls:>9}{tg:>8}{tf:>7}{tp*100:>6.0f}%")

    tg_list = [r[7] for r in rows]; tf_list = [r[8] for r in rows]; tp_list = [r[9] for r in rows]
    print(f"--- [{split}] 摘要：tailGap(frame) 中位={statistics.median(tg_list):.0f} "
          f"平均={statistics.mean(tg_list):.1f} 最大={max(tg_list)} | "
          f"tail% 中位={statistics.median(tp_list)*100:.0f}% 平均={statistics.mean(tp_list)*100:.0f}% "
          f"最大={max(tp_list)*100:.0f}% | HEAD={b.HEAD_WINDOW} TAIL={b.TAIL_WINDOW}")
    return rows


if __name__ == "__main__":
    for sp in ["train", "valid"]:
        analyze(sp)
