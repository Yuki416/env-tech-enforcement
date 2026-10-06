#!/usr/bin/env python3
"""
訓練監控腳本：每 5 epoch 把 loss 和 val AP 寫進 training_log.md
用法：在 container 外執行，掛載路徑已對應。
"""
import json
import os
import glob
import time
import re
from datetime import datetime
from pathlib import Path

WORK_DIR = "/path/to/repo/co-detr/results/chiayi_v4_36e"
LOG_MD   = os.path.join(WORK_DIR, "training_log.md")
LOG_INTERVAL = 5       # 每幾個 epoch append 一次
MAX_EPOCHS   = 36
CHECK_EVERY  = 120     # 每 120 秒掃描一次

logged_epochs = set()
train_start_time = time.time()


def find_json_log(work_dir):
    """找最新的 mmdetection JSON scalar log"""
    pattern = os.path.join(work_dir, "*/vis_data/scalars.json")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_json_log(json_path):
    """從 scalars.json 解析每個 epoch 的 loss 和 val AP"""
    epochs = {}
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = entry.get("epoch")
            if epoch is None:
                continue

            if epoch not in epochs:
                epochs[epoch] = {}

            # 訓練 loss
            for k in ["loss_cls", "loss_bbox", "loss_iou",
                      "loss", "grad_norm"]:
                if k in entry:
                    epochs[epoch][k] = entry[k]

            # 驗證 mAP（key 依 mmdet 版本可能有差異）
            for k in ["coco/bbox_mAP", "bbox_mAP",
                      "coco/bbox_mAP_50", "bbox_mAP_50"]:
                if k in entry:
                    epochs[epoch][k] = entry[k]

    return epochs


def format_row(epoch, info):
    loss_cls  = f"{info.get('loss_cls', '-'):.4f}"   if 'loss_cls'  in info else "-"
    loss_bbox = f"{info.get('loss_bbox', '-'):.4f}"  if 'loss_bbox' in info else "-"
    loss_iou  = f"{info.get('loss_iou', '-'):.4f}"   if 'loss_iou'  in info else "-"
    # val AP (try both key formats)
    map_val   = info.get("coco/bbox_mAP", info.get("bbox_mAP", None))
    map50_val = info.get("coco/bbox_mAP_50", info.get("bbox_mAP_50", None))
    val_map   = f"{map_val:.4f}"  if map_val  is not None else "-"
    val_map50 = f"{map50_val:.4f}" if map50_val is not None else "-"
    ts = datetime.now().strftime("%m-%d %H:%M")
    return f"| {epoch:5d} | {loss_cls:8} | {loss_bbox:9} | {loss_iou:8} | {val_map:7} | {val_map50:9} | {ts} |"


def append_to_log(rows):
    with open(LOG_MD, "r") as f:
        content = f.read()

    # 找到表格 header 後面插入新行
    marker = "| Epoch | loss_cls | loss_bbox | loss_iou | val mAP | val mAP50 | 時間戳 |"
    sep    = "|-------|----------|-----------|----------|---------|-----------|--------|"
    if marker not in content:
        print("[monitor] 找不到 table marker，跳過")
        return

    insert_after = sep
    idx = content.find(insert_after)
    if idx == -1:
        return
    idx += len(insert_after)

    new_rows_str = "\n" + "\n".join(rows)
    content = content[:idx] + new_rows_str + content[idx:]

    with open(LOG_MD, "w") as f:
        f.write(content)
    print(f"[monitor] 寫入 {len(rows)} 行到 training_log.md")


def write_final(epochs):
    """訓練結束後更新最終結果區塊"""
    total_sec = int(time.time() - train_start_time)
    h, m, s = total_sec // 3600, (total_sec % 3600) // 60, total_sec % 60

    # 找最後一個有 val mAP 的 epoch
    best_map = None
    best_epoch = None
    for ep, info in sorted(epochs.items()):
        m_val = info.get("coco/bbox_mAP", info.get("bbox_mAP"))
        if m_val is not None:
            if best_map is None or m_val > best_map:
                best_map = m_val
                best_epoch = ep

    last_ep = max(epochs.keys()) if epochs else MAX_EPOCHS
    last_info = epochs.get(last_ep, {})
    final_map  = last_info.get("coco/bbox_mAP", last_info.get("bbox_mAP", "—"))
    final_map50 = last_info.get("coco/bbox_mAP_50", last_info.get("bbox_mAP_50", "—"))

    best_ckpt = f"{WORK_DIR}/best_coco_bbox_mAP_epoch_{best_epoch}.pth" if best_epoch else "—"

    footer = f"""
- **總訓練時間**：{h}h {m}m {s}s
- **最終 val mAP**：{final_map}
- **最終 val mAP50**：{final_map50}
- **最佳 checkpoint epoch**：{best_epoch}（mAP={best_map}）
- **最佳 checkpoint 路徑**：`{best_ckpt}`
"""
    with open(LOG_MD, "r") as f:
        content = f.read()
    content = content.replace(
        "- **總訓練時間**：—\n- **最終 val mAP**：—\n- **最終 val mAP50**：—\n- **各類別 AP**：—\n- **最佳 checkpoint**：—",
        footer.strip()
    )
    with open(LOG_MD, "w") as f:
        f.write(content)
    print("[monitor] 已更新訓練結束記錄")


def main():
    print(f"[monitor] 啟動，監控 {WORK_DIR}")
    print(f"[monitor] 每 {CHECK_EVERY}s 掃描，每 {LOG_INTERVAL} epoch append 一次")

    pending_rows = []
    finished = False

    while True:
        json_log = find_json_log(WORK_DIR)
        if json_log:
            epochs = parse_json_log(json_log)
            new_rows = []
            for ep in sorted(epochs.keys()):
                if ep in logged_epochs:
                    continue
                if ep % LOG_INTERVAL == 0 or ep == MAX_EPOCHS:
                    row = format_row(ep, epochs[ep])
                    new_rows.append(row)
                    logged_epochs.add(ep)
            if new_rows:
                append_to_log(new_rows)

            # 判斷是否訓練結束
            if not finished and MAX_EPOCHS in epochs:
                write_final(epochs)
                finished = True
                print("[monitor] 訓練已完成，結束監控")
                break
        else:
            print(f"[monitor] 尚未找到 JSON log，等待訓練啟動...")

        time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    main()
