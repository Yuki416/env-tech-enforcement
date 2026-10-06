import os
from datetime import datetime

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class TrainingMarkdownLogger(Hook):
    """每個 epoch 結束後將訓練與驗證指標寫入 Markdown 檔案。"""

    priority = "VERY_LOW"

    def __init__(self, log_file):
        self.log_file = log_file
        self._epoch_start = None

    # ── 訓練開始：寫入標頭 ────────────────────────────────────────────────────

    def before_run(self, runner):
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
        cfg_name = getattr(runner.cfg, "filename", "N/A")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("# Training Log\n\n")
            f.write(f"**Config:** `{cfg_name}`\n\n")
            f.write(f"**Work dir:** `{runner.work_dir}`\n\n")
            f.write(f"**Started:** {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
            f.write("---\n\n")

    def before_train_epoch(self, runner):
        self._epoch_start = datetime.now()

    # ── 每個 epoch 驗證結束後寫入詳細紀錄 ────────────────────────────────────

    def after_val_epoch(self, runner, metrics):
        epoch = runner.epoch
        now = datetime.now()

        duration = ""
        if self._epoch_start:
            delta = now - self._epoch_start
            duration = str(delta).split(".")[0]

        # 學習率
        try:
            lr_dict = runner.optim_wrapper.get_lr()
            lr_val = list(lr_dict.values())[0][0] if isinstance(lr_dict, dict) else lr_dict[0]
            lr_str = f"{lr_val:.2e}"
        except Exception:
            lr_str = "N/A"

        # 訓練 loss（從 message_hub 取移動平均值）
        hub = runner.message_hub

        def get_loss(name):
            for key in (f"train/{name}", name):
                try:
                    buf = hub.log_scalars.get(key)
                    if buf is not None:
                        return f"{buf.mean:.4f}"
                except Exception:
                    pass
            return "N/A"

        losses = {
            "loss_cls":        get_loss("loss_cls"),
            "loss_bbox":       get_loss("loss_bbox"),
            "loss_iou":        get_loss("loss_iou"),
            "loss_centerness": get_loss("loss_centerness"),
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"## Epoch {epoch}\n\n")
            f.write(f"- **Time:** {now:%Y-%m-%d %H:%M:%S}  |  **Duration:** {duration}\n")
            f.write(f"- **LR:** {lr_str}\n\n")

            f.write("### Train Losses\n\n")
            f.write("| loss_cls | loss_bbox | loss_iou | loss_centerness |\n")
            f.write("|----------|-----------|----------|-----------------|\n")
            f.write(
                f"| {losses['loss_cls']} | {losses['loss_bbox']} "
                f"| {losses['loss_iou']} | {losses['loss_centerness']} |\n\n"
            )

            f.write("### Validation Metrics\n\n")

            # 主要 mAP 指標
            def fmt(key):
                for k in (key, f"coco/{key}"):
                    v = metrics.get(k)
                    if v is not None:
                        return f"{float(v):.4f}"
                return "N/A"

            f.write("| mAP | mAP@50 | mAP@75 | mAP_s | mAP_m | mAP_l |\n")
            f.write("|-----|--------|--------|-------|-------|-------|\n")
            f.write(
                f"| {fmt('bbox_mAP')} | {fmt('bbox_mAP_50')} | {fmt('bbox_mAP_75')}"
                f" | {fmt('bbox_mAP_s')} | {fmt('bbox_mAP_m')} | {fmt('bbox_mAP_l')} |\n\n"
            )

            # 全部 val metrics（含 classwise AP，key 名稱依 MMDetection 輸出而定）
            f.write("**All val metrics:**\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for k, v in sorted(metrics.items()):
                v_str = f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)
                f.write(f"| {k} | {v_str} |\n")
            f.write("\n---\n\n")

    # ── 訓練結束 ──────────────────────────────────────────────────────────────

    def after_run(self, runner):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"**Finished:** {datetime.now():%Y-%m-%d %H:%M:%S}\n")
