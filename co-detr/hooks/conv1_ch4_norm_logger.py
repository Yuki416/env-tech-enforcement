"""診斷探針：追蹤 backbone.conv1 第 4 通道（changemap）權重的 L2 norm。

用途：驗證 stem 真的有解凍、第 4 通道權重有在更新。
若 norm 從頭到尾不動 → 代表 conv1 仍被凍住（frozen_stages 設錯）。
同時記錄前 3 通道（RGB）norm 做對照。

在 before_train（初值）與每個 train epoch 結束時各記一次，寫進 runner.logger
與 message_hub（key: conv1_ch4_norm / conv1_rgb_norm），可在 log 直接看到變化。
"""
import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet.registry import HOOKS


@HOOKS.register_module()
class Conv1Ch4NormLogger(Hook):
    priority = "LOW"

    def _conv1_weight(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        # ResNet backbone: backbone.conv1 是第一層 7x7 conv，weight 形狀 [64,4,7,7]
        return model.backbone.conv1.weight

    @torch.no_grad()
    def _log(self, runner, tag):
        w = self._conv1_weight(runner)
        if w.shape[1] < 4:
            runner.logger.warning(
                f"[Conv1Ch4NormLogger] conv1 in_channels={w.shape[1]} < 4，跳過。")
            return
        ch4 = w[:, 3].norm().item()          # 第 4 通道（changemap）
        rgb = w[:, :3].norm().item()         # 前 3 通道（RGB）
        runner.logger.info(
            f"[Conv1Ch4NormLogger][{tag}] conv1 ch4(L2)={ch4:.6f} | "
            f"rgb(L2)={rgb:.6f} | ratio(ch4/rgb)={ch4/ (rgb + 1e-12):.4f}")
        mh = runner.message_hub
        mh.update_scalar("train/conv1_ch4_norm", ch4)
        mh.update_scalar("train/conv1_rgb_norm", rgb)

    def before_train(self, runner):
        self._log(runner, "init")

    def after_train_epoch(self, runner):
        self._log(runner, f"epoch_{runner.epoch}")
