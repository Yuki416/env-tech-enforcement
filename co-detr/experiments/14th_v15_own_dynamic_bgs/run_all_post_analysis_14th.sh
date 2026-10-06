#!/usr/bin/env bash
# 第 14 次（v15 BGS 4ch）訓練後全套分析總入口：依序跑三階段（共用 GPU，不並行）。
#   Stage 1  166 動態子集評估（v15 vs 第13次）—— 輔助判定，快
#   Stage 2  6-part post-analysis（誤差分析/AP曲線/場景/score分布/對比/TP視覺化）—— 最花時間
#   Stage 3  影片推論 TP/TN/FP/FN 分類（4ch 版）
# 主判定（594 靜止 test）已先跑完：eval_static_test_vs_13th/
#
# 用法：
#   nohup bash experiments/14th_v15_own_dynamic_bgs/run_all_post_analysis_14th.sh \
#     > /path/to/data/14th_expt_v15_own_dynamic_bgs_4ch_36e/post_analysis_chain.log 2>&1 &
#   tail -f /path/to/data/14th_expt_v15_own_dynamic_bgs_4ch_36e/post_analysis_chain.log

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/14th_v15_own_dynamic_bgs"
RESULT=/path/to/data/14th_expt_v15_own_dynamic_bgs_4ch_36e

echo "############### Stage 1/3：166 動態子集評估 ###############"
bash "$EXP/eval_dynamic166_14th_vs_13th.sh" || echo "⚠️ Stage 1 失敗（續跑後續）"

echo; echo "############### Stage 2/3：6-part post-analysis ###############"
bash "$EXP/post_analysis_14th_v15.sh" || echo "⚠️ Stage 2 失敗（續跑後續）"

echo; echo "############### Stage 3/3：影片推論（4ch，TP/TN/FP/FN）###############"
docker run --rm --gpus '"device=0"' --ipc=host --shm-size=16g \
  -v "$ROOT":/workspace/project \
  -v /path/to/data/coco:/workspace/datasets/coco \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e PYTHONUTF8=1 \
  co-detr:latest \
  python /workspace/project/experiments/14th_v15_own_dynamic_bgs/infer_14th_v15_4ch_video_classification.py \
    --splits test valid train || echo "⚠️ Stage 3 失敗"

echo; echo "############### Stage 4/4：影片分類 IoU≥0.5 校正 ###############"
# Stage 3 的分類只看「有沒有預測 litter/garbage」→ 隨便吐一個框就算 TP，會高估。
# 用 error_analysis_score03 的 IoU≥0.5 匹配結果重算，並重渲染被改判影片的 HUD。
# （第13次曾遺漏此步，導致 valid/test 各 3/1 部影片被誤標 TP —— 勿再漏。）
docker run --rm \
  -v "$ROOT":/workspace/project \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONUTF8=1 -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 \
  co-detr:latest \
  python /workspace/project/experiments/14th_v15_own_dynamic_bgs/fix_video_classifications_14th.py \
  || echo "⚠️ Stage 4 失敗"

# 容器產出屬 root → 修正權限，否則後續 rclone 上傳/瀏覽會卡
docker run --rm -v /path/to/data:/path/to/data co-detr:latest \
  chown -R "$(id -u):$(id -g)" "$RESULT/video_inference/" 2>/dev/null || true

echo; echo "=================================================="
echo " 全部分析完成。產出："
echo "  主判定(594靜止):  $RESULT/eval_static_test_vs_13th/"
echo "  輔助(166動態):    $RESULT/eval_dynamic166_vs_13th/"
echo "  誤差分析:         $RESULT/error_analysis_score05/ (及 score03)"
echo "  13th vs 14th對比: $RESULT/error_analysis_score05/comparison_v14_13th_vs_v15_14th.md"
echo "  TP視覺化:         $RESULT/litter_tp_comparison/"
echo "  影片推論:         $RESULT/video_inference/"
echo "=================================================="
echo "ALL_POST_ANALYSIS_DONE"
