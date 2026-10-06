# YOLO baseline 執行環境（YOLO_baseline_定案規劃.md 步驟 1）
#
# 為什麼不裝進 co-detr:latest：那個 image 卡在 torch 2.1 + numpy<2，裝 Ultralytics 可能把它弄壞。
#
# 底層選 pytorch 2.9.1 + CUDA 12.6：本機 (ccu2) driver 535 官方只到 CUDA 12.2，
# 但 2026-10-01 實測此 image 靠 CUDA minor version compatibility 能正常在 RTX 4090 上運算
# （conv 與 CPU 結果最大差 7e-7、AMP fp16 正常），且本機已有此 image，不必另外下載。
#
# 建置：docker build -t yolo-baseline:latest Envir_project/baseline/yolo
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# opencv-python（Ultralytics 的相依套件）需要 libGL 與 glib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 版本寫死，報告才能重現。torch 已由底層 image 提供，pip 不會重裝。
RUN pip install --no-cache-dir "ultralytics==8.4.171" pycocotools

# 資料集檔名含中文：強制 UTF-8，否則 pycocotools 以 ascii 讀 json 會出錯。
# 容器以一般使用者身分執行（--user，避免在 HDD 上留下 root 擁有的檔案），
# 所以 Ultralytics 與 matplotlib 的設定目錄要預先建好並開放寫入；
# 畫圖用的 Arial 字型也先下載進 image，不必每次執行都上網抓。
ENV PYTHONUTF8=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    YOLO_CONFIG_DIR=/opt/ultralytics_cfg \
    MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p /opt/ultralytics_cfg/Ultralytics /tmp/matplotlib \
    && python -c "import urllib.request as u; u.urlretrieve('https://ultralytics.com/assets/Arial.ttf', '/opt/ultralytics_cfg/Ultralytics/Arial.ttf')" \
    && chmod -R 777 /opt/ultralytics_cfg /tmp/matplotlib

WORKDIR /workspace
