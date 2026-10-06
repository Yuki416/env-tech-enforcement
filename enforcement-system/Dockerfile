# 環保執法系統 — VLM pipeline 執行環境
#
# 只裝 enforcement_app（VLM pipeline）需要的依賴：cv2 / numpy / PIL / requests / ultralytics。
# 傳統 pipeline（server/）需要 paddle + SAHI + transformers，且缺權重無法跑，故不裝。
#
# pyproject 要求 Python >=3.12,<3.13；ccu2 系統是 3.10，所以用官方 3.12 image。
# torch 走 cu124 wheel（自帶 CUDA runtime），host 只需 nvidia container runtime。

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# libgl1 / libglib2.0-0 : opencv 執行期依賴
# ffmpeg               : 影片解碼與 clip 匯出
# fonts-wqy-microhei   : vlm_enforcement._put_chinese_text 的 CJK 字型候選之一，
#                        沒有它截圖上的中文會變方框
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        fonts-wqy-microhei \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 先裝 CUDA 版 torch，避免後續 ultralytics 拉到 CPU-only 版本
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision

RUN pip install \
        ultralytics \
        opencv-python-headless \
        pillow \
        numpy \
        requests

# ── GUI 遠端操作能力（獨立一層，改動時不用重跑上面的 pip）─────────────
# ccu2 沒有實體螢幕也沒有桌面環境，用 Xvfb 開一個純記憶體的虛擬螢幕，
# 再用 x11vnc + noVNC 串到瀏覽器，就能在自己電腦上實際操作這個 Tkinter 程式。
#   tk        : slim image 缺 libtk8.6.so，沒有它 import tkinter 會失敗
#   xvfb      : 虛擬 X11 顯示裝置
#   x11vnc    : 把 X11 畫面轉成 VNC
#   novnc     : 瀏覽器端的 VNC 客戶端（HTML5）
#   websockify: 把 VNC 的 TCP 轉成 WebSocket 給瀏覽器用
#   scrot     : 截圖
#   xdotool   : 模擬滑鼠鍵盤（自動化截圖用）
RUN apt-get update && apt-get install -y --no-install-recommends \
        tk \
        xvfb \
        x11vnc \
        novnc \
        websockify \
        scrot \
        xdotool \
        x11-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 程式碼由 docker-compose 以 volume 掛入，方便邊改邊跑
CMD ["sleep", "infinity"]
