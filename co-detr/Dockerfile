FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# ── 修復 apt 金鑰 & 安裝系統套件 ────────────────────────────────────────────
RUN rm -rf /etc/apt/sources.list.d/cuda*.list && \
    apt-key del 7fa2af80 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends \
        git wget curl vim \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 基礎 Python 套件 ──────────────────────────────────────────────────────────
# numpy<2 必須先鎖定：PyTorch 2.1.0 是用 numpy 1.x 編譯的，
# numpy 升到 2.x 會導致 torch 無法載入
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "numpy<2" \
        jupyter jupyterlab \
        matplotlib "opencv-python-headless" pillow scipy timm \
        pycocotools

# ── MMEngine ─────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir openmim && \
    mim install mmengine

# ── MMCV v2.1.0 (CUDA 11.8 + PyTorch 2.1.0)
#    mmdet 3.3.0 要求 mmcv <2.2.0，所以用 2.1.0
#    pip 必須讀到完整 wheel 檔名才能解析 metadata，不可改名
RUN cd /tmp && \
    wget -c --tries=20 --timeout=120 --waitretry=10 \
        "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl" && \
    pip install --no-cache-dir mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl && \
    rm mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl

# ── MMDetection v3（相容 mmcv 2.x + PyTorch 2.x）────────────────────────────
# Co-DETR 在官方 mmdetection v3 的 projects/CO_DETR/ 目錄下
RUN mim install "mmdet>=3.0.0,<4.0.0"

# ── 確保 numpy 維持 1.x（後面套件可能把它升回去）──────────────────────────────
RUN pip install --no-cache-dir "numpy<2"

# ── Clone mmdetection 原始碼（取得 Co-DETR 的 config 與 model 定義）──────────
RUN git clone --depth=1 https://github.com/open-mmlab/mmdetection.git /opt/mmdetection

# 讓 Python 能找到 projects/CO_DETR 底下的 custom registry
ENV PYTHONPATH="/opt/mmdetection"

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", \
     "--notebook-dir=/workspace/project"]
