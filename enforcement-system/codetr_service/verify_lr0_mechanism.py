#!/usr/bin/env python3
"""驗證 learningRate=0 的內部機制：α → 整數更新週期的換算公式。

## 假說

《learningRate=0 不是「凍結背景」》第 06 節從行為反推出一個假說：
OpenCV 的 KNN 把 α 換算成一個**整數的更新週期**，α=0 時那個換算退化成「每次都更新」。
原文明說那是假說、沒有讀原始碼。這支腳本把它變成可證偽的預測並實測。

## 具體公式（假說）

    Kshort       = int( log(0.7) / log(1 - α) ) + 1
    nUpdateShort = Kshort ÷ NSamples          （整數除法）
    nUpdateShort == 0  →  每次 apply 都更新（最快）

α = 0 時 log(1-0) = 0，除法得到 −inf，取整後是負數 → 一樣落在「每次都更新」。
這解釋了為什麼 α=0 的行為跟 α=0.1、α=0.7 完全一樣，而 α=0.0002 慢 100 倍以上。

## 兩個可證偽的預測

  預測一：存在一個臨界 α，跨過去行為會**階梯式**跳到「跟 α=0 相同」，不是平滑變化。
          NSamples=7 時臨界值 = 0.0577。
  預測二：臨界 α 會隨 NSamples 移動，且位置由上面的公式決定。

兩個預測都命中（NSamples ∈ {3,5,7,10,15,20} 六組，相對誤差 0.0%）→ 機制成立。

用法：python verify_lr0_mechanism.py
"""
import math
import cv2
import numpy as np

WARM = 150          # 暖機幀數（走自動模式，與訓練 pass1 相同）
PATCH = (slice(80, 120), slice(80, 120))    # 40×40 靜止色塊的位置
BG_LEVEL, PATCH_LEVEL, NOISE_SIGMA = 170, 20, 2.0


def absorb(alpha, nsamples, knn_k=2, max_apply=600, seed=0):
    """貼上靜止色塊後，量「幾次 apply 之後它被吃進背景」。回傳次數，未吸收回 None。"""
    rng = np.random.default_rng(seed)
    knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=500.0,
                                            detectShadows=True)
    knn.setNSamples(nsamples)
    knn.setkNNSamples(knn_k)
    base = np.full((200, 200, 3), BG_LEVEL, np.uint8)

    def frame(with_patch):
        f = np.clip(base.astype(np.int16) + rng.normal(0, NOISE_SIGMA, base.shape),
                    0, 255).astype(np.uint8)
        if with_patch:
            f[PATCH] = PATCH_LEVEL
        return f

    for _ in range(WARM):
        knn.apply(frame(False))                     # 自動模式暖機
    for i in range(1, max_apply + 1):
        m = knn.apply(frame(True), learningRate=alpha)
        if (m[PATCH] == 255).mean() < 0.5:
            return i
    return None


def predicted_threshold(nsamples):
    """假說預測的臨界 α：Kshort < NSamples 時 nUpdateShort 變成 0。"""
    lo, hi = 1e-5, 0.9
    for _ in range(90):
        mid = (lo + hi) / 2
        if int(math.log(0.7) / math.log(1 - mid)) + 1 < nsamples:
            hi = mid
        else:
            lo = mid
    return hi


def measured_threshold(nsamples):
    """二分搜實測的階梯位置：吸收次數掉到與 α=0 相同的最小 α。"""
    baseline = absorb(0.0, nsamples)
    lo, hi = 1e-4, 0.9
    for _ in range(22):
        mid = (lo + hi) / 2
        if absorb(mid, nsamples) == baseline:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2, baseline


def main():
    print(f"OpenCV {cv2.__version__}")
    print("\n=== 預測一：NSamples=7 時臨界 α 應該是 0.0577，而且是階梯 ===")
    print(f"{'α':>9}{'預測更新週期':>13}{'實測吸收次數':>13}")
    for a in [0.0, 0.05, 0.056, 0.0577, 0.058, 0.06, 0.1, 0.7]:
        per = 1 if a == 0 else int(math.log(0.7) / math.log(1 - a)) // 7 + 1 + (0 if a == 0 else 0)
        per = 1 if a == 0 else (int(math.log(0.7) / math.log(1 - a)) + 1) // 7 + 1
        print(f"{a:>9.4f}{per:>13}{str(absorb(a, 7)):>13}")

    print("\n=== 預測二：臨界 α 隨 NSamples 移動，位置由公式決定 ===")
    print(f"{'NSamples':>9}{'預測臨界 α':>13}{'實測階梯':>11}{'相對誤差':>10}{'α=0 吸收次數':>14}")
    for nN in (3, 5, 7, 10, 15, 20):
        p = predicted_threshold(nN)
        m, base = measured_threshold(nN)
        print(f"{nN:>9}{p:>13.4f}{m:>11.4f}{abs(m - p) / p * 100:>9.1f}%{base:>14}")


if __name__ == "__main__":
    main()
