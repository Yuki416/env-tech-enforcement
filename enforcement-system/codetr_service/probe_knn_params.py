#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐一實測 OpenCV BackgroundSubtractorKNN 的每個參數是否符合理論意義、是否可控。

每個參數各設計一個「它應該要影響的量」去量，而不是全部都量吸收次數 ——
用錯的量去測一個參數，會得到「這個參數沒用」的假結論（2026-08-23 我對 NSamples 就犯過）。
"""
import cv2, numpy as np

BG = 170
def scene(rng, bg=BG, sigma=2, size=60):
    return np.clip(np.full((size,size,3), bg, np.float64)
                   + rng.normal(0, sigma, (size,size,3)), 0, 255).astype(np.uint8)

def warm(k, n=150, sigma=2, bg=BG):
    rng = np.random.default_rng(0)
    for _ in range(n): k.apply(scene(rng, bg, sigma))
    return rng

def mk(**kw):
    p = dict(history=500, dist2Threshold=500.0, detectShadows=True)
    k = cv2.createBackgroundSubtractorKNN(
        history=p['history'], dist2Threshold=p['dist2Threshold'],
        detectShadows=p['detectShadows'])
    for a, v in kw.items():
        # setter 命名不一致：setDist2Threshold 首字大寫，但 setkNNSamples 是小寫 k
        fn = getattr(k, 'set'+a, None) or getattr(k, 'set'+a[0].upper()+a[1:])
        fn(v)
    return k

def absorb(lr=0.0, maxn=3000, **kw):
    """靜止色塊要幾次 apply 才被吃進背景"""
    k = mk(**kw); rng = warm(k)
    for i in range(1, maxn+1):
        f = scene(rng); f[20:40,20:40] = 20
        m = k.apply(f) if lr is None else k.apply(f, learningRate=lr)
        if (m[20:40,20:40] == 255).mean() < 0.5: return i
    return None

print("="*78)
print("① dist2Threshold — 理論：顏色距離平方超過它就算前景")
print("="*78)
print("  預測：門檻亮度差 Δ = sqrt(dist2 / 3)   （BGR 三通道，距離平方 ≈ 3Δ²）")
print(f"  {'dist2':>8} {'預測 Δ':>8} {'實測最小可偵測 Δ':>18}")
for d2 in [100.0, 400.0, 500.0, 1200.0]:
    k = mk(dist2Threshold=d2); rng = warm(k)
    found = None
    for delta in range(1, 60):
        f = scene(rng); f[20:40,20:40] = BG - delta
        if (k.apply(f, learningRate=0)[20:40,20:40] == 255).mean() > 0.5:
            found = delta; break
    print(f"  {d2:>8} {np.sqrt(d2/3):>8.1f} {str(found):>18}")

print()
print("="*78)
print("② kNNSamples — 理論：要幾個記憶樣本『夠像』才判為背景（KNN 的 K）")
print("="*78)
print(f"  {'kNNSamples':>11} {'吸收次數':>10} {'2K+2 預測':>10}")
for kn in [1,2,3,4,5]:
    print(f"  {kn:>11} {str(absorb(kNNSamples=kn)):>10} {2*kn+2:>10}")

print()
print("="*78)
print("③ NSamples — 理論：每像素記幾個樣本。多樣本 = 記得住『會來回變化』的背景")
print("="*78)
print("  用『搖晃的樹葉』模擬多模態背景：背景在兩個顏色間交替")
print(f"  {'NSamples':>9} {'誤判為前景的比例':>18}   說明")
for ns in [2,3,5,7,15]:
    k = mk(NSamples=ns); rng = np.random.default_rng(0)
    for i in range(300):                       # 暖機：兩色交替
        bg = 170 if i % 2 == 0 else 90
        k.apply(scene(rng, bg))
    fp = []
    for i in range(40):                        # 量測：同樣兩色交替，應全判背景
        bg = 170 if i % 2 == 0 else 90
        m = k.apply(scene(rng, bg), learningRate=0)
        fp.append((m == 255).mean())
    note = "記不住兩種背景" if np.mean(fp) > 0.05 else ""
    print(f"  {ns:>9} {np.mean(fp)*100:>17.2f}%   {note}")

print()
print("="*78)
print("④ detectShadows / shadowThreshold / shadowValue")
print("="*78)
print("  理論：比背景暗、但暗得不夠多的像素判為『陰影』(輸出 shadowValue)")
print("  shadowThreshold=0.5 → 亮度落在背景的 50%~100% 之間算陰影")
print(f"  {'色塊亮度':>8} {'佔背景比例':>10} {'理論':>6} {'實測輸出值':>10}")
k = mk(); rng = warm(k)
for v in [150, 120, 100, 85, 60, 30]:
    f = scene(rng); f[20:40,20:40] = v
    m = k.apply(f, learningRate=0)
    val = int(np.bincount(m[20:40,20:40].ravel()).argmax())
    ratio = v/BG
    theory = "陰影" if 0.5 <= ratio < 1.0 else "前景"
    print(f"  {v:>8} {ratio:>10.2f} {theory:>6} {val:>10}  {'(127=陰影)' if val==127 else '(255=前景)' if val==255 else ''}")

print()
print("  shadowThreshold 可調嗎（色塊亮度 60 = 背景的 0.35）：")
for st in [0.2, 0.35, 0.5]:
    k = mk(shadowThreshold=st); rng = warm(k)
    f = scene(rng); f[20:40,20:40] = 60
    val = int(np.bincount(k.apply(f, learningRate=0)[20:40,20:40].ravel()).argmax())
    print(f"    shadowThreshold={st} → 輸出 {val} {'(陰影)' if val==127 else '(前景)'}")

print()
print("  shadowValue 可調嗎：")
for sv in [127, 60, 200]:
    k = mk(shadowValue=sv); rng = warm(k)
    f = scene(rng); f[20:40,20:40] = 120
    val = int(np.bincount(k.apply(f, learningRate=0)[20:40,20:40].ravel()).argmax())
    print(f"    shadowValue={sv} → 輸出 {val} {'✓' if val==sv else '✗'}")

print()
print("  detectShadows 關掉：")
for ds in [True, False]:
    k = mk(detectShadows=ds); rng = warm(k)
    f = scene(rng); f[20:40,20:40] = 120
    val = int(np.bincount(k.apply(f, learningRate=0)[20:40,20:40].ravel()).argmax())
    print(f"    detectShadows={ds} → 輸出 {val}")

print()
print("="*78)
print("⑤ history — 理論：模型記憶多長，越長學越慢")
print("="*78)
print(f"  {'history':>8} {'預設呼叫 apply(f)':>18} {'明確傳 learningRate=0.002':>26}")
for h in [50, 200, 500, 2000]:
    print(f"  {h:>8} {str(absorb(lr=None, history=h)):>18} {str(absorb(lr=0.002, history=h)):>26}")
