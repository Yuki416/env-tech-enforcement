#!/usr/bin/env python3
"""以 HEAD=10 為基底 + 逐 clip 手動覆寫，建「手動微調版」動態子集（第 16 次訓練）。

設計（使用者決定）：
  - 基底 = HEAD=10（`chiayi_v17_own_dynamic_litter_head10` 的配方，即 HEAD_WINDOW=10）。
  - 對某幾部 clip 用 **RANGE_OVERRIDE 直接指定最終擷取頭尾幀**（絕對幀號、所見即所得、與 HEAD 無關）。
  - 沒被手動調的 clip → 與純 HEAD=10 完全相同。
  - 輸出到**新資料夾**，不動純版：`chiayi_v17_own_dynamic_litter_head10_manual/`。

本檔只放「手動覆寫的值」，機制在共用的 `build_v17_dynamic_litter.py`（RANGE_OVERRIDE 空表）。
→ 純 head10/head15 版重建時不受這裡的值影響（值只在本腳本注入 B.RANGE_OVERRIDE）。

用法：PYTHONUTF8=1 python3 build_v17_manual_head10.py
"""
import importlib.util
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_v17", HERE / "build_v17_dynamic_litter.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

# ---- 基底 = HEAD=10、輸出到手動版新資料夾 ----
B.HEAD_WINDOW = 10
B.DST = Path("/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10_manual")

# ---- 手動覆寫表：(clip 名) -> (最終擷取起幀, 訖幀)，絕對幀號、所見即所得 ----
# 每筆註明使用者的判斷理由。key 用完整 clip 名（含 _RoboFlow 者照寫）。
B.RANGE_OVERRIDE = {
    # 黑袋拋出→落地的動態段(f6–24)；砍掉 f24–68 落地靜止長尾與 f69–70 冒出的第二包
    #（第二包由姊妹 clip 00m19s 擷取）。起=框首次出現(f6)、訖=落地+靜止一幀。
    "Day_Garbage_16_8_loc16_bureau_20240828064620_00m12s_RoboFlow": (6, 24),
    # 04m49s(train,重標)：litter f10–12 下墜→f17 重現→f18–20 微調→f21–24 完全靜止(落地)。
    # 自動「種子10–12+TAIL5」只收到 f17，切在落地前 → 使用者要 tail +5。
    # 起=1(同純head10)、訖=落地(f21)+靜止一幀(f22) → 1–22。
    "Day_Litter_12_8_loc12_bureau_20240603181958_04m49s": (1, 22),
    # 00m27s(train,garbage)：落地後 f36–41 完全靜止，f42 又移動一次(disp38)→自動種子含f41、tail拖到46。
    # 使用者要 tail −5、end=41：保留完整落地休息、砍掉 f42+ 第二次移動。起=1(同純head10)。
    "Day_Garbage_16_8_loc16_bureau_20240826063130_00m27s_RoboFlow": (1, 41),
    # 01m00s(train,garbage)：f1–10 拋出落地→f11–29 無框空檔→f30–32 garbage重現→f33+靜止。
    # 自動只收到 f13(種子1–8+TAIL5)。使用者要 tail 拉到 32，含重現的 3 幀(f30–32)。起=1。
    #（1–32 中間 f11–29 為 negative 填充。）
    "Day_Garbage_16_8_loc16_bureau_20240906054955_01m00s_RoboFlow": (1, 32),
    # 00m54s(train,litter,有人工種子(22,48))：litter f42 起完全靜止；tail+5 拖到 53。
    # 使用者要 end=44（trim 靜止長尾 f45–53）。start 保持 12(同純head10)。
    "Day_Litter_12_8_loc12_20240802173010_00m54s": (12, 44),
    # 24m02s(train,litter)：litter 拋物線 f2–19 移動→f26 再現(仍在動)。自動只偵測到 ~f11→收在 f16，
    # 漏掉後半軌跡。使用者要「種子取到26 + tail+2」→ end=28（f26 為最後 litter 幀）。start 保持 1。
    "Day_Litter_537_8_loc537_20240804171613_24m02s": (1, 28),
    # 04m39s(train,litter)：唯一種子 f16→自動收 6–21，切在掉落前；litter f16→f22(disp48)掉落、
    # f25 落地、f26–29 靜止。使用者要「種子取到25 + tail+5」→ end=30。start 保持 6。
    "Day_Litter_537_8_loc537_20240911175753_04m39s": (6, 30),
    # 06m22s(train,litter)：f9–13 掉落(f13 disp42)→f18 重現→f25+完全靜止(到f101)。自動僅種子f12→收2–17，
    # 切在掉落當下。使用者要「種子取到18 + tail+4」→ end=22（砍 f25–101 超長靜止尾）。start 保持 2。
    "Day_Litter_538_8_loc538_20240803182230_06m22s_RoboFlow": (2, 22),
}

if __name__ == "__main__":
    print(f"手動微調版：HEAD={B.HEAD_WINDOW}（基底）, 輸出 → {B.DST}")
    print(f"RANGE_OVERRIDE {len(B.RANGE_OVERRIDE)} 部：")
    for k, v in B.RANGE_OVERRIDE.items():
        print(f"  {k[:60]}  → {v}")
    print()

    # 收集 train+valid 所有 clip，驗證覆寫表（含 RANGE_OVERRIDE 命中數）
    all_clips = set()
    import json
    for sp in ["train", "valid"]:
        j = json.load(open(B.SRC / "annotations" / f"{sp}.json", encoding="utf-8"))
        all_clips |= {B.parse(i["file_name"])[0] for i in j["images"]}
    B.validate_overrides(all_clips)

    for sp in ["train", "valid"]:
        if (B.DST / sp).exists():
            shutil.rmtree(B.DST / sp)

    for sp in ["train", "valid"]:
        B.build_split(sp)
        print()
    B.copy_test()
    print(f"\n完成 → {B.DST}")
