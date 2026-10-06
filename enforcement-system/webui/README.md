# webui — 網頁版執法介面

把原本的 Tkinter 桌面程式換成網頁介面。**推論邏輯完全沒動**，只換人機介面那一層。

外觀來自 Gemini 依 `gui_redesign_bundle/README_for_AI.md` 產出的設計稿
（原稿保留在 `web_draft/index.html`，未修改）。這個資料夾負責讓它接上真的後端。

---

## 快速啟動

```bash
# mock 模式：不載入任何模型，秒開。用來看介面互動
docker compose exec -d enforcement bash -c \
  'cd /app && python -m webui.server --mock --host 0.0.0.0 --port 8080 \
     --videos /videos /testset /testset2 --output-root /output > /tmp/webui.log 2>&1'

# real 模式：接上真的 VLM pipeline
docker compose exec -d enforcement bash -c \
  'cd /app && python -m webui.server --real --host 0.0.0.0 --port 8080 \
     --videos /videos /testset /testset2 --output-root /output > /tmp/webui.log 2>&1'

docker compose exec enforcement tail -f /tmp/webui.log
```

`docker-compose.yml` 已把 `127.0.0.1:8080` 對外，**只綁 127.0.0.1**。
從自己的電腦看要透過 SSH 通道或 VSCode 的 PORTS 面板轉發 8080，然後開
<http://localhost:8080>。

停掉（容器內沒有 `pkill`，用 `/proc` 找）：

```bash
docker compose exec enforcement python -c "
import os,signal
for p in os.listdir('/proc'):
    if p.isdigit() and 'webui.server' in open(f'/proc/{p}/cmdline','rb').read().decode('utf-8','replace'):
        os.kill(int(p), signal.SIGKILL)"
```

---

## 檔案

| 檔案 | 職責 |
|---|---|
| `store.py` | `JobStore` — 所有狀態的單一事實來源；執行緒安全；註記持久化 |
| `engines.py` | `MockEngine`（時間軸模擬）與 `RealEngine`（包 `PipelineEngine`），同一個介面 |
| `server.py` | HTTP 路由、SSE 推播、Range 串流、路徑逃逸防護 |
| `static/index.html` | 設計稿改成資料驅動後的版本 |
| `static/app.js` | 渲染、SSE 接收、播放器、資料夾瀏覽器 |

### 為什麼用 stdlib `http.server` 而不是 FastAPI

1. 容器裡目前只有 torch / cv2 / ultralytics / pillow / requests。為了幾個端點多裝一套
   web 框架不划算，而且會拉長 image build。
2. `PipelineEngine` 本來就是 thread-based（`ThreadPoolExecutor`），
   `ThreadingHTTPServer` 的模型跟它一致，不必把同步程式碼硬塞進 async。

---

## API

| 端點 | 用途 |
|---|---|
| `GET /api/state?offset=&limit=` | 一頁 jobs（預設 100、上限 500）＋ events / stats / logs / config。頁面載入、翻頁與重連時用 |
| `GET /api/stream` | SSE。**只推變動**：`job`（單筆）／`reset`（整批變動，前端重抓自己那頁）／`log` |
| `GET /api/browse?path=` | 伺服器端目錄瀏覽。**瀏覽器的檔案選擇器看不到伺服器檔案系統**，所以必須後端做 |
| `POST /api/scan` | 遞迴掃描資料夾加入佇列（同路徑不重複）。超過 `--scan-confirm-limit`（預設 200）會先回 `needs_confirm`，帶 `confirm:true` 再送才真的加入 |
| `POST /api/start` | 用送來的 `JobOptions` 開跑 |
| `POST /api/clear` | 清掉已完成／失敗的列 |
| `POST /api/note` | 儲存稽查註記 |
| `POST /api/retry` | 重跑單一工作 |
| `GET /api/export.csv` | 整批總表（utf-8-sig，Excel 開中文不亂碼） |
| `GET /api/media?path=` | 取截圖或影片片段，支援 `Range`（時間軸拖曳要靠它） |

`/api/browse` 與 `/api/media` 都只允許白名單根目錄底下的路徑（`--videos` 與 `--output-root`），
路徑逃逸會回 403。

---

## 對設計稿只動了四處

CSS（978 行）、版面配置、圖示、文案**完全沒碰**，因為那是已經確認過的外觀。

1. 工作佇列的 7 列靜態 HTML → `<template id="tplJobRow">`，由 JS 依資料渲染
2. 違規事件的 3 列靜態 HTML → `<template id="tplEventRow">`
3. 移除設計稿裡兩個模擬用的 `setInterval`（改由伺服器狀態驅動）
4. 播放器加一個真的 `<video>`：有片段檔就播真影片，沒有就退回原本的 SVG 示意圖

另外附加了一段 CSS（資料夾瀏覽器、空狀態、連線指示），加在 `</style>` 之前，
沒有更動任何既有規則。

---

## 設計稿多出來的三個功能怎麼落地

| 功能 | 後端原本 | 實作方式 |
|---|---|---|
| 稽查註記 | `VideoJob` 沒有這個欄位 | 存兩份：`{output_root}/officer_notes.json`（索引，用影片路徑當 key）＋各舉證目錄的 `note.json`（跟證物一起帶走）。**不開資料庫** —— 舉證資料本來就是一案一資料夾，註記跟著走最自然 |
| 違規片段播放器 | 片段 mp4 有產出但沒有串流端點 | `GET /api/media` 支援 Range；時間軸紅標由真實事件的 `frame_time` 生成 |
| 證據截圖 Modal | 截圖有產出但沒有取圖端點 | 同 `/api/media`；有真截圖顯示照片，沒有退回示意圖 |

---

## 實跑驗證（2026-08-05）

### mock 模式
掃描 12 部影片、平行度 3 確實只跑 3 部、SSE 推送正常、`/etc` 被 403 擋下、
CSV 帶 BOM 且欄位沿用 `vlm_enforcement._write_csv` 格式多一欄稽查註記。

### real 模式
`/testset` 三部 avi、平行度 2：

- 引擎載入 2 秒；平行度 2 確實只跑 2 部，第 3 部維持「排隊中」
- VLM 遙測即時跳動，平均延遲 ~4.1 s/call
- 進度單調遞增：10.1% → 33.1% → 44.6%
- 輸出落在 `/output/{站別}/{時間戳}_{檔名}/`，含 `違規紀錄.csv` 與 `vlm_decisions.csv`

---

## 兩個實跑才發現、已修掉的問題

### 1. 進度條會前後跳動

`vlm_enforcement.py` 的 `progress_callback` 有三個來源，回報的 `frame` 語意不同：

| 來源 | stage | frame 的意義 |
|---|---|---|
| 第 1226 行（預取執行緒） | `prefilter` | **讀取前緣** |
| `HeartbeatEmitter.emit_vlm_call` | `vlm` | **讀取前緣**（取自 `frame_index_ref[0]`，第 1218 行由讀取迴圈寫入） |
| 第 1318 行（判定迴圈） | `vlm` | **判定前緣** |

兩種前緣交錯回報，後到的會把進度覆蓋回較小的值。實測 `frame` 欄在
483 → 504 → 567 與 21 → 147 → 63 → 84 之間來回跳。

**修法**：`JobStore.update()` 加進度單調遞增保護，執行中的階段只取
`max(新值, 舊值)`。`retry` 會把 status 設回 `queued`，那時才允許歸零。

**讀取前緣不會脫韁**：候選幀佇列上限是 `_VLM_PREFETCH_BATCH * 3 = 24`
（第 1185-1186 行），佇列滿了讀取執行緒就阻塞，所以讀取前緣最多領先判定前緣
24 個候選幀。實測 421 幀的短片進度為 34.9% → 74.8% → 89.5% → 完成，
分段跳但持續前進；2 小時的長片上這個領先量可忽略。

> 一度誤判成「讀取前緣會在幾秒內衝到 100% 然後卡住」，並據此寫了一版
> 依 `vlm_calls_done > 0` 分辨回呼來源、只採用判定前緣的邏輯。實測推翻了那個前提
> （佇列會阻塞讀取），該版本已撤回 —— 單調保護就夠，且不需要脆弱的來源判別。

### 2. 首次 VLM 呼叫期間，介面看起來像當機

`HeartbeatEmitter.emit_vlm_call()` 只在**呼叫回來之後**才發遙測。
實測 ollama 冷啟動時第一次呼叫花了 **67.5 秒**（佔整部片 85.8 秒的 79%），
那段期間 `vlm_calls_done` 一直是 0、`last_vlm_at` 也還沒有值，
介面顯示「尚未呼叫 AI」且不會觸發閒置警示 —— 正是廠商回報「像當機」的情境。

**修法**：
- `last_vlm_at` 從**開工**就起算，不等第一次呼叫完成。這樣卡住的首次呼叫
  也會正常觸發 2 分鐘閒置紅標（那正是 ollama 退回 CPU 的症狀）
- 新增 `awaiting_first_vlm` 旗標，介面顯示「等待首次判定 N 秒（模型載入中）」
  而不是誤導的「上次呼叫 N 秒前」

---

### 3. 佇列一多，介面就卡死（2026-08-07）

使用者掃了 `/videos` 根目錄 —— 那是 105GB 的完整影片庫，遞迴掃出 **1526 部**全部進佇列。
伺服器端其實很快（首頁 1.2ms），卡的是瀏覽器：

| | 修之前 | 修之後 |
|---|---|---|
| `GET /api/state` | **1,069,290 bytes**（1527 筆全送） | **72,457 bytes**（一頁 100 筆） |
| SSE 每次狀態變動 | **推整份 1MB 快照** | 推**單筆 job**，約 1.3KB |
| 前端 DOM | 渲染 1527 列，每秒 tick 全掃 | 只渲染當頁 100 列 |

進度回呼一秒會打好幾次，每次都推 1MB 給瀏覽器 `JSON.parse` + 重繪整張表 —— 這就是卡的原因。
**我當初只用 12 部影片測過，沒設想過有人會掃整個影片庫。**

三層修法：

1. **掃描門檻**（止血）：超過 `SCAN_CONFIRM_LIMIT` 先回 `needs_confirm`，前端跳確認框。
2. **`/api/state` 分頁**：`offset` / `limit`，預設 100、上限 500（擋 `limit=999999` 把 1MB 拉回來）。
   `JobStore.snapshot(offset, limit)` 只對當頁做 `to_dict()`。
3. **SSE 改推增量**：
   - `JobStore._push_job(job_id)` → `{type:"job", job, stats}`，進度／階段／遙測走這條
   - `JobStore._broadcast()` → `{type:"reset", stats}`，整批變動（掃描、清除、事件產生）
     只叫前端重抓自己那頁 —— 伺服器不知道各訂閱者停在第幾頁，推整份既錯又大
   - 前端對 `reset` 做 250ms debounce，批次掃描不會觸發重抓風暴

另外 `JobStore.stats()` 改成直接讀 `JobRecord` 欄位，不再對每個 job 呼叫 `to_dict()`；
`add_video` 的重複檢查也從每筆線性掃描（1526 部 ≈ 116 萬次比對）改成掃描前先取一次
路徑集合（`existing_video_paths()`）。

## 已知限制

- **`heavy_frame_stride` 與 `max_vlm_calls_per_video` 已接上，但 `require_plate_for_csv`
  只是傳給 `build_job_options`**，實際是否生效取決於 pipeline，本層沒有驗證過。
- **沒有工作持久化**。伺服器重啟後佇列會清空（註記不會，那個存在檔案裡）。
  長批次跑到一半重啟伺服器，正在跑的工作會遺失狀態。
- **`retry` 只是把狀態重設後重新送出**，不會中斷正在跑的執行緒。
- **stage 會在「YOLO 偵測中」與「VLM 判定中」之間來回切**。這是 pipeline 的真實行為
  （讀幀與判定交錯），沒有刻意壓平 —— 壓平等於發明後端沒有的行為。
