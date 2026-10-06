"""網頁介面後端 —— 純 stdlib，無新增依賴。

為什麼不用 FastAPI：
  1. 容器裡目前只有 torch / cv2 / ultralytics / pillow / requests，多裝一套 web 框架
     只是為了幾個端點，不划算。
  2. PipelineEngine 本來就是 thread-based（ThreadPoolExecutor），
     ThreadingHTTPServer 的模型跟它一致，不需要把同步程式碼硬塞進 async。

用法：
    python -m webui.server --mock           # 假資料，不載模型，秒開
    python -m webui.server --real           # 接真的 VLM pipeline
    python -m webui.server --real --port 8080 --videos /videos /testset2
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import queue
import re
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.engines import BaseEngine, MockEngine, RealEngine  # noqa: E402
from webui.store import JobStore  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".ts"}

# 這些在 module 層由 main() 填好，handler 才讀得到
STORE: JobStore
ENGINE: BaseEngine
ALLOWED_ROOTS: list[Path] = []
DEFAULT_SOURCE = ""
# 單次掃描超過這個數量就先要求確認（見 _scan 的說明）。用 --scan-confirm-limit 調整。
SCAN_CONFIRM_LIMIT = 200
# /api/state 的 jobs 分頁大小。前端一次只渲染這麼多列。
JOBS_PAGE_SIZE = 100
JOBS_PAGE_MAX = 500


def infer_station(path: Path) -> str:
    """站別解析。用 enforcement_app 既有規則，讓網頁版與 Tkinter 版輸出路徑一致。"""
    try:
        from enforcement_app.station import infer_station_id

        return infer_station_id(path)
    except Exception:  # noqa: BLE001 — mock 模式下不強制依賴
        match = re.search(r"(?:IPC|CH)\s*0*(\d+)", path.name, re.IGNORECASE)
        if match:
            return f"CH{match.group(1)}"
        return "unknown"


def is_allowed(path: Path) -> bool:
    """路徑逃逸防護：只允許存取白名單根目錄底下的檔案。

    瀏覽器端的檔案選擇器看不到伺服器的檔案系統（README §7），
    所以目錄瀏覽必須由後端做 —— 也就必須自己擋 `../`。
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in ALLOWED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


class Handler(BaseHTTPRequestHandler):
    server_version = "EnforcementWebUI/1.0"
    protocol_version = "HTTP/1.1"

    # ── 基礎回應工具 ──────────────────────────────────────────────

    def log_message(self, fmt: str, *args: Any) -> None:  # 靜音預設 access log
        pass

    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json({"ok": False, "error": message}, code)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    # ── 路由 ─────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            return self._serve_static("index.html")
        if route.startswith("/static/"):
            return self._serve_static(route[len("/static/"):])
        if route == "/api/state":
            def _int(name: str, default: int) -> int:
                try:
                    return int(params.get(name, [str(default)])[0])
                except (TypeError, ValueError):
                    return default

            # jobs 一律分頁。上限擋住「limit=999999 又把 1MB 拉回來」。
            limit = max(1, min(_int("limit", JOBS_PAGE_SIZE), JOBS_PAGE_MAX))
            snap = STORE.snapshot(offset=_int("offset", 0), limit=limit)
            snap["logs"] = STORE.logs()
            snap["config"] = {
                "default_source": DEFAULT_SOURCE,
                "output_root": str(STORE.output_root),
                "allowed_roots": [str(p) for p in ALLOWED_ROOTS],
                "mode": "mock" if isinstance(ENGINE, MockEngine) else "real",
            }
            return self._json(snap)
        if route == "/api/stream":
            return self._serve_sse()
        if route == "/api/browse":
            return self._browse(params.get("path", [DEFAULT_SOURCE])[0])
        if route == "/api/export.csv":
            return self._export_csv()
        if route == "/api/media":
            return self._serve_media(params)
        return self._error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        route = urllib.parse.urlparse(self.path).path
        body = self._read_json()

        if route == "/api/scan":
            return self._scan(
                body.get("path") or DEFAULT_SOURCE,
                confirmed=bool(body.get("confirm")),
            )
        if route == "/api/start":
            return self._start(body)
        if route == "/api/clear":
            removed = STORE.clear_finished()
            STORE.log(f"[QUEUE] 已清除 {removed} 筆已完成/失敗紀錄")
            return self._json({"ok": True, "removed": removed})
        if route == "/api/note":
            ok = STORE.set_note(str(body.get("job_id", "")), str(body.get("note", "")))
            return self._json({"ok": ok})
        if route == "/api/retry":
            return self._retry(str(body.get("job_id", "")))
        if route in ("/api/clip/delete", "/api/clip/update", "/api/clip/add"):
            return self._clip_edit(route.rsplit("/", 1)[1], body)
        return self._error(404, "not found")

    # ── 片段人工編輯 ─────────────────────────────────────────────
    # 刪除誤判、調整偵測器抓歪的起訖、補上偵測器漏掉的違規。
    # 三種操作都會重新匯出證物檔並重寫 CSV —— 見 clip_edit.py 開頭。
    def _clip_edit(self, action: str, body: dict) -> None:
        from .clip_edit import ClipEditError, ClipEditor

        editor = ClipEditor(STORE)
        job_id = str(body.get("job_id", ""))
        try:
            if action == "delete":
                result = editor.delete(job_id, int(body.get("event_index", -1)))
            elif action == "update":
                result = editor.update(
                    job_id,
                    int(body.get("event_index", -1)),
                    float(body.get("seg_start_sec", 0)),
                    float(body.get("seg_end_sec", 0)),
                    behavior=(str(body["behavior"]) if body.get("behavior") else None),
                    plate=str(body.get("plate") or ""),
                    reason=str(body.get("reason") or ""),
                )
            else:
                result = editor.add(
                    job_id,
                    float(body.get("seg_start_sec", 0)),
                    float(body.get("seg_end_sec", 0)),
                    behavior=str(body.get("behavior") or "亂丟垃圾"),
                    plate=str(body.get("plate") or ""),
                    reason=str(body.get("reason") or ""),
                )
        except ClipEditError as exc:
            return self._error(400, str(exc))
        except (TypeError, ValueError) as exc:
            return self._error(400, f"參數不正確：{exc}")
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            STORE.log(f"[EDIT] 片段編輯失敗：{exc}", level="error")
            return self._error(500, f"片段編輯失敗：{exc}")
        return self._json({"ok": True, "count": len(result["events"])})

    # ── 靜態檔 ───────────────────────────────────────────────────

    def _serve_static(self, rel: str) -> None:
        target = (STATIC_DIR / rel).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return self._error(403, "forbidden")
        if not target.is_file():
            return self._error(404, "not found")
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/javascript":
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    # ── SSE 即時推播 ─────────────────────────────────────────────

    def _serve_sse(self) -> None:
        """伺服器推送取代輪詢。斷線後瀏覽器的 EventSource 會自動重連，
        重連時前端會重新拉一次 /api/state，所以短斷不會造成狀態不一致。"""
        outbox: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=200)

        def on_event(payload: dict[str, Any]) -> None:
            try:
                outbox.put_nowait(payload)
            except queue.Full:
                pass

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        STORE.subscribe(on_event)
        try:
            # 連上／重連時只推 reset，讓前端自己去抓「它目前那一頁」。
            # 伺服器不知道這個訂閱者停在第幾頁，推整份快照既錯又大。
            outbox.put_nowait({"type": "reset", "stats": STORE.stats()})
            while True:
                try:
                    payload = outbox.get(timeout=15)
                except queue.Empty:
                    # 心跳：避免中間的 proxy 因閒置切斷連線
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                if payload is None:
                    break
                data = json.dumps(payload, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            STORE.unsubscribe(on_event)

    # ── 目錄瀏覽與掃描 ────────────────────────────────────────────

    def _browse(self, raw_path: str) -> None:
        path = Path(raw_path) if raw_path else (ALLOWED_ROOTS[0] if ALLOWED_ROOTS
                                                else Path("/"))
        if not is_allowed(path):
            return self._error(403, f"路徑不在允許範圍內：{path}")
        if not path.is_dir():
            return self._error(404, f"找不到資料夾：{path}")
        dirs: list[str] = []
        videos: list[dict[str, Any]] = []
        try:
            for entry in sorted(path.iterdir(), key=lambda p: p.name):
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    dirs.append(entry.name)
                elif entry.suffix.lower() in VIDEO_EXT:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    videos.append(
                        {
                            "name": entry.name,
                            "size_mb": round(size / 1024 / 1024, 1),
                            "station": infer_station(entry),
                        }
                    )
        except PermissionError:
            return self._error(403, "沒有讀取權限")

        parent = str(path.parent) if is_allowed(path.parent) else ""
        self._json(
            {
                "ok": True,
                "path": str(path),
                "parent": parent,
                "dirs": dirs,
                "videos": videos,
                "roots": [str(p) for p in ALLOWED_ROOTS],
            }
        )

    def _scan(self, raw_path: str, confirmed: bool = False) -> None:
        path = Path(raw_path)
        if not is_allowed(path):
            return self._error(403, f"路徑不在允許範圍內：{path}")
        if not path.is_dir():
            return self._error(404, f"找不到資料夾：{path}")

        existing = STORE.existing_video_paths()
        found: list[Path] = []
        skipped = 0
        for entry in sorted(path.rglob("*")):
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXT:
                if str(entry) in existing:
                    skipped += 1
                else:
                    found.append(entry)

        # 一次加入太多會讓介面卡住：/api/state 是全量快照、SSE 每次狀態變動都推一次，
        # 前端還要重繪整張表。實測掃 /videos 根目錄拉進 1526 部 → 快照 1MB、瀏覽器卡死。
        # 這裡先擋下來要使用者確認；根治要做分頁與增量推播（見 README 的已知限制）。
        if not confirmed and len(found) > SCAN_CONFIRM_LIMIT:
            STORE.log(
                f"[SCANNER] {path} 底下有 {len(found)} 部影片，超過 {SCAN_CONFIRM_LIMIT} 部，"
                "等待確認",
                level="warn",
            )
            return self._json(
                {
                    "ok": False,
                    "needs_confirm": True,
                    "found": len(found),
                    "skipped": skipped,
                    "limit": SCAN_CONFIRM_LIMIT,
                    "path": str(path),
                }
            )

        added = 0
        for entry in found:
            if STORE.add_video(entry, infer_station(entry)) is not None:
                added += 1
        STORE.log(
            f"[SCANNER] 掃描 {path} 完成：新增 {added} 部影片"
            + (f"（{skipped} 部已在佇列中，略過）" if skipped else ""),
            level="success" if added else "warn",
        )
        STORE._broadcast()  # noqa: SLF001 — 掃描是唯一批次新增 job 的入口
        self._json({"ok": True, "added": added, "skipped": skipped})

    # ── 執行控制 ─────────────────────────────────────────────────

    def _start(self, body: dict[str, Any]) -> None:
        count = ENGINE.start(body)
        if count == 0:
            STORE.log("[ENGINE] 佇列中沒有待處理的影片", level="warn")
            return self._json({"ok": False, "error": "佇列中沒有待處理的影片"})
        STORE.log(
            f"[ENGINE] 已送出 {count} 部影片，平行度 "
            f"{body.get('parallel_jobs_count', 1)}",
            level="success",
        )
        self._json({"ok": True, "submitted": count})

    def _retry(self, job_id: str) -> None:
        job = STORE.get(job_id)
        if job is None:
            return self._error(404, "找不到該工作")
        STORE.update(
            job_id,
            status="queued",
            progress=0.0,
            frame=0,
            error="",
            vlm_calls_done=0,
            vlm_latency_sum_ms=0,
            vlm_timeouts=0,
            last_vlm_at=0.0,
            started_at=0.0,
            finished_elapsed=0.0,
        )
        STORE.log(f"[RETRY] {job.video_name} 已重新排入佇列", level="warn")
        ENGINE.start(ENGINE.options or {"parallel_jobs_count": 1})
        self._json({"ok": True})

    # ── 匯出與媒體 ───────────────────────────────────────────────

    def _export_csv(self) -> None:
        """整批總表。欄位沿用 vlm_enforcement._write_csv，多一欄稽查註記。"""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["來源影片", "站別", "幀時間", "畫面時間", "車牌", "違規行為",
             "信心分數", "判定理由", "截圖路徑", "影片片段路徑", "稽查註記"]
        )
        for job in STORE.all():
            for event in job.events:
                writer.writerow(
                    [
                        job.video_name,
                        job.station_id,
                        event.get("frame_time", ""),
                        event.get("overlay_time", ""),
                        event.get("plate", ""),
                        event.get("behavior", ""),
                        f"{float(event.get('confidence') or 0):.3f}",
                        event.get("reason", ""),
                        event.get("snapshot", ""),
                        event.get("clip", ""),
                        job.job_note,
                    ]
                )
        # utf-8-sig：Excel 開啟中文 CSV 不亂碼，與 pipeline 輸出一致
        body = buf.getvalue().encode("utf-8-sig")
        self._send(
            200,
            body,
            "text/csv; charset=utf-8",
            {"Content-Disposition": 'attachment; filename="violations_all.csv"'},
        )

    def _serve_media(self, params: dict[str, list[str]]) -> None:
        """供截圖預覽與片段播放取檔。支援 Range，瀏覽器才能拖曳時間軸。"""
        raw = params.get("path", [""])[0]
        if not raw:
            return self._error(400, "missing path")
        path = Path(raw)
        if not is_allowed(path) or not path.is_file():
            return self._error(404, "not found")

        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        range_header = self.headers.get("Range", "")
        match = re.match(r"bytes=(\d*)-(\d*)", range_header)

        if not match:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with path.open("rb") as handle:
                self._pump(handle, size)
            return

        start = int(match.group(1)) if match.group(1) else 0
        end = int(match.group(2)) if match.group(2) else size - 1
        start = max(0, min(start, size - 1))
        end = max(start, min(end, size - 1))
        length = end - start + 1

        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            self._pump(handle, length)

    def _pump(self, handle: Any, remaining: int) -> None:
        try:
            while remaining > 0:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> int:
    global STORE, ENGINE, ALLOWED_ROOTS, DEFAULT_SOURCE, SCAN_CONFIRM_LIMIT

    parser = argparse.ArgumentParser(description="環保科技執法系統 — 網頁介面")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true",
                      help="假資料模式，不載入模型（預設）")
    mode.add_argument("--real", action="store_true",
                      help="接上真的 VLM pipeline")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--scan-confirm-limit", type=int, default=SCAN_CONFIRM_LIMIT,
                        help="單次掃描超過這個數量就先要求確認（0 = 不限制）")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output-root", default=os.environ.get("WEBUI_OUTPUT_ROOT",
                                                                "/output"))
    parser.add_argument("--videos", nargs="*", default=None,
                        help="允許瀏覽的影片根目錄（可多個）")
    args = parser.parse_args()

    roots = args.videos or ["/videos", "/testset", "/testset2"]
    ALLOWED_ROOTS = [Path(p) for p in roots if Path(p).exists()]
    SCAN_CONFIRM_LIMIT = max(0, int(args.scan_confirm_limit))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root not in ALLOWED_ROOTS:
        ALLOWED_ROOTS.append(output_root)
    DEFAULT_SOURCE = str(ALLOWED_ROOTS[0]) if ALLOWED_ROOTS else ""

    STORE = JobStore(output_root)
    ENGINE = RealEngine(STORE) if args.real else MockEngine(STORE)
    STORE.log(
        f"[SYSTEM] 網頁介面啟動（{'真實推論' if args.real else 'mock 模擬'}模式）"
    )
    STORE.log(f"[SYSTEM] 可瀏覽根目錄：{', '.join(str(p) for p in ALLOWED_ROOTS)}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"[webui] listening on http://{args.host}:{args.port}  "
          f"mode={'real' if args.real else 'mock'}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[webui] 停止", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
