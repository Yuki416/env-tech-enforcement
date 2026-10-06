#!/usr/bin/env bash
# 網頁介面的啟停腳本。在 ccu2 的 bash 直接跑，不用記那串 /proc 掃描指令。
#
#   bash webui/webui.sh start      啟動（會自動偵測 Co-DETR sidecar 有沒有開）
#   bash webui/webui.sh stop       停掉
#   bash webui/webui.sh restart    重開（改了 webui/ 的程式碼之後用這個）
#   bash webui/webui.sh status     看狀態
#   bash webui/webui.sh logs       跟著看 log（Ctrl-C 離開）
#   bash webui/webui.sh mock       用假資料開，不載模型、秒開，純看介面
#
# 連線：服務只綁 127.0.0.1:8080，不對外。用 VSCode PORTS 面板轉發 8080，
#       或在自己電腦跑  ssh -N -L 8080:127.0.0.1:8080 user@<server-ip>
#       然後開 http://localhost:8080
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP=enforce-app
CODETR=enforce-codetr
PORT=8080
LOG=/tmp/webui.log

cd "$REPO" || exit 1

# 容器裡沒有 pkill / ps（沒裝 procps），只能掃 /proc 的 cmdline。
# 注意這段字串會被拆開組合，否則會比對到自己這行指令而自殺。
kill_webui() {
    docker exec "$APP" python -c "
import os, signal
pat = 'webui' + '.server'
me = os.getpid()
for p in os.listdir('/proc'):
    if not p.isdigit() or int(p) == me:
        continue
    try:
        cmd = open(f'/proc/{p}/cmdline', 'rb').read().decode('utf-8', 'replace')
    except OSError:
        continue
    if pat in cmd:
        os.kill(int(p), signal.SIGKILL)
" 2>/dev/null
}

webui_pid() {
    docker exec "$APP" python -c "
import os
pat = 'webui' + '.server'
me = os.getpid()
for p in os.listdir('/proc'):
    if not p.isdigit() or int(p) == me:
        continue
    try:
        cmd = open(f'/proc/{p}/cmdline', 'rb').read().decode('utf-8', 'replace')
    except OSError:
        continue
    if pat in cmd:
        print(p)
        break
" 2>/dev/null
}

require_container() {
    if ! docker ps --format '{{.Names}}' | grep -qx "$APP"; then
        echo "✗ 容器 $APP 沒在跑。先起容器："
        echo "    cd $REPO && docker compose up -d ollama enforcement codetr"
        exit 1
    fi
}

wait_ready() {
    echo -n "等待服務就緒"
    for _ in $(seq 1 60); do
        if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" 2>/dev/null | grep -q 200; then
            echo " ✓"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    echo " ✗ 逾時。看 log：bash webui/webui.sh logs"
    return 1
}

start() {
    local mode="${1:---real}"
    require_container
    if [ -n "$(webui_pid)" ]; then
        echo "服務已經在跑了（pid $(webui_pid)）。要套用程式碼改動請用 restart。"
        exit 0
    fi

    # Co-DETR sidecar 有開才啟用；沒開就走原本的 YOLO，不會因此起不來。
    local envflag=()
    if docker ps --format '{{.Names}}' | grep -qx "$CODETR"; then
        envflag=(-e "ENFORCE_CODETR_URL=http://codetr:8091")
        echo "偵測到 $CODETR → 啟用第 18 次訓練的 Co-DETR 垃圾偵測"
    else
        echo "沒有偵測到 $CODETR → 垃圾偵測走 YOLO"
        echo "  （要用 Co-DETR：docker compose up -d codetr，再 restart）"
    fi

    docker exec -d "${envflag[@]}" -e PYTHONPATH=/app "$APP" bash -c \
        "cd /app && python -m webui.server $mode --host 0.0.0.0 --port $PORT \
         --videos /videos /testset /testset2 --output-root /output > $LOG 2>&1"

    wait_ready || exit 1
    status
}

stop() {
    require_container
    local pid
    pid="$(webui_pid)"
    if [ -z "$pid" ]; then
        echo "服務本來就沒在跑"
        return 0
    fi
    kill_webui
    sleep 2
    if [ -z "$(webui_pid)" ]; then
        echo "✓ 已停止（原 pid $pid）"
    else
        echo "✗ 停不掉，pid $(webui_pid)"
        exit 1
    fi
}

status() {
    require_container
    local pid
    pid="$(webui_pid)"
    echo "──────────────────────────────────────────"
    if [ -z "$pid" ]; then
        echo "  網頁服務  : 沒在跑"
    else
        echo "  網頁服務  : 執行中（pid $pid）"
        curl -s "http://127.0.0.1:$PORT/api/state?limit=1" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('  狀態      : 讀不到 /api/state'); raise SystemExit
print(f\"  模式      : {d['config']['mode']}\")
print(f\"  佇列      : {d['jobs_total']} 部影片\")
s = d['stats']
print(f\"  執行中    : {s['running_jobs']} ｜ 排隊 {s['queued_jobs']} ｜ 完成 {s['completed_jobs']}\")
" 2>/dev/null
    fi
    if docker ps --format '{{.Names}}' | grep -qx "$CODETR"; then
        echo "  Co-DETR   : 執行中（$CODETR）"
    else
        echo "  Co-DETR   : 沒在跑"
    fi
    echo "──────────────────────────────────────────"
    echo "  網址      : http://localhost:$PORT"
    echo "              （先在 VSCode PORTS 面板轉發 $PORT，或開 SSH 通道）"
    echo "  log       : bash webui/webui.sh logs"
}

case "${1:-status}" in
    start)   start --real ;;
    mock)    start --mock ;;
    stop)    stop ;;
    restart) stop; sleep 1; start --real ;;
    status)  status ;;
    logs)    require_container; docker exec "$APP" tail -f "$LOG" ;;
    *)
        echo "用法：bash webui/webui.sh {start|stop|restart|status|logs|mock}"
        exit 1
        ;;
esac
