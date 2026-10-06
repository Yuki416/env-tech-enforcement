/* 環保科技執法系統 — 前端資料層
 *
 * 設計稿（Gemini 產出）負責外觀，這支檔案負責讓它動起來：
 *   1. SSE 接收伺服器推送的狀態，重繪工作佇列與事件表
 *   2. 本地每秒 tick，讓「耗時」與「上次呼叫」在兩次推送之間仍持續跳動
 *   3. 伺服器端資料夾瀏覽（瀏覽器的檔案選擇器看不到伺服器檔案系統）
 *   4. 違規片段播放：有真檔就播真影片，沒有就退回設計稿的 SVG 示意圖
 *
 * 刻意不碰設計稿的 CSS 與版面 —— 那是使用者已經確認過的外觀。
 */
'use strict';

// ── 全域狀態 ──────────────────────────────────────────────────────
const App = {
    jobs: [],          // 只有「目前這一頁」的 job，不是全部
    jobsTotal: 0,
    jobsOffset: 0,
    jobsLimit: 100,
    events: [],
    stats: {},
    config: {},
    connected: false,
    browserTarget: 'source',
    browserPath: '',
    player: { jobId: '', eventIndex: -1, hasRealVideo: false, fps: 0 },
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function setField(root, name, value) {
    const el = root.querySelector(`[data-field="${name}"]`);
    if (el) el.textContent = value;
}

function fmtElapsed(sec) {
    sec = Math.max(0, Math.floor(sec));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${h}小時${m}分${sec % 60}秒`;
}

function fmtClock(sec) {
    sec = Math.max(0, Math.floor(sec));
    const h = String(Math.floor(sec / 3600)).padStart(2, '0');
    const m = String(Math.floor((sec % 3600) / 60)).padStart(2, '0');
    return `${h}:${m}:${String(sec % 60).padStart(2, '0')}`;
}

function hmsToSec(str) {
    const parts = String(str || '').split(':').map(Number);
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    return 0;
}

async function api(path, options) {
    const res = await fetch(path, options);
    const text = await res.text();
    try { return JSON.parse(text); } catch (e) { return { ok: false, error: text }; }
}

function post(path, body) {
    return api(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}),
    });
}

// ── 工作佇列渲染 ──────────────────────────────────────────────────

/* 這張表原本對不上系統實際會產出的字串：
 *   - '亂丟菸蒂'   系統從來不會產出（prompt 只允許「亂丟垃圾」與「便溺」）
 *   - '隨地便溺'   實際輸出是「便溺」，所以永遠對不到 → 掉進預設的灰色
 * 對不到就會用 status-queued（灰），看起來像「排隊中」而不是違規。
 *
 * 顏色語意：紅＝行為（有人做了什麼）、橘＝狀態（現場的既成事實）。
 */
const BADGE_BY_BEHAVIOR = {
    '亂丟垃圾': 'status-error',     // 紅：行為
    '便溺': 'status-merging',       // 琥珀：行為（沿用設計稿給便溺的色）
    '隨地便溺': 'status-merging',   // 別名，避免日後改字串又對不上
    '垃圾堆置': 'status-prefilter', // 藍：狀態，不是當下的行為，刻意跟紅色區隔
};

/* 第 6 欄「閒置/狀態」的內容隨狀態改變。整格重建會沖掉本地 tick，
 * 所以只在「形狀」變了才重建，數值更新走 data 屬性。 */
function idleCellHtml(job) {
    if (job.status === 'completed')
        return '<span style="color: var(--success); font-weight: 600;">已產出舉證檔</span>';
    if (job.status === 'error')
        return '<span style="color: var(--danger); font-weight: 600;">系統異常</span>';
    if (job.status === 'queued')
        return '<span class="vlm-tag">等待排程</span>';
    if (job.status === 'preparing')
        return '<span class="vlm-tag">載入影片中</span>';
    if (job.status === 'prefilter')
        return '<span class="vlm-tag">YOLO 抽幀過濾中</span>';
    if (job.idle_alert)
        return `<strong style="color: var(--danger);" data-field="seconds_since_last_vlm" data-raw-idle="${job.seconds_since_last_vlm}">${job.seconds_since_last_vlm} 秒無回應</strong>`;
    if (job.seconds_since_last_vlm < 0)
        return '<span class="vlm-tag">尚未呼叫 AI</span>';
    // 首次呼叫尚未回來：模型冷啟動可能要 60 秒以上，要明確說明而不是顯示「上次呼叫」
    if (job.awaiting_first_vlm)
        return `<span class="vlm-tag">等待首次判定 <span class="vlm-val" data-field="seconds_since_last_vlm" data-raw-idle="${job.seconds_since_last_vlm}">${job.seconds_since_last_vlm}</span> 秒<br><span style="font-size:10.5px;color:var(--text-dim);">（模型載入中）</span></span>`;
    return `<span class="vlm-tag">上次呼叫: <span class="vlm-val" data-field="seconds_since_last_vlm" data-raw-idle="${job.seconds_since_last_vlm}">${job.seconds_since_last_vlm}</span> 秒前</span>`;
}

function violationsCellHtml(job) {
    if (job.violations > 0)
        return `<strong style="color: var(--danger); font-size: 15px;">${job.violations}</strong> 件`;
    return '<span style="color: var(--text-muted);">0</span> 件';
}

function actionCellHtml(job) {
    if (job.status === 'error')
        return `<button class="btn btn-outline btn-sm" onclick="showErrorLog('${job.job_id}')">查看錯誤</button>
                <button class="btn btn-secondary btn-sm" onclick="restartJob('${job.job_id}')">重試</button>`;
    if (job.idle_alert)
        return `<button class="btn btn-primary btn-sm" style="background:var(--danger)" onclick="restartJob('${job.job_id}')">重試/重啟</button>`;
    if (job.violations > 0)
        return `<button class="btn btn-primary btn-sm" onclick="openClip('${job.job_id}', 0)">播放違規片段</button>`;
    if (job.status === 'completed')
        return '<span style="font-size:12px;color:var(--text-dim);">無違規事件</span>';
    return `<button class="btn btn-secondary btn-sm" onclick="viewJobDetails('${job.job_id}')">詳情</button>`;
}

function renderJobs() {
    const tbody = $('#jobsTbody');
    if (!tbody) return;

    if (!App.jobs.length) {
        tbody.innerHTML = `<tr><td colspan="9"><div class="empty-state">
            佇列是空的。先在左側選擇伺服器上的影片資料夾，按「掃描」載入影片。
        </div></td></tr>`;
        return;
    }
    // 清掉空狀態列
    if (tbody.querySelector('.empty-state')) tbody.innerHTML = '';

    const seen = new Set();
    App.jobs.forEach(job => {
        seen.add(job.job_id);
        let tr = tbody.querySelector(`tr[data-job-id="${job.job_id}"]`);
        if (!tr) {
            tr = $('#tplJobRow').content.firstElementChild.cloneNode(true);
            tr.dataset.jobId = job.job_id;
            tbody.appendChild(tr);
        }
        updateJobRow(tr, job);
    });
    // 移除已被清單刪掉的列
    $$('tr[data-job-id]', tbody).forEach(tr => {
        if (!seen.has(tr.dataset.jobId)) tr.remove();
    });
}

function updateJobRow(tr, job) {
    tr.classList.toggle('row-idle-alert', !!job.idle_alert);

    setField(tr, 'video_name', job.video_name);
    setField(tr, 'output_dir', job.output_dir || `${job.station_id}/（尚未建立）`);
    setField(tr, 'stage_display', job.stage_display);

    const badge = tr.querySelector('[data-field="status"]');
    if (badge) badge.className = `status-badge ${job.badge_class}`;

    const pill = tr.querySelector('[data-field="idle_alert"]');
    if (pill) pill.style.display = job.idle_alert ? '' : 'none';

    const pct = (job.progress * 100).toFixed(1);
    const bar = tr.querySelector('[data-field="progress_bar"]');
    if (bar) {
        bar.style.width = `${pct}%`;
        bar.className = 'progress-bar-fill'
            + (job.status === 'completed' ? ' success'
                : (job.status === 'error' || job.idle_alert) ? ' danger' : '');
    }
    setField(tr, 'progress', `${pct}%`);

    const info = tr.querySelector('[data-field="frame_info"]');
    if (info) {
        if (job.status === 'error') {
            info.textContent = job.error || '處理失敗';
            info.style.color = 'var(--danger)';
        } else {
            info.style.color = '';
            info.textContent = job.status === 'completed'
                ? '已完成彙整'
                : (job.total_frames ? `Frame ${job.frame}/${job.total_frames}` : '準備中');
        }
    }

    // 耗時：伺服器值為權威，本地 tick 補足兩次推送之間的間隔
    const el = tr.querySelector('[data-field="elapsed_sec"]');
    if (el) {
        el.setAttribute('data-raw-seconds', String(Math.floor(job.elapsed_sec)));
        el.setAttribute('data-live', job.status === 'completed' || job.status === 'error' ? '0' : '1');
        el.textContent = fmtElapsed(job.elapsed_sec);
    }

    setField(tr, 'vlm_calls_done', job.vlm_calls_done);
    setField(tr, 'vlm_avg_latency', job.vlm_avg_latency.toFixed(2));
    setField(tr, 'vlm_timeouts', job.vlm_timeouts);

    // 延遲與逾時異常時轉紅，對應 §5「平均延遲是唯一能及早發現 CPU 退化的訊號」
    const lat = tr.querySelector('[data-field="vlm_avg_latency"]');
    if (lat) lat.style.color = job.vlm_avg_latency > 8 ? 'var(--danger)' : '';
    const to = tr.querySelector('[data-field="vlm_timeouts"]');
    if (to) to.style.color = job.vlm_timeouts > 0 ? 'var(--danger)' : '';

    // 只在形狀改變時重建這三格，避免沖掉輸入焦點與本地 tick
    const shape = `${job.status}|${job.idle_alert}|${job.violations > 0}|${job.awaiting_first_vlm}`;
    if (tr.dataset.shape !== shape) {
        tr.dataset.shape = shape;
        tr.querySelector('[data-cell="idle"]').innerHTML = idleCellHtml(job);
        tr.querySelector('[data-cell="violations"]').innerHTML = violationsCellHtml(job);
        tr.querySelector('[data-cell="action"]').innerHTML = actionCellHtml(job);
    } else {
        const idleEl = tr.querySelector('[data-field="seconds_since_last_vlm"]');
        if (idleEl && job.seconds_since_last_vlm >= 0) {
            idleEl.setAttribute('data-raw-idle', String(job.seconds_since_last_vlm));
            idleEl.textContent = job.idle_alert
                ? `${job.seconds_since_last_vlm} 秒無回應`
                : job.seconds_since_last_vlm;
        }
        const vc = tr.querySelector('[data-cell="violations"]');
        if (vc) vc.innerHTML = violationsCellHtml(job);
    }

    // 註記輸入框：使用者正在打字時不覆寫
    const note = tr.querySelector('[data-field="job_note"]');
    if (note && document.activeElement !== note) {
        note.value = job.job_note || '';
        if (!note.dataset.bound) {
            note.dataset.bound = '1';
            note.addEventListener('change', () => saveTableJobNote(job.job_id, note.value));
        }
    }
}

// ── 違規事件表 ────────────────────────────────────────────────────

function renderEvents() {
    const tbody = $('#eventsTbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    if (!App.events.length) {
        tbody.innerHTML = `<tr><td colspan="9"><div class="empty-state">
            目前沒有偵測到違規事件。工作跑完後，AI 判定的違規會列在這裡。
        </div></td></tr>`;
        return;
    }
    App.events.forEach(ev => {
        const tr = $('#tplEventRow').content.firstElementChild.cloneNode(true);
        setField(tr, 'source_video', ev.source_video);
        setField(tr, 'screen_time', ev.overlay_time || '—');
        setField(tr, 'frame_time', ev.frame_time || '—');

        const plate = tr.querySelector('[data-field="plate_number"]');
        if (ev.plate) {
            plate.textContent = ev.plate;
            plate.style.color = '#b45309';
        } else {
            plate.textContent = '未讀取到車牌';
            plate.style.color = '#64748b';
        }

        const type = tr.querySelector('[data-field="violation_type"]');
        type.textContent = ev.behavior || '—';
        type.className = `status-badge ${BADGE_BY_BEHAVIOR[ev.behavior] || 'status-queued'}`;

        setField(tr, 'confidence', Number(ev.confidence || 0).toFixed(2));
        setField(tr, 'reason', ev.reason || '—');
        setField(tr, 'job_note', ev.job_note || '—');

        tr.querySelector('[data-cell="action"]').innerHTML =
            `<button class="btn btn-secondary btn-sm" onclick="previewEvent('${ev.job_id}', ${ev.event_index})">證據照片</button>
             <button class="btn btn-primary btn-sm" onclick="openClip('${ev.job_id}', ${ev.event_index})">播放片段</button>`;
        tbody.appendChild(tr);
    });
}

// ── 統計數字與頁籤徽章 ────────────────────────────────────────────

function renderStats() {
    const s = App.stats;
    const map = {
        total_jobs: s.total_jobs, running_jobs: s.running_jobs,
        queued_jobs: s.queued_jobs, completed_jobs: s.completed_jobs,
        idle_alert_count: s.idle_alert_count, total_violations: s.total_violations,
        jobs_count: s.total_jobs, events_count: App.events.length,
        clips_count: App.events.length,
    };
    Object.entries(map).forEach(([name, value]) => {
        $$(`[data-field="${name}"]`).forEach(el => {
            if (el.closest('#jobsTbody') || el.closest('#eventsTbody')) return;
            el.textContent = value == null ? '0' : value;
        });
    });
}

// ── 執行紀錄 ──────────────────────────────────────────────────────

const LOG_CLASS = { info: 'info', warn: 'warn', error: 'error', success: 'success' };

function appendLogEntry(entry) {
    const box = $('#system-log-container');
    if (!box) return;
    const div = document.createElement('div');
    div.className = `log-entry ${LOG_CLASS[entry.level] || 'info'}`;
    div.innerHTML = `<span class="time">[${entry.time}]</span> `;
    div.appendChild(document.createTextNode(entry.message));
    box.appendChild(div);
    while (box.children.length > 400) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
}

// 設計稿的 appendLog 是本地假訊息；改成只寫本地畫面，伺服器訊息走 SSE
window.appendLog = function (msg, type = 'info') {
    appendLogEntry({ time: new Date().toTimeString().slice(0, 8), level: type, message: msg });
};

// ── SSE 連線 ──────────────────────────────────────────────────────

function applyState(payload) {
    App.jobs = payload.jobs || [];
    if (typeof payload.jobs_total === 'number') App.jobsTotal = payload.jobs_total;
    if (typeof payload.jobs_offset === 'number') App.jobsOffset = payload.jobs_offset;
    if (payload.jobs_limit) App.jobsLimit = payload.jobs_limit;
    App.events = payload.events || [];
    App.stats = payload.stats || {};
    if (payload.config) App.config = payload.config;
    // 清除／重試之後總數可能縮水，讓 offset 落在空白頁 → 退回最後一頁
    if (!App.jobs.length && App.jobsTotal > 0 && App.jobsOffset > 0) {
        App.jobsOffset = Math.max(0,
            (Math.ceil(App.jobsTotal / App.jobsLimit) - 1) * App.jobsLimit);
        schedulePageRefresh();
        return;
    }
    renderJobs();
    renderPager();
    renderEvents();
    renderStats();
    renderClips();
}

// ── 分頁 ──────────────────────────────────────────────────────────
// 佇列可能上千部。整包送到瀏覽器會讓 JSON.parse 與 DOM 重繪雙雙卡死
// （實測 1526 部 → /api/state 1MB），所以一次只抓一頁、也只渲染一頁。

async function fetchPage(offset = App.jobsOffset) {
    const max = Math.max(0, App.jobsTotal - 1);
    const off = Math.max(0, Math.min(offset, max));
    const state = await api(`/api/state?offset=${off}&limit=${App.jobsLimit}`);
    applyState(state);
}

// SSE 的 reset 可能連續來好幾個（例如批次掃描），合併成一次重抓。
let pageRefreshTimer = null;
function schedulePageRefresh() {
    if (pageRefreshTimer) return;
    pageRefreshTimer = setTimeout(() => {
        pageRefreshTimer = null;
        fetchPage().catch(() => {});
    }, 250);
}

window.gotoJobsPage = function (offset) {
    App.jobsOffset = offset;
    fetchPage(offset).catch(() => {});
};

function renderPager() {
    const tbody = $('#jobsTbody');
    if (!tbody) return;
    const table = tbody.closest('table');
    if (!table) return;

    let pager = $('#jobsPager');
    if (!pager) {
        pager = document.createElement('div');
        pager.id = 'jobsPager';
        pager.className = 'jobs-pager';
        table.parentNode.insertBefore(pager, table.nextSibling);
    }
    // 一頁裝得下就不顯示分頁列，介面維持設計稿原樣
    if (App.jobsTotal <= App.jobsLimit) {
        pager.style.display = 'none';
        return;
    }
    pager.style.display = '';
    const from = App.jobsOffset + 1;
    const to = Math.min(App.jobsOffset + App.jobsLimit, App.jobsTotal);
    const prev = Math.max(0, App.jobsOffset - App.jobsLimit);
    const next = App.jobsOffset + App.jobsLimit;
    const atStart = App.jobsOffset <= 0;
    const atEnd = next >= App.jobsTotal;
    pager.innerHTML =
        `<button class="btn btn-secondary btn-sm" ${atStart ? 'disabled' : ''} ` +
        `onclick="gotoJobsPage(${prev})">← 上一頁</button>` +
        `<span class="jobs-pager-info">第 ${from}–${to} 筆，共 ${App.jobsTotal} 筆</span>` +
        `<button class="btn btn-secondary btn-sm" ${atEnd ? 'disabled' : ''} ` +
        `onclick="gotoJobsPage(${next})">下一頁 →</button>`;
}

function setConnected(ok) {
    App.connected = ok;
    let pill = $('#connPill');
    if (!pill) {
        const host = $('.tabs-header');
        if (!host) return;
        pill = document.createElement('span');
        pill.id = 'connPill';
        pill.style.cssText = 'margin-left:auto;align-self:center;margin-right:8px;';
        host.appendChild(pill);
    }
    pill.className = `conn-pill ${ok ? 'conn-ok' : 'conn-bad'}`;
    pill.textContent = ok
        ? `● 已連線（${App.config.mode === 'real' ? '真實推論' : 'mock 模擬'}）`
        : '● 連線中斷，重試中…';
}

function connect() {
    const es = new EventSource('/api/stream');
    es.onopen = () => setConnected(true);
    es.onmessage = (msg) => {
        let payload;
        try { payload = JSON.parse(msg.data); } catch (e) { return; }
        if (payload.type === 'log') { appendLogEntry(payload.entry); return; }
        // 單筆 job 變動（進度、階段、遙測）—— 這是最頻繁的路徑，只動那一列。
        // 不在目前這一頁的 job 就只更新統計，不必重抓。
        if (payload.type === 'job') {
            App.stats = payload.stats || App.stats;
            const idx = App.jobs.findIndex(j => j.job_id === payload.job.job_id);
            if (idx >= 0) {
                App.jobs[idx] = payload.job;
                const tr = $(`#jobsTbody tr[data-job-id="${payload.job.job_id}"]`);
                if (tr) updateJobRow(tr, payload.job);
                else renderJobs();
            }
            renderStats();
            return;
        }
        // 整批變動（掃描新增、清除、事件產生）或剛連上 → 重抓自己那一頁
        if (payload.type === 'reset') {
            App.stats = payload.stats || App.stats;
            renderStats();
            schedulePageRefresh();
            return;
        }
        if (payload.type === 'state') applyState(payload);   // 舊格式，保留相容
    };
    es.onerror = () => {
        setConnected(false);
        // EventSource 會自行重連；重連後伺服器會立刻推一份完整快照，
        // 所以短暫斷線不會造成狀態不一致（README §7 的要求）。
    };
}

// ── 每秒 tick：耗時與閒置秒數在兩次推送之間持續跳動 ──────────────

setInterval(() => {
    $$('#jobsTbody [data-field="elapsed_sec"]').forEach(el => {
        if (el.getAttribute('data-live') !== '1') return;
        const sec = parseInt(el.getAttribute('data-raw-seconds') || '0', 10) + 1;
        el.setAttribute('data-raw-seconds', String(sec));
        el.textContent = fmtElapsed(sec);
    });
    $$('#jobsTbody [data-field="seconds_since_last_vlm"]').forEach(el => {
        const raw = el.getAttribute('data-raw-idle');
        if (raw === null) return;
        const sec = parseInt(raw, 10) + 1;
        el.setAttribute('data-raw-idle', String(sec));
        el.textContent = el.textContent.includes('無回應') ? `${sec} 秒無回應` : sec;
    });
}, 1000);

// ── 左側控制列 ────────────────────────────────────────────────────

// ── 處理模式（完整 / 測試）──────────────────────────────────────
function currentMode() {
    const picked = document.querySelector('input[name="pipeline_mode"]:checked');
    return picked ? picked.value : 'full';
}

// 測試模式用不到的參數淡化，並提示這個模式實際會做什麼。
// 不用 disabled 是刻意的：欄位仍要看得到，只是這個模式不看它。
function syncModeUI() {
    const test = currentMode() === 'test';
    document.querySelectorAll('[data-opt-full]').forEach(el => {
        el.classList.toggle('opt-dimmed', test);
    });
    const note = document.getElementById('modeNote');
    if (!note) return;
    note.innerHTML = test
        ? '測試模式：<b>不呼叫 VLM</b>，違規完全由 Co-DETR 的 litter/garbage 偵測決定。'
          + '「抽幀間隔」在這個模式下直接決定掃描密度（1 = 每幀都掃，約 0.42 秒/幀）。'
          + '便溺與車牌欄位不會有內容。'
        : '';
}

function collectOptions() {
    return {
        mode: currentMode(),
        enable_litter: $('[data-field="enable_litter"]').checked,
        enable_urination: $('[data-field="enable_urination"]').checked,
        require_plate_for_csv: $('[data-field="require_plate_for_csv"]').checked,
        heavy_frame_stride: parseInt($('[data-field="heavy_frame_stride"]').value || '1', 10),
        max_vlm_calls_per_video: parseInt($('[data-field="max_vlm_calls_per_video"]').value || '2400', 10),
        parallel_jobs_count: parseInt($('[data-field="parallel_jobs_count"]').value || '1', 10),
        output_root: $('[data-field="output_root_path"]').value,
    };
}

window.scanFolder = async function () {
    const path = $('[data-field="source_folder_path"]').value;
    appendLog(`[SCANNER] 掃描 ${path} …`);
    let res = await post('/api/scan', { path });

    // 一次拉進太多影片會讓介面卡住（/api/state 是全量快照、SSE 每次變動都推一次，
    // 前端還要重繪整張表）。後端超過門檻會擋下來，這裡問過使用者再送一次。
    if (res.needs_confirm) {
        appendLog(`[SCANNER] ${path} 底下有 ${res.found} 部影片，等待確認`, 'warn');
        const ok = confirm(
            `這個資料夾底下有 ${res.found} 部影片（上限 ${res.limit} 部）。\n\n` +
            `全部加入佇列會讓介面明顯變慢甚至卡住。\n` +
            `建議改選底下比較小的子資料夾。\n\n` +
            `仍要全部加入嗎？`
        );
        if (!ok) {
            appendLog('[SCANNER] 已取消，佇列未變動', 'warn');
            return;
        }
        res = await post('/api/scan', { path, confirm: true });
    }
    if (!res.ok) appendLog(`[SCANNER] 掃描失敗：${res.error}`, 'error');
};

window.startBatchProcessing = async function () {
    const res = await post('/api/start', collectOptions());
    if (!res.ok) appendLog(`[ENGINE] ${res.error}`, 'warn');
};

window.clearQueue = async function () {
    await post('/api/clear', {});
};

window.exportAllCSV = function () {
    if (!App.events.length) {
        appendLog('[CSV] 目前沒有違規事件可匯出', 'warn');
        return;
    }
    window.location.href = '/api/export.csv';
    appendLog(`[CSV] 已匯出 ${App.events.length} 筆違規事件（含稽查註記）`, 'success');
};

window.saveTableJobNote = async function (jobId, value) {
    const res = await post('/api/note', { job_id: jobId, note: value });
    if (res.ok) appendLog(`[NOTE] 已儲存 ${jobId} 的稽查註記`, 'success');
};

window.viewJobDetails = function (jobId) {
    switchTab('events');
};

window.showErrorLog = function (jobId) {
    const job = App.jobs.find(j => j.job_id === jobId);
    if (job) appendLog(`[ERROR] ${job.video_name}: ${job.error || '未知錯誤'}`, 'error');
    switchTab('logs');
};

window.restartJob = async function (jobId) {
    await post('/api/retry', { job_id: jobId });
};

// ── 伺服器端資料夾瀏覽器 ──────────────────────────────────────────

window.openBrowser = function (target) {
    App.browserTarget = target;
    $('#browserTitle').textContent = target === 'source'
        ? '選擇伺服器上的影片資料夾' : '選擇舉證檔案輸出根目錄';
    $('#browserHint').textContent = target === 'source'
        ? '選定後按「掃描」載入資料夾內的影片' : '所有舉證資料會依站別分層存放於此';
    const current = target === 'source'
        ? $('[data-field="source_folder_path"]').value
        : $('[data-field="output_root_path"]').value;
    $('#browserModal').classList.add('active');
    loadBrowser(current || (App.config.allowed_roots || [])[0] || '/');
};

window.closeBrowser = function () {
    $('#browserModal').classList.remove('active');
};

async function loadBrowser(path) {
    const res = await api(`/api/browse?path=${encodeURIComponent(path)}`);
    const list = $('#browserList');
    const roots = $('#browserRoots');

    roots.innerHTML = '<span style="font-size:11.5px;color:var(--text-muted);font-weight:600;margin-right:6px;">快速跳轉:</span>'
        + (App.config.allowed_roots || []).map(r =>
            `<span class="browser-root-chip" onclick="loadBrowser('${r}')">${r}</span>`).join('');

    if (!res.ok) {
        $('#browserPath').textContent = path;
        list.innerHTML = `<div class="empty-state">${res.error}</div>`;
        return;
    }
    App.browserPath = res.path;
    $('#browserPath').textContent = res.path;

    const rows = [];
    if (res.parent) {
        rows.push(`<div class="browser-item" onclick="loadBrowser('${res.parent}')">
            <span>⬆️</span><span>.. 上一層</span></div>`);
    }
    res.dirs.forEach(d => {
        const full = `${res.path.replace(/\/$/, '')}/${d}`;
        rows.push(`<div class="browser-item" onclick="loadBrowser('${full}')">
            <span>📁</span><span>${d}</span></div>`);
    });
    res.videos.forEach(v => {
        rows.push(`<div class="browser-item" style="cursor:default;">
            <span>🎬</span><span>${v.name}</span>
            <span class="bi-meta">${v.station} · ${v.size_mb} MB</span></div>`);
    });
    list.innerHTML = rows.length ? rows.join('')
        : '<div class="empty-state">這個資料夾是空的</div>';
    $('#browserHint').textContent =
        `此資料夾有 ${res.videos.length} 部影片、${res.dirs.length} 個子資料夾`;
}
window.loadBrowser = loadBrowser;

window.chooseCurrentFolder = function () {
    const field = App.browserTarget === 'source' ? 'source_folder_path' : 'output_root_path';
    $(`[data-field="${field}"]`).value = App.browserPath;
    closeBrowser();
    if (App.browserTarget === 'source') scanFolder();
};

// ── 違規片段播放器 ────────────────────────────────────────────────

function jobEvents(jobId) {
    return App.events.filter(e => e.job_id === jobId);
}

function renderClips() {
    const grid = $('#clipsGrid');
    if (!grid) return;
    if (!App.events.length) {
        grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1;">
            尚未擷取到任何違規片段。</div>`;
        return;
    }
    grid.innerHTML = App.events.map((ev, i) => {
        const active = (ev.job_id === App.player.jobId && ev.event_index === App.player.eventIndex)
            ? ' active-clip' : '';
        // 違規區間的起訖。舊事件（完整模式）沒有這兩個欄位 → 整列不顯示，
        // 不要拿代表幀時間硬湊，那是不同的東西。
        const hasSeg = typeof ev.seg_start_sec === 'number' && typeof ev.seg_end_sec === 'number';
        const segRow = hasSeg ? `
            <div class="clip-seg-row">
                <span class="clip-seg-label">違規區間</span>
                <button class="clip-seg-btn" title="定位到起始（不自動播放）"
                        onclick="seekClipMark(event, '${ev.job_id}', ${ev.event_index}, 'start')">
                    ▏起 ${ev.seg_start_time}
                </button>
                <span class="clip-seg-dash">–</span>
                <button class="clip-seg-btn" title="定位到結束（不自動播放）"
                        onclick="seekClipMark(event, '${ev.job_id}', ${ev.event_index}, 'end')">
                    終 ${ev.seg_end_time} ▕
                </button>
                <span class="clip-seg-dur">${(ev.seg_end_sec - ev.seg_start_sec).toFixed(1)} 秒</span>
            </div>
            <div class="clip-edit-row">
                ${ev.edited ? '<span class="clip-edited-tag" title="這筆紀錄經稽查人員調整過">人工調整</span>' : ''}
                <button class="clip-act-btn" onclick="openClipEditor(event, '${ev.job_id}', ${ev.event_index})">編輯</button>
                <button class="clip-act-btn danger" onclick="deleteClip(event, '${ev.job_id}', ${ev.event_index})">刪除</button>
            </div>` : '';
        return `<div class="clip-card${active}" onclick="openClip('${ev.job_id}', ${ev.event_index})">
            <div class="clip-card-header">
                <span class="clip-card-title">片段 #${String(i + 1).padStart(2, '0')}: ${ev.behavior || '違規'}</span>
                <span class="status-badge ${BADGE_BY_BEHAVIOR[ev.behavior] || 'status-queued'}"
                      style="font-size:10px; padding:2px 6px;">${ev.frame_time || '—'}</span>
            </div>
            <div class="clip-card-body">
                車牌: <strong>${ev.plate || '未讀取到車牌'}</strong> | 信心: ${Number(ev.confidence || 0).toFixed(2)}<br>
                來源: ${ev.source_video}
            </div>
            ${segRow}
        </div>`;
    }).join('') + addClipCardHtml();
}

/* 「新增片段」卡：只有在目前有選中的工作時才給，因為新片段一定要掛在某部影片上。 */
function addClipCardHtml() {
    if (!App.player.jobId) return '';
    const job = App.jobs.find(j => j.job_id === App.player.jobId);
    if (!job) return '';
    return `<div class="clip-card clip-card-add" onclick="openClipEditor(event, '${job.job_id}', -1)">
        <div class="clip-add-plus">＋</div>
        <div class="clip-add-text">新增違規片段<br><span>${job.video_name}</span></div>
    </div>`;
}

/* 點擊卡片上的「起 / 終」：定位到該時間點但**不播放**，讓稽查人員自己決定何時開始。
 *
 * 兩個容易出錯的地方：
 *   1. 必須 stopPropagation，否則會一併觸發卡片本身的 openClip（那個會自動播放）
 *   2. 事件時間是「原始影片」座標，片段是從事件前幾秒剪起的 → 要扣掉 clip_start_sec
 */
/* ── 片段人工編輯 ──────────────────────────────────────────────────
 * 稽查人員要能修正機器的輸出：刪掉誤判、調整抓歪的起訖、補上漏掉的違規。
 * 三種操作後端都會重新匯出 mp4 與截圖並重寫 CSV（見 clip_edit.py），
 * 所以這裡只負責收集參數與回報結果。
 */
const BEHAVIOR_CHOICES = ['亂丟垃圾', '垃圾堆置', '便溺', '其他'];

// 接受 "MM:SS"、"HH:MM:SS" 或純秒數（可含小數）。空字串回 null。
function parseTimeInput(str) {
    const t = String(str || '').trim();
    if (!t) return null;
    if (t.includes(':')) {
        const p = t.split(':').map(Number);
        if (p.some(isNaN)) return null;
        return p.length === 3 ? p[0] * 3600 + p[1] * 60 + p[2] : p[0] * 60 + p[1];
    }
    const n = Number(t);
    return isNaN(n) ? null : n;
}

window.deleteClip = async function (e, jobId, eventIndex) {
    if (e) e.stopPropagation();
    const ev = jobEvents(jobId).find(x => x.event_index === eventIndex);
    const what = ev ? `${ev.behavior} ${ev.seg_start_time}–${ev.seg_end_time}` : '這個片段';
    // 刪除會一併移除 mp4 與截圖，不可復原 —— 一定要確認
    if (!confirm(`確定刪除「${what}」？\n\n片段影片與截圖檔會一起刪除，無法復原。`)) return;
    const res = await post('/api/clip/delete', { job_id: jobId, event_index: eventIndex });
    if (!res.ok) appendLog(`[EDIT] 刪除失敗：${res.error || '未知錯誤'}`, 'error');
};

/* eventIndex = -1 代表新增。編輯既有片段時預填原本的起訖。 */
window.openClipEditor = function (e, jobId, eventIndex) {
    if (e) e.stopPropagation();
    const isNew = eventIndex < 0;
    const ev = isNew ? null : jobEvents(jobId).find(x => x.event_index === eventIndex);
    if (!isNew && !ev) return;

    const dlg = $('#clipEditModal');
    dlg.dataset.jobId = jobId;
    dlg.dataset.eventIndex = String(eventIndex);
    $('#clipEditTitle').textContent = isNew ? '新增違規片段' : '編輯違規片段';
    $('#clipEditStart').value = isNew ? '' : ev.seg_start_time;
    $('#clipEditEnd').value = isNew ? '' : ev.seg_end_time;
    $('#clipEditPlate').value = isNew ? '' : (ev.plate || '');
    $('#clipEditReason').value = isNew ? '' : (ev.reason || '');

    const sel = $('#clipEditBehavior');
    sel.innerHTML = BEHAVIOR_CHOICES
        .map(b => `<option value="${b}"${!isNew && ev.behavior === b ? ' selected' : ''}>${b}</option>`)
        .join('');
    $('#clipEditHint').textContent = isNew
        ? '時間是「原始影片」的時間，可輸入 MM:SS、HH:MM:SS 或秒數。判定理由留空會自動產生。'
        : '儲存後會重新產生片段影片與截圖，並更新違規紀錄.csv。判定理由留空會自動產生。';
    dlg.classList.add('active');
};

window.closeClipEditor = function () { $('#clipEditModal').classList.remove('active'); };

/* 「用目前播放位置」：播放器裡的時間是片段相對的，要換算回原始影片座標。
 * 新增片段時播放器可能放的是別的片段 —— 那就用它自己的 clip_start_sec 換算，
 * 這樣不論目前在播哪一段，取到的都是同一條原始影片的絕對時間。 */
window.useCurrentTime = function (which) {
    const v = $('#realVideo');
    if (!App.player.hasRealVideo || !v || !isFinite(v.currentTime)) {
        appendLog('[EDIT] 播放器目前沒有載入片段，無法取用播放位置', 'warn');
        return;
    }
    const cur = jobEvents(App.player.jobId).find(x => x.event_index === App.player.eventIndex);
    const offset = Number(cur && cur.clip_start_sec) || 0;
    const abs = offset + v.currentTime;
    $(which === 'end' ? '#clipEditEnd' : '#clipEditStart').value = fmtClock(abs);
};

window.saveClipEdit = async function () {
    const dlg = $('#clipEditModal');
    const jobId = dlg.dataset.jobId;
    const eventIndex = Number(dlg.dataset.eventIndex);
    const start = parseTimeInput($('#clipEditStart').value);
    const end = parseTimeInput($('#clipEditEnd').value);
    if (start === null || end === null) {
        $('#clipEditHint').textContent = '起始與結束都要填。格式可用 MM:SS、HH:MM:SS 或秒數。';
        return;
    }
    const payload = {
        job_id: jobId,
        seg_start_sec: start,
        seg_end_sec: end,
        behavior: $('#clipEditBehavior').value,
        plate: $('#clipEditPlate').value.trim(),
        // 留空 → 後端自動產生；有填 → 原樣存下，不會被系統改寫
        reason: $('#clipEditReason').value.trim(),
    };
    $('#clipEditHint').textContent = '處理中：重新匯出片段影片與截圖…';
    let res;
    if (eventIndex < 0) {
        res = await post('/api/clip/add', payload);
    } else {
        res = await post('/api/clip/update', { ...payload, event_index: eventIndex });
    }
    if (res.ok) { closeClipEditor(); }
    else { $('#clipEditHint').textContent = `失敗：${res.error || '未知錯誤'}`; }
};

window.seekClipMark = function (e, jobId, eventIndex, which) {
    if (e) e.stopPropagation();
    const ev = jobEvents(jobId).find(x => x.event_index === eventIndex);
    if (!ev || !ev.clip) { appendLog('[PLAYER] 這個事件沒有片段檔可定位', 'warn'); return; }

    const srcSec = which === 'end' ? ev.seg_end_sec : ev.seg_start_sec;
    const target = Math.max(0, srcSec - (Number(ev.clip_start_sec) || 0));
    const label = which === 'end' ? '結束' : '起始';

    const same = App.player.jobId === jobId
        && App.player.eventIndex === eventIndex
        && App.player.hasRealVideo;
    if (same) {
        // 已經是這個片段 → 直接 seek，不重載（重載會閃一下且回到開頭）
        const v = $('#realVideo');
        v.pause();
        v.currentTime = Math.max(0, Math.min(v.duration || 0, target));
        switchTab('player');
    } else {
        openClip(jobId, eventIndex, { autoplay: false, seekTo: target });
    }
    appendLog(`[PLAYER] 定位到違規${label} ${which === 'end' ? ev.seg_end_time : ev.seg_start_time}`, 'info');
};

window.openClip = function (jobId, eventIndex, opts) {
    opts = opts || {};
    const evs = jobEvents(jobId);
    if (!evs.length) {
        switchTab('player');
        appendLog('[PLAYER] 這部影片沒有違規片段可播放', 'warn');
        return;
    }
    const ev = evs.find(e => e.event_index === eventIndex) || evs[0];
    App.player.jobId = jobId;
    App.player.eventIndex = ev.event_index;

    switchTab('player');
    const job = App.jobs.find(j => j.job_id === jobId);
    setField(document, 'selected_clip_name',
        ev.clip ? ev.clip.split('/').pop() : (job ? job.video_name : ''));

    const video = $('#realVideo');
    const svg = $('#playerSvgCanvas');

    if (ev.clip) {
        // 有真的片段檔 → 播真影片
        App.player.hasRealVideo = true;
        App.player.fps = Number(ev.clip_fps) || 0;
        isPlaying = false;
        video.style.display = 'block';
        svg.style.display = 'none';
        video.src = `/api/media?path=${encodeURIComponent(ev.clip)}`;
        video.playbackRate = playbackRate;
        // 時間軸與讀數要跟著「這個片段」走。設計稿裡那些
        // 00:01:23 / 2490 / 9000 / 05:00 都是靜態假資料，沒有任何東西在更新它們。
        // seek 必須等 metadata 到齊，否則 currentTime 會被忽略（duration 還是 NaN）
        video.onloadedmetadata = () => {
            renderTimeline(jobId, ev);
            if (typeof opts.seekTo === 'number') {
                video.currentTime = Math.max(0, Math.min(video.duration || 0, opts.seekTo));
            }
        };
        video.ontimeupdate = () => syncPlayerReadout(ev);
        // 播放按鈕的標示一律跟著影片的真實狀態走（含自動播放被瀏覽器擋掉的情況）
        video.onplay = () => setPlayButton(true);
        video.onpause = () => setPlayButton(false);
        setPlayButton(false);
        if (opts.autoplay !== false) video.play().catch(() => { });
        appendLog(`[PLAYER] 載入違規片段 ${ev.clip.split('/').pop()}`, 'success');
    } else {
        // 沒有片段檔（mock 模式，或 pipeline 未產出）→ 用設計稿的示意圖
        App.player.hasRealVideo = false;
        video.style.display = 'none';
        video.removeAttribute('src');
        svg.style.display = 'block';
        $('#player-violation-label').textContent =
            `違規: ${ev.behavior} [${Number(ev.confidence || 0).toFixed(2)}]`;
        $('#player-plate-label').textContent = ev.plate || '無車牌';
        currentSeconds = hmsToSec(ev.frame_time);
        isPlaying = true;
        appendLog(`[PLAYER] ${job ? job.video_name : ''} ${ev.frame_time} ${ev.behavior}（無片段檔，顯示示意畫面）`, 'info');
    }

    // 註記面板載入該工作的註記
    const ta = $('#activeVideoNoteTextarea');
    if (ta && job) { ta.value = job.job_note || ''; ta.dataset.jobId = jobId; }
    $('#noteSavedTag').style.display = 'none';

    renderTimeline(jobId, ev);
    renderClips();
};

// ── 全螢幕 ────────────────────────────────────────────────────────
// 全螢幕的對象是 wrapper 而不是 <video>：這樣之後要在影片上疊 AI 標註框或
// 浮動控制列時還有容器可用；直接對 <video> 全螢幕的話只剩瀏覽器原生控制列。
window.toggleFullscreen = function () {
    const el = document.querySelector('.player-screen-wrapper');
    if (!el) return;
    const active = document.fullscreenElement || document.webkitFullscreenElement;
    if (active) {
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    } else {
        const req = el.requestFullscreen || el.webkitRequestFullscreen;
        if (!req) { appendLog('[PLAYER] 這個瀏覽器不支援全螢幕', 'warn'); return; }
        // Safari 舊版的 webkitRequestFullscreen 不回 Promise，所以要防呆
        const p = req.call(el);
        if (p && p.catch) p.catch(err => appendLog(`[PLAYER] 全螢幕失敗：${err}`, 'warn'));
    }
};

function syncFsLabel() {
    const on = !!(document.fullscreenElement || document.webkitFullscreenElement);
    const label = $('#fsBtnLabel');
    if (label) label.textContent = on ? '離開全螢幕' : '全螢幕';
}

// 播放中的讀數：目前時間 / 幀號 / 總幀數 / 滑桿位置。
// 幀號需要 fps —— 測試模式的事件會帶 clip_fps；沒有就不顯示幀號（不要瞎猜）。
function syncPlayerReadout(ev) {
    const v = $('#realVideo');
    if (!v || !isFinite(v.duration)) return;
    const fps = Number(ev && ev.clip_fps) || 0;
    setField(document, 'current_time', fmtClock(v.currentTime));
    if (fps > 0) {
        setField(document, 'current_frame', String(Math.round(v.currentTime * fps)));
        setField(document, 'total_frames', String(Math.round(v.duration * fps)));
    } else {
        setField(document, 'current_frame', '—');
        setField(document, 'total_frames', '—');
    }
    const slider = $('#timelineSlider');
    if (slider && document.activeElement !== slider) slider.value = String(v.currentTime);
}

function renderTimeline(jobId, activeEvent) {
    const wrapper = $('#timelineWrapper');
    if (!wrapper) return;
    $$('.timeline-marker-highlight, .timeline-marker-flag', wrapper).forEach(n => n.remove());

    const slider = $('#timelineSlider');
    const evs = jobEvents(jobId);
    const video = $('#realVideo');
    const real = App.player.hasRealVideo && video && isFinite(video.duration) && video.duration > 0;

    // 有真片段時，時間軸的長度就是「這個片段」的長度。
    // 沒有時才退回舊的估算（用事件時間往後推 30 秒），那是給示意圖模式用的。
    const maxSec = real ? video.duration
                        : Math.max(60, ...evs.map(e => hmsToSec(e.frame_time) + 30));
    slider.max = String(maxSec);
    slider.step = real ? '0.05' : '1';
    setField(document, 'total_duration', fmtClock(maxSec));
    setField(document, 'clip_end_label', fmtClock(maxSec));

    // 片段是從「事件前幾秒」開始剪的，所以事件在片段裡的位置要扣掉起始偏移，
    // 否則標記會跑到錯的地方（事件在原片 60 秒 ≠ 在片段裡的第 60 秒）。
    const offset = real ? (Number(activeEvent && activeEvent.clip_start_sec) || 0) : 0;
    const shown = real ? [activeEvent] : evs;

    shown.filter(Boolean).forEach(ev => {
        const sec = real ? Math.max(0, hmsToSec(ev.frame_time) - offset)
                         : hmsToSec(ev.frame_time);
        if (sec > maxSec) return;
        const left = (sec / maxSec) * 100;
        const hl = document.createElement('div');
        hl.className = 'timeline-marker-highlight';
        hl.style.left = `${Math.max(0, left - 2)}%`;
        hl.style.width = '4%';
        hl.title = `${ev.behavior} ${ev.frame_time}`;
        wrapper.insertBefore(hl, slider);

        const flag = document.createElement('div');
        flag.className = 'timeline-marker-flag';
        flag.style.left = `${left}%`;
        flag.textContent = `📍 ${ev.frame_time} ${ev.behavior}`;
        flag.onclick = () => seekToSeconds(sec);
        wrapper.insertBefore(flag, slider);
    });

    if (real) syncPlayerReadout(activeEvent);
}

// 真影片模式下，播放器控制要作用在 <video> 上而不是假的 ticker
// 按鈕標示＝「按下去會發生什麼」：暫停時顯示「播放」，播放時顯示「暫停」。
//
// 關鍵是由 <video> 的 play/pause 事件驅動，而不是在點擊時順手改。
// 原本的寫法在片段載入自動播放、影片播完、或瀏覽器擋掉自動播放時
// 都不會更新，標示會跟實際狀態對不起來。
const _ICON_PLAY = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
const _ICON_PAUSE = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6zm8 0h4v16h-4z"/></svg>';

function setPlayButton(playing) {
    const btn = $('#playPauseBtn');
    if (!btn) return;
    btn.innerHTML = playing ? `${_ICON_PAUSE} 暫停` : `${_ICON_PLAY} 播放`;
    btn.title = playing ? '暫停（空白鍵）' : '播放（空白鍵）';
}

const origToggle = window.togglePlayPause;
window.togglePlayPause = function () {
    if (App.player.hasRealVideo) {
        const v = $('#realVideo');
        if (v.paused) v.play().catch(() => { }); else v.pause();
        return;   // 標示交給 play/pause 事件處理
    }
    origToggle();
};

const origStep = window.stepFrame;
window.stepFrame = function (direction) {
    if (App.player.hasRealVideo) {
        const v = $('#realVideo');
        v.pause();
        // 原本寫死 /30。這支影片是 10 fps，按一次「下一幀」會跳 3 幀。
        // 事件現在會帶 clip_fps，用真的幀率；拿不到才退回 30。
        const fps = App.player.fps || 30;
        v.currentTime = Math.max(0, Math.min(v.duration || 1e9, v.currentTime + direction / fps));
        appendLog(`[PLAYER] 逐格調整 ${direction > 0 ? '+1' : '-1'} 幀（${fps} fps）`, 'info');
        return;
    }
    origStep(direction);
};

const origPreset = window.setSpeedPreset;
window.setSpeedPreset = function (rate) {
    origPreset(rate);
    if (App.player.hasRealVideo) $('#realVideo').playbackRate = rate;
};

const origSpeed = window.changePlaybackSpeed;
window.changePlaybackSpeed = function (delta) {
    origSpeed(delta);
    if (App.player.hasRealVideo) $('#realVideo').playbackRate = playbackRate;
};

const origScrub = window.onTimelineScrub;
window.onTimelineScrub = function (val) {
    if (App.player.hasRealVideo) {
        $('#realVideo').currentTime = parseFloat(val);
        return;
    }
    origScrub(val);
};

const origSeek = window.seekToSeconds;
window.seekToSeconds = function (sec) {
    if (App.player.hasRealVideo) {
        $('#realVideo').currentTime = sec;
        return;
    }
    origSeek(sec);
};

// ── 註記面板 ──────────────────────────────────────────────────────

window.saveActiveNote = async function () {
    const ta = $('#activeVideoNoteTextarea');
    const jobId = ta.dataset.jobId || App.player.jobId;
    if (!jobId) { appendLog('[NOTE] 請先選擇一個違規片段', 'warn'); return; }
    const res = await post('/api/note', { job_id: jobId, note: ta.value });
    if (res.ok) {
        const badge = $('#noteSavedTag');
        badge.textContent = `✓ 已儲存 ${new Date().toTimeString().slice(0, 5)}`;
        badge.style.display = 'inline';
        appendLog('[NOTE] 稽查註記已寫入舉證資料夾（note.json）', 'success');
    }
};

window.addQuickNoteTag = function (tagText) {
    const ta = $('#activeVideoNoteTextarea');
    ta.value = ta.value.trim() ? `${ta.value} [${tagText}]` : `[${tagText}]`;
    saveActiveNote();
};

// ── 證據照片預覽 Modal ────────────────────────────────────────────

window.previewEvent = function (jobId, eventIndex) {
    const ev = jobEvents(jobId).find(e => e.event_index === eventIndex);
    if (!ev) return;

    $('#detail-plate').textContent = ev.plate || '未讀取到車牌';
    $('#detail-type').textContent = ev.behavior || '—';
    $('#detail-conf').textContent = Number(ev.confidence || 0).toFixed(2);
    $('#detail-time').textContent = ev.overlay_time || ev.frame_time || '—';
    $('#detail-reason').textContent = ev.reason || '—';
    $('#modal-plate-text').textContent = ev.plate || '無車牌';
    $('#modal-type-text').textContent = `違規: ${ev.behavior || ''}`;
    $('#modal-time-text').textContent = ev.overlay_time || '';

    // 有真的截圖檔就顯示照片，沒有就退回設計稿的 SVG 示意圖
    const box = $('.preview-img-box');
    let img = $('#modalRealImg');
    if (!img) {
        img = document.createElement('img');
        img.id = 'modalRealImg';
        // 只給 width:100% 會讓 4:3 的截圖高度超出容器（360px）而被
        // overflow:hidden 裁掉上下。要同時夾住兩個方向，由 object-fit 保持比例。
        img.style.cssText =
            'max-width:100%;max-height:100%;object-fit:contain;'
            + 'border-radius:6px;display:none;cursor:zoom-in;';
        img.title = '點擊開啟原始解析度截圖';
        img.onclick = () => { if (img.src) window.open(img.src, '_blank'); };
        box.appendChild(img);
    }
    const svg = $('.preview-svg-placeholder');
    if (ev.snapshot) {
        img.src = `/api/media?path=${encodeURIComponent(ev.snapshot)}`;
        img.style.display = 'block';
        svg.style.display = 'none';
    } else {
        img.style.display = 'none';
        svg.style.display = 'block';
    }
    $('#previewModal').dataset.clip = ev.clip || '';
    $('#previewModal').classList.add('active');
};

// 設計稿的下載鈕原本是 alert；接上真檔案
document.addEventListener('click', (e) => {
    const btn = e.target.closest('#previewModal .btn-primary');
    if (!btn) return;
    e.preventDefault();
    const clip = $('#previewModal').dataset.clip;
    if (clip) window.open(`/api/media?path=${encodeURIComponent(clip)}`, '_blank');
    else appendLog('[DOWNLOAD] 這個事件沒有對應的影片片段檔', 'warn');
}, true);

// ── 啟動 ──────────────────────────────────────────────────────────

(async function init() {
    $('#system-log-container').innerHTML = '';
    isPlaying = false;   // 停掉設計稿預設的「播放中」狀態

    document.querySelectorAll('input[name="pipeline_mode"]').forEach(
        el => el.addEventListener('change', syncModeUI)
    );
    syncModeUI();

    ['fullscreenchange', 'webkitfullscreenchange'].forEach(
        e => document.addEventListener(e, syncFsLabel)
    );
    // 快捷鍵：F 全螢幕、空白鍵播放/暫停。
    // 在輸入框裡打字時不攔截，否則稽查註記打到 f 就跳全螢幕、打空白就暫停。
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        const t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
        // 空白鍵在按鈕上有原生的「觸發」語意，別重複觸發
        if (e.key === ' ' && t && t.tagName === 'BUTTON') return;
        if (e.key === 'f' || e.key === 'F') { e.preventDefault(); toggleFullscreen(); }
        else if (e.key === ' ') { e.preventDefault(); togglePlayPause(); }
    });

    const state = await api(`/api/state?offset=0&limit=${App.jobsLimit}`);
    App.config = state.config || {};
    (state.logs || []).forEach(appendLogEntry);
    applyState(state);

    if (App.config.default_source) {
        $('[data-field="source_folder_path"]').value = App.config.default_source;
    }
    if (App.config.output_root) {
        $('[data-field="output_root_path"]').value = App.config.output_root;
    }
    setConnected(false);
    connect();
})();
