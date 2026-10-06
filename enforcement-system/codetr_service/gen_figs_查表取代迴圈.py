#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""產生 查表取代迴圈.md 用的三張靜態 SVG。

LAB / AREA 與網頁版同一份資料，來源是實際跑 cv2.connectedComponentsWithStats 的輸出。

注意：色碼一律直接寫死，不要用 CSS 自訂屬性（var(--x)）。
不支援自訂屬性的算圖器會把 fill:var(--ink) 當成無效值，整張圖變黑。
深色模式靠 @media 區塊重寫同一組 class，同樣寫死色碼。
"""
import os
os.makedirs('assets', exist_ok=True)

LAB = [[0,0,0,0,0,0,0,0],
       [0,1,1,0,0,2,2,0],
       [0,1,1,0,0,2,2,0],
       [0,1,0,0,0,2,0,0],
       [0,0,0,4,0,0,0,5],
       [3,0,0,0,0,0,0,0],
       [0,0,6,0,0,7,0,0],
       [0,0,0,0,0,0,0,8]]
AREA = [0,5,5,1,1,1,1,1,1]
THR  = 5
KEEP = [1 if i > 0 and AREA[i] >= THR else 0 for i in range(9)]

MONO = 'ui-monospace,"SF Mono",Menlo,Consolas,monospace'
SANS = '"Noto Sans TC",system-ui,-apple-system,"PingFang TC","Microsoft JhengHei",sans-serif'

# (class, 屬性, 淺色, 深色)
PAL = [
    ('paper',   'fill',   '#ffffff', '#1c1a23'),
    ('off',     'fill',   '#f4f2f7', '#232029'),   # 背景格
    ('fgc',     'fill',   '#dcd8e6', '#343040'),   # 前景格（未處理）
    ('kp',      'fill',   '#ebe8f9', '#251f42'),   # 保留
    ('ct',      'fill',   '#f7e6d5', '#3d2a12'),   # 這一輪塗上去
    ('t-ink',   'fill',   '#1a1922', '#eae7f2'),
    ('t-off',   'fill',   '#a5a1b0', '#6b6678'),
    ('t-keep',  'fill',   '#4a3fb0', '#9b8ff0'),
    ('t-cost',  'fill',   '#a8571d', '#d68f4e'),
    ('t-soft',  'fill',   '#55525f', '#a5a0b3'),
    ('s-keep',  'stroke', '#b9b1ea', '#4c3f8c'),
    ('s-cost',  'stroke', '#a8571d', '#d68f4e'),
    ('s-rule',  'stroke', '#e2dfea', '#312d3d'),
]

def style():
    def rules(idx):
        return '\n'.join('    .%s{%s:%s}' % (c, p, v[idx]) for c, p, *v in
                         [(c, p, l, d) for c, p, l, d in PAL])
    base = ('\n    text{font-family:%s}\n'
            '    .lbl{font-family:%s;font-size:12px}\n'
            '    .lbl-b{font-family:%s;font-size:12.5px;font-weight:700}\n'
            '    .kp,.ct{stroke-width:1}\n'
            '    .maskbg{fill:#1d1b24} .maskfg{fill:#ffffff}\n    .maskgrid{fill:#2b2933}\n') % (MONO, SANS, SANS)
    return ('  <style>' + base + rules(0) +
            '\n\n    @media (prefers-color-scheme:dark){\n' +
            '\n'.join('  ' + l for l in rules(1).split('\n')) + '\n    }\n  </style>\n')

def svg(fn, w, h, body, alt):
    out = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
           'role="img" aria-label="%s">\n' % (w, h, w, h, alt)) + style() + \
          '  <rect class="paper" width="%d" height="%d"/>\n' % (w, h) + body + '\n</svg>\n'
    open('assets/' + fn, 'w', encoding='utf-8').write(out)
    print('  %-32s %6d bytes' % (fn, os.path.getsize('assets/' + fn)))

def grid(x0, y0, cells, cs=26, gap=3, fs=13):
    """cells[r][c] = (文字, 方格 class, 文字 class)"""
    o = []
    for r, row in enumerate(cells):
        for c, (txt, box, tc) in enumerate(row):
            x, y = x0 + c * (cs + gap), y0 + r * (cs + gap)
            o.append('<rect class="%s" x="%g" y="%g" width="%g" height="%g" rx="3"/>' % (box, x, y, cs, cs))
            if txt != '':
                o.append('<text class="%s" x="%g" y="%g" font-size="%g" text-anchor="middle">%s</text>'
                         % (tc, x + cs / 2, y + cs / 2 + fs * .36, fs, txt))
    return '\n'.join(o), 8 * cs + 7 * gap

W = 8 * 26 + 7 * 3     # 229

# ── 圖 1：遮罩 與 lab ─────────────────────────────────────
# 遮罩照它實際的樣子畫：整片黑底，只在前景畫白方塊，不隨主題變色。
# （先前每格都畫深色圓角磚，結果白色縫隙反而被讀成圖案，等於畫反了。）
def maskpanel(x0, y0, cs=26, gap=3):
    o = ['<rect class="maskbg" x="%g" y="%g" width="%g" height="%g" rx="4"/>'
         % (x0 - 4, y0 - 4, 8 * (cs + gap) - gap + 8, 8 * (cs + gap) - gap + 8)]
    for r in range(8):
        for c in range(8):
            x, y = x0 + c * (cs + gap), y0 + r * (cs + gap)
            if LAB[r][c]:
                o.append('<rect class="maskfg" x="%g" y="%g" width="%g" height="%g" rx="2"/>'
                         % (x, y, cs, cs))
            else:   # 極淡的格線，讓人數得出格數
                o.append('<rect class="maskgrid" x="%g" y="%g" width="%g" height="%g" rx="2"/>'
                         % (x, y, cs, cs))
    return '\n'.join(o)

labg = [[(str(v) if v else '·', 'fgc' if v else 'off', 't-ink' if v else 't-off') for v in row]
        for row in LAB]
g1 = maskpanel(16, 46)
g2, _ = grid(16 + W + 58, 46, labg)
body = ('<text class="lbl-b t-ink" x="%g" y="32" text-anchor="middle">遮罩（白＝前景）</text>'
        '<text class="lbl-b t-ink" x="%g" y="32" text-anchor="middle">lab（元件編號，· ＝背景 0）</text>'
        % (16 + W / 2, 16 + W + 58 + W / 2)) + g1 + g2
svg('玩具範例_遮罩與lab.svg', 32 + W * 2 + 58, 46 + W + 18, body,
    '8x8 玩具範例：左邊是黑底白點的遮罩，右邊是每格標上元件編號的標籤圖')

# ── 圖 2：迴圈版逐步（八格小圖）────────────────────────────
CS, GAP, PW, COLS = 12, 2, 8 * 12 + 7 * 2, 4
panels = []
for i in range(1, 9):
    keep = KEEP[i] == 1
    cells = []
    for r in range(8):
        row = []
        for c in range(8):
            v = LAB[r][c]
            if v == i:
                # 只有通過門檻的元件才會被塗上去；被跳過的維持原樣
                row.append((str(v), 'ct' if keep else 'fgc', 't-cost' if keep else 't-ink'))
            elif v:
                row.append((str(v), 'fgc', 't-off'))
            else:
                row.append(('', 'off', ''))
        cells.append(row)
    px = 20 + ((i - 1) % COLS) * (PW + 30)
    py = 96 + ((i - 1) // COLS) * (PW + 56)
    g, _ = grid(px, py, cells, cs=CS, gap=GAP, fs=8)
    if keep:
        g += ('<rect class="s-cost" fill="none" stroke-dasharray="3 2" x="%g" y="%g" '
              'width="%g" height="%g" rx="4"/>' % (px - 5, py - 5, PW + 10, PW + 10))
    panels.append(
        '<text class="lbl-b %s" x="%g" y="%g" text-anchor="middle">元件 %d</text>'
        % ('t-cost' if keep else 't-soft', px + PW / 2, py - 13, i) + g +
        '<text class="lbl %s" x="%g" y="%g" text-anchor="middle">%s</text>'
        % ('t-cost' if keep else 't-keep', px + PW / 2, py + PW + 18,
           '面積 5 ≥ 5 → 掃全圖' if keep else '面積 1 &lt; 5 → 跳過'))
W2 = 40 + COLS * PW + (COLS - 1) * 30
body2 = ('<text class="lbl-b t-ink" x="%g" y="28" text-anchor="middle" font-size="14">'
         '迴圈版：八個元件輪流檢查，只有兩個真的掃了全圖</text>'
         '<text class="lbl t-soft" x="%g" y="48" text-anchor="middle">'
         '橘色虛線框＝這一輪掃過全部 64 格；橘底＝掃描時發現屬於當前元件而塗上去</text>'
         '<text class="lbl t-soft" x="%g" y="64" text-anchor="middle">'
         '灰底＝其他元件的格子，這一輪沒被碰到</text>'
         % (W2 / 2, W2 / 2, W2 / 2)) + '\n'.join(panels)
svg('迴圈版逐步.svg', W2, 96 + 2 * (PW + 56) - 22, body2,
    '八個元件各一格小圖：面積達標的兩個有橘色虛線框代表掃過全圖，其餘六個標示跳過')

# ── 圖 3：keep[lab] 展開 ──────────────────────────────────
outg = [[(str(KEEP[v]), 'kp' if KEEP[v] else 'off', 't-keep' if KEEP[v] else 't-off')
         for v in row] for row in LAB]
g3, _ = grid(16, 46, labg)
g4, _ = grid(16 + W + 100, 46, outg)
TOT = 32 + W * 2 + 100
kx = (TOT - (9 * 26 + 8 * 3)) / 2
kg = []
for i in range(9):
    x = kx + i * 29
    kg.append('<text class="lbl t-soft" x="%g" y="%g" text-anchor="middle" font-size="11">[%d]</text>'
              % (x + 13, 46 + W + 46, i))
    kg.append('<rect class="%s" x="%g" y="%g" width="26" height="26" rx="3"/>'
              % ('kp' if KEEP[i] else 'off', x, 46 + W + 52))
    kg.append('<text class="%s" x="%g" y="%g" font-size="13" text-anchor="middle">%d</text>'
              % ('t-keep' if KEEP[i] else 't-off', x + 13, 46 + W + 52 + 17, KEEP[i]))
ay = 46 + W / 2
body3 = ('<text class="lbl-b t-ink" x="%g" y="32" text-anchor="middle">lab（每格寫著元件編號）</text>'
         '<text class="lbl-b t-keep" x="%g" y="32" text-anchor="middle">keep[lab]（查完的結果）</text>'
         % (16 + W / 2, 16 + W + 100 + W / 2)) + g3 + g4 + \
        ('<text class="t-keep" x="%g" y="%g" text-anchor="middle" font-size="22">→</text>'
         '<text class="lbl t-soft" x="%g" y="%g" text-anchor="middle" font-size="10.5">逐格查小抄</text>'
         % (16 + W + 50, ay + 6, 16 + W + 50, ay + 28)) + \
        ('<text class="lbl-b t-ink" x="%g" y="%g" text-anchor="middle">'
         '小抄 keep（長度 9；上排是格號＝元件編號，下排是值）</text>' % (TOT / 2, 46 + W + 30)) + \
        '\n'.join(kg)
svg('keep查表展開.svg', TOT, 46 + W + 52 + 26 + 18, body3,
    '把 lab 每格的元件編號當索引去查長度 9 的 keep 小抄，一次得到整張結果')
