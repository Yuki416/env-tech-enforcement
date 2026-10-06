# -*- coding: utf-8 -*-
"""把本專案這幾份 HTML 文件轉成 Markdown。只處理我自己寫過的那幾種結構。"""
import re, html as H

def unesc(t):
    return H.unescape(t)

def inline(t):
    """處理行內元素。"""
    t = re.sub(r'<span class="tag[^"]*">([^<]*)</span>', r'（\1）', t)
    t = re.sub(r'<span class="lbl">(.*?)</span>', r'**\1**　', t, flags=re.S)
    t = re.sub(r'<(strong|b)>(.*?)</\1>', r'**\2**', t, flags=re.S)
    t = re.sub(r'<em>(.*?)</em>', r'*\1*', t, flags=re.S)
    t = re.sub(r'<code>(.*?)</code>', lambda m: '`' + re.sub(r'<[^>]+>', '', m.group(1)) + '`', t, flags=re.S)
    t = re.sub(r'<a [^>]*href="#[^"]*"[^>]*>(.*?)</a>', r'\1', t, flags=re.S)
    t = re.sub(r'<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>', r'[\2](\1)', t, flags=re.S)
    t = re.sub(r'<br\s*/?>', ' ', t)
    t = re.sub(r'<small>(.*?)</small>', r'\1', t, flags=re.S)
    t = re.sub(r'<[^>]+>', '', t)
    t = unesc(t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n +', '\n', t)          # 續行不要留行首空白
    return t.strip()

def cell(t):
    """表格儲存格：<br> 要保留成換行，用 Markdown 表格允許的 <br>。"""
    t = re.sub(r'<br\s*/?>', '###BR###', t)
    t = inline(t)
    return t.replace('###BR###', '<br>').replace('|', '\\|')

def table(blk):
    rows = []
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', blk, re.S):
        cs = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', tr, re.S)
        rows.append([cell(c) for c in cs])
    if not rows:
        return ''
    head = re.search(r'<thead>(.*?)</thead>', blk, re.S)
    nh = len(re.findall(r'<tr[^>]*>', head.group(1))) if head else 1
    w = max(len(r) for r in rows)
    rows = [r + [''] * (w - len(r)) for r in rows]
    # 多列表頭就把它們併成一列
    if nh > 1:
        merged = [' / '.join(x for x in col if x) for col in zip(*rows[:nh])]
        rows = [merged] + rows[nh:]
        nh = 1
    align = ['---'] * w
    for i in range(w):
        vals = [r[i].replace('**', '').strip() for r in rows[nh:]]
        if any(re.match(r'^[\d.,%≥≤×+\-–— ]+$', v) and v for v in vals):
            align[i] = '---:'
    out = ['| ' + ' | '.join(rows[0]) + ' |', '|' + '|'.join(align) + '|']
    for r in rows[nh:]:
        out.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(out)

def convert(path, svgmap, title_prefix=''):
    s = open(path, encoding='utf-8').read()
    b = s.split('</style>', 1)[1]
    b = re.sub(r'<script.*?</script>', '', b, flags=re.S)
    out, svg_i = [], 0

    # 開場
    m = re.search(r'<h1[^>]*>(.*?)</h1>', b, re.S)
    if m:
        out.append('# ' + inline(m.group(1)) + '\n')
    m = re.search(r'<p class="eyebrow">(.*?)</p>', b, re.S)
    if m:
        out.append('*' + inline(m.group(1)) + '*\n')
    m = re.search(r'<p class="standfirst">(.*?)</p>', b, re.S) or \
        re.search(r'<p class="lede">(.*?)</p>', b, re.S)
    if m:
        out.append(inline(m.group(1)) + '\n')

    body = b
    m = re.search(r'</header>(.*)', b, re.S)
    if m:
        body = m.group(1)

    # 逐塊掃描
    pat = re.compile(
        r'<h2[^>]*>(?P<h2>.*?)</h2>'
        r'|<h3[^>]*>(?P<h3>.*?)</h3>'
        r'|<p class="sub">(?P<sub>.*?)</p>'
        r'|<div class="tw">(?P<tw>.*?)</div>'
        r'|<figure>(?P<fig>.*?)</figure>'
        r'|<div class="formula">(?P<fml>.*?)</div>'
        r'|<div class="hero">(?P<hero>.*?)</div>\s*</div>'
        r'|<pre[^>]*>(?P<pre>.*?)</pre>'
        r'|<(?P<lt>ul|ol)[^>]*>(?P<list>.*?)</(?P=lt)>'
        r'|<p class="note(?P<nk>[^"]*)">(?P<note>.*?)</p>'
        r'|<div class="note(?P<nk2>[^"]*)">(?P<note2>.*?)</div>\s*(?=<)'
        r'|<p(?P<pa>(?: [^>]*)?)>(?P<p>.*?)</p>'
        r'|<footer[^>]*>(?P<foot>.*?)</footer>', re.S)

    for m in pat.finditer(body):
        g = m.groupdict()
        if g['h2'] is not None:
            t = re.sub(r'<span class="n">(.*?)</span>', r'\1. ', g['h2'], flags=re.S)
            out.append('\n---\n\n## ' + inline(t).replace('. ', '. ', 1) + '\n')
        elif g['h3'] is not None:
            out.append('\n### ' + inline(g['h3']) + '\n')
        elif g['sub'] is not None:
            out.append('*' + inline(g['sub']) + '*\n')
        elif g['tw'] is not None:
            out.append(table(g['tw']) + '\n')
        elif g['fig'] is not None:
            f = g['fig']
            t = re.search(r'<p class="fig-t(?:itle)?">(.*?)</p>', f, re.S)
            n = re.search(r'<p class="fig-n(?:ote)?">(.*?)</p>', f, re.S)
            c = re.search(r'<figcaption>(.*?)</figcaption>', f, re.S)
            if t: out.append('**' + inline(t.group(1)) + '**\n')
            if n: out.append(inline(n.group(1)) + '\n')
            if '<svg' in f or svg_i < len(svgmap):
                if svg_i < len(svgmap):
                    alt, fn = svgmap[svg_i]; svg_i += 1
                    out.append(f'![{alt}](assets/{fn})\n')
            # figure 內的表格也要帶出來
            for tw in re.findall(r'<div class="tw">(.*?)</div>', f, re.S):
                out.append(table(tw) + '\n')
            if c: out.append(inline(c.group(1)) + '\n')
        elif g['fml'] is not None:
            txt = g['fml']
            main = re.split(r'<small>', txt, 1)
            out.append('> **' + inline(main[0]) + '**')
            if len(main) > 1:
                out.append('>\n> ' + inline(main[1]) + '\n')
            else:
                out.append('')
        elif g['hero'] is not None:
            v = re.search(r'<div class="v">(.*?)</div>', g['hero'], re.S)
            k = re.search(r'<div class="k">(.*?)</div>', g['hero'], re.S)
            if v and k:
                out.append(f'> **{inline(v.group(1))}** — {inline(k.group(1))}\n')
        elif g['pre'] is not None:
            code = re.sub(r'<[^>]+>', '', g['pre'])
            out.append('```\n' + unesc(code).strip('\n') + '\n```\n')
        elif g['list'] is not None:
            mark = '1. ' if g['lt'] == 'ol' else '- '
            items = re.findall(r'<li[^>]*>(.*?)</li>', g['list'], re.S)
            for i, it in enumerate(items):
                pre_ = f'{i+1}. ' if g['lt'] == 'ol' else '- '
                out.append(pre_ + inline(it))
            out.append('')
        elif g.get('note2') is not None:
            kind = (g['nk2'] or '').strip()
            lbl = {'warn': '⚠️ '}.get(kind, '💡 ')
            inner = re.sub(r'</?p[^>]*>', '\n', g['note2'])
            txt = '\n'.join(x for x in (inline(l) for l in inner.split('\n')) if x)
            out.append('> ' + lbl + txt.replace('\n', '\n>\n> ') + '\n')
        elif g['note'] is not None:
            kind = (g['nk'] or '').strip()
            lbl = {'cost': '⚠️ ', 'sig': '💡 '}.get(kind, '')
            out.append('> ' + lbl + inline(g['note']).replace('\n', '\n> ') + '\n')
        elif g['p'] is not None:
            a = g['pa'] or ''
            if 'fig-title' in a or 'fig-note' in a or 'sub' in a:
                continue
            t = inline(g['p'])
            if t:
                out.append(t + '\n')
        elif g['foot'] is not None:
            out.append('\n---\n')
            for p in re.findall(r'<p[^>]*>(.*?)</p>', g['foot'], re.S):
                t = inline(p)
                if t: out.append(t + '\n')

    md = '\n'.join(out)
    md = re.sub(r'\n{4,}', '\n\n\n', md)
    return md
