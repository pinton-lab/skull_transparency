#!/usr/bin/env python3
"""Convert tutorial.tex -> tutorial.md so the LaTeX stays the single source.

Zero third-party dependencies. It is NOT a general LaTeX engine: it understands
exactly the constructs tutorial.tex uses (sections, the \\code macro, lstlisting,
booktabs tabular, figure+caption+ref, align/\\[..\\] math, quote, enumerate/
itemize, paragraph, textbf/emph). Math is emitted as GitHub $...$/$$...$$ and the
figure PNGs are linked from figs/. Keep new prose within these constructs and the
Markdown will track the PDF automatically.

    python tex2md.py tutorial.tex tutorial.md
"""
import re
import sys

MATH = "\x00M{}\x00"


def _protect_math(s, store):
    """Replace $...$ and \\(..\\) inline math with placeholders (kept verbatim)."""
    out = []
    i = 0
    while i < len(s):
        if s[i] == "$":
            j = s.find("$", i + 1)
            if j == -1:
                out.append(s[i:]); break
            store.append(s[i:j + 1])
            out.append(MATH.format(len(store) - 1))
            i = j + 1
        else:
            out.append(s[i]); i += 1
    return "".join(out)


def _brace_groups(text, start, n):
    """Extract up to ``n`` balanced ``{...}`` group contents from ``text`` at/after ``start``."""
    groups, i = [], start
    while len(groups) < n and i < len(text):
        while i < len(text) and text[i] != "{":
            i += 1
        if i >= len(text):
            break
        depth, buf = 0, []
        while i < len(text):
            c = text[i]
            if c == "{":
                depth += 1
                if depth == 1:
                    i += 1; continue
            elif c == "}":
                depth -= 1
                if depth == 0:
                    i += 1; break
            buf.append(c); i += 1
        groups.append("".join(buf))
    return groups


def inline(s, refs):
    """Inline-markup substitution on a run of ordinary (non-verbatim) text."""
    s = re.sub(r"(?<!\\)%.*$", "", s)              # strip comments
    s = s.replace("``", '"').replace("''", '"')    # LaTeX smart quotes -> "
    store = []
    s = _protect_math(s, store)                    # shield inline math
    s = re.sub(r"\\(?:code|texttt|lstinline)\{([^{}]*)\}", r"`\1`", s)
    s = re.sub(r"\\(?:textbf|paragraph)\{([^{}]*)\}", r"**\1**", s)
    s = re.sub(r"\\(?:emph|textit)\{([^{}]*)\}", r"*\1*", s)
    s = re.sub(r"\\href\{([^{}]*)\}\{([^{}]*)\}", r"[\2](\1)", s)
    s = re.sub(r"\\ref\{([^{}]*)\}", lambda m: refs.get(m.group(1), "?"), s)
    s = re.sub(r"\\label\{[^{}]*\}", "", s)
    s = re.sub(r"\\(?:small|large|itshape|noindent|centering|bfseries|hrule)\b", "", s)
    s = re.sub(r"\\\\(\[[^\]]*\])?", " ", s)        # stray line breaks -> space
    s = s.replace("~", " ")
    for a, b in [(r"\dots", "…"), (r"\ldots", "…"), (r"\textbullet", "•"),
                 (r"\%", "%"), (r"\&", "&"), (r"\_", "_"), (r"\#", "#"),
                 (r"\,", " "), (r"\;", " "), (r"\ ", " "), (r"\@", "")]:
        s = s.replace(a, b)
    s = s.replace("{", "").replace("}", "")         # drop leftover grouping braces
    s = s.replace("---", "—")                  # em-dash
    s = re.sub(r"(?<=\w)--(?=\w)", "–", s)      # en-dash for ranges
    for k, frag in enumerate(store):                # restore math
        s = s.replace(MATH.format(k), frag)
    return re.sub(r"[ \t]+", " ", s).strip()


def render_table(rows):
    """rows: list of cell-lists; the row before \\midrule is the header."""
    if not rows:
        return []
    hdr, body = rows[0], rows[1:]
    out = ["| " + " | ".join(hdr) + " |",
           "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in body:
        r = (r + [""] * len(hdr))[:len(hdr)]
        out.append("| " + " | ".join(r) + " |")
    return out


def main(src, dst):
    text = open(src).read()
    # ---- metadata ----------------------------------------------------------
    def grab(cmd):
        m = re.search(r"\\" + cmd + r"\{", text)
        if not m:
            return ""
        i = m.end(); depth = 1; buf = []
        while i < len(text) and depth:
            c = text[i]
            depth += {"{": 1, "}": -1}.get(c, 0)
            if depth:
                buf.append(c)
            i += 1
        return "".join(buf)

    raw_title = re.sub(r"\\vspace\{[^{}]*\}", "", grab("title"))
    _tp = re.split(r"\\\\\s*\[[^\]]*\]", raw_title, maxsplit=1)   # split title vs subtitle

    def _clean_title(p):
        p = p.replace("\\\\", " ")
        p = re.sub(r"\\(?:textbf|large|Large|huge)\b", "", p)
        p = p.replace("{", "").replace("}", "")
        return re.sub(r"\s+", " ", p).strip()

    title = _clean_title(_tp[0])
    subtitle = _clean_title(_tp[1]) if len(_tp) > 1 else ""

    body = text[text.index(r"\begin{document}") + len(r"\begin{document}"):
                text.index(r"\end{document}")]

    # ---- pass 1: number figures (by \label order) and sections -------------
    refs, fign = {}, 0
    for blk in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", body, re.S):
        m = re.search(r"\\label\{([^{}]*)\}", blk)
        if m:
            fign += 1
            refs[m.group(1)] = str(fign)
    secn = 0
    for ln in body.split("\n"):
        if "\\section{" in ln:
            secn += 1
            m = re.search(r"\\label\{(sec:[^{}]*)\}", ln)
            if m:
                refs[m.group(1)] = str(secn)

    # ---- pass 2: line state machine ----------------------------------------
    lines = body.split("\n")
    out, para = [], []
    list_stack = []           # entries: {'kind','n','marker_emitted','text'}

    def flush_para():
        if para:
            t = inline(" ".join(para), refs)
            if t:
                out.append(t); out.append("")
            para.clear()

    def cur():
        return list_stack[-1] if list_stack else None

    def emit_item_marker():
        it = cur()
        if it and not it["marker_emitted"]:
            ind = "  " * (len(list_stack) - 1)
            mark = f"{it['n']}." if it["kind"] == "ol" else "-"
            out.append(f"{ind}{mark} {inline(' '.join(it['text']), refs)}".rstrip())
            it["marker_emitted"] = True

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        s = ln.strip()

        if s.startswith("\\section{"):
            flush_para(); out.append("## " + inline(_brace_groups(s, s.index("{"), 1)[0], refs)); out.append("")
        elif s.startswith("\\subsection{"):
            flush_para(); out.append("### " + inline(_brace_groups(s, s.index("{"), 1)[0], refs)); out.append("")
        elif s.startswith("\\maketitle") or re.match(r"\\vspace\b", s) or s in ("", "\\centering"):
            if s == "":
                if list_stack:
                    pass
                else:
                    flush_para()
            # else: skip spacing-only command lines
        elif s.startswith("\\hrule"):
            flush_para(); out.append("---"); out.append("")
        elif s.startswith("\\appendix"):
            flush_para()
        elif s.startswith("\\callout{"):
            flush_para()
            buf, depth, started = [], 0, False
            while i < len(lines):
                for ch in lines[i]:
                    if ch == "{":
                        depth += 1; started = True
                    elif ch == "}":
                        depth -= 1
                buf.append(lines[i])
                if started and depth <= 0:
                    break
                i += 1
            block = "\n".join(buf)
            g = _brace_groups(block, block.index("\\callout") + 8, 2)
            out.append("> **" + (inline(g[0], refs) if g else "") + "**"); out.append(">")
            bodyraw = g[1] if len(g) > 1 else ""
            for item in re.split(r"\\\\", bodyraw):
                item = re.sub(r"^%\s*", "", item.replace("\n", " ").strip())
                item = re.sub(r"\\textbullet\\?\s*", "", item)
                item = inline(item, refs)
                if item:
                    out.append("> - " + item)
            out.append("")
        elif s.startswith("\\begin{quote}"):
            flush_para(); buf = []
            i += 1
            while "\\end{quote}" not in lines[i]:
                buf.append(lines[i].strip()); i += 1
            txt = inline(" ".join(buf), refs)
            for q in re.split(r"(?<=\.)\s+", txt) if False else [txt]:
                out.append("> " + q)
            out.append("")
        elif s.startswith("\\begin{align}") or s.startswith("\\["):
            flush_para(); buf = []
            end = "\\end{align}" if "align" in s else "\\]"
            aligned = "align" in s
            i += 1
            while end not in lines[i]:
                buf.append(lines[i].rstrip()); i += 1
            inner = "\n".join(buf).strip()
            out.append("$$")
            if aligned:
                out.append("\\begin{aligned}"); out.append(inner); out.append("\\end{aligned}")
            else:
                out.append(inner)
            out.append("$$"); out.append("")
        elif s.startswith("\\begin{lstlisting}"):
            m = re.search(r"style=(\w+)", s)
            lang = {"sh": "bash", "py": "python"}.get(m.group(1) if m else "", "")
            buf = []
            i += 1
            while "\\end{lstlisting}" not in lines[i]:
                buf.append(lines[i]); i += 1
            ind = "    " if list_stack else ""
            if list_stack:
                emit_item_marker()
            else:
                flush_para()
            out.append(ind + "```" + lang)
            out += [ind + b for b in buf]
            out.append(ind + "```"); out.append("")
        elif s.startswith("\\begin{tabular}"):
            rows, cells, cur_cells = [], [], []
            i += 1
            while "\\end{tabular}" not in lines[i]:
                t = lines[i].strip()
                if t.startswith("\\toprule") or t.startswith("\\bottomrule"):
                    i += 1; continue
                if t.startswith("\\midrule"):
                    i += 1; continue
                for piece in re.split(r"\\\\", t):
                    if piece.strip() == "":
                        if cur_cells:
                            rows.append([inline(c, refs) for c in cur_cells]); cur_cells = []
                        continue
                    cur_cells = piece.split("&")
                    rows.append([inline(c, refs) for c in cur_cells]); cur_cells = []
                i += 1
            flush_para()
            out += render_table(rows); out.append("")
        elif s.startswith("\\begin{figure}"):
            buf = []
            i += 1
            while "\\end{figure}" not in lines[i]:
                buf.append(lines[i]); i += 1
            blk = "\n".join(buf)
            img = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]*)\}", blk)
            anim = re.search(r"\\animategraphics(?:\[[^\]]*\])?\{[^{}]*\}\{([^{}]*)\}"
                             r"\{([^{}]*)\}\{([^{}]*)\}", blk)
            lab = re.search(r"\\label\{([^{}]*)\}", blk)
            blk_nolab = re.sub(r"\\label\{[^{}]*\}", "", blk)   # so caption regex stops correctly
            cap = re.search(r"\\caption\{(.*)\}", blk_nolab, re.S)
            num = refs.get(lab.group(1), "") if lab else ""
            flush_para()
            if img:
                out.append(f"![Figure {num}]({img.group(1)})"); out.append("")
            elif anim:
                f0, f1 = int(anim.group(2)), int(anim.group(3))
                out.append(f"![Figure {num}]({anim.group(1)}{(f0 + f1) // 2}.png)"); out.append("")
                out.append(f"*(animation, {f1 - f0 + 1} frames — plays in the PDF; "
                           "one representative frame is shown here.)*"); out.append("")
            if cap:
                ctxt = inline(re.sub(r"\s+", " ", cap.group(1)).strip(), refs)
                out.append(f"**Figure {num}.** {ctxt}"); out.append("")
        elif s.startswith("\\begin{enumerate}"):
            flush_para(); list_stack.append({"kind": "ol", "n": 0,
                                             "marker_emitted": True, "text": []})
        elif s.startswith("\\begin{itemize}"):
            flush_para(); list_stack.append({"kind": "ul", "n": 0,
                                             "marker_emitted": True, "text": []})
        elif s.startswith("\\end{enumerate}") or s.startswith("\\end{itemize}"):
            emit_item_marker(); list_stack.pop()
            if not list_stack:
                out.append("")
        elif s.startswith("\\item"):
            emit_item_marker()
            it = cur()
            it["n"] += 1
            it["text"] = [s[len("\\item"):].strip()]
            it["marker_emitted"] = False
        elif s.startswith("\\begin{center}") or s.startswith("\\end{center}"):
            flush_para()
        else:
            if list_stack and not cur()["marker_emitted"]:
                cur()["text"].append(s)
            else:
                para.append(s)
        i += 1
    flush_para()

    # collapse 3+ blank lines
    header = "# " + title + "\n\n"
    if subtitle:
        header += "*" + subtitle + "*\n\n"
    md = re.sub(r"\n{3,}", "\n\n", header + "\n".join(out)).rstrip() + "\n"
    open(dst, "w").write(md)
    print(f"wrote {dst}  ({md.count(chr(10))} lines, {fign} figures, {len(refs)} refs)")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "tutorial.tex"
    b = sys.argv[2] if len(sys.argv) > 2 else "tutorial.md"
    main(a, b)
