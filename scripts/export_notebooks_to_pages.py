#!/usr/bin/env python3
"""将 notebooks/*.ipynb 转为可在 GitHub Pages 直接查看的静态 HTML。

说明：
- 不依赖 jupyter/nbconvert，使用标准库解析 JSON。
- 目标是“可读性优先”的教学展示页面。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "docs" / "notebooks"

STYLE = """
:root {
  --bg: #f6f8fb;
  --text: #17202a;
  --muted: #5c6874;
  --panel: #ffffff;
  --border: #dfe5ee;
  --code-bg: #101418;
  --code-fg: #f2f5f9;
  --accent: #0b6a4f;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.65;
}
.container {
  width: min(980px, 92vw);
  margin: 0 auto;
  padding: 24px 0 40px;
}
.topbar {
  margin-bottom: 14px;
}
.topbar a {
  color: var(--accent);
  text-decoration: none;
}
h1 {
  margin: 0 0 6px;
  font-size: 30px;
}
.sub {
  margin: 0 0 18px;
  color: var(--muted);
}
.cell {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  margin-bottom: 12px;
}
.badge {
  display: inline-block;
  font-size: 12px;
  color: var(--accent);
  border: 1px solid #b7d7cc;
  border-radius: 999px;
  padding: 2px 8px;
  margin-bottom: 8px;
}
pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
}
.code {
  background: var(--code-bg);
  color: var(--code-fg);
  border-radius: 10px;
  padding: 10px;
  overflow: auto;
}
.markdown {
  background: #ffffff;
}
.note {
  margin-top: 20px;
  color: var(--muted);
  font-size: 14px;
}
"""


def join_source(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    if isinstance(value, str):
        return value
    return ""


def render_markdown_cell(source: str) -> str:
    # 为了零依赖，先以可读的 pre 展示 markdown 原文。
    # 不做完整 markdown 渲染，避免引入第三方库。
    safe = html.escape(source)
    return (
        '<section class="cell markdown">'
        '<div class="badge">Markdown</div>'
        f"<pre>{safe}</pre>"
        "</section>"
    )


def render_code_cell(source: str, outputs: list[object]) -> str:
    safe_src = html.escape(source)
    out_blocks: list[str] = []

    for output in outputs:
        if not isinstance(output, dict):
            continue

        text = ""
        if "text" in output:
            text = join_source(output.get("text"))
        elif "data" in output and isinstance(output["data"], dict):
            if "text/plain" in output["data"]:
                text = join_source(output["data"]["text/plain"])

        if text:
            out_blocks.append(
                '<div style="margin-top:8px;">'
                '<div class="badge">Output</div>'
                f'<pre class="code">{html.escape(text)}</pre>'
                "</div>"
            )

    out_html = "".join(out_blocks)

    return (
        '<section class="cell">'
        '<div class="badge">Code</div>'
        f'<pre class="code">{safe_src}</pre>'
        f"{out_html}"
        "</section>"
    )


def convert_notebook(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])

    body_parts: list[str] = []

    title = path.stem
    if cells and isinstance(cells[0], dict) and cells[0].get("cell_type") == "markdown":
        first = join_source(cells[0].get("source", "")).strip().splitlines()
        if first:
            title = first[0].lstrip("#").strip() or title

    for cell in cells:
        if not isinstance(cell, dict):
            continue
        ctype = cell.get("cell_type")
        source = join_source(cell.get("source", ""))

        if ctype == "markdown":
            body_parts.append(render_markdown_cell(source))
        elif ctype == "code":
            body_parts.append(render_code_cell(source, cell.get("outputs", [])))

    html_doc = f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>{STYLE}</style>
</head>
<body>
  <main class=\"container\">
    <div class=\"topbar\"><a href=\"../index.html#notebooks\">← 返回 Pages 首页（Notebook 区域）</a></div>
    <h1>{html.escape(title)}</h1>
    <p class=\"sub\">文件：{html.escape(path.name)}（静态导出版）</p>
    {''.join(body_parts)}
    <p class=\"note\">说明：这是为了 GitHub Pages 展示生成的静态版本。原始可交互版本请在仓库 notebooks/ 目录中打开 .ipynb。</p>
  </main>
</body>
</html>
"""

    out = OUT_DIR / f"{path.stem}.html"
    out.write_text(html_doc, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not files:
        print("未找到 notebooks/*.ipynb")
        return

    for f in files:
        convert_notebook(f)
        print(f"已生成: docs/notebooks/{f.stem}.html")


if __name__ == "__main__":
    main()
