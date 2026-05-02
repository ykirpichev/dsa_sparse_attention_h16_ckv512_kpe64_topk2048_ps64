"""Render REPORT_TEMPLATE.md to REPORT_TEMPLATE.pdf.

Pipeline:
  1. Read the markdown.
  2. Find every ``` ```mermaid ... ``` `` block; render to SVG via the public
     Kroki service (https://kroki.io/mermaid/svg, deflate+base64 GET API).
     Embedded inline as raw SVG so weasyprint vectorises it.
  3. Convert remaining markdown to HTML via the `markdown` library
     (extensions: `extra` for tables/fenced code, `tables`,
     `fenced_code`, `attr_list`, `toc`).
  4. Wrap in an A4 HTML/CSS template.
  5. Render to PDF via WeasyPrint.

Re-creates the exact tooling used for the original
`docs(report): fix mermaid quoting and add rendered PDF` commit (a9d8eea1):
WeasyPrint A4 + Kroki Mermaid SVGs.
"""

from __future__ import annotations

import base64
import re
import sys
import urllib.request
import zlib
from pathlib import Path

import markdown as md_lib
import weasyprint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "REPORT_TEMPLATE.md"
PDF_PATH = PROJECT_ROOT / "REPORT_TEMPLATE.pdf"

KROKI_URL = "https://kroki.io"
MERMAID_BLOCK_RE = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)


def render_mermaid(source: str) -> str:
    """Render a Mermaid source block to inline SVG via Kroki (POST endpoint)."""
    url = f"{KROKI_URL}/mermaid/svg"
    req = urllib.request.Request(
        url,
        data=source.encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "text/plain",
            "Accept": "image/svg+xml",
            "User-Agent": "render_report.py (mlsys26-contest writeup builder)",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        svg = resp.read().decode("utf-8")
    # Strip XML prolog if present so it can sit inline inside the HTML.
    svg = re.sub(r"^<\?xml[^?]*\?>\s*", "", svg)
    # Make Kroki output responsive within the page width.
    svg = re.sub(
        r"<svg",
        '<svg style="max-width: 100%; height: auto;" preserveAspectRatio="xMidYMid meet"',
        svg,
        count=1,
    )
    return svg


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
@page {{
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
  @bottom-center {{
    content: counter(page) " / " counter(pages);
    font-family: "DejaVu Sans", Arial, sans-serif;
    font-size: 8pt;
    color: #888;
  }}
}}
body {{
  font-family: "DejaVu Sans", Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.40;
  color: #222;
}}
h1 {{ font-size: 18pt; margin: 0 0 0.4em 0; }}
h2 {{ font-size: 14pt; margin: 1.0em 0 0.4em 0; border-bottom: 1px solid #ccc; padding-bottom: 0.15em; }}
h3 {{ font-size: 12pt; margin: 0.8em 0 0.3em 0; }}
p  {{ margin: 0.35em 0; text-align: justify; }}
code {{ font-family: "DejaVu Sans Mono", Consolas, monospace; font-size: 9.5pt; background: #f4f4f4; padding: 0 3px; border-radius: 2px; }}
pre {{ background: #f6f6f6; padding: 8pt 10pt; border-radius: 4pt; overflow-x: auto; font-size: 9pt; line-height: 1.35; }}
pre code {{ background: transparent; padding: 0; }}
table {{
  border-collapse: collapse;
  margin: 0.6em 0;
  font-size: 9.5pt;
  width: 100%;
}}
th, td {{
  border: 1px solid #c8c8c8;
  padding: 3pt 6pt;
  text-align: left;
  vertical-align: top;
}}
th {{ background: #ececec; font-weight: 600; }}
blockquote {{
  border-left: 3px solid #c0c0c0;
  margin: 0.5em 0;
  padding: 0.1em 0.8em;
  color: #555;
  background: #fafafa;
}}
a {{ color: #1f4e79; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.mermaid-svg {{
  display: block;
  margin: 0.6em auto;
  max-width: 100%;
}}
.page-break {{ break-before: page; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> int:
    md_text = MD_PATH.read_text(encoding="utf-8")

    # 1. Render mermaid blocks first; substitute back into the markdown as raw HTML.
    blocks = list(MERMAID_BLOCK_RE.finditer(md_text))
    print(f"render_report: found {len(blocks)} mermaid block(s); rendering via Kroki...",
          file=sys.stderr)
    rendered = {}
    for idx, m in enumerate(blocks, 1):
        source = m.group(1).strip()
        try:
            svg = render_mermaid(source)
        except Exception as exc:
            print(f"  [{idx}/{len(blocks)}] FAILED: {exc!r}", file=sys.stderr)
            return 2
        # Wrap in a div so we can style it.
        rendered[m.start()] = f'<div class="mermaid-svg">{svg}</div>'
        print(f"  [{idx}/{len(blocks)}] ok ({len(svg)} bytes)", file=sys.stderr)

    # Substitute in reverse to preserve byte offsets.
    out = md_text
    for m in reversed(blocks):
        out = out[: m.start()] + rendered[m.start()] + out[m.end():]

    # 2. Markdown -> HTML.
    html_body = md_lib.markdown(
        out,
        extensions=[
            "extra",       # tables, fenced code, def lists, attr_list
            "tables",
            "fenced_code",
            "attr_list",
            "toc",
            "sane_lists",
        ],
        output_format="html5",
    )

    # 3. Wrap in template.
    title = "DSA Sparse Attention - Process Write-Up"
    html = HTML_TEMPLATE.format(title=title, body=html_body)

    # 4. Render to PDF.
    print(f"render_report: rendering PDF to {PDF_PATH} ...", file=sys.stderr)
    weasyprint.HTML(string=html, base_url=str(PROJECT_ROOT)).write_pdf(str(PDF_PATH))
    print(f"render_report: wrote {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
