#!/usr/bin/env python3
"""
md_to_pdf.py — Convert Markdown with LaTeX math to PDF
=======================================================

Converts Markdown files containing $...$ (inline) and $$...$$ (display)
LaTeX math into beautifully rendered PDFs.

Pipeline:
  1. markdown-it-py  → Markdown to HTML (tables, code, lists, etc.)
  2. KaTeX (CDN)     → LaTeX math rendering (loaded in the HTML)
  3. Chrome headless  → HTML to PDF

Usage:
    python md_to_pdf.py input.md                   # → input.pdf
    python md_to_pdf.py input.md output.pdf         # → output.pdf
    python md_to_pdf.py input.md --html-only        # → input.html (no PDF)

Requirements:
    pip install markdown-it-py
"""

import re
import sys
import os
import subprocess
from html import escape as html_escape

# ---------------------------------------------------------------------------
# 1. Math Protection — extract $...$ and $$...$$ before markdown-it parsing
# ---------------------------------------------------------------------------

def protect_and_extract_math(text):
    """
    Extract math expressions from markdown, replacing them with HTML-comment
    placeholders that survive markdown-it processing without mangling.

    This prevents issues like:
     - '_' inside math being treated as emphasis
     - '|' inside math breaking table cells
     - '*' inside math becoming bold/italic

    Returns (processed_text, math_store)
    """
    math_store = []

    # --- Phase 1: protect code blocks and inline code from math scanning ---
    code_blocks = []
    def _save_code_block(m):
        i = len(code_blocks); code_blocks.append(m.group(0))
        return f'\x00CBLK{i}\x00'
    text = re.sub(r'```[\s\S]*?```', _save_code_block, text)

    inline_codes = []
    def _save_inline_code(m):
        i = len(inline_codes); inline_codes.append(m.group(0))
        return f'\x00ICODE{i}\x00'
    text = re.sub(r'`[^`\n]+?`', _save_inline_code, text)

    # --- Phase 2: extract display math  $$...$$ ---
    def _save_display(m):
        i = len(math_store)
        math_store.append(('display', m.group(1).strip()))
        return f'\n\n<!--DMATH_{i}-->\n\n'
    text = re.sub(r'\$\$([\s\S]*?)\$\$', _save_display, text)

    # --- Phase 3: extract inline math  $...$ ---
    # Match $...$ that:
    #   - doesn't start/end with space (standard LaTeX convention)
    #   - doesn't cross newlines
    #   - isn't preceded by a backslash (escaped dollar)
    def _save_inline(m):
        i = len(math_store)
        math_store.append(('inline', m.group(1)))
        return f'<!--IMATH_{i}-->'
    text = re.sub(r'(?<!\\)\$([^\n$]+?)\$', _save_inline, text)

    # --- Phase 4: restore code blocks and inline code ---
    for i, code in enumerate(inline_codes):
        text = text.replace(f'\x00ICODE{i}\x00', code)
    for i, code in enumerate(code_blocks):
        text = text.replace(f'\x00CBLK{i}\x00', code)

    return text, math_store


def restore_math_in_html(html, math_store):
    """Replace placeholders in rendered HTML with KaTeX-renderable elements."""
    for i, (mode, content) in enumerate(math_store):
        escaped = html_escape(content)
        if mode == 'display':
            tag = f'<div class="math-block">{escaped}</div>'
            html = html.replace(f'<!--DMATH_{i}-->', tag)
        else:
            tag = f'<span class="math-inline">{escaped}</span>'
            html = html.replace(f'<!--IMATH_{i}-->', tag)
    return html


# ---------------------------------------------------------------------------
# 2. Markdown → HTML
# ---------------------------------------------------------------------------

def markdown_to_html(md_text):
    """Convert markdown text to HTML body, with math properly preserved."""
    from markdown_it import MarkdownIt

    protected_text, math_store = protect_and_extract_math(md_text)

    md = MarkdownIt('default', {
        'html': True,
        'linkify': False,
        'typographer': False,
        'breaks': False,
    })
    html_body = md.render(protected_text)

    html_body = restore_math_in_html(html_body, math_store)
    return html_body


# ---------------------------------------------------------------------------
# 3. Full HTML document template
# ---------------------------------------------------------------------------

KATEX_VERSION = "0.16.21"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>{title}</title>

<!-- KaTeX CSS -->
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@{katex_version}/dist/katex.min.css"
      crossorigin="anonymous">

<!-- KaTeX JS -->
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@{katex_version}/dist/katex.min.js"
        crossorigin="anonymous"></script>

<style>
/* ============================================================
   Page layout
   ============================================================ */
@page {{
    size: A4;
    margin: 2cm 2.2cm;
}}

@media print {{
    body {{ margin: 0; }}
    h2 {{ page-break-before: auto; }}
    pre, table, .math-block {{ page-break-inside: avoid; }}
}}

/* ============================================================
   Body & Typography
   ============================================================ */
body {{
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.65;
    color: #1a1a1a;
    max-width: 100%;
    margin: 0 auto;
    padding: 0;
}}

h1 {{
    font-size: 1.8em;
    border-bottom: 2px solid #2c3e50;
    padding-bottom: 0.3em;
    margin-top: 0.5em;
    color: #2c3e50;
}}

h2 {{
    font-size: 1.45em;
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 0.25em;
    margin-top: 1.8em;
    color: #2c3e50;
}}

h3 {{
    font-size: 1.2em;
    margin-top: 1.4em;
    color: #34495e;
}}

h4 {{
    font-size: 1.05em;
    margin-top: 1.2em;
    color: #34495e;
}}

a {{
    color: #2980b9;
    text-decoration: none;
}}

blockquote {{
    border-left: 4px solid #3498db;
    margin: 1em 0;
    padding: 0.5em 1em;
    background: #f0f7fd;
    color: #2c3e50;
}}

hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 2em 0;
}}

/* ============================================================
   Tables
   ============================================================ */
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 0.92em;
}}

th, td {{
    border: 1px solid #cbd5e0;
    padding: 0.5em 0.75em;
    text-align: left;
}}

thead th {{
    background-color: #edf2f7;
    font-weight: 600;
    color: #2d3748;
}}

tbody tr:nth-child(even) {{
    background-color: #f7fafc;
}}

/* ============================================================
   Code
   ============================================================ */
pre {{
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 1em;
    overflow-x: auto;
    font-size: 0.88em;
    line-height: 1.5;
}}

code {{
    font-family: "Cascadia Code", "Fira Code", "Consolas", "Monaco", monospace;
    font-size: 0.9em;
}}

:not(pre) > code {{
    background: #f0f0f0;
    padding: 0.15em 0.35em;
    border-radius: 3px;
    color: #c7254e;
}}

/* ============================================================
   Math
   ============================================================ */
.math-block {{
    text-align: center;
    margin: 1.2em 0;
    overflow-x: auto;
}}

.math-inline {{
    /* KaTeX will fill this element */
}}

/* ============================================================
   Lists
   ============================================================ */
ul, ol {{
    padding-left: 1.8em;
}}

li {{
    margin-bottom: 0.25em;
}}
</style>
</head>
<body>
{content}

<!-- Render all math elements with KaTeX after page loads -->
<script>
// Wait for KaTeX script to load (it's deferred)
document.addEventListener('DOMContentLoaded', function() {{
    function renderAllMath() {{
        document.querySelectorAll('.math-inline').forEach(function(el) {{
            try {{
                katex.render(el.textContent, el, {{
                    displayMode: false,
                    throwOnError: false,
                    strict: false
                }});
            }} catch(e) {{ console.error('KaTeX inline error:', e, el.textContent); }}
        }});
        document.querySelectorAll('.math-block').forEach(function(el) {{
            try {{
                katex.render(el.textContent, el, {{
                    displayMode: true,
                    throwOnError: false,
                    strict: false
                }});
            }} catch(e) {{ console.error('KaTeX block error:', e, el.textContent); }}
        }});
    }}

    // KaTeX is loaded with defer, so it's available by DOMContentLoaded
    if (typeof katex !== 'undefined') {{
        renderAllMath();
    }} else {{
        // Fallback: wait a bit for script loading
        setTimeout(renderAllMath, 500);
    }}
}});
</script>
</body>
</html>"""


def generate_full_html(md_file_path):
    """Read a markdown file and produce a complete HTML string."""
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    html_body = markdown_to_html(md_text)
    title = os.path.splitext(os.path.basename(md_file_path))[0].replace('_', ' ')

    return HTML_TEMPLATE.format(
        title=html_escape(title),
        katex_version=KATEX_VERSION,
        content=html_body,
    )


# ---------------------------------------------------------------------------
# 4. HTML → PDF  (Chrome headless)
# ---------------------------------------------------------------------------

def find_chrome():
    """Locate Chrome / Chromium executable on this system."""
    candidates = [
        os.path.join(os.environ.get('ProgramFiles', ''), 'Google', 'Chrome', 'Application', 'chrome.exe'),
        os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'Google', 'Chrome', 'Application', 'chrome.exe'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google', 'Chrome', 'Application', 'chrome.exe'),
        os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
        os.path.join(os.environ.get('ProgramFiles', ''), 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def html_to_pdf(html_path, pdf_path, chrome_path=None):
    """Convert an HTML file to PDF via Chrome headless mode."""
    if chrome_path is None:
        chrome_path = find_chrome()
    if chrome_path is None:
        raise FileNotFoundError(
            "Could not find Chrome or Edge.  Install Google Chrome or pass "
            "the path explicitly with --chrome."
        )

    abs_html = os.path.abspath(html_path).replace('\\', '/')
    abs_pdf  = os.path.abspath(pdf_path)

    cmd = [
        chrome_path,
        '--headless',
        '--disable-gpu',
        '--no-sandbox',
        f'--print-to-pdf={abs_pdf}',
        '--no-pdf-header-footer',          # Chrome 112+  (silently ignored on older)
        '--run-all-compositor-stages-before-draw',
        f'file:///{abs_html}',
    ]

    print(f"  Chrome: {chrome_path}")
    print(f"  HTML → PDF …")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        # Retry without --no-pdf-header-footer for older Chrome
        cmd_retry = [c for c in cmd if c != '--no-pdf-header-footer']
        result = subprocess.run(cmd_retry, capture_output=True, text=True, timeout=60)

    if not os.path.isfile(abs_pdf):
        print("  ⚠ PDF file was not created. Chrome stderr:")
        print(result.stderr)
        sys.exit(1)

    size_kb = os.path.getsize(abs_pdf) / 1024
    print(f"  PDF saved: {abs_pdf}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert Markdown (with LaTeX math) to PDF')
    parser.add_argument('input', help='Input .md file')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output .pdf file  (default: same name as input)')
    parser.add_argument('--html-only', action='store_true',
                        help='Generate .html only, skip PDF conversion')
    parser.add_argument('--chrome', default=None,
                        help='Path to Chrome / Edge executable')
    args = parser.parse_args()

    md_path = args.input
    if not os.path.isfile(md_path):
        print(f"Error: file not found — {md_path}")
        sys.exit(1)

    base = os.path.splitext(md_path)[0]
    html_path = base + '.html'
    pdf_path  = args.output or (base + '.pdf')

    # Step 1 — Markdown → HTML
    print("[1/2] Converting Markdown → HTML …")
    full_html = generate_full_html(md_path)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"  HTML saved: {html_path}")

    if args.html_only:
        print("Done (--html-only).")
        return

    # Step 2 — HTML → PDF
    print("[2/2] Converting HTML → PDF …")
    html_to_pdf(html_path, pdf_path, chrome_path=args.chrome)
    print("Done!")


if __name__ == '__main__':
    main()

# COMO USAR:

# # Gerar PDF (cria também o .html intermediário)
# python md_to_pdf.py SMX_perturbation_theory.md

# # Escolher nome do PDF de saída
# python md_to_pdf.py SMX_perturbation_theory.md meu_arquivo.pdf

# # Gerar apenas o HTML (para preview no navegador)
# python md_to_pdf.py SMX_perturbation_theory.md --html-only