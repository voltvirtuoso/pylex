"""Build docs/index.html: one scrollable page from docs/*.md and CHANGELOG.md.

    pip install markdown      # or: pip install -e ".[docs]"
    python scripts/build_docs.py            # write docs/index.html
    python scripts/build_docs.py --check    # exit 1 if the file is out of date (used in CI)
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = DOCS / "index.html"
SITE = "https://voltvirtuoso.github.io/pylex/"
BLOB = "https://github.com/voltvirtuoso/pylex/blob/main/docs/"
SUMMARY = (
    "Pylex is a Python OCR library and command-line tool that makes scanned PDFs searchable and extracts "
    "text, confidence and position from images. It runs on CPU and NVIDIA GPU and installs from GitHub."
)
# One-line description of each doc for llms.txt (kept next to the page list so they cannot drift apart).
DESCRIPTIONS = {
    "overview": "What Pylex is, when to use it, and when to choose something else.",
    "installation": "Install profiles for CPU, Intel, NVIDIA CUDA 12 and 13; GPU checks; TPU and other hardware status.",
    "quickstart": "Searchable PDF from the command line and from Python in five minutes.",
    "python-api": "Python SDK reference: extract_text, create_searchable_document, Config, callbacks, GPU use.",
    "cli": "Every command line option, defaults, exit codes, recipes, and pylex doctor.",
    "how-it-works": "Rendering, tiling, vertical-text retry, and how the invisible text layer is built.",
    "accuracy-and-benchmarks": "Choosing settings, recorded measurements, and a reproducible comparison harness.",
    "limitations": "What Pylex does not do, script and font limits, hardware support.",
    "troubleshooting": "Fixes for install, GPU, model-download, and text-layer problems.",
    "changelog": "Release history.",
}

# (group, [(slug, source file, nav title, kind)]). kind: sdk / cli filter the two main docs.
PAGES = [
    ("Start", [("overview", "index.md", "Overview", "all"), ("installation", "installation.md", "Installation", "all"),
               ("quickstart", "quickstart.md", "Quickstart", "all")]),
    ("Python SDK", [("python-api", "python-api.md", "Python SDK reference", "sdk")]),
    ("Command line", [("cli", "cli.md", "CLI reference", "cli")]),
    ("Concepts", [("how-it-works", "how-it-works.md", "How it works", "all"),
                  ("accuracy-and-benchmarks", "accuracy-and-benchmarks.md", "Accuracy and benchmarks", "all"),
                  ("limitations", "limitations.md", "Limitations", "all")]),
    ("Help", [("troubleshooting", "troubleshooting.md", "Troubleshooting", "all"),
              ("changelog", "../CHANGELOG.md", "Changelog", "all")]),
]
SLUGS = {src.split("/")[-1]: slug for _g, items in PAGES for slug, src, _t, _k in items}
SLUGS["index.md"] = "overview"
PICKER = (
    '<div id="pick"><label for="hw">Choose your hardware</label><select id="hw">'
    '<option value="cpu">Any CPU</option><option value="g13">NVIDIA GPU, CUDA 13+</option>'
    '<option value="g12">NVIDIA GPU, CUDA 12.x</option><option value="intel">Intel CPU (OpenVINO)</option>'
    '<option value="auto">Not sure / Apple / AMD / TPU</option></select>'
    '<pre><code></code></pre><p class="n"></p></div>'
)


def version() -> str:
    match = re.search(r'__version__ = "([^"]+)"', (ROOT / "src/pylex/__init__.py").read_text(encoding="utf-8"))
    return match.group(1) if match else ""


def render(slug: str, text: str):
    def make_slug(value, sep):
        return f"{slug}-{markdown.extensions.toc.slugify(value, sep)}"

    md = markdown.Markdown(extensions=["fenced_code", "tables", "toc"],
                           extension_configs={"toc": {"slugify": make_slug, "toc_depth": "2"}})
    body = md.convert(text)
    heads = [(t["id"], html.unescape(re.sub("<[^>]+>", "", t["name"]))) for t in md.toc_tokens[0]["children"]] \
        if md.toc_tokens and md.toc_tokens[0]["level"] == 1 else []
    title_id = md.toc_tokens[0]["id"] if md.toc_tokens else ""
    body = body.replace(f'id="{title_id}"', "", 1)

    def fix(m):
        href = m.group(1)
        if href.startswith("#"):
            return f'href="#{slug}-{href[1:]}"'
        doc, _, frag = href.partition("#")
        target = SLUGS.get(doc.split("/")[-1])
        if target and doc.endswith(".md"):
            return f'href="#{target}-{frag}"' if frag else f'href="#{target}"'
        return m.group(0)

    body = re.sub(r'href="([^"]+)"', fix, body)
    body = re.sub(r"<table>", '<div class="tw"><table>', body).replace("</table>", "</table></div>")
    return body, heads


def build() -> str:
    nav, sections = [], []
    for group, items in PAGES:
        nav.append(f"<h4>{group}</h4>")
        for slug, src, title, kind in items:
            body, heads = render(slug, (DOCS / src).read_text(encoding="utf-8"))
            if slug == "installation":
                body = body.replace("</h1>", "</h1>", 1)
                body = re.sub(r"(<h1[^>]*>.*?</h1>\s*<p>.*?</p>)", lambda m: m.group(1) + PICKER, body, count=1, flags=re.DOTALL)
            subs = "".join(f'<a class="s" href="#{i}">{html.escape(n)}</a>' for i, n in heads) if slug != "changelog" else ""
            nav.append(f'<div class="grp" data-kind="{kind}"><a class="d" href="#{slug}">{html.escape(title)}</a>{subs}</div>')
            label = {"sdk": "Python SDK", "cli": "Command line"}.get(kind, "")
            tag = f'<div class="kind">{label}</div>' if label else ""
            sections.append(f'<section id="{slug}" data-kind="{kind}">{tag}{body}</section>')
    page = (ROOT / "scripts/docs_template.html").read_text(encoding="utf-8")
    return page.replace("{{NAV}}", "\n".join(nav)).replace("{{CONTENT}}", "\n".join(sections)).replace("{{VERSION}}", version())


def llms_txt() -> str:
    """llms.txt: a short, link-based map of the docs for AI tools (https://llmstxt.org)."""
    lines = ["# Pylex", "", f"> {SUMMARY}", "", "## Docs", ""]
    for _group, items in PAGES:
        for slug, src, title, _kind in items:
            if slug == "changelog":
                continue
            url = BLOB + src.split("/")[-1]
            lines.append(f"- [{title}]({url}): {DESCRIPTIONS[slug]}")
    lines += ["", "## Optional", "",
              f"- [Single-page docs site]({SITE}): All pages in one scrollable page.",
              "- [Repository](https://github.com/voltvirtuoso/pylex): Source code, issues, license (MIT).",
              f"- [Full text of the docs]({SITE}llms-full.txt): Every page below in one plain-text file.", ""]
    return "\n".join(lines)


def llms_full() -> str:
    parts = [f"# Pylex documentation (full text)\n\n> {SUMMARY}\n"]
    for _group, items in PAGES:
        for slug, src, _title, _kind in items:
            if slug != "changelog":
                parts.append((DOCS / src).read_text(encoding="utf-8").strip() + "\n")
    return "\n---\n\n".join(parts)


def sitemap() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{SITE}</loc></url>\n"
        "</urlset>\n"
    )


def outputs() -> dict[Path, str]:
    return {OUT: build(), DOCS / "llms.txt": llms_txt(), DOCS / "llms-full.txt": llms_full(), DOCS / "sitemap.xml": sitemap()}


def main() -> int:
    files = outputs()
    if "--check" in sys.argv:
        stale = [p.name for p, text in files.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            print(f"docs/{', docs/'.join(stale)} out of date; run: python scripts/build_docs.py")
            return 1
        return 0
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({max(len(text) // 1024, 1)} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
