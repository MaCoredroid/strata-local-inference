"""Render the editable article into dependency-free HTML. Requires Pandoc."""
import html
from pathlib import Path
import re
import subprocess

SITE = Path(__file__).resolve().parent
source = (SITE / "article.md").read_text(encoding="utf-8")
heading, _, article = source.partition("\n\n")
if not heading.startswith("# ") or not article:
    raise ValueError("article.md must start with a title")
title = html.escape(heading.removeprefix("# "))
display_title = title.replace("RTX 5090", "RTX&nbsp;5090").replace("DGX Spark", "DGX&nbsp;Spark")

diagram = """<figure class="switching">
  <div class="origin">OpenCode media request</div>
  <div class="routes">
    <div><strong>RTX 5090</strong><span>Switch chat to Qwen Image</span></div>
    <div><strong>DGX Spark</strong><span>Switch chat to MiniMax H3</span></div>
  </div>
  <figcaption>Wait for idle chat → render and save → restore the prior model → return the attachment.</figcaption>
</figure>"""
article, count = re.subn(r"```mermaid\n.*?\n```", lambda _: diagram, article, flags=re.S)
if count != 1:
    raise ValueError("Expected one media switching diagram")
body = subprocess.run(
    ["pandoc", "--from=gfm+raw_html", "--to=html5", "--shift-heading-level-by=1"],
    input=article, check=True, capture_output=True, text=True,
).stdout
body = body.replace("<table>", '<div class="table-wrap" tabindex="0" role="region" aria-label="Article table"><table>')
body = body.replace("</table>", "</table></div>")
(SITE / "index.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="How Strata serves Qwen 27B and Flash-Next on two local GPUs and switches them to image and video generation.">
  <title>{title} · Strata</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <a class="skip-link" href="#article">Skip to article</a>
  <header class="masthead">
    <div class="masthead-inner">
      <span class="wordmark">Strata</span>
      <a class="series" href="https://macoredroid.github.io/collective-intelligence/">Collective Intelligence companion</a>
    </div>
  </header>
  <main id="article">
    <article>
      <header class="hero">
        <div class="eyebrow">Field notes · Local inference</div>
        <h1>{display_title}</h1>
        <p class="deck">An RTX 5090, a DGX Spark, and the difference between a faster model and a faster turn.</p>
        <nav aria-label="In this article" class="jump-links">
          <a href="#what-strata-owns">Architecture</a>
          <a href="#what-the-experiments-changed">Results</a>
          <a href="#try-the-serving-profiles-on-your-hardware">Run it locally</a>
        </nav>
      </header>
      <div class="article-body">
{body}      </div>
    </article>
  </main>
  <footer>Strata engineering notes</footer>
</body>
</html>
""", encoding="utf-8")
