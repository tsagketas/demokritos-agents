# -*- coding: utf-8 -*-
"""Insert 3 Zero-shot heatmap slides after slide 14 and renumber following slides."""
import html as html_lib
from pathlib import Path

SLIDES_DIR = Path(__file__).resolve().parent
HTML_PATH = SLIDES_DIR / "project_presentation.html"

# One slide HTML template: placeholders TITLE, IMG_SRC
TEMPLATE = """<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&amp;display=swap" rel="stylesheet"/>
<style>
body {{ margin:0; font-family: 'Montserrat', sans-serif; background:#fff; }}
.slide-container {{ width: 1280px; height: 720px; display: flex; flex-direction: column; background: white; }}
.header {{ padding: 24px 60px; border-bottom: 1px solid #e2e8f0; }}
.title {{ font-family: 'Montserrat', sans-serif; font-size: 2rem; font-weight: 700; color: #1e3a8a; margin: 0; }}
.main-content {{ flex: 1; padding: 20px 60px; display: flex; align-items: center; justify-content: center; min-height: 0; }}
.main-content img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
</style>
</head>
<body>
<div class="slide-container">
<header class="header"><h1 class="title">{title}</h1></header>
<div class="main-content">
<img src="{img_src}" alt="{title}"/>
</div>
</div>
</body>
</html>"""

# Order: Σενάριο 1 (80/20), Σενάριο 2 (LOSO), Σενάριο 3 (80/20 + PCA)
SLIDES = [
    ("Zero-shot Results Heatmap – Σενάριο 1", "../workflows/iemocap_80_20/results/Metrics_graphs/Zero_shot_heatmap.png"),
    ("Zero-shot Results Heatmap – Σενάριο 2", "../workflows/iemocap_loso/results/Metrics_graphs/Zero_shot_heatmap.png"),
    ("Zero-shot Results Heatmap – Σενάριο 3", "../workflows/iemocap_pca/results/Metrics_graphs/Zero_shot_heatmap.png"),
]

def main():
    text = HTML_PATH.read_text(encoding="utf-8")

    # Build the 3 new slide sections with escaped srcdoc
    new_sections = []
    for title, img_src in SLIDES:
        content = TEMPLATE.format(title=title, img_src=img_src)
        escaped = html_lib.escape(content, quote=True)
        new_sections.append(
            '<section class="slide" data-index="{}">\n<iframe srcdoc="{}"></iframe>\n</section>'.format(
                14 + len(new_sections) + 1, escaped
            )
        )

    insertion = "\n".join(new_sections) + '\n<section class="slide" data-index="18">'

    marker = '</section>\n<section class="slide" data-index="15">'
    if marker not in text:
        raise SystemExit("Marker not found in HTML")
    text = text.replace(marker, "</section>\n" + insertion, 1)

    # Rename old slide 16 -> 19, 17 -> 20, 18 -> 21 (only the OLD ones; new 15,16,17,18 stay)
    # Replace from last to first so positions don't shift.
    def replace_second_occurrence(s, old, new):
        first = s.find(old)
        if first == -1:
            return s
        second = s.find(old, first + 1)
        if second == -1:
            return s
        return s[:second] + new + s[second + len(old):]

    tag_18 = '<section class="slide" data-index="18">'
    tag_17 = '<section class="slide" data-index="17">'
    tag_16 = '<section class="slide" data-index="16">'
    text = replace_second_occurrence(text, tag_18, '<section class="slide" data-index="21">')
    text = replace_second_occurrence(text, tag_17, '<section class="slide" data-index="20">')
    text = replace_second_occurrence(text, tag_16, '<section class="slide" data-index="19">')

    # Update comment
    text = text.replace("<!-- Slides in order (1..18) -->", "<!-- Slides in order (1..21) -->", 1)

    HTML_PATH.write_text(text, encoding="utf-8")
    print("Inserted 3 heatmap slides after slide 14 and renumbered. Total slides: 21.")

if __name__ == "__main__":
    main()
