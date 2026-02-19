#!/usr/bin/env python3
"""
Compose slide_*.html files into a single project_presentation.html (HTML-only)

This script reads files named slide_<n>.html in the presentation directory,
orders them by <n>, and writes a combined project_presentation.html with
previous/next navigation. Run from the presentation folder.
"""
from pathlib import Path
import re
import argparse
import sys
import html as html_lib


def extract_head_body(text: str):
    head_m = re.search(r"<head[^>]*>(.*?)</head>", text, re.S | re.I)
    body_m = re.search(r"<body[^>]*>(.*?)</body>", text, re.S | re.I)
    head = head_m.group(1).strip() if head_m else ""
    body = body_m.group(1).strip() if body_m else text.strip()
    return head, body


def read_text_fallback(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    for enc in ("utf-8", "utf-8-sig", "cp1253"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("cp1253", errors="replace")


def make_blank_slide():
    return (
        '<div style="width:1280px;height:720px;display:flex;align-items:center;justify-content:center;">'
        '<h2 style="color:#9CA3AF;">Blank slide</h2></div>'
    )


def compose(slides_dir: Path, out_name: str = "project_presentation.html"):
    files = list(slides_dir.glob("slide_*.html"))
    slides = {}
    number_re = re.compile(r"slide_(\d+)")

    for f in files:
        m = number_re.search(f.name)
        if not m:
            continue
        idx = int(m.group(1))
        slides[idx] = f

    if not slides:
        print("No slide_*.html files found in", slides_dir)
        return None

    max_idx = max(slides.keys())
    slide_blocks = []
    for i in range(1, max_idx + 1):
        if i in slides:
            slide_blocks.append(read_text_fallback(slides[i]))
        else:
            slide_blocks.append(
                f'<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Blank</title></head><body>{make_blank_slide()}</body></html>'
            )

    nav_css = (
        '<style>'
        'body{margin:0;font-family:"Open Sans",sans-serif;background:#e8e8e8}'
        '#slideshow{position:relative;width:1280px;height:720px;margin:20px auto;overflow:hidden;background:#fff}'
        '.slides{width:100%;height:100%}'
        '.slide{display:none;width:100%;height:100%}'
        '.slide.active{display:block}'
        '.slide iframe{width:100%;height:100%;border:0;display:block}'
        '#prev,#next{position:absolute;top:50%;transform:translateY(-50%);background:rgba(0,0,0,.5);color:#fff;border:0;padding:12px 16px;cursor:pointer;font-size:20px;border-radius:6px}'
        '#prev{left:10px}#next{right:10px}'
        '#indicator{position:absolute;right:16px;bottom:12px;background:rgba(255,255,255,.9);padding:6px 10px;border-radius:6px;font-weight:600}'
        '</style>'
    )

    nav_js = """<script>
(function(){
  var slides=document.querySelectorAll('.slide');
  var current=0;
  function show(n){
    if(!slides.length)return;
    for(var i=0;i<slides.length;i++)slides[i].classList.remove('active');
    slides[n].classList.add('active');
    var ind=document.getElementById('indicator');
    if(ind)ind.textContent=(n+1)+' / '+slides.length;
  }
  function next(){current=(current+1)%slides.length;show(current);}
  function prev(){current=(current-1+slides.length)%slides.length;show(current);}
  document.getElementById('next').onclick=next;
  document.getElementById('prev').onclick=prev;
  document.onkeydown=function(e){if(e.key==='ArrowRight')next();if(e.key==='ArrowLeft')prev();};
  show(0);
  window.showSlide=function(n){if(n>=0&&n<slides.length){current=n;show(n);}};
})();
</script>"""

    out_path = slides_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>')
        f.write('<title>Fictitious Play και Reinforcement Learning</title>')
        f.write(nav_css)
        f.write('</head><body><!-- Slides 1..%d --><div id="slideshow"><div class="slides">' % max_idx)
        for idx, block in enumerate(slide_blocks, 1):
            srcdoc = html_lib.escape(block, quote=True)
            f.write(f'<section class="slide" data-index="{idx}"><iframe srcdoc="{srcdoc}"></iframe></section>')
        f.write('</div><button id="prev" aria-label="Previous">◀</button><button id="next" aria-label="Next">▶</button>')
        f.write('<div id="indicator"></div></div>')
        f.write(nav_js)
        f.write('</body></html>')

    print(f"Wrote: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--slides-dir", default=".", help="directory with slide_*.html")
    parser.add_argument("-o", "--out", default="project_presentation.html", help="output file")
    args = parser.parse_args()
    slides_dir = Path(args.slides_dir).resolve()
    if not slides_dir.exists():
        print("Directory does not exist:", slides_dir)
        sys.exit(1)
    compose(slides_dir, args.out)


if __name__ == "__main__":
    main()
