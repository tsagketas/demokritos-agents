#!/usr/bin/env python3
"""
Convert project_presentation.html to PDF — όπως φαίνεται στο browser.

Χρήση: pip install playwright img2pdf && playwright install chromium
Μετά: python html_to_pdf.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HTML_FILE = SCRIPT_DIR / "game-learning" / "project_presentation.html"
OUTPUT_PDF = SCRIPT_DIR / "game-learning" / "project_presentation.pdf"

HIDE_NAV_CSS = "#prev, #next, #indicator { display: none !important; }"


async def capture_slides_to_pdf():
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Playwright not installed:")
        print("  pip install playwright img2pdf")
        print("  playwright install chromium")
        sys.exit(1)
    try:
        import img2pdf
    except ImportError:
        print("img2pdf not installed: pip install img2pdf")
        sys.exit(1)

    if not HTML_FILE.exists():
        print(f"HTML file not found: {HTML_FILE}")
        print("Run compose_slides.py first.")
        sys.exit(1)

    html_url = HTML_FILE.resolve().as_uri()

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1280, "height": 720})
        await page.goto(html_url, wait_until="networkidle")
        await page.wait_for_selector(".slide", timeout=10000)
        await page.add_style_tag(content=HIDE_NAV_CSS)

        slides_selector = page.locator(".slide")
        n_slides = await slides_selector.count()
        if n_slides == 0:
            await browser.close()
            raise SystemExit("No slides found.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            paths = []
            for i in range(n_slides):
                await page.evaluate("window.showSlide(%d)" % i)
                await asyncio.sleep(0.4)
                fp = tmp / ("slide_%02d.png" % i)
                await page.locator("#slideshow").screenshot(path=str(fp))
                paths.append(fp)
                print("  slide %d/%d" % (i + 1, n_slides))
            await browser.close()

            with open(OUTPUT_PDF, "wb") as f:
                f.write(img2pdf.convert([str(p) for p in paths]))

    print("PDF saved:", OUTPUT_PDF)
    return OUTPUT_PDF


def main():
    os.chdir(SCRIPT_DIR)
    asyncio.run(capture_slides_to_pdf())


if __name__ == "__main__":
    main()
