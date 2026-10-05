"""Record the README demo (GIF, MP4, screenshots) by scripting a Chrome window.

The local site and Ollama must be running. Recording tools are kept out of
requirements.txt since only this script needs them:

    python3 -m venv /tmp/demo-tools
    /tmp/demo-tools/bin/pip install playwright imageio-ffmpeg
    /tmp/demo-tools/bin/playwright install ffmpeg
    /tmp/demo-tools/bin/python scripts/record_demo.py           # full README demo
    /tmp/demo-tools/bin/python scripts/record_demo.py --short   # ~15s clip for LinkedIn
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
DOCS = Path(__file__).parent.parent / "docs"
REPO = "https://github.com/karpathy/micrograd"
QUESTIONS = ["What does this project do?", "Where is the Value class defined?"]
SHORT_QUESTIONS = ["Where is the Value class defined?"]
SIZE = {"width": 1440, "height": 900}


def record(video_dir: Path, questions: list[str], short: bool) -> Path:
    typing_delay = 20 if short else 45
    pause = 0.5 if short else 1.0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport=SIZE, device_scale_factor=2, color_scheme="dark",
            record_video_dir=str(video_dir), record_video_size=SIZE,
        )
        page = context.new_page()

        page.goto(URL)
        page.locator(".rg-name").wait_for(timeout=60_000)
        page.wait_for_timeout(int(2500 * pause))
        if not short:
            page.screenshot(path=DOCS / "screenshot-home.png")
            print("home screenshot saved")

        box = page.get_by_placeholder("https://github.com/psf/requests")
        box.click()
        box.press_sequentially(REPO, delay=typing_delay)
        page.wait_for_timeout(500)
        box.press("Enter")

        chat = page.get_by_placeholder("Ask about karpathy/micrograd...")
        chat.wait_for(timeout=180_000)
        print("repo indexed")
        page.wait_for_timeout(int(1500 * pause))

        for i, question in enumerate(questions, start=1):
            chat.click()
            chat.press_sequentially(question, delay=typing_delay + 10)
            page.wait_for_timeout(400)
            chat.press("Enter")
            # One "Citations checked" box appears per answer.
            page.wait_for_function(
                f"document.body.innerText.split('Citations checked').length > {i}",
                timeout=240_000,
            )
            print(f"answer {i} received")
            page.wait_for_timeout(int(3500 * pause))
            if i == 1 and not short:
                page.screenshot(path=DOCS / "screenshot-answer.png")
                print("answer screenshot saved")

        sources = page.get_by_text("Sources", exact=False).last
        sources.scroll_into_view_if_needed()
        sources.click()
        page.wait_for_timeout(1500)
        page.mouse.wheel(0, 300)
        page.wait_for_timeout(3000 if short else 4000)

        video = Path(page.video.path())
        context.close()
        browser.close()
    return video


def convert(video: Path, short: bool) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = DOCS / ("repoguide-demo-short.mp4" if short else "repoguide-demo.mp4")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4)],
        check=True,
    )
    print(f"saved {mp4.relative_to(DOCS.parent)}")
    if short:
        return
    gif_filter = (
        "fps=8,scale=960:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-vf", gif_filter,
         str(DOCS / "demo.gif")],
        check=True,
    )
    print("saved docs/demo.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", action="store_true", help="one question, faster pacing")
    args = parser.parse_args()

    DOCS.mkdir(exist_ok=True)
    video_dir = Path(tempfile.mkdtemp())
    try:
        questions = SHORT_QUESTIONS if args.short else QUESTIONS
        convert(record(video_dir, questions, args.short), args.short)
    finally:
        shutil.rmtree(video_dir, ignore_errors=True)
