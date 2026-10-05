"""Record the README demo (GIF, MP4, screenshots) by scripting a Chrome window.

The local site and Ollama must be running. Recording tools are kept out of
requirements.txt since only this script needs them:

    python3 -m venv /tmp/demo-tools
    /tmp/demo-tools/bin/pip install playwright imageio-ffmpeg
    /tmp/demo-tools/bin/playwright install ffmpeg
    /tmp/demo-tools/bin/python scripts/record_demo.py
"""

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
SIZE = {"width": 1440, "height": 900}


def record(video_dir: Path) -> Path:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport=SIZE, device_scale_factor=2, color_scheme="dark",
            record_video_dir=str(video_dir), record_video_size=SIZE,
        )
        page = context.new_page()

        page.goto(URL)
        page.locator(".rg-name").wait_for(timeout=60_000)
        page.wait_for_timeout(2500)
        page.screenshot(path=DOCS / "screenshot-home.png")
        print("home screenshot saved")

        box = page.get_by_placeholder("https://github.com/psf/requests")
        box.click()
        box.press_sequentially(REPO, delay=45)
        page.wait_for_timeout(500)
        box.press("Enter")

        chat = page.get_by_placeholder("Ask about karpathy/micrograd...")
        chat.wait_for(timeout=180_000)
        print("repo indexed")
        page.wait_for_timeout(1500)

        for i, question in enumerate(QUESTIONS, start=1):
            chat.click()
            chat.press_sequentially(question, delay=55)
            page.wait_for_timeout(400)
            chat.press("Enter")
            # One "Citations checked" box appears per answer.
            page.wait_for_function(
                f"document.body.innerText.split('Citations checked').length > {i}",
                timeout=240_000,
            )
            print(f"answer {i} received")
            page.wait_for_timeout(3500)
            if i == 1:
                page.screenshot(path=DOCS / "screenshot-answer.png")
                print("answer screenshot saved")

        sources = page.get_by_text("Sources", exact=False).last
        sources.scroll_into_view_if_needed()
        sources.click()
        page.wait_for_timeout(1500)
        page.mouse.wheel(0, 300)
        page.wait_for_timeout(4000)

        video = Path(page.video.path())
        context.close()
        browser.close()
    return video


def convert(video: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
         str(DOCS / "repoguide-demo.mp4")],
        check=True,
    )
    gif_filter = (
        "fps=8,scale=960:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-vf", gif_filter,
         str(DOCS / "demo.gif")],
        check=True,
    )
    print("saved docs/repoguide-demo.mp4 and docs/demo.gif")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    video_dir = Path(tempfile.mkdtemp())
    try:
        convert(record(video_dir))
    finally:
        shutil.rmtree(video_dir, ignore_errors=True)
