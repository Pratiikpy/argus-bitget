"""The Track 3 walkthrough: one research task from question to actionable insight, recorded live.

    python scripts/record_walkthrough.py --out <folder outside the repository>

The handbook requires, for the AI Trading Desk track, a demo of "a complete research task, the
full flow from question to actionable insight". This drives the public console the way a trader
would, with nothing staged: it types a question about a real book into the live console, waits for
the real answer, reads it top to bottom, asks a question about the desk's own record,
opens the research task page, then the pages that say what was beaten, what was lost and which
Bitget services answered. Every frame is the hosted site answering at the moment of recording.

**How it is recorded** (`Downloads/merit-4k-recording-playbook.md`, the lessons of an earlier
recording): Playwright's ``recordVideo`` at a native 1600x900 viewport equal to the video size,
``deviceScaleFactor`` 1, because ``recordVideo`` ignores the scale factor and CSS zoom; one
context and one page, so the video is one continuous clip; an in-page cursor drawn from real
``mousemove`` events, since headless Chromium has no pointer; eased cursor moves and typed
questions, so a viewer can follow. ``ffmpeg`` then upscales to 3840x2160 with lanczos.

**Subtitles.** Each step's start is timed as it happens and written as English and Chinese
subtitle files (``walkthrough.en.srt``, ``walkthrough.zh.srt``); the upload copies have them
burned in, one per language (X shows no soft subtitle track), and a third carries both as selectable
tracks. English is the one to upload first: the product is verified in English before any other
language. The caption text is written here, per step, and says only what that step shows.

Outputs, in ``--out``: ``walkthrough.webm`` (the raw recording), ``walkthrough-4k.mp4``,
``walkthrough-4k-en.mp4`` and ``walkthrough-4k-zh.mp4`` (captions burned in),
``walkthrough-4k-subs.mp4`` (both tracks), the two
``.srt`` files and ``steps.json`` (each step's start and end, and what the page answered).
"""

# The Chinese captions use Chinese punctuation, which RUF001 reads as lookalikes of ASCII.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SITE = "https://deploy-topaz-seven-64.vercel.app"
WIDTH, HEIGHT = 1600, 900
MAX_SECONDS = 180.0

QUESTION = "I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?"
QUESTION_RECORD = "Why did you do nothing all weekend?"
"""English first: the product is verified in English before any other language (2026-09-26)."""

CURSOR = """(() => {
  if (window.__cursor) return; window.__cursor = true;
  const install = () => {
    if (!document.body) return requestAnimationFrame(install);
    if (document.getElementById('__cursor')) return;
    const style = document.createElement('style');
    style.textContent = '@keyframes __ripple{from{opacity:.8;'
      + 'transform:translate(-50%,-50%) scale(1)}'
      + 'to{opacity:0;transform:translate(-50%,-50%) scale(3.4)}}';
    document.head.appendChild(style);
    const c = document.createElement('div');
    c.id = '__cursor';
    c.style.cssText = 'position:fixed;left:-40px;top:-40px;width:26px;height:26px;'
      + 'z-index:2147483647;pointer-events:none;opacity:0;'
      + 'filter:drop-shadow(0 1px 2px rgba(0,0,0,.35))';
    c.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24"><path d="M5 2 L5 19.5 '
      + 'L9.6 15.6 L12.4 21.8 L15.2 20.4 L12.4 14.4 L18.5 14 Z" fill="#fff" stroke="#1a1714" '
      + 'stroke-width="1.3" stroke-linejoin="round"/></svg>';
    document.body.appendChild(c);
    addEventListener('mousemove', e => { c.style.opacity = '1';
      c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; },
      {passive: true, capture: true});
    addEventListener('mousedown', e => {
      const r = document.createElement('div');
      r.style.cssText = 'position:fixed;left:' + e.clientX + 'px;top:' + e.clientY + 'px;'
        + 'width:10px;height:10px;border-radius:999px;border:2px solid rgba(26,23,20,.65);'
        + 'z-index:2147483646;pointer-events:none;transform:translate(-50%,-50%);'
        + 'animation:__ripple .5s ease-out forwards';
      document.body.appendChild(r); setTimeout(() => r.remove(), 600);
    }, {capture: true});
  };
  install();
  document.addEventListener('DOMContentLoaded', install);
})();"""


@dataclass
class Motion:
    """Eased cursor moves, typing and scrolling on one page, the way a person would."""

    page: Any
    x: float = WIDTH / 2
    y: float = HEIGHT / 2

    def move(self, x: float, y: float, ms: float | None = None) -> None:
        distance = math.hypot(x - self.x, y - self.y)
        duration = ms if ms is not None else min(900.0, 250.0 + distance * 0.6)
        steps = max(8, int(duration / 16))
        start_x, start_y = self.x, self.y
        bow = min(60.0, distance * 0.08)
        for i in range(1, steps + 1):
            t = i / steps
            eased = 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
            arc = math.sin(math.pi * t) * bow
            self.page.mouse.move(start_x + (x - start_x) * eased,
                                 start_y + (y - start_y) * eased - arc)
            time.sleep(duration / steps / 1000)
        self.x, self.y = x, y

    def to(self, selector: str) -> tuple[float, float]:
        box = self.page.locator(selector).first.bounding_box()
        if box is None:
            raise RuntimeError(f"{selector} is not on the page")
        x, y = box["x"] + min(box["width"] / 2, 220), box["y"] + box["height"] / 2
        self.move(x, y)
        return x, y

    def click(self, selector: str) -> None:
        self.to(selector)
        time.sleep(0.15)
        self.page.mouse.down()
        time.sleep(0.07)
        self.page.mouse.up()

    def type(self, selector: str, text: str, per_char_ms: float = 45.0) -> None:
        self.click(selector)
        if text.isascii():
            self.page.keyboard.type(text, delay=per_char_ms)
            return
        # Key events cannot carry CJK characters (there is no key for them; an IME composes
        # them), so each character is inserted as composed text at a typing pace.
        for char in text:
            self.page.keyboard.insert_text(char)
            time.sleep(per_char_ms * 2.5 / 1000)

    def scroll(self, pixels: float, ms: float) -> None:
        steps = max(10, int(ms / 16))
        done = 0.0
        for i in range(1, steps + 1):
            t = i / steps
            eased = 0.5 - math.cos(math.pi * t) / 2
            target = pixels * eased
            self.page.mouse.wheel(0, target - done)
            done = target
            time.sleep(ms / steps / 1000)

    def read_down(self, seconds: float, *, stop_at_end: bool = True) -> None:
        """Scroll through the page at reading pace for ``seconds``."""
        height = float(self.page.evaluate("document.documentElement.scrollHeight"))
        remaining = max(0.0, height - HEIGHT - float(self.page.evaluate("scrollY")))
        chunks = max(1, int(seconds / 3.2))
        for _ in range(chunks):
            step = min(remaining, 260.0) if stop_at_end else 260.0
            if step > 0:
                self.scroll(step, 900)
                remaining -= step
            time.sleep(2.3)


@dataclass(frozen=True)
class Step:
    key: str
    en: str
    zh: str
    run: Callable[[Motion], dict[str, Any]]


def _answer_cards(page: Any) -> int:
    return int(page.locator("#out .card").count())


def _ask(motion: Motion, question: str, *, read_for: float) -> dict[str, Any]:
    page = motion.page
    before = _answer_cards(page)
    # After reading an answer the page is scrolled down and the question box is off-screen; the
    # cursor clicks where the box is, so it has to be on screen first.
    if float(page.evaluate("scrollY")) > 0:
        motion.scroll(-float(page.evaluate("scrollY")), 1400)
        time.sleep(0.6)
    page.locator("#q").fill("")
    motion.type("#q", question)
    time.sleep(0.4)
    motion.click("#go")
    asked = time.monotonic()
    # A locator wait, not wait_for_function: the console's CSP forbids 'unsafe-eval', as it should.
    # The console clears earlier answers first, so the new card is found by its question.
    del before
    page.locator("#out .card .q", has_text=question[:40]).first.wait_for(timeout=90_000)
    page.locator("#out .card .line").first.wait_for(timeout=90_000)
    answered_in = time.monotonic() - asked
    card = page.locator("#out .card").first
    text = card.inner_text()
    motion.to("#out .card")
    motion.read_down(read_for)
    return {"question": question, "answered_in_s": round(answered_in, 1),
            "lines": len(text.splitlines()), "first_line": text.splitlines()[:3]}


def _visit(
    motion: Motion, path: str, *, read_for: float, wait_for: str = ".wrap"
) -> dict[str, Any]:
    page = motion.page
    page.goto(SITE + path, wait_until="networkidle", timeout=120_000)
    page.locator(wait_for).first.wait_for(timeout=60_000)
    time.sleep(1.2)
    motion.move(WIDTH * 0.62, HEIGHT * 0.45)
    motion.read_down(read_for)
    return {"url": SITE + path, "title": page.title()}


STEPS: tuple[Step, ...] = (
    Step("open", "ARGUS, a research desk for Bitget's tokenized US stocks and crypto. No account, "
         "no key.", "ARGUS：面向 Bitget 美股代币与加密资产的研究台。无需账户、无需密钥。",
         lambda m: _visit(m, "/", read_for=3.0, wait_for="#q")),
    Step("ask", "A trader asks about their own book, in plain words.",
         "交易者用日常语言，询问关于自己持仓的问题。",
         lambda m: _ask(m, QUESTION, read_for=34.0)),
    Step("record", "Asked about the desk's own record, it answers from the ledger.",
         "询问研究台自己的决策记录，它依据账本作答。",
         lambda m: _ask(m, QUESTION_RECORD, read_for=12.0)),
    Step("research", "The research task page: seven engines, one question, ending in what to do.",
         "研究任务页：七个引擎回答同一个问题，最后给出该怎么做。",
         lambda m: _visit(m, "/research", read_for=27.0, wait_for=".concl")),
    Step("proof", "What it beat: each capability run against a named rival on the same input.",
         "胜过了谁：每项能力都在相同输入上与具名对手实际对比。",
         lambda m: _visit(m, "/proof", read_for=20.0)),
    Step("wrong", "What it got wrong: every recorded loss stays published.",
         "做错了什么：每一次记录在案的失败都会保持公开。",
         lambda m: _visit(m, "/wrong", read_for=14.0)),
    Step("status", "Which Bitget services answered today, checked live.",
         "今天哪些 Bitget 服务有响应，实时检查。",
         lambda m: _visit(m, "/status", read_for=9.0)),
)


def _srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, timed: list[dict[str, Any]], lang: str) -> None:
    blocks = []
    for i, step in enumerate(timed, start=1):
        blocks.append(f"{i}\n{_srt_time(step['start'])} --> {_srt_time(step['end'])}\n"
                      f"{step[lang]}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def record(out: Path) -> list[dict[str, Any]]:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    timed: list[dict[str, Any]] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": WIDTH, "height": HEIGHT},
                                  device_scale_factor=1, color_scheme="light",
                                  record_video_dir=str(out),
                                  record_video_size={"width": WIDTH, "height": HEIGHT})
        ctx.add_init_script(CURSOR)
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        motion = Motion(page)
        started = time.monotonic()
        for step in STEPS:
            begin = time.monotonic() - started
            detail = step.run(motion)
            timed.append({"key": step.key, "en": step.en, "zh": step.zh,
                          "start": round(begin, 2),
                          "end": round(time.monotonic() - started, 2), "detail": detail})
        time.sleep(1.5)
        video = page.video
        ctx.close()
        raw = Path(video.path()) if video is not None else None
        browser.close()
    if raw is None:
        raise SystemExit("Playwright wrote no video")
    shutil.move(str(raw), out / "walkthrough.webm")
    (out / "steps.json").write_text(json.dumps({"steps": timed, "page_errors": errors},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    return timed


def transcode(out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is not on PATH; the webm is in the output folder")
    src = out / "walkthrough.webm"
    base = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-r", "30"]
    scale = "scale=3840:2160:flags=lanczos,format=yuv420p"
    subprocess.run([ffmpeg, "-y", "-i", str(src), *base, "-vf", scale,
                    str(out / "walkthrough-4k.mp4")], check=True)
    # Burned in: X plays no subtitle track. The filter path is relative so the Windows drive
    # colon never reaches ffmpeg's filter parser.
    style = "FontSize=10,BorderStyle=3,Outline=1,Shadow=0,BackColour=&H99000000,MarginV=8"
    for lang, font in (("en", "Segoe UI"), ("zh", "Microsoft YaHei")):
        burn = f"{scale},subtitles=walkthrough.{lang}.srt:force_style='FontName={font},{style}'"
        subprocess.run([ffmpeg, "-y", "-i", "walkthrough.webm", *base, "-vf", burn,
                        f"walkthrough-4k-{lang}.mp4"], check=True, cwd=out)
    subprocess.run([ffmpeg, "-y", "-i", "walkthrough-4k.mp4", "-i", "walkthrough.zh.srt",
                    "-i", "walkthrough.en.srt", "-map", "0:v", "-map", "1", "-map", "2",
                    "-c:v", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=chi",
                    "-metadata:s:s:1", "language=eng", "walkthrough-4k-subs.mp4"],
                   check=True, cwd=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-transcode", action="store_true")
    args = parser.parse_args(argv)
    timed = record(args.out)
    total = timed[-1]["end"]
    write_srt(args.out / "walkthrough.en.srt", timed, "en")
    write_srt(args.out / "walkthrough.zh.srt", timed, "zh")
    print(f"recorded {total:.1f} s in {len(timed)} steps")
    if total > MAX_SECONDS:
        print(f"over the {MAX_SECONDS:.0f} s budget", file=sys.stderr)
    if not args.no_transcode:
        transcode(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
