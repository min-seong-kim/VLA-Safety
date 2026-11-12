"""Make browser-compatible copies of actual rollout videos, retaining originals."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess


def encode(source):
    source = Path(source)
    target = source.with_name("views_browser.mp4")
    if target.exists() and target.stat().st_size > 1024 and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    temporary = target.with_name("views_browser.pending.mp4")
    result = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-threads", "1", "-movflags", "+faststart", str(temporary)],
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Video conversion failed for {source}: {result.stderr[-2000:]}")
    temporary.replace(target)
    return target


def encode_all(sources):
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(encode, sources))
