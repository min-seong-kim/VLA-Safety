"""Check the offline comparison viewer in desktop and mobile Chromium."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    errors, checks = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((run / "review.html").as_uri())
        for benchmark, count in [("test", 264), ("v6", 132), ("confirmation", 66), ("native", 16)]:
            page.select_option("#bench", benchmark)
            assert page.locator("#scene option").count() == count
            for index in [0, count - 1]:
                page.select_option("#scene", str(index))
                page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v => v.readyState >= 1 && v.duration > 0 && !v.error)")
                assert page.locator("#grid article").count() == 4
                videos = page.locator("video").evaluate_all("nodes => nodes.map(v => ({width:v.videoWidth,height:v.videoHeight,duration:v.duration}))")
                assert all(v["width"] > 0 and v["height"] > 0 for v in videos)
                checks.append(dict(benchmark=benchmark, index=index, videos=videos))
            if benchmark == "native":
                assert "XY 운반" not in page.locator("#grid").inner_text()
        page.select_option("#bench", "test")
        page.click("#next")
        assert page.locator("#scene").input_value() == "1"
        page.click("#previous")
        assert page.locator("#scene").input_value() == "0"
        page.click("#play")
        page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v => !v.paused && v.currentTime > 0)")
        page.locator("video").evaluate_all("nodes => nodes.forEach(v => v.pause())")
        page.screenshot(path=str(run / "review_desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        for benchmark in ["test", "v6", "confirmation", "native"]:
            page.select_option("#bench", benchmark)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), benchmark
        assert page.locator("#grid").evaluate("e => getComputedStyle(e).gridTemplateColumns.split(' ').length") == 1
        page.select_option("#bench", "test")
        page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v => v.readyState >= 1 && !v.error)")
        page.screenshot(path=str(run / "review_mobile.png"), full_page=True)
        browser.close()
    assert not errors, errors
    audit = dict(passed=True, offline_html=True, checks=checks, javascript_errors=errors,
                 navigation_passed=True, simultaneous_playback_passed=True, mobile_no_overflow=True)
    (run / "browser_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(dict(passed=True, sampled_scene_sets=len(checks), videos_checked=4 * len(checks))))


if __name__ == "__main__":
    main()
