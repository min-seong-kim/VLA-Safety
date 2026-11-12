"""Exercise real offline catalog filtering, review persistence, export and import."""

import json
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/semantic_safety_catalog_v3"


def main():
    rows = [json.loads(line) for line in (OUT / "scenario_pairs.jsonl").read_text().splitlines()]
    errors = []
    with sync_playwright() as p, tempfile.TemporaryDirectory(prefix="semantic-v3-review-") as temp:
        browser = p.chromium.launch(args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((OUT / "preview.html").as_uri())
        assert page.locator("article").count() == 16
        first_id = page.locator("article").first.get_attribute("data-id")
        for category in sorted({r["category"] for r in rows}):
            page.select_option("#category", category)
            expected = sum(r["category"] == category for r in rows)
            assert f"검색 결과 {expected}쌍" in page.locator("#pageinfo").inner_text()
        page.select_option("#category", "")
        page.select_option("#verb", "cut")
        assert page.locator("article").count() > 0
        page.select_option("#verb", "")
        page.fill("#search", first_id)
        assert page.locator("article").count() == 1
        page.locator(".review-status").select_option("revise")
        note = "UI 검증용 임시 메모 — 사람 검수 아님"
        page.locator(".review-notes").fill(note)
        page.reload()
        page.fill("#search", first_id)
        assert page.locator(".review-status").input_value() == "revise"
        assert page.locator(".review-notes").input_value() == note
        with page.expect_download() as download:
            page.click("#export")
        saved = Path(temp) / "review.json"
        download.value.save_as(saved)
        data = json.loads(saved.read_text())
        assert data["reviews"][first_id]["notes"] == note
        page.evaluate("localStorage.clear()")
        page.reload()
        page.set_input_files("#file", str(saved))
        page.wait_for_function("document.querySelector('#notice').textContent.includes('가져왔습니다')")
        page.fill("#search", first_id)
        assert page.locator(".review-status").input_value() == "revise"
        page.evaluate("localStorage.clear()")
        page.reload()
        assert "검토 상태 지정 0 /" in page.locator("#reviewcount").inner_text()
        page.click("#next")
        assert page.locator("article").first.get_attribute("data-id") != first_id
        page.click("#prev")
        assert page.locator("article").first.get_attribute("data-id") == first_id
        for flag in ["context_required", "verb_mismatch", "shared_physical_setup", "prior_v6_request_overlap", "instruction_variant_not_new_setup"]:
            page.select_option("#flag", flag)
            expected = sum(flag in r["flags"] for r in rows)
            assert f"검색 결과 {expected}쌍" in page.locator("#pageinfo").inner_text()
        page.select_option("#flag", "")
        page.screenshot(path=str(OUT / "review_desktop.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        for category in sorted({r["category"] for r in rows}):
            page.select_option("#category", category)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), category
        page.select_option("#category", "")
        page.screenshot(path=str(OUT / "review_mobile.png"))
        browser.close()
    assert not errors, errors
    audit = dict(passed=True, offline=True, categories_tested=len({r["category"] for r in rows}),
                 filtering=True, pagination=True, local_review_persistence=True, export_import=True,
                 mobile_no_overflow=True, javascript_errors=errors, human_semantic_validation=False,
                 temporary_review_notes_removed=True)
    (OUT / "browser_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
