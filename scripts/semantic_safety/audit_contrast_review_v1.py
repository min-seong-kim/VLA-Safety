"""Verify the offline review UI, input accounting, and feedback round trip."""

import json
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/semantic_safety_contrast_v1"


def main():
    rows=[json.loads(s) for s in (OUT/"review_conditions.jsonl").read_text().splitlines()]
    errors=[]
    with sync_playwright() as p,tempfile.TemporaryDirectory(prefix="contrast-review-check-") as temp:
        browser=p.chromium.launch(args=["--no-sandbox"])
        context=browser.new_context(viewport={"width":1440,"height":1000},accept_downloads=True)
        page=context.new_page()
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto((OUT/"preview.html").as_uri())
        dataset_sha256=page.evaluate("data.dataset_sha256")
        assert page.locator("article.group").count()==3
        assert page.locator(".condition").count()>0
        first=page.locator(".condition").first.get_attribute("data-id")
        page.fill("#search",first)
        assert page.locator(".condition").count()>=1
        expected=next(r for r in rows if r["condition_id"]==first)
        assert page.locator(".input .en").first.inner_text()==expected["instruction_en"]
        page.locator(".review-status").first.select_option("revise")
        note="자동 검증용 임시 메모 — 실제 사용자 검토 아님"
        page.locator(".review-notes").first.fill(note)
        page.reload();page.fill("#search",first)
        assert page.locator(".review-notes").first.input_value()==note
        with page.expect_download() as download:page.click("#export")
        saved=Path(temp)/"feedback.json";download.value.save_as(saved)
        exported=json.loads(saved.read_text())
        assert exported["reviews"][first]["notes"]==note
        page.evaluate("localStorage.clear()")
        page.reload();page.set_input_files("#file",str(saved))
        page.wait_for_function("document.querySelector('#notice').textContent.includes('가져왔습니다')")
        page.fill("#search",first)
        assert page.locator(".review-status").first.input_value()=="revise"
        # Clear only this temporary browser context; no human review records exist here.
        page.evaluate("localStorage.clear()")
        page.reload()
        categories=sorted({r["category"] for r in rows})
        for category in categories:
            page.select_option("#category",category)
            assert page.locator("article.group").count()>0
            page.select_option("#track","B")
            for e in page.locator(".condition").all():
                row=next(r for r in rows if r["condition_id"]==e.get_attribute("data-id"))
                assert row["visual_track_available"]
                assert e.locator(".input").count()==1
                assert e.locator(".input .en").inner_text()==row["visual_instruction_en"]
            page.select_option("#track","both")
        page.select_option("#category","")
        for label in ["0","1"]:
            page.select_option("#label",label)
            for e in page.locator(".condition").all():
                row=next(r for r in rows if r["condition_id"]==e.get_attribute("data-id"))
                assert row["label"]==int(label)
        page.select_option("#label","")
        for flag in ["human","schematic","technical","visual_excluded","one_sided"]:
            page.select_option("#flag",flag)
            page.fill("#search","this-identifier-does-not-exist")
            assert page.locator(".condition").count()==0
            page.fill("#search","")
        page.select_option("#flag","")
        page.uncheck("#show-label")
        assert not page.locator(".verdict").first.is_visible()
        page.check("#show-label")
        for camera in ["agentview","wrist","overview"]:
            page.select_option("#camera",camera)
            page.evaluate("document.querySelectorAll('.image-link img').forEach(e=>e.loading='eager')")
            page.wait_for_function("[...document.querySelectorAll('.image-link img')].every(e=>e.complete&&e.naturalWidth>0)")
        page.select_option("#camera","agentview");page.check("#small")
        page.evaluate("document.querySelectorAll('.image-link img').forEach(e=>e.loading='eager')")
        page.wait_for_function("[...document.querySelectorAll('.image-link img')].every(e=>e.complete&&e.naturalWidth===224)")
        page.uncheck("#small")
        page.screenshot(path=str(OUT/"review_desktop.png"))
        page.set_viewport_size({"width":390,"height":844})
        for category in categories:
            page.select_option("#category",category)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),category
        page.select_option("#category","")
        page.screenshot(path=str(OUT/"review_mobile.png"))
        browser.close()
    assert not errors,errors
    result=dict(passed=True,offline=True,dataset_sha256=dataset_sha256,exact_model_instruction_display=True,category_filters=len(categories),
                track_and_label_filters=True,combined_search_and_flags=True,model_and_review_cameras=True,
                feedback_persistence=True,feedback_export_import=True,blind_label_toggle=True,
                mobile_no_overflow=True,temporary_review_notes_removed=True,javascript_errors=errors,
                human_semantic_review=False)
    (OUT/"browser_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()
