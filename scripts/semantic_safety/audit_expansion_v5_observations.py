"""Audit delivered RGB/state files and the offline scene-review UI."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/semantic_safety_observations_v5"


from expansion_v5_feedback_audit import run as feedback_audit


def file_audit():
    import numpy as np
    from PIL import Image
    scenes=[json.loads(l) for l in (OUT/"scenes.jsonl").read_text().splitlines()]
    manifest=json.loads((OUT/"assets/manifest.json").read_text())
    catalog=[json.loads(l) for l in (ROOT/"data/semantic_safety_catalog_v5/scenario_pairs.jsonl").read_text().splitlines()]
    assert len(scenes)==2*len(catalog) and len({s["scene_id"] for s in scenes})==len(scenes)
    by_id={p["pair_id"]:p for p in catalog}
    protocol=json.loads((OUT/"protocol.json").read_text())
    assert protocol["source_catalog_sha256"]==hashlib.sha256(
        (ROOT/"data/semantic_safety_catalog_v5/scenario_pairs.jsonl").read_bytes()).hexdigest()
    script=ROOT/"scripts/semantic_safety/build_expansion_v5_observations.py"
    code_hash=hashlib.sha256(script.read_bytes()).hexdigest()
    asset_code_hash=hashlib.sha256(script.with_name("expansion_v5_assets.py").read_bytes()).hexdigest()
    native_code_hash=hashlib.sha256(script.with_name("expansion_v5_native.py").read_bytes()).hexdigest()
    for alias,asset in manifest.items():
        assert hashlib.sha256((OUT/"assets"/(alias+".xml")).read_bytes()).hexdigest()==asset["xml_sha256"]
    technical=[];rows=[]
    for scene in scenes:
        folder=OUT/"observations"/scene["scene_id"]
        report=json.loads((folder/"validation.json").read_text())
        assert scene["request"]==by_id[scene["pair_id"]][scene["side"]]["instruction_en"]
        assert scene["catalog_branch"]==by_id[scene["pair_id"]][scene["side"]]
        signature=hashlib.sha256((json.dumps(scene,sort_keys=True)+code_hash+asset_code_hash+native_code_hash).encode()).hexdigest()
        assert report["signature"]==signature,scene["scene_id"]
        assert report["rendered"] and report["finite_state"],scene["scene_id"]
        if not report["technical_pass"]:technical.append(scene["scene_id"])
        with np.load(folder/"initial_observation.npz") as obs:
            assert obs["state"].shape==(8,) and np.isfinite(obs["sim_state"]).all()
            for key,name in [("image","agentview"),("wrist_image","wrist")]:
                rgb=np.asarray(Image.open(folder/(name+".png")))
                assert rgb.shape==(512,512,3)
                assert np.array_equal(rgb,obs[key][::-1,::-1]),scene["scene_id"]
            assert Image.open(folder/"overview.png").size==(768,768)
        for name,entry in report["images"].items():
            assert hashlib.sha256((OUT/entry["path"]).read_bytes()).hexdigest()==entry["sha256"]
        rows.append(dict(pair_id=scene["pair_id"],condition=scene["side"],category=scene["category_ko"],
                         title=scene["title_ko"],instruction=scene["request"],
                         agentview=report["images"]["agentview"]["path"],wrist=report["images"]["wrist"]["path"],
                         overview=report["images"]["overview"]["path"],
                         objects=" / ".join(p["object_name"] for p in scene["props"]),
                         review_flags=";".join(scene["review_flags"]),review_status="pending",review_notes=""))
    with (OUT/"observation_review_sheet.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with zipfile.ZipFile(OUT/"observation_review.zip") as z:
        assert z.testzip() is None
        assert z.read("preview.html")== (OUT/"preview.html").read_bytes()
    result=dict(passed=True,conditions=len(scenes),rgb_images=3*len(scenes),
                raw_observation_matches_png=True,asset_hashes_match=True,zip_crc=True,
                catalog_scene_requests_match=True,all_render_signatures_current=True,
                technical_review_needed=technical,semantic_labels_validated=False,task_skills_validated=False)
    (OUT/"delivery_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


def contact_sheets():
    from PIL import Image, ImageDraw
    pairs=[json.loads(l) for l in (OUT/"review_pairs.jsonl").read_text().splitlines()]
    dest=OUT/"contact_sheets";dest.mkdir(exist_ok=True)
    # One row per contrast; 18 pairs per sheet. Captions never enter observations.
    for start in range(0,len(pairs),18):
        batch=pairs[start:start+18];canvas=Image.new("RGB",(1440,6*268),(243,245,248));d=ImageDraw.Draw(canvas)
        for i,pair in enumerate(batch):
            x=(i%3)*480;y=(i//3)*268
            d.text((x+5,y+3),pair["pair_id"],fill=(30,40,50))
            for j,side in enumerate(["safe","unsafe"]):
                path=OUT/pair["observations"][side]["validation"]["images"]["overview"]["path"]
                im=Image.open(path).resize((236,236))
                canvas.paste(im,(x+2+j*240,y+24))
        canvas.save(dest/f"sheet_{start//18+1:02d}.jpg",quality=88)
    print(json.dumps(dict(contact_sheets=(len(pairs)+17)//18)))


def browser_audit():
    from playwright.sync_api import sync_playwright
    rows=[json.loads(l) for l in (OUT/"review_pairs.jsonl").read_text().splitlines()]
    errors=[]
    with sync_playwright() as p,tempfile.TemporaryDirectory(prefix="observation-v5-audit-") as temp:
        browser=p.chromium.launch(args=["--no-sandbox"])
        context=browser.new_context(viewport={"width":1440,"height":1200},accept_downloads=True)
        page=context.new_page();page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto((OUT/"preview.html").as_uri())
        assert page.locator("article").count()==6
        first=page.locator("article").first.get_attribute("data-id")
        for category in sorted({r["category"] for r in rows}):
            page.select_option("#category",category)
            n=sum(r["category"]==category for r in rows)
            assert f"검색 결과 {n}쌍" in page.locator("#pageinfo").inner_text()
            for camera in ["overview","agentview","wrist"]:
                page.select_option("#camera",camera)
                page.wait_for_function("Array.from(document.querySelectorAll('article:first-child img')).every(i=>i.complete&&i.naturalWidth>0)")
        page.select_option("#category","");page.select_option("#camera","overview")
        page.fill("#search",first);assert page.locator("article").count()==1
        page.locator(".review-status").select_option("revise")
        note="임시 UI 검증 메모 — 사용자 검수 아님"
        page.locator(".review-notes").fill(note);page.reload();page.fill("#search",first)
        assert page.locator(".review-notes").input_value()==note
        with page.expect_download() as download:page.click("#export")
        path=Path(temp)/"reviews.json";download.value.save_as(path)
        assert json.loads(path.read_text())["reviews"][first]["status"]=="revise"
        page.evaluate("localStorage.clear()");page.reload();page.set_input_files("#file",str(path))
        page.wait_for_function("document.querySelector('#notice').textContent.includes('가져왔습니다')")
        page.fill("#search",first);assert page.locator(".review-notes").input_value()==note
        page.evaluate("localStorage.clear()");page.reload()
        assert "검토 상태 지정 0 /" in page.locator("#reviewcount").inner_text()
        for flag in ["identical","technical","schematic"]:
            page.select_option("#flag",flag)
            assert "검색 결과" in page.locator("#pageinfo").inner_text()
        page.select_option("#flag","")
        page.click("#next");assert page.locator("article").first.get_attribute("data-id")!=first
        page.click("#prev")
        page.wait_for_function("Array.from(document.querySelectorAll('article:first-child img')).every(i=>i.complete&&i.naturalWidth>0)")
        page.screenshot(path=str(OUT/"browser_desktop.png"))
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
        page.screenshot(path=str(OUT/"browser_mobile.png"))
        browser.close()
    assert not errors,errors
    result=dict(passed=True,categories_tested=len({r["category"] for r in rows}),cameras_tested=3,image_loading=True,
                review_persistence=True,export_import=True,pagination=True,mobile_no_overflow=True,
                temporary_notes_removed=True,javascript_errors=errors,human_semantic_review=False)
    (OUT/"browser_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


def complete_bundle():
    """Add decision records and audits to the image-review bundle once validated."""
    files=["observation_review_sheet.csv","feedback_audit.json","delivery_audit.json","browser_audit.json",
           "revision_report.md","revision_summary.json","revision_decisions.jsonl","excluded_pairs.csv",
           "protocol.json","browser_desktop.png","browser_mobile.png","feedback_applied.txt",
           "native_reference/reference.json","native_reference/agentview.png"]
    # Replace an existing entry on reruns instead of creating ambiguous ZIP duplicates.
    path=OUT/"observation_review.zip"
    temp=OUT/"observation_review.complete.zip"
    with zipfile.ZipFile(path) as old,zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as new:
        for item in old.infolist():
            if item.filename not in files and not item.filename.startswith("contact_sheets/"):
                new.writestr(item,old.read(item.filename))
        for name in files:
            assert (OUT/name).is_file(),name
            new.write(OUT/name,name)
        for sheet in sorted((OUT/"contact_sheets").glob("*.jpg")):
            new.write(sheet,str(sheet.relative_to(OUT)))
    temp.replace(path)
    with zipfile.ZipFile(path) as bundle:
        assert bundle.testzip() is None
        assert len(bundle.namelist())==len(set(bundle.namelist()))
        assert all(bundle.read(name)==(OUT/name).read_bytes() for name in files)
    print(json.dumps(dict(review_zip_bytes=path.stat().st_size,decision_records_included=True,audits_included=True)))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["files","sheets","browser","feedback","bundle"])
    args=p.parse_args();{"files":file_audit,"sheets":contact_sheets,"browser":browser_audit,
                        "feedback":feedback_audit,"bundle":complete_bundle}[args.command]()
