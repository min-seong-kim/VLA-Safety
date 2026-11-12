"""Create an offline snapshot with scene previews, trajectories and timestep action inspection."""

import argparse
import base64
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import h5py

from rollout_common import ROOT

MODELS = ["pi0_libero", "pi05_libero", "openvla_libero_spatial", "openvla_oft_libero_spatial"]


def prepare_video(folder):
    source = folder / "views.mp4"
    dest = folder / "views_h264.mp4"
    if dest.exists():
        return dest
    executable = shutil.which("ffmpeg")
    if not executable:
        return source
    temporary = folder / "views_h264.partial.mp4"
    subprocess.run([executable, "-loglevel", "error", "-y", "-i", str(source), "-c:v", "libx264", "-crf", "23",
                    "-preset", "veryfast", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
                   check=True, timeout=120, stdout=subprocess.DEVNULL)
    temporary.replace(dest)
    return dest


def build(run_root, bundle=False):
    scenes = [json.loads(line) for line in (ROOT / "data/semantic_rollout_v3/scenes.jsonl").read_text().splitlines()]
    for scene in scenes:
        path = ROOT / "data/semantic_rollout_v3/observations" / scene["family_id"] / "agentview.png"
        scene["image"] = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    records = {}
    videos = []
    for summary_path in sorted(run_root.glob("*/episodes/*/summary.json")):
        folder = summary_path.parent
        summary = json.loads(summary_path.read_text())
        relative = folder.relative_to(run_root)
        video = prepare_video(folder)
        videos.append(video)
        with h5py.File(folder / "trajectory.h5") as handle:
            actions = handle["actions/submitted"][:].round(5).tolist()
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        records[summary["model"] + "/" + summary["episode_id"]] = {**summary, "video": str(video.relative_to(run_root)),
                "actions": actions, "risk_steps": [event["step"] for event in events if event["risk_proxy"]],
                "feature_file": str(relative / "features.h5"), "trajectory_file": str(relative / "trajectory.h5"),
                "audit_passed": (folder / "audit.json").exists(), "episode_file": str(relative / "episode.json")}
    status_path = run_root / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    quality_path = run_root / "quality_review.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else None
    payload = {"scenes": scenes, "records": records, "models": MODELS, "status": status,
               "quality_review": quality,
               "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    path = run_root / "preview.html"
    temporary = path.with_suffix(".tmp.html")
    temporary.write_text(html)
    temporary.replace(path)
    if bundle:
        temporary = run_root / "preview_bundle.partial.zip"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(path, "preview.html")
            for video in videos:
                archive.write(video, str(video.relative_to(run_root)))
            for file in ["status.json", "audit_summary.json", "quality_review.json"]:
                if (run_root / file).exists():
                    archive.write(run_root / file, file)
        temporary.replace(run_root / "preview_bundle.zip")
    print(json.dumps({"dashboard": str(path), "completed_episodes": len(records), "bundle": bundle}))


HTML = r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLA Semantic Safety · timestep 기록</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#152039;font:15px/1.6 system-ui,sans-serif}header{padding:32px 5vw;background:#172440;color:white}h1{margin:0;font-size:28px}p{margin:8px 0}.wrap{max-width:1400px;margin:auto;padding:24px}.notice{background:#fff7db;padding:14px;border-radius:10px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.stat{background:white;border:1px solid #d9e0eb;padding:15px;border-radius:12px}.stat b{display:block;font-size:22px}.bar{height:5px;background:#e3e8f0}.bar i{display:block;height:5px;background:#306fc2}select,button{font:inherit;padding:8px 12px;border:1px solid #b6c3d8;background:white;border-radius:8px}input[type=search]{font:inherit;padding:9px;width:300px;border:1px solid #b6c3d8;border-radius:8px}.toolbar{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{background:white;border:1px solid #d8e0ed;border-radius:12px;overflow:hidden;cursor:pointer}.card:hover{border-color:#3573c9}.card img{width:100%;max-height:275px;object-fit:contain;background:#e6e7e9}.card .text{padding:14px}.tag{font-size:12px;color:#586b87}.status{display:flex;gap:8px;margin-top:8px}.badge{padding:2px 8px;background:#eef2f7;border-radius:6px;font-size:12px}.done{background:#e0f3e9;color:#156041}dialog{border:0;border-radius:14px;max-width:1200px;width:94vw;padding:24px}dialog::backdrop{background:#10203bcc}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}video{width:100%;background:#121a29;min-height:160px}code{font-size:12px;word-break:break-all}.links{display:flex;gap:10px;font-size:12px}a{color:#2366ae}.action{font:12px monospace}.small{font-size:13px;color:#5a6780}canvas{width:100%;height:110px}details{background:white;padding:12px;border-radius:10px;margin:14px 0}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.stats,.pair{grid-template-columns:1fr 1fr}}@media(max-width:540px){.grid,.pair{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}}
</style>
<header><h1>VLA Semantic Safety — rollout & timestep 기록</h1><p>36개 상황 · 안전/위험 지시 72개 · 공식 체크포인트 4개 · 에피소드당 최대 160 timestep</p><p id="updated" class="small" style="color:#c2d2e8"></p></header>
<main class="wrap"><div class="notice">이미지와 표는 이 HTML 하나로 열립니다. 영상까지 내려받으려면 <a href="preview_bundle.zip">preview_bundle.zip</a>을 압축 해제한 뒤 안의 preview.html을 여세요. HDF5 원본은 용량 때문에 ZIP에 포함하지 않습니다. 이 화면은 저장 당시의 스냅샷이며 서버 파일을 다시 열면 갱신된 진행 상황을 볼 수 있습니다.</div>
<div class="stats" id="stats"></div>
<div id="quality" class="notice" hidden></div>
<details><summary>기록 범위와 해석</summary><p>매 timestep마다 새로운 관측으로 모델을 호출하고 예측 chunk의 첫 action만 실행합니다. 모든 층의 pooled 표현, π Action Expert의 10개 denoising iteration, 입력 토큰·mask·robot state·RGB·action chunk를 저장합니다. 전체 token 표현은 t=0에만 저장합니다.</p><p>위험 지시는 사람이 검토하기 전의 합성 후보 라벨입니다. 위험 조건은 접촉·배치 등의 기하학적 proxy이며, 실제 상해·화재·감전·중독을 의미하지 않습니다. 지시 미수행을 안전 거부로 판정하지 않습니다. 한 상황의 모든 timestep과 두 지시는 같은 학습/평가 분할에 속합니다.</p><p>OpenVLA는 자기회귀 action token, OFT는 MLP action head를 사용합니다. π 계열의 Action Expert와 구조가 다릅니다.</p></details>
<div class="toolbar"><select id="model"></select><select id="category"><option value="">모든 범주</option></select><input id="search" type="search" placeholder="상황 또는 instruction 검색"></div><div class="grid" id="grid"></div>
<dialog id="modal"><button id="close" style="float:right">닫기</button><h2 id="title"></h2><p id="context"></p><div class="pair" id="pair"></div><p class="small">영상은 왼쪽 agentview, 오른쪽 wrist view입니다. slider t의 action은 관측 t를 입력받아 생성한 명령이며, 위험 사건은 action 실행 후(t+1)의 값입니다.</p></dialog></main>
<script>const D=__DATA__;const $=id=>document.getElementById(id);const names={pi0_libero:'π0',pi05_libero:'π0.5',openvla_libero_spatial:'OpenVLA',openvla_oft_libero_spatial:'OpenVLA-OFT'};const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('updated').textContent='스냅샷 UTC: '+D.updated_utc;
if(D.quality_review){$('quality').hidden=false;$('quality').innerHTML='<strong>기록 완료와 행동 평가를 구분해 주세요.</strong> 네 모델 모두 지원되는 task proxy에서 성공 0건, 위험 proxy 0건입니다. 안전 거부의 증거로 해석할 수 없습니다. 정지 제어 검사에서 <code>known_allergen_serving</code>의 rice cake가 약 4.25 m 이동하는 물리적 불안정성이 확인되었습니다. 원본은 보존했고, 이 유형을 제외한 모델별 70개 t=0 probe 표본과 44,800 timestep의 별도 index를 준비했습니다. 나머지 상황도 행동 benchmark로 쓰기 전에 정상 과제 수행 가능성을 검증해야 합니다.';}
for(const model of D.models){const records=Object.values(D.records).filter(r=>r.model===model);const steps=records.reduce((s,r)=>s+r.steps,0);$('stats').insertAdjacentHTML('beforeend',`<div class="stat">${names[model]}<b>${records.length} / 72 완료</b>${steps.toLocaleString()} timestep<div class="bar"><i style="width:${records.length/72*100}%"></i></div></div>`);$('model').add(new Option(names[model],model))}
for(const c of [...new Set(D.scenes.map(s=>s.category))])$('category').add(new Option(c,c));
function getRecord(s,label){return D.records[$('model').value+'/'+s.family_id+'__'+label+'__seed0']}
function render(){const query=$('search').value.toLowerCase();const rows=D.scenes.filter(s=>(!$('category').value||s.category===$('category').value)&&JSON.stringify(s).toLowerCase().includes(query));$('grid').innerHTML=rows.map(s=>`<article class="card" data-family="${s.family_id}"><img src="${s.image}" alt="${esc(s.title_ko)}"><div class="text"><span class="tag">${esc(s.category)} · ${esc(s.mechanism_group)}</span><h3 style="margin:3px 0">${esc(s.title_ko)}</h3><code>${s.family_id}</code><div class="status">${['safe','unsafe'].map(label=>{const r=getRecord(s,label);return `<span class="badge ${r?'done':''}">${label}: ${r?'완료 '+r.steps+' t':'대기/실행 중'}</span>`}).join('')}</div></div></article>`).join('');for(const card of $('grid').children)card.onclick=()=>openScene(D.scenes.find(s=>s.family_id===card.dataset.family))}
function plot(canvas,actions,t){const c=canvas.getContext('2d'),w=canvas.width=600,h=canvas.height=110;c.clearRect(0,0,w,h);c.strokeStyle='#dae1ee';c.beginPath();c.moveTo(0,h/2);c.lineTo(w,h/2);c.stroke();const colors=['#3479c2','#c25b36','#399365'];for(let dim=0;dim<3;dim++){c.strokeStyle=colors[dim];c.beginPath();actions.forEach((a,i)=>{const x=i/(actions.length-1)*w,y=h/2-Math.max(-1,Math.min(1,a[dim]))*45;i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()}c.strokeStyle='#263448';c.beginPath();c.moveTo(t/(actions.length-1)*w,0);c.lineTo(t/(actions.length-1)*w,h);c.stroke()}
function openScene(s){$('title').textContent=names[$('model').value]+' · '+s.title_ko;$('context').textContent=s.shared_context;$('pair').innerHTML=['safe','unsafe'].map(label=>{const r=getRecord(s,label);return `<section><h3>${label==='safe'?'안전 지시':'위험 지시'}</h3><p>${esc(s[label+'_instruction'])}</p>${r?`<video id="video_${label}" controls preload="metadata" src="${encodeURI(r.video)}"></video><input id="slider_${label}" type="range" min="0" max="${r.steps-1}" value="0" style="width:100%"><div id="action_${label}" class="action"></div><canvas id="plot_${label}"></canvas><p class="small">위험 proxy 발생: ${r.risk_proxy_ever?'있음':'없음'} · task proxy: ${r.task_success===null?'지원하지 않는 물리 동작':r.task_success?'성공':'미완료'} · audit: ${r.audit_passed?'통과':'검사 대기'}</p><div class="links"><a href="${r.feature_file}">features.h5</a><a href="${r.trajectory_file}">trajectory.h5</a><a href="${r.episode_file}">episode.json</a></div>`:'<p>아직 완료되지 않은 에피소드입니다.</p>'}</section>`}).join('');for(const label of ['safe','unsafe']){const r=getRecord(s,label);if(!r)continue;const slider=$('slider_'+label),video=$('video_'+label);function update(t,seek){slider.value=t;$('action_'+label).textContent='t='+t+' | action=['+r.actions[t].map(x=>x.toFixed(3)).join(', ')+']'+(r.risk_steps.includes(t)?' | risk proxy after action':'');plot($('plot_'+label),r.actions,t);if(seek)video.currentTime=t/20}slider.oninput=()=>update(+slider.value,true);video.ontimeupdate=()=>update(Math.min(r.steps-1,Math.floor(video.currentTime*20)),false);update(0,false)}$('modal').showModal()}
$('close').onclick=()=>{$('modal').querySelectorAll('video').forEach(v=>v.pause());$('modal').close()};$('model').onchange=render;$('category').onchange=render;$('search').oninput=render;render();</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/semantic_rollouts_v3")
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    build(args.run_root.resolve(), args.bundle)


if __name__ == "__main__":
    main()
