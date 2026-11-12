"use strict";
const DATA = JSON.parse(document.getElementById("dataset-payload").textContent);
document.getElementById("dataset-payload").remove();
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (n) => Number(n).toLocaleString("ko-KR");
const gb = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + " GB" : (n / 1e6).toFixed(1) + " MB";
const image = (key) => DATA.images[key] || "";
const ds = new Map(DATA.datasets.map(d => [d.id, d]));
const records = new Map(DATA.records.map(r => [r.id, r]));
const groups = new Map();
DATA.records.forEach(r => { if(r.group) { if(!groups.has(r.group)) groups.set(r.group, []); groups.get(r.group).push(r); } });
const searchIndex = new Map(DATA.records.map(r => [r.id, [r.id,r.instruction,r.context,r.category,r.subset,ds.get(r.dataset).name,JSON.stringify(r.annotation || {})].join(" ").normalize("NFKC").toLowerCase()]));
const state = {view:"overview", family:"", page:1, pageSize:24, filtered:DATA.records, pinned:[], current:null};
let toastTimer;
function toast(message) { $("#toast").textContent = message; $("#toast").classList.add("show"); clearTimeout(toastTimer); toastTimer = setTimeout(() => $("#toast").classList.remove("show"), 2800); }
function badge(label, draft=false) { return `<span class="badge ${draft ? "draft" : label || "none"}">${draft ? "제안 " : ""}${label === "safe" ? "SAFE" : label === "unsafe" ? "UNSAFE" : "이진 라벨 없음"}</span>`; }
function localPath(path) { return path ? encodeURI(path.startsWith("data/safety_sources/") ? path.slice("data/safety_sources/".length) : "../../" + path).replace(/#/g,"%23").replace(/\?/g,"%3F") : ""; }
function originalFile(r) { const s=r.source; return s.observation_ref?.file || s.observation_ref?.archive || s.file || ""; }
function title(r) { return r.instruction || (r.dataset === "robojailbench" ? "대응 instruction이 없는 원본 장면" : "자연어 instruction이 없는 원본 episode"); }
function preview(r, cls="") { return r.media.length ? `<img class="${cls}" src="${image(r.media[0])}" alt="${esc(ds.get(r.dataset).name)} 관측 미리보기" loading="lazy">` : `<div class="placeholder"><b>${r.kind === "text" ? "Aa" : "∅"}</b><span>${r.kind === "text" ? "텍스트 전용 QA" : "원본 이미지 누락"}</span></div>`; }
function chip(r) { return r.kind === "video" ? "VIDEO · t=0" : r.kind === "video_frames" ? "VIDEO · 3 FRAMES" : r.has_action ? "ROBOT OBSERVATION" : r.kind === "text" ? "TEXT QA" : r.media.length > 1 ? `${r.media.length} VIEWS` : "IMAGE"; }
function showView(name, scroll=true) {
  if(!["overview","explore","compare","quality"].includes(name)) name="overview";
  state.view=name;
  $$(".view").forEach(v => v.hidden = v.id !== "view-" + name);
  $$(".main-nav [data-view]").forEach(b => {b.classList.toggle("active",b.dataset.view===name); b.setAttribute("aria-current", b.dataset.view===name ? "page" : "false");});
  $("#breadcrumb").textContent = {overview:"전체 개요",explore:"레코드 탐색",compare:"조건 비교",quality:"구조와 품질"}[name];
  history.replaceState(null,"","#"+name);
  if(name === "explore") renderResults();
  if(scroll) window.scrollTo({top:0,behavior:"instant"});
}
function chooseDataset(id) {
  state.family=""; $("#dataset-filter").value=id; updateSubsets(); state.page=1;
  $$(".family-nav button").forEach(b=>b.classList.remove("active"));
  filter(); showView("explore");
}
function chooseFamily(name) {
  resetFilters(false); state.family=name;
  $$(".family-nav button").forEach(b=>b.classList.toggle("active",b.dataset.family===name));
  updateSubsets(); filter(); showView("explore");
}
function initOverview() {
  const q=DATA.quality;
  $("#nav-total").textContent=fmt(DATA.records.length);
  const families=[...new Set(DATA.datasets.map(d=>d.family))];
  $("#family-nav").innerHTML=families.map(f=>`<button data-family="${esc(f)}">${esc(f)}<b>${fmt(DATA.records.filter(r=>ds.get(r.dataset).family===f).length)}</b></button>`).join("");
  $("#stats").innerHTML=[
    ["확보한 원본",gb(q.downloaded_bytes),"전체 배포본 + 명시한 샘플"],
    ["탐색 가능한 레코드",fmt(DATA.records.length),"행 · 지시 · episode 단위가 다름"],
    ["데이터 계열",families.length+"개",DATA.datasets.length+"개 세부 자료"],
    ["검증한 다운로드",fmt(q.downloaded_files)+"개","다운로드 오류 0건 · 내용 검사 별도"]
  ].map(([label,value,note])=>`<article class="stat"><span>${label}</span><strong>${value}</strong><small>${note}</small></article>`).join("");
  const babyUnsafe=records.get("hazardarena:sim:sem_pick_moveto_baby:unsafe_refuse");
  const babySafe=records.get("hazardarena:sim:sem_pick_moveto_baby:safe2");
  $("#hero-pair").innerHTML=`<p><span>SAME TASK FAMILY</span><span>01 / CHILD SAFETY</span></p><div class="pair-images">${[[babySafe,"장난감 → 아기"],[babyUnsafe,"칼 → 아기"]].map(([r,text])=>`<figure>${preview(r)}<figcaption><span>${text}</span>${badge(r.label)}</figcaption></figure>`).join("")}</div><span class="pair-note">HazardArena 원본 시뮬레이션 · 행동 전 관측</span>`;
  $("#recommendations").innerHTML=["embguardtest_v3","hazardarena","asimov_multimodal_auto_val"].map((id,i)=>{const d=ds.get(id);return `<article class="recommendation"><span class="number">0${i+1} / ${["위험의 조합","로봇 상태와 궤적","로봇 시각 맥락"][i]}</span><h3>${esc(d.name)}</h3><p>${esc(d.why)}</p><button class="text-button" data-dataset="${id}">${fmt(d.count)}개 레코드 보기 ↗</button></article>`;}).join("");
  $("#inventory-count").textContent=`${DATA.datasets.length}개 세부 자료 · 2026.11.09 snapshot`;
  $("#inventory-body").innerHTML=DATA.datasets.map(d=>`<tr><td><strong>${esc(d.name)}</strong><small>${gb(d.bytes)}</small></td><td>${esc(d.scope)}<small>인덱스 ${fmt(d.count)}개</small></td><td>${esc(d.input)}<small>${esc(d.format)}</small></td><td>${esc(d.labels)}</td><td><button class="text-button" data-dataset="${d.id}" aria-label="${esc(d.name)} 탐색">↗</button></td></tr>`).join("");
}
function updateSubsets() {
  const id=$("#dataset-filter").value;
  const items=DATA.records.filter(r=>(!id||r.dataset===id)&&(!state.family||ds.get(r.dataset).family===state.family));
  const names=[...new Set(items.map(r=>r.subset).filter(Boolean))].sort();
  $("#subset-filter").innerHTML='<option value="">전체 구성</option>'+names.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join("");
}
function filter() {
  const query=$("#search").value.trim().normalize("NFKC").toLowerCase();
  const dataset=$("#dataset-filter").value, subset=$("#subset-filter").value, label=$("#label-filter").value, media=$("#media-filter").value, action=$("#action-filter").checked;
  state.filtered=DATA.records.filter(r=>{
    if(state.family&&ds.get(r.dataset).family!==state.family) return false;
    if(dataset&&r.dataset!==dataset || subset&&r.subset!==subset) return false;
    if(label&&(label==="none" ? r.label!=null : r.label!==label)) return false;
    if(action&&!r.has_action) return false;
    if(media==="video"&&!r.kind.startsWith("video") || media==="text"&&r.kind!=="text" || media==="image"&&r.kind!=="image" || media==="missing"&&(r.kind!=="image"||r.media.length)) return false;
    return !query||searchIndex.get(r.id).includes(query);
  });
  state.page=1; renderResults();
}
function resetFilters(render=true) {
  state.family=""; state.page=1;
  ["search","dataset-filter","subset-filter","label-filter","media-filter"].forEach(id=>$("#"+id).value="");
  $("#action-filter").checked=false;
  $$(".family-nav button").forEach(b=>b.classList.remove("active"));
  updateSubsets(); if(render) filter();
}
function card(r) {
  const d=ds.get(r.dataset), selected=state.pinned.includes(r.id);
  return `<article class="record-card ${selected?"selected":""}"><button class="card-open" data-record="${esc(r.id)}" aria-label="${esc(d.name)}: ${esc(title(r).slice(0,110))}"><div class="thumb">${preview(r)}<span class="media-chip">${chip(r)}</span></div><div class="card-content"><div class="card-top"><span class="card-dataset">${esc(d.name)}</span>${badge(r.label)}</div><p class="instruction">${esc(title(r))}</p><div class="card-sub">${esc(r.category||r.subset||r.id.split(":").slice(1).join(":"))}</div></div></button><div class="card-bottom"><span>${r.has_action?"● state / action 포함":r.context_role==="annotation"?"위험 설명은 annotation":esc(r.kind==="text"?"image 없음":"원본 참조 포함")}</span><button class="pin-button" data-pin="${esc(r.id)}" aria-pressed="${selected}" aria-label="${esc(r.id)} 비교 선택">${selected?"선택됨 ✓":"비교 +"}</button></div></article>`;
}
function renderResults() {
  const total=state.filtered.length, pages=Math.max(1,Math.ceil(total/state.pageSize));
  state.page=Math.min(state.page,pages);
  const start=(state.page-1)*state.pageSize;
  $("#record-grid").innerHTML=state.filtered.slice(start,start+state.pageSize).map(card).join("");
  $("#empty-state").hidden=total>0;
  $("#result-count").innerHTML=`<b>${fmt(total)}</b>개 결과 <span class="muted">/ 전체 ${fmt(DATA.records.length)} · ${total?fmt(start+1):0}–${fmt(Math.min(start+state.pageSize,total))}</span>`;
  $("#compare-selected b").textContent=state.pinned.length+" / 2";
  const dataset=$("#dataset-filter").value;
  $("#dataset-context").hidden=!dataset&&!state.family;
  if(dataset) {const d=ds.get(dataset);$("#dataset-context").innerHTML=`<div><strong>${esc(d.name)} · ${esc(d.scope)}</strong><p>${esc(d.note)}</p></div><a class="text-button" href="${esc(d.url)}" target="_blank" rel="noopener">공식 출처 ↗</a>`;}
  else if(state.family) $("#dataset-context").innerHTML=`<div><strong>${esc(state.family)} 계열</strong><p>선택한 계열의 레코드를 보고 있습니다. 데이터셋별로 라벨 의미와 확보 범위를 확인하세요.</p></div><button class="text-button" id="clear-family">계열 해제</button>`;
  const indices=[...new Set([1,pages,state.page-1,state.page,state.page+1].filter(x=>x>=1&&x<=pages))].sort((a,b)=>a-b);
  let previous=0;
  $("#pagination").innerHTML=`<button data-page="${state.page-1}" ${state.page===1?"disabled":""} aria-label="이전 페이지">←</button>`+indices.map(n=>{const ellipsis=n-previous>1?"<span>…</span>":"";previous=n;return ellipsis+`<button data-page="${n}" class="${n===state.page?"active":""}" ${n===state.page?'aria-current="page"':""}>${n}</button>`;}).join("")+`<button data-page="${state.page+1}" ${state.page===pages?"disabled":""} aria-label="다음 페이지">→</button>`;
}
function togglePin(id) {
  if(state.pinned.includes(id)) state.pinned=state.pinned.filter(x=>x!==id);
  else if(state.pinned.length<2) state.pinned.push(id);
  else {toast("두 개가 선택돼 있습니다. 기존 선택을 해제한 뒤 추가하세요."); return;}
  renderResults();
}
function detail(id) {
  const r=records.get(id); if(!r) return;
  state.current=id;
  const d=ds.get(r.dataset), original=originalFile(r), related=(groups.get(r.group)||[]).filter(x=>x.id!==id);
  $("#dialog-eyebrow").textContent=d.name+" / RECORD DETAIL";
  $("#detail-content").innerHTML=`<div class="detail-layout"><div class="detail-media">${r.media.length?`<img id="detail-image" class="detail-image" src="${image(r.media[0])}" alt="원본 관측 미리보기">`:preview(r)}${r.media.length>1?`<div class="media-tabs">${r.media.map((key,i)=>`<button data-media="${i}" class="${i===0?"active":""}">${esc(r.media_names[i]||"view "+(i+1))}</button>`).join("")}</div>`:""}${r.video?`<button class="secondary" id="play-video">원본 영상 재생 ▷</button><div id="video-region" class="video-region"></div>`:""}<p class="footnote">${r.kind.startsWith("video")?"썸네일과 영상의 시간 범위를 구분하세요.":"원본 내용을 바꾸지 않은 축소 미리보기입니다."}${r.media.length>1?" 탭으로 다른 카메라·프레임을 선택할 수 있습니다.":""}</p></div><div class="detail-info">${badge(r.label)}${r.has_action?' <span class="badge none">STATE / ACTION</span>':""}<h2>${esc(d.name)}</h2><p class="detail-id">${esc(r.id)}</p><p class="label-meaning">${esc(r.label_meaning)}</p><div class="detail-label">${r.kind==="text"||r.dataset.startsWith("asimov_v2")?"원본 PROMPT / 질문":"INSTRUCTION"}</div><p class="detail-text">${esc(title(r))}</p><div class="detail-buttons"><button class="secondary" id="copy-instruction">지시문 복사</button><button class="secondary" id="download-record">레코드 JSON ↓</button>${original?`<a class="secondary" href="${esc(localPath(original))}" target="_blank">원본 파일 ↗</a>`:""}</div>${r.context?`<div class="detail-label">${r.context_role==="annotation"?"상황 설명 · ANNOTATION":"CONTEXT"}</div><p class="detail-text">${esc(r.context)}</p>`:""}${r.annotation?`<details><summary>정답·위험 설명·상태 정보 (모델 입력 제외)</summary><pre>${esc(JSON.stringify(r.annotation,null,2))}</pre></details>`:""}${related.length?`<details open><summary>같은 그룹의 다른 레코드 · ${related.length}개</summary><div class="related-list">${related.slice(0,12).map(x=>`<button data-record="${esc(x.id)}">${badge(x.label)} ${esc(title(x).slice(0,150))}</button>`).join("")}${related.length>12?`<p class="footnote">처음 12개만 표시합니다. group 정보는 원본 메타데이터에서 확인할 수 있습니다.</p>`:""}</div></details>`:""}<details><summary>원본 참조와 전체 메타데이터</summary><pre>${esc(JSON.stringify(r.source,null,2))}</pre></details><p class="footnote">${esc(d.scope)} · <a href="${esc(d.url)}" target="_blank" rel="noopener">공식 배포처 ↗</a></p></div></div>`;
  if(!$("#detail-dialog").open) $("#detail-dialog").showModal();
  $("#detail-dialog").scrollTop=0;
}
function compareSelected() {
  if(state.pinned.length!==2) {toast("레코드 카드에서 비교할 두 개를 선택하세요.");return;}
  $("#selected-comparison").innerHTML=state.pinned.map(id=>{const r=records.get(id);return `<article>${preview(r)}<p class="detail-label">${esc(ds.get(r.dataset).name)}</p>${badge(r.label)}<h3>${esc(title(r))}</h3><p>${esc(r.label_meaning)}</p>${r.context?`<div class="detail-label">${r.context_role==="annotation"?"상황 설명 · annotation":"Context"}</div><p>${esc(r.context)}</p>`:""}<details><summary>메타데이터</summary><pre>${esc(JSON.stringify(r.source,null,2))}</pre></details></article>`;}).join("");
  $("#compare-dialog").showModal();
}
function renderTriple() {
  const scenario=$("#scenario-select").value;
  const variants=[["causal_risky","01 · 기준 장면","위험 요소 + 위험을 만드는 행동"],["decoupled_benign","02 · 행동만 변경","같은 이미지, 안전한 행동"],["absent_benign","03 · 관측만 변경","같은 행동, 위험 요소 제거"]];
  $("#triple-grid").innerHTML=variants.map(([split,number,caption])=>{const r=DATA.records.find(x=>x.dataset==="embguardtest_v3"&&x.source.source_id===split+"-"+scenario);return `<article class="comparison-card">${preview(r)}<div class="inner"><div class="comparison-label"><span>${number}</span>${badge(r.label)}</div><h3>${caption}</h3><p>${esc(r.instruction)}</p><small>${esc(r.source.source_id)} · scene ${scenario}</small><button class="text-button" data-record="${esc(r.id)}">원본 annotation 확인 ↗</button></div></article>`;}).join("");
}
function initCompare() {
  $("#scenario-select").innerHTML=DATA.quality.v3_triplets.map(t=>t.scenario).sort().map(s=>`<option value="${s}">장면 ${s}</option>`).join("");
  $("#scenario-select").value="059"; renderTriple();
  const labels={clearly_unsafe:["01 · Clearly unsafe","칼 → 아기"],clearly_safe:["02 · Clearly safe","장난감 → 아기"],implicit_unsafe:["03 · Implicit unsafe","물체 이름을 위치 지시로 변경"],implicit_safe_user_example:["04 · 사용자 implicit safe 예시","같은 위험 장면에서 보호 행동"],implicit_safe_matched_control:["추가 대조군 A","같은 암묵적 지시 + 장난감"],safe_with_sharp_object:["추가 대조군 B","위험 물체가 있어도 안전한 행동"]};
  $("#draft-grid").innerHTML=DATA.drafts.map(d=>{const r=records.get(d.source_episode_id); const l=labels[d.condition];return `<article class="comparison-card">${preview(r)}<div class="inner"><div class="comparison-label"><span>${l[0]}</span>${badge(d.proposed_task_safety,true)}</div><h3>${l[1]}</h3><p>${esc(d.instruction)}</p><small>${d.instruction_origin==="original instruction"?"원본 지시 사용":"새 지시문 초안"} · camera 0 · action target 검증 전</small><button class="text-button" data-record="${esc(r.id)}">참조한 원본 episode ↗</button></div></article>`;}).join("");
}
function initQuality() {
  const q=DATA.quality;
  $("#quality-stats").innerHTML=[["다운로드 오류","0건","1,463개 파일 크기·해시 기록"],["HDF5 판독 가능","43 / 45","원본 SHA 일치 · 2개 판독 실패"],["Heldout 이미지 존재","563 / 565","2개 null observation"],["v3 장면 그룹","127개","75 triple + 52 selective pair"]].map(([label,value,note])=>`<article class="stat"><span>${label}</span><strong>${value}</strong><small>${note}</small></article>`).join("");
  $("#quality-issues").innerHTML=`<article class="issue"><p class="issue-number">01 / MISSING OBSERVATIONS</p><h3>EMBGuard heldout · 이미지 2개 누락</h3><p>레코드는 탐색기에 포함하고, 이미지가 없는 상태를 표시했습니다. 실험 입력에서는 별도 처리해야 합니다.</p><ul>${q.missing_images.map(x=>`<li><code>${esc(x.source_id)}</code></li>`).join("")}</ul><button class="text-button" id="show-missing">누락 레코드 보기 ↗</button></article><article class="issue"><p class="issue-number">02 / SOURCE READ ERRORS</p><h3>HazardArena · 원본 HDF5 2개 판독 실패</h3><p>공식 배포 SHA-256과 일치하지만 h5py 3.16.0에서 열리지 않습니다. 아래 파일의 레코드는 탐색 인덱스에 포함하지 않았습니다.</p><ul>${q.hazardarena_unreadable.map(x=>`<li><code>${esc(x.file.replace("data/safety_sources/raw/hazardarena/",""))}</code></li>`).join("")}</ul><details><summary>오류와 해시 보기</summary><pre>${esc(JSON.stringify(q.hazardarena_unreadable,null,2))}</pre></details></article><article class="issue"><p class="issue-number">03 / RELEASE COUNTS</p><h3>RoboJailBench · 배포 ZIP에서 직접 집계</h3><p>JSON/JSONL 중복을 제외한 490쌍, 980개 지시입니다. car·drone·humanoid에는 현재 내려받은 ZIP 안에 대응 instruction 파일이 없습니다. 웹페이지 안내 수량과 다릅니다.</p><details><summary>하위 자료별 실제 지시 쌍</summary><pre>${esc(JSON.stringify(q.rjb_pairs_by_subset,null,2))}</pre></details></article><article class="issue"><p class="issue-number">04 / DOWNLOAD SCOPE</p><h3>아직 받지 않은 대용량 자료</h3><p>HazardArena_pi0_real 약 708.49 GB, HazardArena assets 약 17.10 GB, LIBERO assets 약 10.67 GB는 파일 목록만 확보했습니다. EMBHazard 전체와 LIBERO 전체 학습셋은 이번에 받은 샘플 범위 밖입니다.</p><p style="margin-top:10px">HTML은 현재 확보한 원본과 annotation을 확인하는 도구입니다. 학습·평가 split을 확정한 benchmark는 아닙니다.</p></article>`;
  if(DATA.media_errors.length) $("#quality-issues").insertAdjacentHTML("beforeend",`<article class="issue"><h3>영상 썸네일 생성 실패 · ${DATA.media_errors.length}개</h3><pre>${esc(JSON.stringify(DATA.media_errors,null,2))}</pre></article>`);
  renderSchema();
}
function renderSchema() {
  const id=$("#schema-select").value||"embguardtest_v3", d=ds.get(id), r=DATA.records.find(x=>x.dataset===id);
  let schema;
  if(id.startsWith("emb")) schema=DATA.schemas.parquet.filter(x=>x.dataset===id).map(x=>({file:x.file,rows:x.rows,schema:x.schema}));
  else if(id.startsWith("asimov")) schema=DATA.schemas.asimov.find(x=>x.dataset===id)?.fields;
  else if(id==="libero_safety") schema=DATA.schemas.libero;
  else if(id==="hazardarena") schema={simulation:{rgb:"[T,4,480,480,3]",ee_state:"[T,8]",q_state:"[T,7]",trajectory:"[T,8]",instruction:"text",safety_label:"SAFE / UNSAFE"},real:{rgb:"third / wrist camera",actions:"[T,7]",instruction:null},note:"trajectory 좌표계와 policy action 정의를 맞춰야 함"};
  else schema=r.source;
  $("#schema-panel").innerHTML=`<h3>${esc(d.name)}</h3><p class="footnote">${esc(d.note)}</p><div class="schema-meta"><span>${esc(d.format)}</span><span>${esc(d.scope)}</span><span>${gb(d.bytes)}</span></div><p class="detail-id">Revision: ${esc(d.revision)}</p><p class="footnote"><a href="${esc(d.url)}" target="_blank" rel="noopener">공식 데이터셋 / 배포처 ↗</a> · <a href="catalog/index.json">전체 원본 catalog ↗</a></p><pre>${esc(JSON.stringify(schema,null,2))}</pre><button class="secondary" data-dataset="${d.id}">이 자료의 레코드 탐색 ↗</button>`;
}
function download(value,name,jsonl=false) {
  const text=jsonl?value.map(v=>JSON.stringify(v)).join("\n")+"\n":JSON.stringify(value,null,2);
  const blob=new Blob([text],{type:jsonl?"application/x-ndjson;charset=utf-8":"application/json;charset=utf-8"});
  const url=URL.createObjectURL(blob), a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function copyText(text) {
  try {await navigator.clipboard.writeText(text);}
  catch {const input=document.createElement("textarea"); input.value=text;$("#detail-dialog").append(input);input.select();document.execCommand("copy");input.remove();}
  toast("지시문을 복사했습니다.");
}
document.addEventListener("click",e=>{
  const view=e.target.closest("[data-view]");if(view) return showView(view.dataset.view);
  const dataset=e.target.closest("[data-dataset]");if(dataset) return chooseDataset(dataset.dataset.dataset);
  const family=e.target.closest("[data-family]");if(family) return chooseFamily(family.dataset.family);
  const row=e.target.closest("[data-record]");if(row) return detail(row.dataset.record);
  const pin=e.target.closest("[data-pin]");if(pin) return togglePin(pin.dataset.pin);
  const page=e.target.closest("[data-page]");if(page&&!page.disabled){state.page=Number(page.dataset.page);renderResults();$(".filters").scrollIntoView({block:"start"});return;}
  const close=e.target.closest("[data-close-dialog]");if(close) return close.closest("dialog").close();
  const tab=e.target.closest("[data-compare-tab]");if(tab){const selected=tab.dataset.compareTab;$$("[data-compare-tab]").forEach(x=>{x.classList.toggle("active",x===tab);x.setAttribute("aria-selected",String(x===tab));});$("#compare-original").hidden=selected!=="original";$("#compare-drafts").hidden=selected!=="drafts";return;}
  const media=e.target.closest("[data-media]");if(media){const r=records.get(state.current);$("#detail-image").src=image(r.media[Number(media.dataset.media)]);$$("[data-media]").forEach(x=>x.classList.toggle("active",x===media));return;}
  if(e.target.closest("#copy-instruction")) return copyText(records.get(state.current).instruction);
  if(e.target.closest("#download-record")) return download(records.get(state.current).source,"safety-record.json");
  if(e.target.closest("#play-video")){const r=records.get(state.current);$("#video-region").innerHTML=`<video controls preload="metadata" src="${esc(encodeURI(r.video))}"></video><p>원본 영상 재생은 HTML을 원래 폴더에서 열었을 때 사용할 수 있습니다.</p>`;$("#play-video").hidden=true;return;}
  if(e.target.closest("#clear-family")){state.family="";$$(".family-nav button").forEach(b=>b.classList.remove("active"));updateSubsets();filter();}
  if(e.target.closest("#show-missing")){resetFilters(false);$("#dataset-filter").value="embguard_heldout";$("#media-filter").value="missing";updateSubsets();filter();showView("explore");}
});
let searchTimer;
$("#search").addEventListener("input",()=>{clearTimeout(searchTimer);searchTimer=setTimeout(filter,130);});
$("#dataset-filter").addEventListener("change",()=>{state.family="";$$(".family-nav button").forEach(b=>b.classList.remove("active"));updateSubsets();filter();});
["subset-filter","label-filter","media-filter","action-filter"].forEach(id=>$("#"+id).addEventListener("change",filter));
$("#page-size").addEventListener("change",()=>{state.pageSize=Number($("#page-size").value);state.page=1;renderResults();});
$("#reset-filters").addEventListener("click",()=>resetFilters());
$("#empty-reset").addEventListener("click",()=>resetFilters());
$("#compare-selected").addEventListener("click",compareSelected);
$("#scenario-select").addEventListener("change",renderTriple);
$("#schema-select").addEventListener("change",renderSchema);
$("#guide-button").addEventListener("click",()=>$("#guide-dialog").showModal());
$("#download-filtered").addEventListener("click",()=>{if(!state.filtered.length)return toast("내보낼 레코드가 없습니다.");download(state.filtered.map(r=>r.source),"safety-records-filtered.jsonl",true);toast(fmt(state.filtered.length)+"개 레코드 메타데이터를 저장합니다.");});
$("#detail-dialog").addEventListener("close",()=>{$$("#detail-dialog video").forEach(v=>v.pause());});
document.addEventListener("keydown",e=>{if(e.key==="/"&&!/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)&&!$("dialog[open]")){e.preventDefault();showView("explore");$("#search").focus();}});
window.addEventListener("hashchange",()=>showView(location.hash.slice(1)));
const options=DATA.datasets.map(d=>`<option value="${d.id}">${esc(d.name)}</option>`).join("");
$("#dataset-filter").insertAdjacentHTML("beforeend",options);$("#schema-select").innerHTML=options;
initOverview();updateSubsets();initCompare();initQuality();filter();showView(location.hash.slice(1)||"overview",false);
window.SAFETY_EXPLORER={data:DATA,state};
