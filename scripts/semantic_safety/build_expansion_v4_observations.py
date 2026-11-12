"""Render the user's reviewed v4 candidates in actual LIBERO/MuJoCo scenes.

Props are fixed for visual review. Motor skills, articulation, consequences,
success predicates, and semantic labels are intentionally not validated here.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import traceback
import xml.etree.ElementTree as ET
import zipfile

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from expansion_v4_assets import ASSETS, OUT, build_asset, choose_kind
from rollout_common import ROOT, bootstrap, observation_payload, write_json

CATALOG = ROOT / "data/semantic_safety_catalog_v4"
SCRIPT = Path(__file__)


def jsonl(path, values):
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in values))


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs=load(CATALOG/"scenario_pairs.jsonl")
    manifest={}; cache={}; scenes=[]; mapping=[]
    def asset(name,state="",material="",kind=None):
        key=(name,state,material,kind)
        if key not in cache:
            cache[key]=build_asset(*key[:3],kind_override=kind)
            if cache[key]["alias"]:manifest[cache[key]["alias"]]=cache[key]
        return cache[key]
    for pair in pairs:
        for side in ["safe","unsafe"]:
            branch=pair[side]; props=[]; changes=[]; issues=[]
            actor=asset(branch["actor"],branch["actor_state"],branch["actor_material"])
            def put(binding,role,xy,z=0):
                if binding["alias"]:
                    props.append(dict(name=f"prop_{len(props)+1}",role=role,asset=binding["alias"],
                                      object_name=binding["name"],xy=xy,z_offset=z,kind=binding["kind"],fidelity=binding["fidelity"],
                                      pose_adjustment=binding.get("pose_adjustment"),task_object=role!="support_fixture"))
            occupancy=pair["family_id"]=="body_path_occupancy"
            same_scene=pair["pair_axis"]=="instruction"
            if occupancy:
                put(actor,"actor_and_target",[.01,-.08])
                if "occupies" in branch["target_state"]:
                    put(asset("person's hand","intact hand in the mechanism path","skin"),"occupied_path",[.13,-.08],.035)
                changes.append("끼임 후보는 도구와 개구부를 하나의 설비로 묶고, 경로 점유 조건에 손을 추가했다. 관절·닫힘 운동은 미구현이다.")
                issues.append("articulation_and_precise_contact_path_pending")
            else:
                put(actor,"actor",[-.055,.15],.006 if actor["kind"]=="saw" else 0)
                if actor["kind"]=="saw":
                    put(asset("upright saw cradle", "fixed neutral support", "rigid support"),
                        "support_fixture",[-.055,.12])
                    changes.append("톱은 별도 고정 받침에 세웠다. 받침은 과제 물체 수에 포함하지 않으며 집을 도구와 분리되어 있다.")
                if same_scene and " and " in branch["target"]:
                    person,work=branch["target"].split(" and ",1)
                    put(asset(person,"intact living recipient represented by a static avatar","skin","person"),"recipient",[.035,-.23])
                    put(asset(work,"work material on table","mixed"),"workpiece",[.13,.02])
                    changes.append("동일 장면의 지시 대조이므로 도구·사람·작업 재료를 함께 배치했다.")
                else:
                    target=asset(branch["target"],branch["target_state"],"")
                    put(target,"target",[.035,-.17])
            assert not any(p["kind"] in {"person","hand","arm","foot","worn_loop"} for p in props)
            if any(p["kind"] in {"mesh:hand","mesh:baby"} for p in props):
                issues.append("human_is_static_visual_proxy_not_living_tissue")
            if any(p["kind"]=="mesh:hand" for p in props):
                changes.append("이전 v6의 스캔 손 mesh로 교체했다. 살아 있는 손이라는 시나리오 조건은 여전히 텍스트에 명시한다.")
            if actor.get("pose_adjustment"):
                changes.append(actor["pose_adjustment"]["note_ko"])
                issues.append("grasp_and_action_feasibility_not_rollout_validated")
            changes.extend(pair.get("revision_notes_ko", []))
            if any(p["kind"] in {"animal_bowl","animal_habitat","plant","aquarium"} for p in props):
                issues.append("organisms_are_stylized_static_proxies")
            if any(p["fidelity"]=="schematic_rigid_proxy" for p in props):
                issues.append("schematic_asset_identity_requires_review")
            if "held" in branch["actor_state"]:
                changes.append("원 명세의 held 표현 대신 도구를 테이블에 놓은 집기 전 상태로 통일했다.")
            # No facts hidden in a label field are treated as observations.
            context={"common":pair["shared_context_en"],"actor":branch["actor_state"],"target":branch["target_state"]}
            context["actor"]=context["actor"].split(", held")[0]
            context["actor"]=context["actor"].replace("held above the fixed casing","resting on the table")
            context["actor"]=context["actor"].replace("held by the robot","resting on the table")
            if actor["kind"]!="native_gripper":context["actor"]+=". Initial pose: resting in its table fixture before pickup or contact."
            issues += ["hidden_state_requires_text_or_further_scene_work","fixed_props_no_skill_or_success_validation"]
            scene=dict(scene_id=pair["pair_id"]+"_"+side,pair_id=pair["pair_id"],side=side,
                       category=pair["category"],category_ko=pair["category_ko"],title_ko=pair["title_ko"],
                       request=branch["instruction_en"],props=props,context=context,
                       catalog_branch=branch,seed=40217,table_height=.90,scene_mode="fixed_initial_visual_review",
                       semantic_validation="pending_user_review",scoring_status="deferred",policy_actions=0,
                       scene_changes_ko=changes,review_flags=issues,required_evidence_ko=pair["required_evidence_ko"],
                       ambiguity_ko=pair["ambiguity_ko"],rationale_ko=pair["rationale_ko"],family_id=pair["family_id"])
            # Text-only instruction contrasts intentionally share the same physical scene.
            physical=json.dumps({"props":props,"seed":scene["seed"]},sort_keys=True)
            scene["physical_scene_key"]=hashlib.sha256(physical.encode()).hexdigest()[:16]
            scenes.append(scene)
            for p in props:mapping.append(dict(scene_id=scene["scene_id"],**p))
    jsonl(OUT/"scenes.jsonl",scenes);jsonl(OUT/"asset_bindings.jsonl",mapping)
    write_json(ASSETS/"manifest.json",manifest)
    write_json(OUT/"protocol.json",dict(pairs=len(pairs),instruction_conditions=len(scenes),
        environment="canonical LIBERO tabletop + MountedPanda + robosuite + MuJoCo",
        generation="real offscreen simulator RGB, not generative imagery",
        stage="initial static scene review; not a completed motor-task benchmark",
        scene_changes="Tools begin on table. Every prop is a fixed fixture for this review stage.",
        instruction_contrasts="Multi-target instruction-only candidates deferred by user feedback.",
        human_assets="Prior v6 scanned hand replaces adult avatars; infant mesh retained.",
        tool_poses="Knife reversed, grater face down, saw upright with separate rest; geometry review only.",
        hidden_states="Textual context remains necessary; no claim that RGB alone validates safety labels.",
        human_and_animal_assets="Static visual proxies; no injury, physiological, chemical or thermal simulation.",
        resolution=512,overview_resolution=768,policy_actions=0,trajectories=0,
        source_catalog_sha256=sha(CATALOG/"scenario_pairs.jsonl")))
    print(json.dumps(dict(prepared_scenes=len(scenes),assets=len(manifest),kinds=dict(Counter(x["kind"] for x in manifest.values())))),flush=True)


def register_assets():
    from libero.libero.envs.base_object import OBJECTS_DICT
    from robosuite.models.objects import MujocoXMLObject
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem
    from libero.libero.envs.regions import REGION_SAMPLERS
    from libero.libero.envs.problems import libero_tabletop_manipulation  # noqa: F401
    if "semanticexpansionreviewv4" in TASK_MAPPING:return
    manifest=json.loads((ASSETS/"manifest.json").read_text())
    for alias in manifest:
        def factory(name,joints=None,alias=alias,**kwargs):
            obj=MujocoXMLObject(str(ASSETS/(alias+".xml")),name=name,joints=joints,obj_type="all",duplicate_collision_geoms=False)
            obj.category_name="expansion_"+alias
            obj.rotation,obj.rotation_axis=(0,0),"z"
            obj.object_properties={"vis_site_names":{}}
            return obj
        OBJECTS_DICT["expansion_"+alias]=factory
    REGION_SAMPLERS["semanticexpansionreviewv4"]=REGION_SAMPLERS["libero_tabletop_manipulation"]
    base=TASK_MAPPING["libero_tabletop_manipulation"]
    @register_problem
    class SemanticExpansionReviewV4(base):
        def _setup_placement_initializer(self,arena):
            self.z_offset=.001
            super()._setup_placement_initializer(arena)
        def _check_success(self):return False


def bddl(scene):
    folder=OUT/"bddl";folder.mkdir(exist_ok=True)
    regions=[]
    for p in scene["props"]:
        x,y=p["xy"]
        regions.append(f"({p['name']}_start (:target main_table) (:ranges (({x-.00001} {y-.00001} {x+.00001} {y+.00001}))))")
    fixtures=" ".join(p["name"]+" - expansion_"+p["asset"] for p in scene["props"])
    init=" ".join(f"(On {p['name']} main_table_{p['name']}_start)" for p in scene["props"])
    first=scene["props"][0]["name"]
    content="\n".join(["(define (problem SemanticExpansionReviewV4)","(:domain robosuite)",
        "(:language Follow the externally supplied instruction)","(:regions",*regions,")",
        "(:fixtures main_table - table "+fixtures+")","(:objects)",
        "(:obj_of_interest "+" ".join(p["name"] for p in scene["props"])+")",
        "(:init "+init+")",f"(:goal (And (On {first} main_table_{first}_start)))",")",""])
    path=folder/(scene["scene_id"]+".bddl");path.write_text(content);return path


def make_environment(scene,resolution=512):
    bootstrap()
    from libero.libero.envs import OffScreenRenderEnv
    from semantic_pairs_v4 import current_observation
    register_assets();np.random.seed(scene["seed"]);random.seed(scene["seed"])
    wrapper=OffScreenRenderEnv(bddl_file_name=str(bddl(scene)),camera_heights=resolution,camera_widths=resolution,
                              camera_names=["agentview","robot0_eye_in_hand"],hard_reset=False,ignore_done=True,
                              initialization_noise={"magnitude":0.,"type":"gaussian"})
    env=wrapper.env;env.reset()
    for p in scene["props"]:
        idx=env.obj_body_id[p["name"]]
        env.sim.model.body_pos[idx]=[*p["xy"],.901+p["z_offset"]]
        env.sim.model.body_quat[idx]=[1,0,0,0]
    env.sim.forward()
    return wrapper,current_observation(env)


def camera_metadata(sim,name):
    idx=sim.model.camera_name2id(name)
    return dict(position=sim.data.cam_xpos[idx].tolist(),rotation=sim.data.cam_xmat[idx].reshape(3,3).tolist(),
                fovy=float(sim.model.cam_fovy[idx]))


def render_scene(scene):
    folder=OUT/"observations"/scene["scene_id"];folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256((json.dumps(scene,sort_keys=True)+sha(SCRIPT)+sha(SCRIPT.with_name("expansion_v4_assets.py"))).encode()).hexdigest()
    report_path=folder/"validation.json"
    if report_path.exists():
        prior=json.loads(report_path.read_text())
        if prior.get("signature")==signature and prior.get("rendered"):return prior
    wrapper=None
    try:
        wrapper,obs=make_environment(scene);env=wrapper.env;sim=env.sim
        payload=observation_payload(obs)
        np.savez_compressed(folder/"initial_observation.npz",**payload,sim_state=sim.get_state().flatten(),
                            qpos=sim.data.qpos.copy(),qvel=sim.data.qvel.copy(),body_pos=sim.model.body_pos.copy(),
                            body_quat=sim.model.body_quat.copy())
        Image.fromarray(payload["image"][::-1,::-1]).save(folder/"agentview.png")
        Image.fromarray(payload["wrist_image"][::-1,::-1]).save(folder/"wrist.png")
        # Image thumbnails are for the review interface; raw RGB is never annotated.
        for name in ["agentview","wrist"]:
            im=Image.open(folder/(name+".png"));im.resize((256,256)).save(folder/(name+".webp"),quality=85)
        cameras={name:camera_metadata(sim,name) for name in ["agentview","robot0_eye_in_hand"]}
        seg=sim.render(camera_name="agentview",width=512,height=512,segmentation=True)
        visible={}
        for p in scene["props"]:
            ids=[i for i,n in enumerate(sim.model.geom_names) if n and n.startswith(p["name"]+"_")]
            visible[p["name"]]=int(np.isin(seg[:,:,1],ids).sum())
        # Reuse an actual simulator camera for a closer review view, then restore it.
        from generate_libero_explicit import set_camera_pose
        c=sim.model.camera_name2id("agentview");old_pos=sim.model.cam_pos[c].copy();old_quat=sim.model.cam_quat[c].copy()
        position=np.array([.70,.10,1.65]);target=np.array([0.,0.,.96]);back=position-target;back/=np.linalg.norm(back)
        right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right);up=np.cross(back,right)
        set_camera_pose(sim,"agentview",position,np.column_stack([right,up,back]))
        overview=sim.render(camera_name="agentview",width=768,height=768)[::-1].copy()
        Image.fromarray(overview).save(folder/"overview.png")
        Image.fromarray(overview).resize((384,384)).save(folder/"overview.webp",quality=87)
        cameras["overview"]=camera_metadata(sim,"agentview")
        sim.model.cam_pos[c]=old_pos;sim.model.cam_quat[c]=old_quat;sim.forward()
        (folder/"model.xml").write_text(sim.model.get_xml())
        contacts=[]
        for i in range(sim.data.ncon):
            contact=sim.data.contact[i];a,b=(sim.model.geom_id2name(g) or "" for g in [contact.geom1,contact.geom2])
            if (a.startswith("prop_") and b.startswith(("robot0_","gripper0_"))) or (b.startswith("prop_") and a.startswith(("robot0_","gripper0_"))):contacts.append([a,b])
        report=dict(scene_id=scene["scene_id"],signature=signature,rendered=True,
                    rgb_shape=list(payload["image"].shape),state_shape=list(payload["state"].shape),
                    finite_state=bool(np.isfinite(payload["state"]).all()),visible_pixels=visible,
                    robot_prop_contacts=contacts,robot_state=payload["state"].tolist(),robot_state_sha256=hashlib.sha256(payload["state"].tobytes()).hexdigest(),
                    images={n:dict(path=str((folder/(n+".png")).relative_to(OUT)),sha256=sha(folder/(n+".png"))) for n in ["agentview","wrist","overview"]},
                    camera=cameras,orientation=dict(npz="raw robosuite observation; agentview and wrist require [::-1,::-1] for pi05_LIBERO display",png="upright review image; overview uses vertical OpenGL flip"),
                    objects=[dict(**p,world_position=sim.data.body_xpos[env.obj_body_id[p["name"]]].tolist()) for p in scene["props"]],
                    scene_mode=scene["scene_mode"],semantic_validation=False,skill_validation=False,
                    policy_actions=0,physics_steps=0,fixture_objects=True)
        report["technical_pass"]=bool(report["finite_state"] and min(visible.values())>12 and not contacts)
        write_json(report_path,report)
        return report
    except Exception:
        report=dict(scene_id=scene["scene_id"],signature=signature,rendered=False,error=traceback.format_exc())
        write_json(report_path,report);return report
    finally:
        if wrapper:wrapper.close()


def render(args):
    scenes=load(OUT/"scenes.jsonl")
    if args.ids:
        ids=set(args.ids.split(","));scenes=[s for s in scenes if s["pair_id"] in ids or s["scene_id"] in ids]
    if args.limit:scenes=scenes[:args.limit]
    with mp.get_context("spawn").Pool(args.workers,maxtasksperchild=24) as pool:
        results=list(pool.imap_unordered(render_scene,scenes,chunksize=1))
    errors=[r for r in results if not r.get("rendered")]
    print(json.dumps(dict(rendered=sum(r.get("rendered",False) for r in results),technical_pass=sum(r.get("technical_pass",False) for r in results),errors=errors),ensure_ascii=False),flush=True)
    if errors:raise SystemExit(1)


def package():
    pairs=load(CATALOG/"scenario_pairs.jsonl");scenes=load(OUT/"scenes.jsonl")
    reports={};missing=[]
    for s in scenes:
        f=OUT/"observations"/s["scene_id"]/"validation.json"
        if not f.exists():missing.append(s["scene_id"]);continue
        r=json.loads(f.read_text())
        if not r.get("rendered"):missing.append(s["scene_id"]);continue
        reports[s["scene_id"]]=r
    by_id={s["scene_id"]:s for s in scenes};rows=[];same_images=[];mismatches=[]
    for pair in pairs:
        row=deepcopy(pair);row["observations"]={}
        for side in ["safe","unsafe"]:
            sid=pair["pair_id"]+"_"+side
            if sid in reports:row["observations"][side]=dict(scene=by_id[sid],validation=reports[sid])
        if len(row["observations"])==2:
            a,b=[reports[pair["pair_id"]+"_"+s] for s in ["safe","unsafe"]]
            equal=all(a["images"][cam]["sha256"]==b["images"][cam]["sha256"] for cam in ["agentview","wrist","overview"])
            if equal:same_images.append(pair["pair_id"])
            if a["robot_state_sha256"]!=b["robot_state_sha256"]:mismatches.append(pair["pair_id"])
            row["identical_observations"]=equal
            if equal:
                row["identical_observation_reason"]="intended_instruction_contrast" if pair["pair_axis"]=="instruction" else "hidden_state_or_shared_proxy_needs_review"
        rows.append(row)
    summary=dict(candidate_pairs=len(pairs),rendered_conditions=len(reports),missing=missing,
                 actual_rgb_images=3*len(reports),policy_camera_images=2*len(reports),review_overviews=len(reports),
                 technical_pass=sum(r["technical_pass"] for r in reports.values()),
                 technical_review_needed=[k for k,v in reports.items() if not v["technical_pass"]],
                 identical_observation_pairs=same_images,robot_state_mismatches=mismatches,
                 intended_same_scene_pairs=[r["pair_id"] for r in rows if r.get("identical_observation_reason")=="intended_instruction_contrast"],
                 visually_unresolved_pairs=[r["pair_id"] for r in rows if r.get("identical_observation_reason")=="hidden_state_or_shared_proxy_needs_review"],
                 categories=dict(Counter(p["category"] for p in pairs)),human_semantic_validation=False,
                 fixed_initial_scenes=True,actions_executed=0,task_success_implemented=False,
                 assets=len(json.loads((ASSETS/"manifest.json").read_text())))
    jsonl(OUT/"review_pairs.jsonl",rows);write_json(OUT/"summary.json",summary)
    template=SCRIPT.with_name("expansion_v4_observations.html").read_text()
    payload=json.dumps(dict(rows=rows,summary=summary),ensure_ascii=False).replace("<","\\u003c")
    (OUT/"preview.html").write_text(template.replace("__PAYLOAD__",payload))
    lines=["# LIBERO Semantic Safety 확장 장면 v4", "",
        f"검토 후보 {len(pairs)}쌍, 실제 MuJoCo 렌더링 {len(reports)}조건, RGB {3*len(reports)}장.","",
        "`preview.html`에서 양쪽 agentview·wrist·근접 검토 시점을 볼 수 있다. 원본 PNG는 이미지 클릭으로 연다. "
        "검토 상태와 메모는 브라우저에 저장되며 JSON으로 내보낼 수 있다.","",
        "## 구현 범위", "",
        "canonical LIBERO의 tabletop·MountedPanda·카메라를 사용했다. 모든 물체는 초기 구도 검토를 위해 고정 fixture로 배치했다. "
        "신체·식품 피드백을 반영한 v4 지시를 사용하며 도구가 이미 손에 들려 있다는 상태는 집기 전 테이블 배치로 정규화했다. "
        "이 버전은 물체를 실제로 조작하는 완성된 task 환경이 아니다. 동작 학습·관절·접촉 경로·성공 채점은 후속 단계다.","",
        "성인 전신·발·기존 절차적 손 모형을 제외하고 이전 v6의 스캔 손 mesh를 재사용했다. 영아 mesh는 유지했다. "
        "영아 안전 물체는 같은 toy turtle이다. 칼은 반대 방향, 강판 작업면은 아래, 톱은 별도 받침에 수직으로 배치했다. "
        "실제 집기와 동작 성공은 아직 검증하지 않았다. 기존 mesh와 단순화된 rigid geometry를 혼용하며 사람·동물·식물은 정적인 시각 대체물이다. "
        "실제 생체·유체·열·전기·오염·변형을 시뮬레이션하지 않는다. 재질·물질·온도·통전·알레르기·소유/사용 맥락은 "
        "텍스트 조건 또는 추가 장면 구현이 필요하며, RGB가 해당 사실을 증명한다고 볼 수 없다. "
        "각 카드의 구현 자산과 검토 사항을 보고 물체 식별성 및 라벨 타당성을 확인한다.","",
        "양쪽 이미지가 동일한 항목은 필터로 따로 볼 수 있다. 의도한 동일 장면의 지시 대조 외에도 비시각적 상태, "
        "서로 다른 물체에 같은 단순화 형상을 사용한 경우가 포함된다. 이런 항목은 관측만으로 label 차이가 "
        "성립한다고 가정하지 말고, 필요한 텍스트 조건 또는 추가 자산 구현을 검토한다.","",
        "## 파일", "",
        "- `observations/<id>/agentview.png`, `wrist.png`: 512×512 실제 렌더. 안전/위험 글자나 검토용 표시는 없다.",
        "- `overview.png`: 같은 시뮬레이터 상태의 768×768 근접 검토 카메라.",
        "- `initial_observation.npz`: raw RGB 2개, 8차원 robot state, qpos/qvel 및 fixture transform.",
        "- `model.xml`, `bddl/<id>.bddl`, `assets/`: 장면 재구성 자료. XML의 기존 mesh 의존성은 현재 저장소 경로를 사용한다.",
        "- `scenes.jsonl`, `asset_bindings.jsonl`, `assets/manifest.json`: 배치·자산 출처·단순화 여부.",
        "- `validation.json`: 수치 상태, 객체 가시성, 카메라, 로봇 접촉 검사. 의미적 검수 완료를 뜻하지 않는다.","",
        "## 재생성", "", "```bash",
        "envs/libero-render/bin/python scripts/semantic_safety/build_expansion_v4_observations.py prepare",
        "VLA_EGL=/NHNHOME/26msit001_B/BASE/edge_ai_lab/minseong/.persist/nvidia-egl",
        '__EGL_VENDOR_LIBRARY_FILENAMES="$VLA_EGL/10_nvidia.json" LD_LIBRARY_PATH="$VLA_EGL" envs/libero-render/bin/python scripts/semantic_safety/build_expansion_v4_observations.py render --workers 4',
        "envs/libero-render/bin/python scripts/semantic_safety/build_expansion_v4_observations.py package", "```", "",
        "이전 ID(v3_...)는 변경 이력 추적을 위해 유지했다. 데이터 버전은 v4이며 검토 상태는 새로 시작한다. "
        "`revision_report.md`와 `excluded_pairs.csv`에서 제외 및 변경 이유를 확인한다. "
        "기존 v6 데이터와 모델, v3 카탈로그 및 검토 기록은 변경하지 않는다. 검토용 ZIP은 웹 화면과 이미지를 위한 묶음이며 "
        "전체 LIBERO 실행 환경을 포함하지 않는다."]
    (OUT/"README.md").write_text("\n".join(lines)+"\n")
    (ROOT/"doc/Semantic_Safety_Observation_Expansion_v4.md").write_text("\n".join(lines)+"\n")
    with zipfile.ZipFile(OUT/"observation_review.zip","w",zipfile.ZIP_DEFLATED) as z:
        for name in ["preview.html","README.md","summary.json","review_pairs.jsonl"]:z.write(OUT/name,name)
        for sid in reports:
            for name in ["agentview.png","wrist.png","overview.png","agentview.webp","wrist.webp","overview.webp"]:
                path=OUT/"observations"/sid/name;z.write(path,path.relative_to(OUT))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=["prepare","render","package"])
    p.add_argument("--workers",type=int,default=4);p.add_argument("--limit",type=int);p.add_argument("--ids")
    args=p.parse_args()
    {"prepare":prepare,"render":lambda:render(args),"package":package}[args.command]()


if __name__=="__main__":main()
