"""Render the user's reviewed v5 candidates in actual LIBERO/MuJoCo scenes.

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

from expansion_v5_assets import ASSETS, OUT, build_asset, choose_kind
from expansion_v5_native import reference, make_native_environment, adapted_bddl
from rollout_common import ROOT, bootstrap, observation_payload, write_json

CATALOG = ROOT / "data/semantic_safety_catalog_v5"
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
    def asset(name,state="",material="",kind=None,max_extent=None,dimensions=None,surface_marks=None):
        key=json.dumps([name,state,material,kind,max_extent,dimensions,surface_marks],sort_keys=True)
        if key not in cache:
            cache[key]=build_asset(name,state,material,kind_override=kind,max_extent=max_extent,
                                   dimensions=dimensions,surface_marks=surface_marks)
            if cache[key]["alias"]:manifest[cache[key]["alias"]]=cache[key]
        return cache[key]
    native=reference()
    def branch_asset(branch,role):
        return asset(branch[role],branch[role+"_state"],branch.get(role+"_material",""),
                     branch.get(role+"_kind"),branch.get(role+"_max_extent"),branch.get(role+"_dimensions"),
                     branch.get(role+"_surface_marks"))
    for pair in pairs:
        for side in ["safe","unsafe"]:
            branch=pair[side]; props=[]; changes=[]; issues=[]
            actor=branch_asset(branch,"actor")
            is_native=pair.get("scene_template")=="libero_long_kitchen_scene6"
            def put(binding,role,xy,z=0):
                if binding["alias"]:
                    props.append(dict(name=f"prop_{len(props)+1}",role=role,asset=binding["alias"],
                                      object_name=binding["name"],xy=xy,z_offset=z,kind=binding["kind"],fidelity=binding["fidelity"],
                                      pose_adjustment=binding.get("pose_adjustment"),task_object=role!="support_fixture"))
            if is_native:
                put(actor,"actor",native["objects"]["white_yellow_mug_1"]["world_position"][:2])
                for name,role,kind,label in [("microwave_1","target","native:microwave","microwave oven"),
                                           ("porcelain_mug_1","background","native:porcelain_mug","original background mug")]:
                    pos=native["objects"][name]["world_position"]
                    props.append(dict(name=name,role=role,asset=None,object_name=label,xy=pos[:2],z_offset=pos[2]-.901,
                                      kind=kind,fidelity="native_libero_asset",pose_adjustment=None,task_object=role!="background"))
                changes.append("실제 LIBERO Long kitchen scene6 및 공식 pruned_init[0]의 로봇·전자레인지·배경 머그 상태 사용")
                issues.append("native_scene_has_original_background_mug_and_articulated_microwave")
            else:
                z=.03 if actor["kind"] in {"mesh:knife","blade"} else (.006 if actor["kind"]=="saw" else 0)
                put(actor,"actor",[-.055,.15],z)
                if actor["kind"]=="saw":
                    put(asset("upright saw cradle", "fixed neutral support", "rigid support"),
                        "support_fixture",[-.055,.12])
                    changes.append("톱은 별도 고정 받침에 세웠다. 받침은 과제 물체 수에 포함하지 않으며 집을 도구와 분리되어 있다.")
                if actor["kind"] in {"mesh:knife","blade"}:
                    put(asset("upright knife cradle","fixed neutral support","rigid support"),"support_fixture",[-.055,.11])
                    changes.append("칼은 별도 고정 받침에 세워 손잡이 아래 집기 공간을 확보했다. 집기 성공은 아직 평가하지 않았다.")
                put(branch_asset(branch,"target"),"target",[.035,-.17])
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
                       catalog_branch=branch,seed=40217,table_height=.90,
                       scene_template=pair.get("scene_template","paired_tabletop"),
                       scene_mode="native_initial_visual_review" if is_native else "fixed_initial_visual_review",
                       native_reference=native if is_native else None,
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
        scene_changes="Tabletop props are fixed. Microwave scenes retain native articulation and movable object joints; no actions are executed.",
        instruction_contrasts="Multi-target instruction-only candidates deferred by user feedback.",
        human_assets="Prior v6 scanned hand replaces adult avatars; infant mesh retained.",
        tool_poses="All knives and saws upright in separate holders; designated contact faces down; geometry review only.",
        native_microwave_reference="native_reference/reference.json",
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
    if "semanticexpansionreviewv5" in TASK_MAPPING:return
    manifest=json.loads((ASSETS/"manifest.json").read_text())
    for alias in manifest:
        def factory(name,joints="default",alias=alias,**kwargs):
            obj=MujocoXMLObject(str(ASSETS/(alias+".xml")),name=name,joints=joints,obj_type="all",duplicate_collision_geoms=False)
            obj.category_name="expansion_"+alias
            obj.rotation,obj.rotation_axis=(0,0),"z"
            obj.object_properties={"vis_site_names":{}}
            return obj
        OBJECTS_DICT["expansion_"+alias]=factory
    REGION_SAMPLERS["semanticexpansionreviewv5"]=REGION_SAMPLERS["libero_tabletop_manipulation"]
    base=TASK_MAPPING["libero_tabletop_manipulation"]
    @register_problem
    class SemanticExpansionReviewV5(base):
        def _setup_placement_initializer(self,arena):
            self.z_offset=.001
            super()._setup_placement_initializer(arena)
        def _check_success(self):return False
    from libero.libero.envs.problems import libero_kitchen_tabletop_manipulation  # noqa: F401
    REGION_SAMPLERS["semanticexpansionkitchenv5"]=REGION_SAMPLERS["libero_kitchen_tabletop_manipulation"]
    kitchen=TASK_MAPPING["libero_kitchen_tabletop_manipulation"]
    @register_problem
    class SemanticExpansionKitchenV5(kitchen):
        def _check_success(self):return False


def bddl(scene):
    if scene["scene_template"]=="libero_long_kitchen_scene6":return adapted_bddl(scene)
    folder=OUT/"bddl";folder.mkdir(exist_ok=True)
    regions=[]
    for p in scene["props"]:
        x,y=p["xy"]
        regions.append(f"({p['name']}_start (:target main_table) (:ranges (({x-.00001} {y-.00001} {x+.00001} {y+.00001}))))")
    fixtures=" ".join(p["name"]+" - expansion_"+p["asset"] for p in scene["props"])
    init=" ".join(f"(On {p['name']} main_table_{p['name']}_start)" for p in scene["props"])
    first=scene["props"][0]["name"]
    content="\n".join(["(define (problem SemanticExpansionReviewV5)","(:domain robosuite)",
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
    if scene["scene_template"]=="libero_long_kitchen_scene6":return make_native_environment(scene,resolution)
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
    signature=hashlib.sha256((json.dumps(scene,sort_keys=True)+sha(SCRIPT)+sha(SCRIPT.with_name("expansion_v5_assets.py"))+sha(SCRIPT.with_name("expansion_v5_native.py"))).encode()).hexdigest()
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
        from mujoco import mjtObj
        np.savez_compressed(folder/"object_segmentation.npz",object_type=seg[:,:,0],object_id=seg[:,:,1])
        visible={};centroids={};geom_ids={}
        for p in scene["props"]:
            # geom_names omits repeated unnamed geoms in the legacy wrapper.
            # Its tuple index is therefore not a valid MuJoCo ID in native kitchens.
            ids=[sim.model.geom_name2id(n) for n in sim.model.geom_names if n and n.startswith(p["name"]+"_")]
            geom_ids[p["name"]]=ids
            mask=(seg[:,:,0]==int(mjtObj.mjOBJ_GEOM)) & np.isin(seg[:,:,1],ids)
            visible[p["name"]]=int(mask.sum())
            yy,xx=np.where(mask)
            centroids[p["name"]]=[float(511-xx.mean()),float(511-yy.mean())] if len(xx) else None
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
            prefixes=tuple(p["name"]+"_" for p in scene["props"])
            if (a.startswith(prefixes) and b.startswith(("robot0_","gripper0_"))) or (b.startswith(prefixes) and a.startswith(("robot0_","gripper0_"))):contacts.append([a,b])
        report=dict(scene_id=scene["scene_id"],signature=signature,rendered=True,
                    rgb_shape=list(payload["image"].shape),state_shape=list(payload["state"].shape),
                    finite_state=bool(np.isfinite(payload["state"]).all()),visible_pixels=visible,
                    visibility_geom_ids=geom_ids,visibility_id_source="actual_mujoco_geom_name2id",
                    agentview_centroids=centroids,
                    robot_prop_contacts=contacts,robot_state=payload["state"].tolist(),robot_state_sha256=hashlib.sha256(payload["state"].tobytes()).hexdigest(),
                    images={n:dict(path=str((folder/(n+".png")).relative_to(OUT)),sha256=sha(folder/(n+".png"))) for n in ["agentview","wrist","overview"]},
                    camera=cameras,orientation=dict(npz="raw robosuite observation; agentview and wrist require [::-1,::-1] for pi05_LIBERO display",png="upright review image; overview uses vertical OpenGL flip"),
                    objects=[dict(**p,world_position=sim.data.body_xpos[env.obj_body_id[p["name"]]].tolist()) for p in scene["props"]],
                    scene_mode=scene["scene_mode"],semantic_validation=False,skill_validation=False,
                    policy_actions=0,physics_steps=0,fixture_objects=scene["scene_template"]!="libero_long_kitchen_scene6")
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
                 initial_scenes_only=True,tabletop_props_fixed=True,native_articulation_retained=True,
                 native_microwave_conditions=sum(s["scene_template"]=="libero_long_kitchen_scene6" for s in scenes),
                 actions_executed=0,task_success_implemented=False,
                 assets=len(json.loads((ASSETS/"manifest.json").read_text())))
    jsonl(OUT/"review_pairs.jsonl",rows);write_json(OUT/"summary.json",summary)
    template=SCRIPT.with_name("expansion_v5_observations.html").read_text()
    payload=json.dumps(dict(rows=rows,summary=summary),ensure_ascii=False).replace("<","\\u003c")
    (OUT/"preview.html").write_text(template.replace("__PAYLOAD__",payload))
    from expansion_v5_documentation import write_docs
    write_docs(summary)
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
