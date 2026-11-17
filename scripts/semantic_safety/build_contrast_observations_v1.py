"""Render fixed, two-task-object LIBERO scenes for the contrast-set review."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import traceback
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image

from contrast_dataset_v1 import OUT, ROOT, author, write_jsonl
from rollout_common import bootstrap, state_vector

ASSETS = OUT / "assets"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def shape(kind, name, state, material):
    import expansion_v5_assets as a
    b, e, c, r = a.box, a.ell, a.cyl, a.rod
    white, dark, metal = a.WHITE, a.DARK, a.METAL
    if kind == "plate":
        return [c(.085,.004,[0,0,.004],white)]+a.ring(.085,.009,white,.005)
    if kind == "basin":
        return a.tray([.48,.61,.65,1],.09,.07,.065)
    if kind == "waste_bin":
        return a.tray([.49,.46,.40,1],.075,.065,.105)
    if kind == "closed_box":
        return [b([.08,.065,.05],[0,0,.05],white),b([.084,.069,.004],[0,0,.105],metal)]
    if kind == "steel_pot":
        return a.bowl(metal,.065,.055)+[c(.052,.002,[0,0,.035],[.56,.72,.76,.8])]+[
            r([x,-.02,.045],[x,.02,.045],.007,dark) for x in [-.085,.085]]
    if kind == "plastic_box":
        return a.tray([.28,.51,.67,1],.07,.055,.04)+[b([.058,.045,.008],[0,0,.014],[.72,.48,.27,1])]
    if kind == "sponge":
        return [b([.052,.034,.012],[0,0,.012],[.84,.70,.23,1]),b([.052,.034,.003],[0,0,.027],[.20,.38,.22,1])]
    if kind == "cotton_pad":
        return [c(.047,.004,[0,0,.004],white)]
    if kind == "bandage":
        return [c(.029,.026,[0,0,.027],white),c(.009,.001,[0,0,.054],[.62,.61,.56,1])]
    if kind == "soft_brush":
        return [r([-.08,0,.014],[.015,0,.014],.009,dark),e([.035,.025,.02],[.04,0,.02],[.78,.66,.51,1])]
    if kind == "cloth_ball":
        return [e([.049,.049,.049],[0,0,.05],[.34,.52,.67,1])]+[
            r([-.035,y,.081],[.035,y,.081],.0008,white) for y in [-.015,.015]]
    if kind == "foam_block":
        return [b([.048,.042,.04],[0,0,.04],[.67,.47,.29,1])]
    if kind == "rubber_duck":
        yellow=[.94,.72,.14,1]
        return [e([.048,.032,.033],[0,0,.034],yellow),e([.025,.023,.026],[-.03,0,.077],yellow),
                e([.02,.015,.006],[-.055,0,.073],[.94,.37,.10,1]),e([.003,.003,.003],[-.04,-.021,.086],dark)]
    if kind == "washer":
        return a.ring(.022,.005,metal,.007)
    if kind == "apple":
        return [e([.037,.035,.038],[0,0,.04],[.70,.15,.12,1]),r([0,0,.072],[.004,0,.087],.002,a.WOOD),
                e([.015,.007,.002],[.014,0,.082],a.GREEN)]
    if kind == "banana":
        return [e([.025,.017,.013],[x,y,.014],[.93,.76,.16,1]) for x,y in [(-.05,.014),(-.025,.003),(0,0),(.025,.003),(.05,.014)]]
    if kind == "cheese":
        return [b([.043,.032,.019],[0,0,.02],[.93,.75,.25,1])]+[
            e([.006,.004,.002],[x,y,.040],[.74,.53,.17,1]) for x,y in [(-.02,.015),(.015,-.01),(.025,.02)]]
    if kind == "painted_panel":
        return [b([.075,.055,.008],[0,0,.009],a.WOOD),b([.074,.054,.0005],[0,0,.018],[.22,.39,.59,1])]
    if kind == "photograph":
        return [b([.075,.052,.0015],[0,0,.002],white),b([.067,.044,.0003],[0,0,.004],[.38,.59,.72,1]),
                e([.045,.023,.0004],[.015,0,.0046],[.26,.43,.25,1]),c(.012,.0004,[-.04,.02,.005],[.91,.78,.25,1])]
    if kind == "cardboard":
        return [b([.075,.055,.004],[0,0,.005],[.59,.42,.25,1])]
    if kind == "power_tool":
        blue=[.13,.36,.49,1]
        return [b([.024,.047,.025],[0,0,.065],blue),
                b([.014,.017,.029],[0,-.025,.033],dark),
                b([.029,.030,.008],[0,-.025,.009],dark),
                c(.018,.013,[0,.057,.065],metal,euler=[np.pi/2,0,0]),
                r([0,.070,.065],[0,.115,.065],.0035,metal),
                b([.006,.004,.006],[.015,-.006,.041],metal)]
    if kind in {"pressure_canister","practice_canister"}:
        p=a.bowl(metal,.043,.105)
        if kind=="pressure_canister":
            p = [c(.043,.054,[0,0,.054],metal),c(.044,.004,[0,0,.109],metal),c(.013,.012,[0,0,.124],dark),
                  c(.023,.004,[.028,0,.14],white, euler=[0,np.pi/2,0]),
                  r([.034,0,.14],[.034,.012,.153],.0015,dark)]
        return p
    return ORIGINAL_GEOMETRY(kind,name,state,material)


def configure_assets():
    import expansion_v5_assets as a
    global ORIGINAL_GEOMETRY
    ORIGINAL_GEOMETRY = a.geometry
    a.OUT, a.ASSETS, a.geometry = OUT, ASSETS, shape
    return a


def build_object(spec, a):
    kind=spec["kind"]
    if kind.startswith("rc:"):
        from rollout_asset_adapter import bounds
        from semantic_pairs_v4 import vector, sites
        source=ROOT/"data/robocasa_libero_import_v1/converted"/(kind.split(":")[1]+".xml")
        root=ET.parse(source).getroot()
        outer=root.find("worldbody/body"); body=root.find(".//body[@name='object']")
        for site in list(outer.findall("site")):outer.remove(site)
        if kind=="rc:knife":
            transform=ET.Element("body",name="blade_vertical",euler="0 1.57079632679 0")
            for child in list(body):body.remove(child);transform.append(child)
            body.append(transform)
        if spec["visual"].get("surface_marks"):
            a.add_surface_marks(root,body,green=True)
        lo,hi=bounds(root); shift=-(lo+hi)/2;shift[2]=-lo[2]
        transform=ET.Element("body",name="centered_visual",pos=vector(shift))
        for child in list(body):body.remove(child);transform.append(child)
        body.append(transform)
        extent=hi-lo
        sites(outer,extent)
        alias="r"+digest(spec)[:14]
        path=ASSETS/(alias+".xml");ET.indent(root);ET.ElementTree(root).write(path,encoding="unicode")
        return dict(alias=alias,kind=kind,name=spec["name_en"],extent=extent.tolist(),
                    fidelity="robocasa_textured_mesh",source_xml=str(source.relative_to(ROOT)))
    result=a.build_asset(spec["name_en"],spec["state"],spec["material"],kind_override=kind,
                         surface_marks=spec["visual"].get("surface_marks"))
    return result


def prepare():
    objects,groups,conditions=author()
    OUT.mkdir(parents=True,exist_ok=True); ASSETS.mkdir(exist_ok=True)
    a=configure_assets(); bindings={}
    used={r[k] for r in conditions for k in ["actor_id","target_id"]}
    for key in sorted(used):
        bindings[key]=build_object(objects[key],a)
    # Tool holders are neutral fixtures, not additional task objects.
    for key in ["knife_cradle","saw_cradle"]:
        bindings[key]=a.build_asset(key.replace("_"," "),kind_override=key)
    scenes={}
    for row in conditions:
        props=[]
        for index,role in enumerate(["actor","target"]):
            oid=row[role+"_id"]; binding=bindings[oid]
            p=dict(name=role,object_id=oid,asset=binding["alias"],xy=[-.055,.15] if index==0 else [.035,-.17],
                   z_offset=.03 if role=="actor" and oid=="knife" else (.006 if role=="actor" and oid=="saw" else 0.),
                   role=role,task_object=True)
            props.append(p)
            if role=="actor" and oid in {"knife","saw"}:
                holder="knife_cradle" if oid=="knife" else "saw_cradle"
                props.append(dict(name="holder",object_id=holder,asset=bindings[holder]["alias"],xy=[-.055,.11],
                                  z_offset=0.,role="support",task_object=False))
        scene_id="s_"+digest(props)[:14]
        if scene_id not in scenes:
            scenes[scene_id]=dict(scene_id=scene_id,props=props,seed=40217,task_objects=2,
                                  scene_mode="fixed_initial_review",policy_actions=0)
        row["scene_id"]=scene_id
        row["actor"]=objects[row["actor_id"]]
        row["target"]=objects[row["target_id"]]
        row["asset_fidelity"]={role:bindings[row[role+"_id"]]["fidelity"] for role in ["actor","target"]}
        if "schematic_rigid_proxy" in row["asset_fidelity"].values():
            row["review_flags"].append("schematic_asset_identity_requires_review")
    write_jsonl(OUT/"objects.jsonl",objects.values())
    write_jsonl(OUT/"contrast_sets.jsonl",groups)
    write_jsonl(OUT/"conditions.jsonl",conditions)
    write_jsonl(OUT/"scenes.jsonl",scenes.values())
    save(ASSETS/"manifest.json",{b["alias"]:b for b in bindings.values()})
    save(OUT/"asset_bindings.json",bindings)
    save(OUT/"protocol.json",dict(version="semantic_safety_contrast_v1",label="request-level harm under normal compliance",
         stage="human review candidates; no policy training, probing, rollout, or consequence simulation",
         track_A="complete factual instruction; no safe/unsafe verdict or rationale in model input",
         track_B="short instruction with required visual cue; candidate availability is not human validation",
         cameras="native LIBERO agentview and wrist; separate closer overview is review-only",
         task_objects=2,static_props=True,semantic_conditions=len(conditions),contrast_sets=len(groups),
         unique_physical_scenes=len(scenes),complex_allergy_crosses=0,layout_augmentation=0,
         previous_data_modified=False,model_rgb_orientation="upright RGB; vertical flip of OpenGL render only",
         split_status="deferred; shared controls must be purged from train/dev for held-out contrast evaluation",
         sources=["https://www.cdc.gov/hygiene/about/cleaning-and-disinfecting-with-bleach.html",
                  "https://www.osha.gov/lithium-ion-battery-safety/hazards"]))
    print(json.dumps(dict(sets=len(groups),conditions=len(conditions),scenes=len(scenes),assets=len(bindings))),flush=True)


def register():
    from libero.libero.envs.base_object import OBJECTS_DICT
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem
    from libero.libero.envs.regions import REGION_SAMPLERS
    from libero.libero.envs.problems import libero_tabletop_manipulation  # noqa: F401
    from robosuite.models.objects import MujocoXMLObject
    if "semanticcontrastreviewv1" in TASK_MAPPING:return
    manifest=json.loads((ASSETS/"manifest.json").read_text())
    for alias in manifest:
        def factory(name,joints=None,alias=alias,**kwargs):
            result=MujocoXMLObject(str(ASSETS/(alias+".xml")),name=name,joints=None,
                                  obj_type="all",duplicate_collision_geoms=False)
            result.category_name="contrast_"+alias
            result.rotation,result.rotation_axis=(0,0),"z"
            result.object_properties={"vis_site_names":{}}
            return result
        OBJECTS_DICT["contrast_"+alias]=factory
    REGION_SAMPLERS["semanticcontrastreviewv1"]=REGION_SAMPLERS["libero_tabletop_manipulation"]
    base=TASK_MAPPING["libero_tabletop_manipulation"]
    @register_problem
    class SemanticContrastReviewV1(base):
        def _setup_placement_initializer(self,arena):
            self.z_offset=.001
            super()._setup_placement_initializer(arena)
        def _check_success(self):return False


def make_environment(scene):
    bootstrap()
    from libero.libero.envs import OffScreenRenderEnv
    register()
    regions=[]
    for p in scene["props"]:
        x,y=p["xy"]
        regions.append(f"({p['name']}_start (:target main_table) (:ranges (({x-.00001} {y-.00001} {x+.00001} {y+.00001}))))")
    fixtures=" ".join(p["name"]+" - contrast_"+p["asset"] for p in scene["props"])
    initial=" ".join(f"(On {p['name']} main_table_{p['name']}_start)" for p in scene["props"])
    content="\n".join(["(define (problem SemanticContrastReviewV1)","(:domain robosuite)",
        "(:language Follow the externally supplied instruction)","(:regions",*regions,")",
        "(:fixtures main_table - table "+fixtures+")","(:objects)",
        "(:obj_of_interest actor target)","(:init "+initial+")",
        "(:goal (And (On actor main_table_actor_start)))", ")", ""])
    folder=OUT/"bddl";folder.mkdir(exist_ok=True)
    path=folder/(scene["scene_id"]+".bddl");path.write_text(content)
    np.random.seed(scene["seed"]);random.seed(scene["seed"])
    wrapper=OffScreenRenderEnv(bddl_file_name=str(path),camera_heights=512,camera_widths=512,
                              camera_names=["agentview","robot0_eye_in_hand"],hard_reset=False,ignore_done=True,
                              initialization_noise={"magnitude":0.,"type":"gaussian"})
    env=wrapper.env;obs=env.reset()
    for p in scene["props"]:
        idx=env.obj_body_id[p["name"]]
        env.sim.model.body_pos[idx]=[*p["xy"],.901+p["z_offset"]]
        env.sim.model.body_quat[idx]=[1,0,0,0]
    env.sim.forward()
    from semantic_pairs_v4 import current_observation
    return wrapper,current_observation(env)


def render_one(scene):
    folder=OUT/"observations"/scene["scene_id"];folder.mkdir(parents=True,exist_ok=True)
    signature=digest([scene,hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      [(ASSETS/(p["asset"]+".xml")).read_text() for p in scene["props"]]])
    report_path=folder/"validation.json"
    if report_path.exists():
        old=json.loads(report_path.read_text())
        if old.get("signature")==signature and old.get("rendered"):return old
    wrapper=None
    try:
        wrapper,obs=make_environment(scene);env=wrapper.env;sim=env.sim
        state=state_vector(obs)
        images={}
        rgbs={}
        for name,camera in [("agentview","agentview"),("wrist","robot0_eye_in_hand")]:
            rgb=sim.render(camera_name=camera,width=512,height=512)[::-1].copy()
            rgbs[name]=rgb
            Image.fromarray(rgb).save(folder/(name+".png"))
            Image.fromarray(rgb).resize((256,256),Image.Resampling.LANCZOS).save(folder/(name+".webp"),quality=88)
            Image.fromarray(rgb).resize((224,224),Image.Resampling.LANCZOS).save(folder/(name+"_224.png"))
            images[name]=dict(path=str((folder/(name+".png")).relative_to(OUT)),
                              sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        np.savez_compressed(folder/"initial_observation.npz",agentview_rgb=rgbs["agentview"],
                            wrist_rgb=rgbs["wrist"],robot_state=state,sim_state=sim.get_state().flatten())
        seg=sim.render(camera_name="agentview",width=512,height=512,segmentation=True)[::-1].copy()
        from mujoco import mjtObj
        pixels={}
        for p in scene["props"]:
            ids=[sim.model.geom_name2id(n) for n in sim.model.geom_names if n and n.startswith(p["name"]+"_")]
            mask=(seg[:,:,0]==int(mjtObj.mjOBJ_GEOM)) & np.isin(seg[:,:,1],ids)
            pixels[p["name"]]=int(mask.sum())
        # A close camera is a review aid; it never replaces the model camera.
        from generate_libero_explicit import set_camera_pose
        position=np.array([.65,.05,1.60]);target=np.array([0.,0.,.97])
        back=position-target;back/=np.linalg.norm(back)
        right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right);up=np.cross(back,right)
        set_camera_pose(sim,"agentview",position,np.column_stack([right,up,back]))
        rgb=sim.render(camera_name="agentview",width=768,height=768)[::-1].copy()
        Image.fromarray(rgb).save(folder/"overview.png")
        Image.fromarray(rgb).resize((384,384)).save(folder/"overview.webp",quality=88)
        images["overview"]=dict(path=str((folder/"overview.png").relative_to(OUT)),sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        contacts=[]
        for index in range(sim.data.ncon):
            contact=sim.data.contact[index]
            names=[sim.model.geom_id2name(i) or "" for i in [contact.geom1,contact.geom2]]
            if any(n.startswith(("actor_","target_")) for n in names) and any(n.startswith(("robot0_","gripper0_")) for n in names):contacts.append(names)
        report=dict(scene_id=scene["scene_id"],signature=signature,rendered=True,images=images,
                    finite_state=bool(np.isfinite(state).all()),visible_pixels=pixels,robot_prop_contacts=contacts,
                    robot_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                    technical_pass=bool(np.isfinite(state).all() and min(pixels.get(k,0) for k in ["actor","target"])>12 and not contacts),
                    semantic_validated=False,skill_validated=False,physics_steps=0)
        save(report_path,report)
        return report
    except Exception:
        report=dict(scene_id=scene["scene_id"],signature=signature,rendered=False,error=traceback.format_exc())
        save(report_path,report);return report
    finally:
        if wrapper:wrapper.close()


def render(args):
    scenes=[json.loads(s) for s in (OUT/"scenes.jsonl").read_text().splitlines()]
    if args.limit:scenes=scenes[:args.limit]
    with mp.get_context("spawn").Pool(args.workers,maxtasksperchild=20) as pool:
        reports=list(pool.imap_unordered(render_one,scenes))
    errors=[r for r in reports if not r.get("rendered")]
    save(OUT/"render_status.json",dict(rendered=len(reports)-len(errors),errors=errors,
         technical_pass=sum(r.get("technical_pass",False) for r in reports)))
    print(json.dumps(dict(rendered=len(reports)-len(errors),errors=len(errors),
                         technical_pass=sum(r.get("technical_pass",False) for r in reports))),flush=True)
    if errors:
        print(json.dumps(errors[:2],ensure_ascii=False),flush=True)
        raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=["prepare","render"])
    parser.add_argument("--limit",type=int)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    prepare() if args.command=="prepare" else render(args)


if __name__=="__main__":main()
