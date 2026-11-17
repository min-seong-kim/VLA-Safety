"""Disjoint benign object/skill curriculum and explicit physical motor criteria."""

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from rollout_common import ROOT, bootstrap, write_json
import expansion_v5_motor as motor

DATA = ROOT / "data/semantic_benign_adaptation_v1"
RUN = ROOT / "runs/semantic_benign_adaptation_v1"
SOURCE = ROOT / "data/semantic_safety_contrast_v1"
BASELINE = ROOT / "runs/semantic_safety_contrast_initial_linear_v1"
SEED = 20261116
OriginalOracle = motor.MotorOracle


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure():
    motor.DATA, motor.RUN, motor.ASSETS = DATA, RUN, DATA / "assets"
    motor.assets.cache_clear()
    motor.conditions.cache_clear()
    motor.MotorOracle = BenignOracle


@lru_cache(None)
def conditions():
    return read_jsonl(DATA / "conditions.jsonl")


def signature():
    return {name: sha(Path(__file__).with_name(name)) for name in
            ["benign_adaptation_v1.py", "collect_benign_adaptation_v1.py", "expansion_v5_motor.py", "expansion_v5_teacher.py"]} | {
                "assets": sha(DATA / "assets/manifest.json")}


def make_spec(index, split, repeat):
    spec = deepcopy(next(row for row in conditions() if row["condition_index"]==index))
    seed = {"calibration": 81100000, "train": 82100000, "validation": 83100000, "test": 84100000}[split]
    seed += index * 1000 + repeat
    rng = np.random.default_rng(seed)
    actor_delta = rng.uniform(-.022, .022, 2) if split != "calibration" else np.zeros(2)
    target_delta = rng.uniform(-.020, .020, 2) if split != "calibration" else np.zeros(2)
    for prop in spec["props"]:
        delta = target_delta if prop["role"] == "target" else actor_delta
        prop["xy"] = (np.asarray(prop["xy"]) + delta).tolist()
    spec.update(item_id=f"{split}__c{index:03d}__{repeat:04d}", split=split, index=repeat,
                seed=seed, noise_seed=seed+90000000, actor_yaw=0.,
                pair_group=f"benign_c{index:03d}", actor_delta=actor_delta.tolist(),
                target_delta=target_delta.tolist())
    return spec


def prepare():
    bootstrap()
    import mujoco
    from semantic_pairs_v4 import primitive_root, sites
    import expansion_v5_assets as a

    DATA.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    assets_dir = DATA / "assets"
    assets_dir.mkdir(exist_ok=True)
    source_objects = {r["object_id"]: r for r in read_jsonl(SOURCE / "objects.jsonl")}
    benchmark = read_jsonl(SOURCE / "conditions.jsonl")
    source_bindings = json.loads((SOURCE / "asset_bindings.json").read_text())
    actor_ids = {r["actor_id"] for r in benchmark}
    grouped = {}
    preferred = {"candle": "unlit", "iron": "ironoff", "hotmetal": "coldmetal", "kettle": "coolwater",
                 "mixing_bowl":"vinegar"}
    for key in sorted(actor_ids):
        obj = source_objects[key]
        if key.startswith("mold_") or key in {"dirtycloth", "aquawater", "hotwater", "iron", "hotmetal", "candle", "glass"}:
            continue
        identity = obj["identity"]
        if identity not in grouped or key == preferred.get(identity) or key == identity:
            grouped[identity] = key
    # Different physical targets, goals and full inputs; object vocabulary reuse is intentional.
    manifest = {}
    for key in [*grouped.values(), "knife_cradle", "saw_cradle"]:
        binding = deepcopy(source_bindings[key]); alias = binding["alias"]
        root = ET.parse(SOURCE / "assets" / (alias+".xml")).getroot()
        compiler=root.find("compiler")
        if compiler is None:compiler=ET.SubElement(root,"compiler")
        compiler.set("angle","radian")
        body = root.find(".//body[@name='object']")
        if body.find("inertial") is None:
            ET.SubElement(body, "inertial", pos="0 0 0", mass=".001", diaginertia=".000001 .000001 .000001")
        scale={"clothball":.70,"fabricbook":.60,"toyblock":.75}.get(key,1.)
        if scale!=1.:
            # Use graspable size variants for objects wider than Panda's 8 cm opening.
            for node in root.iter():
                for attr in ["pos","size","fromto"]:
                    if attr in node.attrib and node.tag in {"geom","body","site","inertial"}:
                        node.set(attr," ".join(str(float(v)*scale) for v in node.get(attr).split()))
                if node.tag=="mesh" and "scale" in node.attrib:
                    node.set("scale"," ".join(str(float(v)*scale) for v in node.get("scale").split()))
                if "mass" in node.attrib:node.set("mass",str(float(node.get("mass"))*scale**3))
                if node.tag=="inertial" and "diaginertia" in node.attrib:
                    node.set("diaginertia"," ".join(str(float(v)*scale**5) for v in node.get("diaginertia").split()))
            binding["extent"]=(np.asarray(binding["extent"])*scale).tolist()
            binding["training_scale"]=scale
        if key=="knife":
            # The static-review knife had a 1 g root and extremely stiff contacts.
            # Declare a stable 40 g carrier without changing any visible geometry.
            body.find("inertial").set("mass",".04")
            body.find("inertial").set("diaginertia",".0001 .000005 .0001")
            for geom in root.iter("geom"):
                if geom.get("group")=="0":
                    geom.set("friction","0.8 0.005 0.0001")
                    geom.set("solref","0.01 1")
                    geom.set("solimp","0.95 0.99 0.001")
            binding["physical_revision"]="40 g root carrier and compliant contacts; visual mesh unchanged"
            from rollout_asset_adapter import bounds
            lo,hi=bounds(root)
            shift=-(lo+hi)/2;shift[2]=.002-lo[2]
            transform=ET.Element("body",name="dynamic_recenter",pos=" ".join(map(str,shift)))
            for child in list(body):
                if child.tag!="inertial":body.remove(child);transform.append(child)
            body.append(transform)
            binding["extent"]=(hi-lo).tolist()
            binding["local_working_point"]=[0.,.035,.002]
            body.find("inertial").set("pos",f"0 0 {(hi-lo)[2]/2+.002}")
            binding["angle_revision"]="explicit radians matching LIBERO; recentered with the same radians before dynamic use"
        for texture in root.findall("asset/texture"):
            original = Path(texture.get("file", ""))
            if original.is_file(): texture.set("file", str(original.resolve()))
        ET.indent(root)
        ET.ElementTree(root).write(assets_dir / (alias+".xml"), encoding="unicode")
        # Bounds are audited against compiled geometry, including imported mesh transforms.
        model = mujoco.MjModel.from_xml_path(str(assets_dir / (alias+".xml")))
        assert model.ngeom > 0
        binding["xml_sha256"] = sha(assets_dir / (alias+".xml"))
        binding["adaptation_source_object_id"] = key
        manifest[alias] = binding

    targets = {
        "tool_pad": ("blue padded tool-sorting mat", [.090,.145,.012], [.18,.39,.64,1]),
        "food_mat": ("green clean meal-preparation mat", [.105,.090,.012], [.25,.55,.34,1]),
        "parts_pad": ("orange padded component-sorting mat", [.095,.080,.014], [.78,.46,.22,1]),
        "textile_pad": ("purple textile-sorting mat", [.105,.085,.014], [.53,.36,.62,1]),
        "bottle_pad": ("blue closed-bottle staging pad", [.090,.080,.020], [.28,.43,.52,1]),
        "rubber_practice": ("replaceable rubber cutting-practice sheet", [.095,.075,.015], [.45,.53,.42,1]),
        "practice_board": ("washable cleaning-practice board", [.100,.080,.018], [.57,.67,.72,1]),
        "plaster_practice": ("replaceable plaster craft-practice tile", [.095,.070,.020], [.77,.74,.67,1]),
        "soap_practice": ("solid soap grating-practice bar", [.080,.065,.023], [.74,.77,.60,1]),
        "press_practice": ("disconnected mechanical press-practice pad", [.080,.065,.021], [.45,.52,.65,1]),
    }
    for key, (name, half, color) in targets.items():
        root, outer, _ = primitive_root(key, [a.box(half, [0,0,half[2]], color)])
        extent = (2*np.asarray(half)).tolist(); sites(outer, extent)
        ET.indent(root); ET.ElementTree(root).write(assets_dir/(key+".xml"), encoding="unicode")
        manifest[key] = dict(kind="benign_practice_fixture", name=name, extent=extent,
                             fidelity="schematic_rigid_motor_fixture", xml_sha256=sha(assets_dir/(key+".xml")))
    # Small empty opening; coverage means overlap with this opening, not cloth simulation.
    key="empty_frame"; half=.025; radius=.004; height=.022
    geoms=[a.box([half+radius,radius,height/2],[0,y,height/2],[.32,.53,.59,1]) for y in [-half,half]]
    geoms += [a.box([radius,half-radius,height/2],[x,0,height/2],[.32,.53,.59,1]) for x in [-half,half]]
    root, outer, _ = primitive_root(key,geoms); sites(outer,[.058,.058,height])
    ET.indent(root); ET.ElementTree(root).write(assets_dir/(key+".xml"),encoding="unicode")
    manifest[key]=dict(kind="empty_practice_frame",name="small empty practice frame",extent=[.058,.058,height],
                       fidelity="schematic_rigid_motor_fixture",xml_sha256=sha(assets_dir/(key+".xml")))
    key="measuring_jug"
    geoms=a.bowl([.55,.69,.75,1],.065,.070)+[
        a.rod([.075,-.025,.025],[.075,.025,.025],.007,[.35,.50,.56,1]),
        a.rod([.075,-.025,.025],[.055,-.025,.055],.007,[.35,.50,.56,1]),
        a.rod([.075,.025,.025],[.055,.025,.055],.007,[.35,.50,.56,1])]
    root,outer,_=primitive_root(key,geoms);sites(outer,[.170,.130,.070]);ET.indent(root)
    ET.ElementTree(root).write(assets_dir/(key+".xml"),encoding="unicode")
    manifest[key]=dict(kind="measuring_jug",name="empty measuring jug",extent=[.170,.130,.070],
                       xml_sha256=sha(assets_dir/(key+".xml")))
    key="knife_pickup_jig"
    root,outer,_=primitive_root(key,[a.box([.045,.010,.035],[0,y,.035],[.32,.35,.37,1]) for y in [-.105,.105]])
    sites(outer,[.090,.230,.070]);ET.indent(root)
    ET.ElementTree(root).write(assets_dir/(key+".xml"),encoding="unicode")
    manifest[key]=dict(kind="pickup_support",name="two-ledge knife pickup jig",extent=[.090,.230,.070],
                       xml_sha256=sha(assets_dir/(key+".xml")))
    for key,height in [("pickup_riser",.044),("pickup_riser_tall",.10)]:
        root,outer,_=primitive_root(key,[a.cyl(.009,height/2,[0,0,height/2],[.32,.35,.37,1])])
        sites(outer,[.018,.018,height]);ET.indent(root)
        ET.ElementTree(root).write(assets_dir/(key+".xml"),encoding="unicode")
        manifest[key]=dict(kind="pickup_support",name="neutral pickup pedestal",extent=[.018,.018,height],
                           xml_sha256=sha(assets_dir/(key+".xml")))
    write_json(assets_dir/"manifest.json",manifest)
    rows=[]
    def add(actor_id,target_key,skill,verb):
        obj=source_objects[actor_id];binding=source_bindings[actor_id]
        kind="mesh:knife" if obj["kind"]=="rc:knife" else obj["kind"]
        # Cold, unpowered and fresh source variants are chosen without teaching a risk verdict.
        actor_name=obj["name_en"]
        instruction=(f"Place the {actor_name} on the {manifest[target_key]['name']}." if skill=="place" else
                     f"Cover the {manifest[target_key]['name']} with the {actor_name}." if skill=="cover" else
                     f"Pour the water from the {actor_name} into the {manifest[target_key]['name']}." if skill=="pour" else
                     f"{verb.capitalize()} the {manifest[target_key]['name']} with the {actor_name}.")
        props=[dict(name="actor",role="actor",asset=binding["alias"],kind=kind,
                    xy=[-.075,.160],z_offset=.095 if actor_id=="knife" else .006 if actor_id=="saw" else 0.,task_object=True),
               dict(name="target",role="target",asset=target_key,kind="benign_practice_fixture",xy=[.035,-.150],
                    z_offset=0.,task_object=True)]
        if actor_id in {"knife","saw"}:
            cradle="knife_pickup_jig" if actor_id=="knife" else "saw_cradle"
            alias=cradle if actor_id=="knife" else source_bindings[cradle]["alias"]
            props.append(dict(name="holder",role="support_fixture",asset=alias,
                              xy=[-.075,.160] if actor_id=="knife" else [-.075,.120],
                              z_offset=0.,kind=cradle,task_object=False))
        extent=manifest[binding["alias"]]["extent"]
        broad=min(extent[:2])>.08 and extent[2]<.06
        thin=extent[2]<.024 or broad
        support=thin and actor_id not in {"knife","razor","saw"}
        if support:
            riser="pickup_riser_tall" if broad or extent[2]<.006 else "pickup_riser"
            props[0]["z_offset"]+=manifest[riser]["extent"][2]
            props.append(dict(name="pickup",role="support_fixture",asset=riser,kind="pickup_support",
                              xy=props[0]["xy"].copy(),z_offset=0.,task_object=False))
        index=len(rows)
        rows.append(dict(condition_index=index,source_scene_id=f"benign_{actor_id}_{skill}_{target_key}",
            actor_id=actor_id,actor_identity=obj["identity"],actor_asset=binding["alias"],actor_kind=kind,
            target_id=target_key,actor_name="actor",target_name="target",scene_template="benign_motor_table",
            instruction=instruction,motor_skill=skill,verb=verb.lower(),skill="pour_proxy" if skill=="pour" else "contact_proxy",props=props,
            flat_side_pinch=broad or extent[2]<.006,paired_yaw_locked=True,stroke_axis=0,
            task_objects=2,safety_label_used=False,physical_effect_simulated=False))
    soft={"cloth","towel","sponge","cotton","brush","bandage","fabricbook","clothball"}
    food={"bread","apple","banana","cookie","cheese","rice"}
    tools={"knife","hammer","grater","saw","sandpaper","wirebrush","screwdriver","scissors","fork","razor","drill","wrench"}
    for identity,key in sorted(grouped.items()):
        target="food_mat" if identity in food else "tool_pad" if identity in tools else "textile_pad" if identity in soft else "bottle_pad" if source_objects[key]["kind"] in {"bottle","pressure_canister"} else "parts_pad"
        add(key,target,"place","place")
    for actor in ["cloth","towel","sponge","cotton","brush"]:
        add(actor,"practice_board","stroke","wipe")
    for actor,verb,target in [("knife","cut","rubber_practice"),("razor","cut","rubber_practice"),
                              ("saw","saw","plaster_practice"),("sandpaper","sand","plaster_practice"),
                              ("wirebrush","brush","plaster_practice"),("grater","grate","soap_practice"),
                              ("screwdriver","scratch","rubber_practice")]:
        add(actor,target,"stroke",verb)
    for actor in ["cloth","towel","paper"]:add(actor,"empty_frame","cover","cover")
    for actor in ["drill","hammer","cotton"]:add(actor,"press_practice","press","press")
    for actor in ["water","coolwater"]:add(actor,"measuring_jug","pour","pour")
    add("drill","plaster_practice","press","drill")
    excluded=[row for row in rows if row["actor_id"]=="paper" and row["motor_skill"]=="cover"]
    rows=[row for row in rows if row not in excluded]
    write_json(DATA/"candidate_exclusions.json",dict(stage="physical teacher feasibility, before policy training",
        excluded=[dict(source_scene_id=r["source_scene_id"],condition_index=r["condition_index"],
                       reason="thin rigid paper did not reliably cover the frame after grasp and release; cloth/towel cover retained") for r in excluded],
        all_original_safety_probe_conditions_retained=True,all_actor_identities_retained=True))
    write_jsonl(DATA/"conditions.jsonl",rows)
    conditions.cache_clear()
    for split in ["validation","test"]:
        write_jsonl(DATA/"holdouts"/(split+".jsonl"),[make_spec(r["condition_index"],split,j) for r in rows for j in range(2)])
    # Fixed validation subset selected by task metadata only; no safety-probe test access.
    selected=[r["condition_index"] for r in rows if r["motor_skill"]!="place"]
    remaining=[r["condition_index"] for r in rows if r["motor_skill"]=="place"]
    selected+=remaining[::max(1,len(remaining)//12)]
    write_jsonl(DATA/"holdouts/selection.jsonl",[make_spec(i,"validation",0) for i in sorted(set(selected))])
    identities=set(grouped)
    audit=dict(passed=all(r["target_id"] not in source_objects for r in rows),
        training_conditions=len(rows),actor_identities=len(identities),motor_skill_counts=dict(Counter(r["motor_skill"] for r in rows)),
        verbs=dict(Counter(r["verb"] for r in rows)),
        exact_instruction_overlap=sorted({r["instruction"] for r in rows}&{r["instruction_en"] for r in benchmark}),
        exact_actor_target_identity_overlap=sorted({(r["actor_identity"],r["target_id"]) for r in rows} &
            {(source_objects[r["actor_id"]]["identity"],source_objects[r["target_id"]]["identity"]) for r in benchmark}),
        target_assets_all_new=True,
        intentionally_shared_actor_identities=sorted(identities),
        omitted_actor_identities={"glass":"broken multi-fragment proxy is unsuitable for this benign single-grasp curriculum",
                                  "candle":"lit candle is omitted; the separate unlit candle identity is included"},
        source_condition_sha256=sha(SOURCE/"conditions.jsonl"),baseline_manifest_sha256=sha(BASELINE/"manifest.json"),
        scope="Known object vocabulary, new benign target/goal contexts; not unseen-object evaluation",
        labels="No request-level safety verdict, rationale or special safety token in policy inputs or targets",
        proxies="Stroke/contact/coverage/release motor criteria; rigid simulation does not implement actual cutting, cleaning, drilling or cloth deformation")
    assert audit["passed"] and not audit["exact_instruction_overlap"]
    write_json(DATA/"semantic_overlap_audit.json",audit)
    protocol=dict(base_checkpoint="checkpoints/pi05_libero_pytorch",scopes=["ae","vlm","all"],
        save_steps=[4000,8000,12000,16000],steps=16000,rank=16,alpha=32,dropout=0,
        learning_rate=2e-4,warmup_steps=300,effective_batch=32,micro_batch=16,
        native_rehearsal_fraction=.1,native_source="existing independently replay-verified successful official-policy train trajectories",
        within_new_data_sampling="50% uniform placement condition; 50% uniform other verb then condition",
        train_episodes_per_condition=32,validation_layouts_per_condition=2,test_layouts_per_condition=2,
        model_selection="benign motor validation success, subject to LIBERO validation retention; no safety-probe test scores",
        adapted_probe="Frozen adapted weights, original first scenes, original A/B tracks and original five connected-group folds; independently refit scaler/probe",
        sites=78,LDA=False,additional_ablation_conditions=False,extra_8000_updates=False,
        autonomous_recovery_authorized=True,
        custom_image_orientation="raw vertical flip only; same upright convention as the fixed contrast probe",
        native_image_orientation="original official-policy rehearsal/evaluation raw 180-degree convention preserved")
    protocol["physical_contact_revision"]="custom env: gripper friction 2.0, actor 0.3, practice target 0.2; same for all teachers and policies"
    protocol["graspable_training_size_variants"]={"clothball":.70,"fabricbook":.60,"toyblock":.75}
    write_json(DATA/"protocol.json",protocol);write_json(RUN/"experiment_protocol.json",protocol)
    if not (RUN/"reporting/delivery_mode.json").exists():
        write_json(RUN/"reporting/delivery_mode.json",{"mode":"foreground"})
    configure()
    print(json.dumps(audit,ensure_ascii=False),flush=True)


class BenignOracle(OriginalOracle):
    def __init__(self,env,spec):
        super().__init__(env,spec)
        if spec["motor_skill"]=="cover":
            self.goal=np.r_[(self.target_low[:2]+self.target_high[:2])/2,self.target_high[2]]
        self.motor_hits=0;self.motor_success=False;self.contact_steps=0
        self.stroke_low=np.inf;self.stroke_high=-np.inf
        self.hold_anchor=None;self.hold_start=None;self.anchor_max_lift=0.

    def measure(self,env):
        metric=super().measure(env)
        # Thin sheets can briefly lose one finger's discrete contact while still
        # physically co-moving with a closed gripper. Verify that hold geometrically.
        robot=env.robots[0]
        eef=env.sim.data.site_xpos[robot.eef_site_id]
        orientation=env.sim.data.site_xmat[robot.eef_site_id].reshape(3,3)
        position=self.position(env)
        relative=orientation.T@(position-eef)
        opening=float(np.max(np.abs(env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes])))
        if metric["grasped"] and self.hold_anchor is None:
            self.hold_anchor=relative.copy();self.hold_start=position.copy();self.anchor_max_lift=0.
        coherent=self.hold_anchor is not None and opening<.0395 and np.linalg.norm(relative-self.hold_anchor)<.020
        if coherent:
            self.anchor_max_lift=max(self.anchor_max_lift,float(position[2]-self.hold_start[2]))
            if self.anchor_max_lift>=.035 and np.linalg.norm(position[:2]-self.hold_start[:2])>=.12:
                self.verified_transport=True
        else:self.hold_anchor=None
        metric["verified_transport"]=self.verified_transport
        metric["closed_gripper_relative_pose_hold"]=bool(coherent)
        held=metric["grasped"] and metric["ever_lifted"] and metric["verified_transport"]
        contact=metric["actor_target_contact"]
        self.contact_steps+=int(held and contact)
        point=self.work_position(env)
        if held and contact:
            position=float(point[self.spec["stroke_axis"]])
            self.stroke_low=min(self.stroke_low,position);self.stroke_high=max(self.stroke_high,position)
        span=max(0.,self.stroke_high-self.stroke_low) if np.isfinite(self.stroke_low) else 0.
        motor_skill=self.spec["motor_skill"]
        released=not metric["grasped"] and np.min(np.abs(env.sim.data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes]))>.020
        if motor_skill=="pour":
            hit=held and np.linalg.norm(position[:2]-self.goal[:2])<.065 and abs(position[2]-self.goal[2])<.08 and metric["tilt_rad"]>=math.radians(50)
        elif motor_skill=="stroke":hit=held and contact and span>=.030 and self.contact_steps>=16
        elif motor_skill=="press":hit=held and contact and self.contact_steps>=14
        else:
            low,high=motor.body_bounds(env,self.actor)
            center=(self.target_low[:2]+self.target_high[:2])/2
            if motor_skill=="cover":
                hit=bool(np.all(low[:2]<=center-.018) and np.all(high[:2]>=center+.018))
            else:
                overlap=np.maximum(0.,np.minimum(high[:2],self.target_high[:2])-np.maximum(low[:2],self.target_low[:2]))
                smaller=min(float(np.prod(high[:2]-low[:2])),float(np.prod(self.target_high[:2]-self.target_low[:2])))
                hit=bool(np.prod(overlap)>=.4*smaller)
            hit=hit and metric["ever_lifted"] and metric["verified_transport"] and contact and released
        self.motor_hits=self.motor_hits+1 if hit else 0
        self.motor_success|=self.motor_hits>=8
        metric.update(proxy_success=self.motor_success,stable_steps=self.motor_hits,motor_skill=motor_skill,
                      contact_steps=self.contact_steps,contact_stroke_span_m=span,released=bool(released),
                      full_instruction_success=None,harm_observed=None)
        return metric


def make_environment(spec,resolution=224):
    configure()
    wrapper,obs,oracle=motor.make_environment(spec,resolution)
    model=wrapper.env.sim.model
    for index in oracle.actor_geoms:model.geom_friction[index]=[.3,.005,.0001]
    for index in oracle.target_geoms:model.geom_friction[index]=[.2,.005,.0001]
    for name in model.geom_names:
        if name and "gripper" in name and "finger" in name:
            model.geom_friction[model.geom_name2id(name)]=[2.,.005,.0001]
    return wrapper,obs,oracle


def teacher_class():
    from expansion_v5_teacher import Teacher
    class BenignTeacher(Teacher):
        def __init__(self,env,oracle,spec,profile):
            original=oracle.working_point.copy()
            super().__init__(env,oracle,spec,profile)
            if spec["motor_skill"] in {"place","cover"}:
                oracle.working_point=original
                self.initial_reference=self.reference(env).copy()
                self.carry_z=max(self.initial_reference[2]+.15,self.target[2]+.09)
            if spec["actor_id"]=="knife" and spec["motor_skill"]=="stroke":
                rotation=env.sim.data.body_xmat[oracle.actor].reshape(3,3)
                self.upright_orientation=rotation.T@self.orientation
                self.reorient=True
                oracle.working_point=np.asarray(motor.assets()[spec["actor_asset"]]["local_working_point"])
                self.initial_reference=self.reference(env).copy()
                self.carry_z=max(self.initial_reference[2]+.17,self.target[2]+.12)

        def action(self,env,obs,metric):
            if self.stage=="hold":
                if self.spec["motor_skill"]=="stroke":
                    if not hasattr(self,"stroke_origin"):self.stroke_origin=self.contact_target.copy()
                    phase=min(self.stage_steps/70.,1.)
                    self.contact_target=self.stroke_origin.copy()
                    self.contact_target[self.spec["stroke_axis"]]+=.040*math.sin(phase*2*math.pi)
                action=super().action(env,obs,metric)
                if self.spec["motor_skill"] in {"place","cover"} and self.stage_steps>=2:
                    action[6]=-1.
                    if self.stage_steps>=24:action[:3]=[0.,0.,.20]
                # Give contact strokes enough time for both directions and stable completion.
                if self.stage_steps<100:self.failed=False
                return action
            return super().action(env,obs,metric)
    return BenignTeacher


def candidate_profiles(spec):
    configure()
    from expansion_v5_teacher import candidates
    proposals=candidates(spec)
    if spec["actor_id"]=="knife":
        # The imported RoboCasa handle is opposite to the older procedural knife.
        explicit=[dict(offset=[0,offset],height_fraction=height,yaw=math.pi/2,tilt=0.,z_adjust=dz)
                  for offset in [-.28,-.34,-.40] for height in [.5,.8] for dz in [.004,-.002,.010]]
        proposals=explicit+[dict(p,offset=[p["offset"][0],-p["offset"][1]]) for p in proposals[:9]]+proposals
    # Broader search is only for training-teacher calibration, never policy-test filtering.
    offsets=[(0,0),(.30,0),(-.30,0),(0,.30),(0,-.30),(.42,0),(-.42,0),(0,.42),(0,-.42)]
    proposals += [dict(offset=list(offset),height_fraction=height,yaw=yaw,tilt=0.,z_adjust=dz)
                  for height in [.90,.15,.65,.35] for offset in offsets for yaw in [0.,math.pi/2]
                  for dz in [.004,-.006,.012]]
    for profile in proposals:
        result=deepcopy(profile)
        if result.get("side_entry"):result["reorient_after_lift"]=False
        yield result
