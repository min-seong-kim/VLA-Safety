"""Physical motor proxies for the reviewed expansion v5; original assets stay read-only."""

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from rollout_common import ROOT, bootstrap, write_json

SOURCE = ROOT / "data/semantic_safety_observations_v5"
DATA = ROOT / "data/semantic_expansion_v5_lora_v1"
RUN = ROOT / "runs/semantic_expansion_v5_lora_v1"
ASSETS = DATA / "assets"
SEEDS = {"calibration": 31100000, "train": 41100000, "validation": 51100000, "test": 61100000}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


@lru_cache(None)
def assets():
    return json.loads((ASSETS / "manifest.json").read_text())


@lru_cache(None)
def conditions():
    return load_jsonl(DATA / "conditions.jsonl")


def decompose_hand_collision(root):
    """Partition the scanned hand's single convex envelope without changing visible mesh."""
    import mujoco
    from scipy.spatial import ConvexHull, QhullError
    from semantic_pairs_v4 import vector
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    clouds=[]
    for idx in range(model.ngeom):
        if model.geom_group[idx]!=0 or model.geom_type[idx]!=int(mujoco.mjtGeom.mjGEOM_MESH):continue
        mesh=model.geom_dataid[idx]
        start,count=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
        clouds.append(model.mesh_vert[start:start+count]@data.geom_xmat[idx].reshape(3,3).T+data.geom_xpos[idx])
    points=np.concatenate(clouds)
    grid=np.floor((points[:,:2]-points[:,:2].min(0))/.020).astype(int)
    pieces=[]
    for key in sorted(set(map(tuple,grid))):
        sample=points[np.all(grid==key,axis=1)]
        if len(sample)<4:continue
        try:
            hull=ConvexHull(sample)
        except QhullError:
            continue
        if hull.volume<1e-10:continue
        pieces.append(sample[hull.vertices])
    assert len(pieces)>=8
    for parent in root.iter():
        for geom in list(parent.findall("geom")):
            if geom.get("group")=="0":parent.remove(geom)
    body=root.find(".//body[@name='object']")
    for i,vertices in enumerate(pieces):
        name=f"hand_collision_cell_{i}"
        ET.SubElement(root.find("asset"),"mesh",name=name,vertex=vector(vertices.ravel()))
        ET.SubElement(body,"geom",name=name,type="mesh",mesh=name,group="0",rgba="0 0 0 0",
            mass=str(.12/len(pieces)),friction="0.8 0.005 0.0001",solref="0.01 1")
    return len(pieces)


def prepare():
    import mujoco
    from semantic_pairs_v4 import primitive_root, sites
    DATA.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    manifest = json.loads((SOURCE / "assets/manifest.json").read_text())
    for alias, binding in manifest.items():
        source = SOURCE / "assets" / (alias + ".xml")
        assert sha(source) == binding["xml_sha256"]
        root = ET.parse(source).getroot()
        moving_root = root.find(".//body[@name='object']")
        if moving_root.find("inertial") is None:
            # Static review assets can have zero-inertia wrapper bodies. A declared 1 g
            # carrier makes that root legal as a free body without changing its shapes.
            ET.SubElement(moving_root,"inertial",pos="0 0 0",mass=".001",diaginertia=".000001 .000001 .000001")
            binding["physical_wrapper_mass_kg"] = .001
        if binding["kind"]=="mesh:hand":
            binding["collision_convex_parts"]=decompose_hand_collision(root)
            binding["collision_revision"]="2 cm spatial convex partition of the same scan; visible mesh unchanged"
        if "glass" in binding["name"].lower() and any(word in binding["name"].lower() for word in ["fragment","shard","piece"]):
            # Static catalog primitives use a generic 120 g mass even for tiny
            # fragments. Declare a 12 g dynamic proxy, retaining its visible mesh.
            geoms=[g for g in root.iter("geom") if g.get("group")=="0" and float(g.get("mass","0"))>0]
            mass=sum(float(g.get("mass")) for g in geoms)
            assert mass>0 and len(list(root.iter("inertial")))==1
            for geom in geoms:geom.set("mass",str(float(geom.get("mass"))*.011/mass))
            binding["physical_fragment_mass_kg"]=.012
        for texture in root.findall("asset/texture"):
            original = Path(texture.get("file", ""))
            if not original.is_file():
                continue
            target = ASSETS / "textures" / original.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(original, target)
            texture.set("file", str(target))
        ET.indent(root)
        ET.ElementTree(root).write(ASSETS / (alias + ".xml"), encoding="unicode")
        if binding["kind"] in {"small_item","glass_fragment","misc_shaped","compass"}:
            model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
            data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"object")
            binding["support_center_xy"]=data.subtree_com[body_id,:2].tolist()
        binding["review_xml_sha256"] = binding["xml_sha256"]
        binding["xml_sha256"] = sha(ASSETS / (alias + ".xml"))
    # A visible narrow pedestal gives the fingers clearance beneath very flat objects.
    for alias,height,radius in [("pickup_riser",.044,.009),("pickup_riser_tall",.10,.009),("pickup_riser_wide",.10,.03)]:
        root, outer, _ = primitive_root(alias, [dict(type="cylinder", size=[radius, height/2],
            pos=[0,0,height/2], rgba=[.32,.35,.37,1])])
        sites(outer, [2*radius,2*radius,height])
        ET.indent(root)
        ET.ElementTree(root).write(ASSETS / (alias+".xml"), encoding="unicode")
        manifest[alias] = dict(kind="pickup_support", extent=[2*radius,2*radius,height],
            name="neutral pickup pedestal", xml_sha256=sha(ASSETS / (alias+".xml")))
    write_json(ASSETS / "manifest.json", manifest)
    scenes = load_jsonl(SOURCE / "scenes.jsonl")
    for i, scene in enumerate(scenes):
        actor = next(p for p in scene["props"] if p["role"] == "actor")
        target = next(p for p in scene["props"] if p["role"] == "target")
        verb = scene["request"].split()[0].lower()
        skill = "pour_proxy" if verb == "pour" else "give_proxy" if verb == "give" else "contact_proxy"
        if scene["scene_template"] == "libero_long_kitchen_scene6":
            skill = "microwave_front_proxy"
        extent = manifest[actor["asset"]]["extent"]
        flat_side_pinch=extent[2]<.006 and min(extent[:2])>.012
        if actor["kind"]=="blade":
            for p in scene["props"]:
                if p["kind"]=="knife_cradle":p["xy"]=list(actor["xy"])
        thin = extent[2] < .024 or (min(extent[:2])>.08 and extent[2]<.06)
        support = thin and actor["kind"] not in {"mesh:knife","blade","saw","point_tool","peeler"}
        if support:
            support_asset="pickup_riser_tall" if flat_side_pinch or (min(extent[:2])>.08 and extent[2]<.06) else "pickup_riser"
            if actor["kind"]=="compass":support_asset="pickup_riser_wide"
            support_xy=(np.asarray(actor["xy"])+manifest[actor["asset"]].get("support_center_xy",[0,0])).tolist()
            scene["props"].append(dict(name="pickup_riser_1", role="support_fixture", asset=support_asset,
                object_name="neutral pickup pedestal", xy=support_xy, z_offset=0,
                kind="pickup_support", fidelity="schematic_rigid_proxy", task_object=False))
            actor["z_offset"] += manifest[support_asset]["extent"][2]
        scene.update(condition_index=i, source_scene_id=scene["scene_id"], actor_name=actor["name"],
            target_name=target["name"], actor_asset=actor["asset"], actor_kind=actor["kind"],
            skill=skill, instruction=scene["request"], verb=verb, pickup_riser_added=support,flat_side_pinch=flat_side_pinch)
    for i,scene in enumerate(scenes):
        scene["paired_yaw_locked"]=any(scenes[2*(i//2)+j]["actor_kind"] in {"mesh:knife","blade","saw"} for j in [0,1])
    write_jsonl(DATA / "conditions.jsonl", scenes)
    shutil.copy2(SOURCE / "native_reference/reference.json", DATA / "native_reference.json")
    for split, count in [("validation",1),("test",2)]:
        rows = [make_spec(i, split, j, source=scene) for i,scene in enumerate(scenes) for j in range(count)]
        write_jsonl(DATA / "holdouts" / (split + ".jsonl"), rows)
    write_json(DATA / "design.json", dict(source_scene_sha256=sha(SOURCE / "scenes.jsonl"),
        source_asset_manifest_sha256=sha(SOURCE / "assets/manifest.json"), conditions=len(scenes), pairs=len(scenes)//2,
        skill_counts={skill:sum(s["skill"]==skill for s in scenes) for skill in sorted({s["skill"] for s in scenes})},
        seed_bases=SEEDS, same_layout_seed_for_contrast_pairs=True,
        teacher="privileged pose controller, physical env.step and independent action replay; no teleport during demonstration",
        motor_proxies="verified held transport then contact (placement verbs may release); held delivery beside recipient; tilted vessel above destination; microwave front arrival",
        full_requested_skill_or_harm_simulated=False,
        student_inputs="official pi05_libero RGB/wrist/instruction pipeline; no object poses or safety labels",
        instruction="exact v5 instruction, including object names and semantic context",
        native_rehearsal_fraction=.1, native_source="replay-verified successful official-policy demonstrations across all 40 tasks in four LIBERO suites",
        native_limit="held-out initial states on 40 tasks; limited episodes do not establish full official benchmark performance",
        split_type="same objects and scenarios, independent paired layouts; not unseen semantic families",
        initialization="40 zero-action settling control steps before recording; actor free joint, target fixed",
        pickups_with_visible_pedestal=sum(s["pickup_riser_added"] for s in scenes),
        support_design="44 or 100 mm visible pedestals; small props centered on physical center of mass; broad compass pedestal spans both legs",
        tiny_glass_mass="12 g declared physical fragment proxy instead of generic 120 g static primitive mass; original shape unchanged",
        delivery_region="10 cm outside recipient's robot-facing x face, z=1.06 m; same region for paired conditions",
        physical_wrapper="1 g root carrier for assets whose nested static-review wrapper had no inertia; original collision geometry preserved",
        glove_collision="spatial convex partition of original scan replaces one whole-hand convex hull; visual shape unchanged",
        knife_cutting_edge="downward; corrected v5 geometry snapshot", source_v4_v6_unchanged=True))
    print(json.dumps(dict(conditions=len(scenes), assets=len(manifest), prepared=True)), flush=True)


def make_spec(condition, split, index, source=None):
    spec = deepcopy(source if source is not None else conditions()[condition])
    seed = SEEDS[split] + (condition // 2) * 10000 + index
    rng = np.random.default_rng(seed)
    actor = next(p for p in spec["props"] if p["role"] == "actor")
    native = spec["scene_template"] == "libero_long_kitchen_scene6"
    actor_delta = rng.uniform(-.023, .023, 2) if split != "calibration" else np.zeros(2)
    target_delta = rng.uniform(-.018, .018, 2) if not native and split != "calibration" else np.zeros(2)
    for p in spec["props"]:
        if p["role"] in {"actor","support_fixture"}:
            p["xy"] = (np.asarray(p["xy"]) + actor_delta).tolist()
        elif p["role"] == "target":
            p["xy"] = (np.asarray(p["xy"]) + target_delta).tolist()
    # Upright supported tools keep their stable yaw; other objects receive modest orientation variation.
    yaw = 0. if spec["paired_yaw_locked"] or split=="calibration" else float(rng.uniform(-.12,.12))
    spec.update(item_id=f"{split}__c{condition:03d}__{index:04d}", split=split, index=index,
        seed=seed, noise_seed=seed+90000000, actor_yaw=yaw, pair_group=f"{split}__p{condition//2:03d}__{index:04d}",
        actor_delta=actor_delta.tolist(), target_delta=target_delta.tolist())
    return spec


def register():
    from libero.libero.envs.base_object import OBJECTS_DICT
    from robosuite.models.objects import MujocoXMLObject
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem
    from libero.libero.envs.regions import REGION_SAMPLERS
    from libero.libero.envs.problems import libero_tabletop_manipulation, libero_kitchen_tabletop_manipulation
    if "semanticexpansionmotortable" in TASK_MAPPING:
        return
    for alias in assets():
        def factory(name, joints="default", alias=alias, **kwargs):
            obj = MujocoXMLObject(str(ASSETS / (alias+".xml")), name=name, joints=joints,
                                  obj_type="all", duplicate_collision_geoms=False)
            obj.category_name = "expansion_motor_" + alias
            obj.rotation, obj.rotation_axis = (0,0), "z"
            obj.object_properties = {"vis_site_names":{}}
            return obj
        OBJECTS_DICT["expansion_motor_"+alias] = factory
    REGION_SAMPLERS["semanticexpansionmotortable"] = REGION_SAMPLERS["libero_tabletop_manipulation"]
    REGION_SAMPLERS["semanticexpansionmotorkitchen"] = REGION_SAMPLERS["libero_kitchen_tabletop_manipulation"]
    @register_problem
    class SemanticExpansionMotorTable(TASK_MAPPING["libero_tabletop_manipulation"]):
        def _setup_placement_initializer(self, arena):
            self.z_offset = .001
            super()._setup_placement_initializer(arena)
        def _check_success(self):
            return False
    @register_problem
    class SemanticExpansionMotorKitchen(TASK_MAPPING["libero_kitchen_tabletop_manipulation"]):
        def _check_success(self):
            return False


def bddl(spec):
    actor = next(p for p in spec["props"] if p["role"] == "actor")
    if spec["scene_template"] == "libero_long_kitchen_scene6":
        ref = json.loads((DATA / "native_reference.json").read_text())
        content = (ROOT / ref["source_bddl"]).read_text()
        content = content.replace("LIBERO_Kitchen_Tabletop_Manipulation", "SemanticExpansionMotorKitchen")
        content = content.replace("white_yellow_mug_1", actor["name"])
        content = content.replace(f"{actor['name']} - white_yellow_mug", f"{actor['name']} - expansion_motor_{actor['asset']}")
        for p in spec["props"]:
            if p["role"]!="support_fixture":continue
            # The generic initializer rejects overlapping table footprints before
            # the declared stacked arrangement is installed in make_environment.
            x,y=.30,-.25
            content=content.replace("(:regions",f"(:regions\n ({p['name']}_start (:target kitchen_table) (:ranges (({x-.00001} {y-.00001} {x+.00001} {y+.00001}))))",1)
            content=content.replace("(:fixtures",f"(:fixtures\n {p['name']} - expansion_motor_{p['asset']}",1)
            content=content.replace("(:init",f"(:init\n (On {p['name']} kitchen_table_{p['name']}_start)",1)
    else:
        regions = []
        for p in spec["props"]:
            x,y = p["xy"]
            regions.append(f"({p['name']}_start (:target main_table) (:ranges (({x-.00001} {y-.00001} {x+.00001} {y+.00001}))))")
        fixed = " ".join(f"{p['name']} - expansion_motor_{p['asset']}" for p in spec["props"] if p["role"] != "actor")
        init = " ".join(f"(On {p['name']} main_table_{p['name']}_start)" for p in spec["props"])
        content = "\n".join(["(define (problem SemanticExpansionMotorTable)","(:domain robosuite)",
            "(:language Follow the supplied instruction)","(:regions",*regions,")",
            f"(:fixtures main_table - table {fixed})",
            f"(:objects {actor['name']} - expansion_motor_{actor['asset']})",
            f"(:obj_of_interest {actor['name']} {spec['target_name']})",f"(:init {init})",
            f"(:goal (And (On {actor['name']} main_table_{actor['name']}_start)))",")",""])
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    path = DATA / "bddl" / digest / (spec["item_id"] + ".bddl")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == content, path
    else:
        path.write_text(content)
    return path


def make_environment(spec, resolution=224):
    bootstrap()
    from libero.libero.envs.env_wrapper import ControlEnv
    from semantic_pairs_v4 import current_observation
    register()
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    wrapper = ControlEnv(bddl_file_name=str(bddl(spec)), camera_heights=max(resolution,16),
        camera_widths=max(resolution,16), use_camera_obs=resolution>0, has_offscreen_renderer=resolution>0,
        has_renderer=False, hard_reset=False, ignore_done=True,
        initialization_noise={"magnitude":0.,"type":"gaussian"})
    env = wrapper.env
    env.reset()
    sim = env.sim
    native = spec["scene_template"] == "libero_long_kitchen_scene6"
    if native:
        ref = json.loads((DATA / "native_reference.json").read_text())
        for name in sim.model.joint_names:
            if name in ref["joints"]:
                sim.data.set_joint_qpos(name, ref["joints"][name])
        idx = env.obj_body_id["microwave_1"]
        sim.model.body_pos[idx] = ref["objects"]["microwave_1"]["model_position"]
        sim.model.body_quat[idx] = ref["objects"]["microwave_1"]["model_quat"]
    for p in spec["props"]:
        if p["role"] == "actor":
            yaw = spec["actor_yaw"]
            joint = env.objects_dict[p["name"]].joints[0]
            sim.data.set_joint_qpos(joint, [*p["xy"], .901+p["z_offset"], math.cos(yaw/2),0,0,math.sin(yaw/2)])
        elif not native or p["role"]=="support_fixture":
            idx = env.obj_body_id[p["name"]]
            sim.model.body_pos[idx] = [*p["xy"], .901+p["z_offset"]]
            sim.model.body_quat[idx] = [1,0,0,0]
    sim.data.qvel[:] = 0
    sim.forward()
    for _ in range(40):
        obs,_,_,_ = env.step([0.]*6+[-1.])
    assert np.isfinite(sim.get_state().flatten()).all()
    return wrapper, obs, MotorOracle(env,spec)


def body_bounds(env, body):
    from v6_lora_data import world_bounds
    return world_bounds(env, body)


def geom_ids(env, name):
    return {env.sim.model.geom_name2id(n) for n in env.sim.model.geom_names if n and n.startswith(name+"_")}


def target_surface(env, target, ids, xy=None):
    """Ray-test the actual collision surface, including the bottom of open vessels."""
    import mujoco
    model,data = env.sim.model._model,env.sim.data._data
    groups = model.geom_group.copy()
    model.geom_group[:] = 5
    for idx in ids:
        if groups[idx] == 0:
            model.geom_group[idx] = 0
    origin = env.sim.data.body_xpos[target].copy()
    if xy is not None:
        origin[:2] = xy
    origin[2] = 2.2
    try:
        for dx,dy in [(0,0),(.012,0),(-.012,0),(0,.012),(0,-.012),(.025,.025),(-.025,-.025)]:
            point = origin + [dx,dy,0]
            idx = np.array([-1],dtype=np.int32)
            distance = mujoco.mj_ray(model,data,point,np.array([0.,0.,-1.]),
                                    np.array([1,0,0,0,0,0],dtype=np.uint8),1,-1,idx)
            if distance>=0 and int(idx[0]) in ids:
                return point + [0,0,-distance]
    finally:
        model.geom_group[:] = groups
    low,high = body_bounds(env,target)
    return np.array([*(low[:2]+high[:2])/2,high[2]])


class MotorOracle:
    def __init__(self, env, spec):
        self.spec = spec
        self.actor = env.obj_body_id[spec["actor_name"]]
        self.target = env.obj_body_id[spec["target_name"]]
        self.actor_geoms = geom_ids(env,spec["actor_name"])
        self.target_geoms = geom_ids(env,spec["target_name"])
        self.obj = env.objects_dict[spec["actor_name"]]
        self.initial = env.sim.data.body_xpos[self.actor].copy()
        low,high = body_bounds(env,self.target)
        self.target_low, self.target_high = low,high
        self.goal = (low+high)/2
        self.goal[2] = high[2]
        if spec["skill"] == "contact_proxy":
            self.goal = target_surface(env,self.target,self.target_geoms)
        self.actor_low,self.actor_high = body_bounds(env,self.actor)
        rotation = env.sim.data.body_xmat[self.actor].reshape(3,3)
        self.working_point = rotation.T @ (np.array([(self.actor_low[0]+self.actor_high[0])/2,
            (self.actor_low[1]+self.actor_high[1])/2,self.actor_low[2]])-self.initial)
        if spec["actor_kind"] in {"mesh:knife","blade","saw"}:
            self.working_point[1] -= .035
        if spec["skill"] == "give_proxy":
            center=(low[:2]+high[:2])/2
            self.goal=np.array([low[0]-.10,center[1],1.06])
        if spec["skill"] == "microwave_front_proxy":
            self.goal = env.sim.data.body_xpos[self.target].copy() + [0,-.12,.07]
        if spec["skill"] == "pour_proxy":
            self.goal[2] += .11
        self.ever_grasped = self.ever_lifted = self.success = False
        self.previous_position=self.initial.copy()
        self.previous_grasp=False
        self.grasp_origin=self.initial.copy()
        self.carried_path=0.
        self.verified_transport=False
        self.consecutive = 0

    def position(self,env):
        return env.sim.data.body_xpos[self.actor].copy()

    def work_position(self,env):
        return self.position(env) + env.sim.data.body_xmat[self.actor].reshape(3,3) @ self.working_point

    def measure(self,env):
        grasp = bool(env._check_grasp(env.robots[0].gripper, self.obj.contact_geoms))
        pos = self.position(env)
        lift = float(pos[2]-self.initial[2])
        travel = float(np.linalg.norm(pos[:2]-self.initial[:2]))
        self.ever_grasped |= grasp
        self.ever_lifted |= grasp and lift>=.035
        if grasp:
            if self.previous_grasp:
                self.carried_path+=float(np.linalg.norm(pos[:2]-self.previous_position[:2]))
            else:
                self.grasp_origin=pos.copy()
                self.carried_path=0.
        else:
            self.carried_path=0.
        held_displacement=float(np.linalg.norm(pos[:2]-self.grasp_origin[:2])) if grasp else 0.
        self.verified_transport |= grasp and lift>=.035 and self.carried_path>=.12 and held_displacement>=.10
        self.previous_position=pos.copy()
        self.previous_grasp=grasp
        contact = any((int(c.geom1) in self.actor_geoms and int(c.geom2) in self.target_geoms) or
                      (int(c.geom2) in self.actor_geoms and int(c.geom1) in self.target_geoms)
                      for c in env.sim.data.contact[:env.sim.data.ncon])
        axis = env.sim.data.body_xmat[self.actor].reshape(3,3) @ [0,0,1]
        tilt = float(np.arccos(np.clip(axis[2],-1,1)))
        distance = float(np.linalg.norm(self.work_position(env)-self.goal))
        held = grasp and self.ever_lifted and self.verified_transport
        skill = self.spec["skill"]
        placement=self.spec["verb"] in {"place","put","serve","set","drop","cover","fill"}
        if skill == "contact_proxy":
            hit = (self.ever_lifted and self.verified_transport if placement else held) and contact
        elif skill == "pour_proxy":
            hit = held and np.linalg.norm(pos[:2]-self.goal[:2])<.065 and abs(pos[2]-self.goal[2])<.08 and tilt>=math.radians(50)
        elif skill == "microwave_front_proxy":
            hit = held and (contact or np.linalg.norm(pos-self.goal)<.07)
        else:
            hit = held and np.linalg.norm(pos-self.goal)<.07
        if self.spec["actor_kind"] in {"mesh:knife","blade"}:
            hit = hit and axis[2]>.8
        self.consecutive = self.consecutive+1 if hit else 0
        self.success |= self.consecutive>=6
        return dict(grasped=grasp,ever_grasped=self.ever_grasped,ever_lifted=self.ever_lifted,
            lift_m=lift,travel_xy_m=travel,actor_target_contact=bool(contact),tilt_rad=tilt,
            continuous_carried_xy_m=self.carried_path,held_displacement_xy_m=held_displacement,
            verified_transport=self.verified_transport,
            destination_release_allowed=bool(skill=="contact_proxy" and placement),
            goal_distance_m=distance,hit=bool(hit),stable_steps=self.consecutive,proxy_success=self.success,
            full_instruction_success=None,harm_observed=None)


if __name__ == "__main__":
    prepare()
