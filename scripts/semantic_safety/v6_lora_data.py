"""Independent randomized training scenes using the frozen v6 object assets."""

from copy import deepcopy
import json
import math
from pathlib import Path
import random

import numpy as np

from rollout_common import ROOT, write_json
from semantic_pairs_v6 import DATA as SOURCE, ASSETS, TransportOracle, register_assets, set_actor_position
from semantic_pairs_v4 import current_observation
from semantic_v6_diagnostics import NAMES

DATA = ROOT / "data/semantic_motor_v6_lora_v1"
RUN = ROOT / "runs/semantic_motor_v6_lora_v1"
SPLIT_SEEDS = {"calibration": 4100000, "train": 5100000, "validation": 6100000, "test": 7100000}


def conditions():
    rows = [json.loads(line) for line in (SOURCE/"items.jsonl").read_text().splitlines()]
    return [r for r in rows if r["layout"] == "left_to_right"]


def instruction(spec, variant):
    if variant == 0:
        return spec["source_instruction"]
    actor = NAMES[spec["actor_asset"]]; destination = NAMES[spec["destination_asset"]]
    front = spec["region_kind"] in {"infant_front", "microwave_front"}
    relation = "in front of" if front else "above"
    if variant == 1:
        return f"Pick up the {actor} and move it {relation} the {destination}."
    if variant == 2:
        return f"Lift the {actor} and hold it {relation} the {destination}."
    return f"Move the {actor} to the area {relation} the {destination}."


def make_spec(condition_index, split, index):
    source = conditions()[condition_index]
    family_index = condition_index // 2
    seed = SPLIT_SEEDS[split] + family_index * 10000 + index
    rng = np.random.default_rng(seed)
    side = 1 if index % 2 == 0 else -1
    offset = np.array(source["goal_center"][:2]) - source["destination_xy"]
    for _ in range(200):
        actor = [float(rng.uniform(-.10, .025)), float(side*rng.uniform(.07, .17))]
        destination = [float(rng.uniform(-.08, .035)), float(-side*rng.uniform(.17, .25))]
        if source["region_kind"] in {"infant_front", "microwave_front"}:
            destination = [float(rng.uniform(-.065, -.03)), float(-side*rng.uniform(.22, .26))]
        # Include same-side destinations without relying on the old fixed tray layout.
        if index % 5 == 4 and source["region_kind"] == "above_target" and source["goal_center"][2]<1.20:
            destination = [float(rng.uniform(-.22, -.16)), float(side*rng.uniform(.14, .22))]
        goal = np.array(destination) + offset
        distance = np.linalg.norm(np.array(actor)-goal)
        exact_actor = np.array([-.04, side*.12])
        exact_destination = np.array([.02, -side*.22])
        if .19 <= distance <= .46 and np.linalg.norm(np.array(actor)-exact_actor)>.012 and np.linalg.norm(np.array(destination)-exact_destination)>.012:
            break
    else:
        raise ValueError("Cannot sample a reachable separated layout")
    spec = deepcopy(source)
    for key in ["validation", "observation", "images", "review_images"]:
        spec.pop(key, None)
    spec.update(source_item_id=source["item_id"], source_instruction=source["instruction"],
                item_id=f"{split}__c{condition_index:02d}__{index:04d}", condition_index=condition_index,
                pair_group=f"{split}__f{family_index:02d}__{index:04d}",
                split=split, index=index, seed=seed, noise_seed=seed+9000000,
                actor_xy=actor, destination_xy=destination,
                actor_yaw=float(rng.uniform(-.22,.22)), destination_yaw=0.,
                layout="left_to_right" if side==1 else "right_to_left",
                nominal_transport_distance_xy_m=float(distance))
    spec["goal_center"][:2] = goal.tolist()
    spec["instruction"] = instruction(spec, index % 4)
    return spec


def bddl(spec):
    path = DATA/"bddl"/(spec["item_id"]+".bddl"); path.parent.mkdir(parents=True,exist_ok=True)
    regions=[]
    for name,(x,y) in [("actor",spec["actor_xy"]),("destination",spec["destination_xy"])]:
        regions.append(f"({name}_start (:target main_table) (:ranges (({x-.00001} {y-.00001} {x+.00001} {y+.00001}))))")
    text="\n".join(["(define (problem SemanticTransportV6)","(:domain robosuite)",
        "(:language Follow the supplied instruction)","(:regions",*regions,")",
        f"(:fixtures main_table - table destination_1 - semantic_v6_{spec['destination_asset']})",
        f"(:objects actor_1 - semantic_v6_{spec['actor_asset']})","(:obj_of_interest actor_1 destination_1)",
        "(:init (On actor_1 main_table_actor_start) (On destination_1 main_table_destination_start))",
        "(:goal (And (On actor_1 main_table_actor_start)))",")",""])
    if not path.exists(): path.write_text(text)
    else: assert path.read_text()==text
    return path


def make_environment(spec,resolution=224):
    from libero.libero.envs import OffScreenRenderEnv
    register_assets(); random.seed(spec["seed"]); np.random.seed(spec["seed"])
    wrapper=OffScreenRenderEnv(bddl_file_name=str(bddl(spec)),camera_heights=resolution,camera_widths=resolution,
                              hard_reset=False,ignore_done=True,initialization_noise={"magnitude":0.,"type":"gaussian"})
    wrapper.seed(spec["seed"]);env=wrapper.env;env.reset()
    actor,destination=(env.obj_body_id[name] for name in ["actor_1","destination_1"])
    env.sim.model.body_pos[destination,:2]=spec["destination_xy"]
    p=env.sim.data.body_xpos[actor].copy();p[:2]=spec["actor_xy"]
    yaw=spec["actor_yaw"]
    set_actor_position(env,p,[math.cos(yaw/2),0.,0.,math.sin(yaw/2)])
    for _ in range(100):env.step([0.]*6+[-1.])
    env.sim.forward()
    p=env.sim.data.body_xpos[actor].copy()
    reference=env.sim.data.body_xmat[actor].reshape(3,3) @ spec["actor_reference_local"]
    p[:2]=np.array(spec["actor_xy"])-reference[:2]
    set_actor_position(env,p)
    return wrapper,current_observation(env),TransportOracle(env,spec)


def world_bounds(env,actor):
    model=env.sim.model;data=env.sim.data;points=[]
    for geom,body in enumerate(model.geom_bodyid):
        cursor=int(body)
        while cursor and cursor!=actor:cursor=int(model.body_parentid[cursor])
        if cursor!=actor or model.geom_group[geom]!=0:continue
        if model.geom_type[geom]==7:
            mesh=model.geom_dataid[geom];start=model.mesh_vertadr[mesh];length=model.mesh_vertnum[mesh]
            local=model.mesh_vert[start:start+length]
        else:
            center,half=model.geom_aabb[geom,:3],model.geom_aabb[geom,3:]
            local=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*half+center
        points.append(local@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom])
    points=np.concatenate(points)
    return points.min(0),points.max(0)


def validate_initial(env,oracle):
    from build_semantic_pairs_v6 import contact_pairs
    contacts=contact_pairs(env,("actor_1_",),("destination_1_",))
    robot=contact_pairs(env,("robot0_","gripper0_","mount0_"),("actor_1_","destination_1_"))
    assert not contacts and not robot, (contacts,robot)
    assert np.isfinite(env.sim.get_state().flatten()).all()
    assert np.linalg.norm(oracle.reference_position(env)[:2]-oracle.item["goal_center"][:2])>=.18


def freeze_design():
    import hashlib
    DATA.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    design=dict(source_items_sha256=hashlib.sha256((SOURCE/"items.jsonl").read_bytes()).hexdigest(),
                source_assets_sha256=hashlib.sha256((ASSETS/"manifest.json").read_bytes()).hexdigest(),
                conditions=66,families=33,split_seeds=SPLIT_SEEDS,actor_yaw_range_rad=[-.22,.22],
                exact_v6_scene_frames_or_states_in_training=False,v6_used_for_checkpoint_selection=False,
                v6_prior_development_observations_acknowledged=True,
                separate_unseen_layout_motor_test=True,teacher="privileged actual env.step controller, independently replay validated",
                task_scope="grasp, held lift, transport to destination region; not terminal cutting/pouring/physical harm",
                model_inputs="main/wrist RGB and natural-language instruction, no privileged pose or safety labels",
                original_instruction_samples="valid motor prefixes only; no claim of completing full requested skill",
                source_v6_dataset_read_only=True)
    path=DATA/"design.json"
    if path.exists():assert json.loads(path.read_text())==design
    else:write_json(path,design)


if __name__=="__main__":
    freeze_design()
