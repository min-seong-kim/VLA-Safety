"""Verify feedback against generated geometry and actual rendered simulator state."""

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/semantic_safety_observations_v5"
CAT=ROOT/"data/semantic_safety_catalog_v5"


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run():
    import mujoco
    import numpy as np

    pairs=load(CAT/"scenario_pairs.jsonl"); scenes=load(OUT/"scenes.jsonl")
    by_id={p["pair_id"]:p for p in pairs}; by_scene={s["scene_id"]:s for s in scenes}
    assets=json.loads((OUT/"assets/manifest.json").read_text())
    revision=json.loads((CAT/"revision_summary.json").read_text())
    decisions=load(CAT/"revision_decisions.jsonl")
    assert len(pairs)==169 and len(scenes)==338
    assert not any(p["category"]=="property" for p in pairs)
    excluded={d["pair_id"] for d in decisions if d["decision"]=="excluded"}
    assert len(excluded)==125 and excluded.isdisjoint(by_id)
    assert hashlib.sha256((ROOT/"data/semantic_safety_catalog_v4/scenario_pairs.jsonl").read_bytes()).hexdigest()==revision["source_sha256"]
    assert hashlib.sha256((ROOT/"feedback_safety_observation_v4.txt").read_bytes()).hexdigest()==revision["feedback_sha256"]
    assert all(a["kind"] not in {"person","hand","arm","foot","worn_loop"} for a in assets.values())

    def prop(pid,side,role):
        return next(p for p in by_scene[pid+"_"+side]["props"] if p["role"]==role)
    def binding(pid,side,role):
        return assets[prop(pid,side,role)["asset"]]

    infant=[p for p in pairs if p["pair_id"].startswith("v3_body_") and 93<=int(p["local_id"][5:])<=108]
    assert len(infant)==13
    assert len({prop(p["pair_id"],"safe","actor")["asset"] for p in infant})==1
    plant_scenes=[s for s in scenes if any(p["kind"]=="plant" for p in s["props"])]
    assert {s["pair_id"] for s in plant_scenes}=={"v5_living_plant_001"}
    knife_checks=0; pose_checks=0
    for alias,a in assets.items():
        if a["kind"] not in {"mesh:knife","blade","pin_pad","grater","brush","abrasive","handheld_hot_plate"}:continue
        model=mujoco.MjModel.from_xml_path(str(OUT/"assets"/(alias+".xml")))
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"v5_tool_rotation")
        assert idx>=0,alias
        rotation=data.xmat[idx].reshape(3,3)
        if a["kind"] in {"mesh:knife","blade"}:
            assert abs((rotation @ [0,0,1])[2])<1e-6
            assert np.allclose(rotation @ a["pose_adjustment"]["cutting_edge_normal_local"], [0,0,-1], atol=1e-6)
            knife_checks+=1
        else:
            original_normal=[0,-1,0] if a["kind"]=="grater" else ([0,0,-1] if a["kind"]=="handheld_hot_plate" else [0,0,1])
            assert np.allclose(rotation @ original_normal,[0,0,-1],atol=1e-6),a["name"]
            pose_checks+=1
    for s in scenes:
        actor=next((p for p in s["props"] if p["role"]=="actor"),None)
        if actor and actor["kind"] in {"mesh:knife","blade"}:
            assert actor["z_offset"]>=.03 and any(p["kind"]=="knife_cradle" and not p["task_object"] for p in s["props"])
    assert np.allclose(binding("v3_food_021","safe","actor")["extent"],[.028,.012,.008],atol=1e-6)
    assert np.allclose(binding("v3_food_021","unsafe","actor")["extent"],[.028,.012,.008],atol=1e-6)
    for a in assets.values():
        if a.get("requested_max_extent"):assert max(a["extent"])<=a["requested_max_extent"]+1e-6
    assert binding("v3_food_052","unsafe","target")["kind"]=="open_jar"
    for n in [65,66,69]:assert binding(f"v3_food_{n:03d}","unsafe","actor")["kind"]=="glass_fragments"
    for side in ["safe","unsafe"]:
        assert binding("v3_food_067",side,"target")["kind"]=="aquarium"
        assert binding("v3_food_092",side,"actor")["glove_coating"]
        assert "bare hand" not in by_id["v3_food_092"][side]["instruction_en"]
    for n in [72,73,74]:assert binding(f"v3_food_{n:03d}","safe","target")["kind"]=="box_container"
    assert binding("v3_food_095","unsafe","actor")["surface_mark_count"]>0
    assert binding("v3_food_095","safe","actor")["surface_mark_count"]==0
    assert binding("v3_food_092","unsafe","actor")["surface_mark_count"]>0
    assert len({prop(f"v3_food_{n:03d}","unsafe","actor")["asset"] for n in [101,102,107,109]})==1
    assert all(binding("v3_food_111",s,"target")["kind"]=="box_container" for s in ["safe","unsafe"])
    assert binding("v3_property_015","unsafe","target")["kind"]=="upright_book"
    for p in pairs:
        if p["category"] in {"food","chemical","living_environment","hygiene"}:
            assert all(p[s]["instruction_en"].split()[0].lower() not in {"spray","spread","mix","stir","drain","rinse","dissolve","serve"} for s in ["safe","unsafe"])
        if "recipient_allergy" in p:
            assert all(p["recipient_allergy"]+" allergy" in p[s]["instruction_en"] for s in ["safe","unsafe"])
        for side in ["safe","unsafe"]:
            b=p[side]
            if b.get("actor_kind")=="bottle":
                assert b["actor"].endswith("bottle") and b["instruction_en"].startswith("Pour ")
    for side in ["safe","unsafe"]:
        assert "de-energized" in by_id["v5_electrical_001"]["safe"]["instruction_en"]
        assert "connected and energized" in by_id["v5_electrical_001"]["unsafe"]["instruction_en"]
        assert prop("v5_electrical_001",side,"actor")["kind"]=="mesh:knife"

    ref=json.loads((OUT/"native_reference/reference.json").read_text())
    native_checked=0; native_camera_reference=None
    for s in scenes:
        report=json.loads((OUT/"observations"/s["scene_id"]/"validation.json").read_text())
        assert report["rendered"] and report["technical_pass"],s["scene_id"]
        if s["pair_id"] in {f"v3_property_{n:03d}" for n in range(1,9)}:
            actor=prop(s["pair_id"],s["side"],"actor"); target=prop(s["pair_id"],s["side"],"target")
            assert target["kind"]=="hotplate"
            assert report["agentview_centroids"][actor["name"]][0]<report["agentview_centroids"][target["name"]][0]
        if s["scene_template"]!="libero_long_kitchen_scene6":continue
        native_checked+=1
        assert np.allclose(report["robot_state"],ref["robot_state"],atol=1e-6)
        for name in ["microwave_1","porcelain_mug_1"]:
            actual=next(p for p in report["objects"] if p["name"]==name)
            assert np.allclose(actual["world_position"],ref["objects"][name]["world_position"],atol=1e-7)
        model=mujoco.MjModel.from_xml_path(str(OUT/"observations"/s["scene_id"]/"model.xml"))
        assert report["visibility_id_source"]=="actual_mujoco_geom_name2id"
        with np.load(OUT/"observations"/s["scene_id"]/"object_segmentation.npz") as mask:
            for name,ids in report["visibility_geom_ids"].items():
                assert all(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i).startswith(name+"_") for i in ids)
                pixels=((mask["object_type"]==int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(mask["object_id"],ids)).sum()
                assert int(pixels)==report["visible_pixels"][name]>12
        with np.load(OUT/"observations"/s["scene_id"]/"initial_observation.npz") as obs:
            for name,val in ref["joints"].items():
                if not name.startswith("microwave_1"):continue
                idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
                assert idx>=0
                offset=model.jnt_qposadr[idx]
                assert np.allclose(obs["qpos"][offset:offset+np.asarray(val).size],np.asarray(val).ravel())
        camera=report["camera"]["agentview"]
        if native_camera_reference is None:native_camera_reference=camera
        assert camera==native_camera_reference
    assert native_checked==8
    result=dict(passed=True,pairs=len(pairs),conditions=len(scenes),excluded_pairs=len(excluded),replacement_pairs=2,
                infant_pairs=13,all_knives_upright_assets=knife_checks,knife_cutting_edges_downward=True,
                downward_contact_face_assets=pose_checks,
                size_and_container_checks=True,allergy_context_retained=True,single_plant_pair=True,
                native_microwave_conditions=native_checked,native_robot_background_and_door_match=True,
                native_visibility_checked_using_actual_geom_ids=True,
                hotplate_actor_left_destination_right=True,contaminated_scissors_visible_marks=True,
                identical_mop_assets=True,source_v4_and_feedback_unchanged=True,
                semantic_labels_validated=False,policy_or_grasp_validated=False)
    (OUT/"feedback_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
