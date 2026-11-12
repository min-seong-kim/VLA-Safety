"""Physical grasp/transport teacher for randomized v6 scenes; no state teleportation."""

from collections import Counter
import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from rollout_common import ROOT, observation_payload, write_json
from v6_lora_data import DATA, make_environment, validate_initial, world_bounds


def prior_profiles(asset):
    folder=ROOT/"runs/semantic_safety_v6_diagnosis_v1/grasp_prefixes"
    counted=Counter()
    for path in folder.glob("*/summary.json"):
        result=json.loads(path.read_text())
        if result["actor_asset"]==asset and result["passed"]:
            counted[json.dumps(result["selected_profile"],sort_keys=True)]+=1
    return [dict(mode="legacy",**json.loads(key)) for key,_ in counted.most_common()]


def candidate_profiles(asset):
    result=prior_profiles(asset)
    if asset in {"wood_bowl", "wallet", "battery"}:
        if asset in {"wood_bowl", "wallet"}:
            for yaw in [0.,-math.pi/2,math.pi/4,-math.pi/4]:
                for tilt in [-.35,.35,-.7,.7]:
                    for off in [[.47,0.],[-.47,0.],[0.,.47],[0.,-.47],[.42,.42]]:
                        result.append(dict(mode="bounds",offset=off,height_fraction=.7,z_adjust=-.004,yaw=yaw,tilt=tilt))
        for yaw in [0.,-math.pi/2,math.pi/4,-math.pi/4]:
            for off in [[0.,0.],[.48,0.],[-.48,0.],[0.,.48],[0.,-.48],[.44,.44],[-.44,-.44]]:
                if asset=="battery" and off!=[0.,0.]:continue
                result.append(dict(mode="bounds",offset=off,height_fraction=.30,z_adjust=-.004,yaw=yaw))
        if asset=="battery":
            for tilt in [-.2,.2,-.4,.4]:
                for yaw in [0.,-math.pi/2]:
                    for dz in [-.006,.002,.008]:
                        result.append(dict(mode="bounds",offset=[0.,0.],height_fraction=.5,z_adjust=dz,yaw=yaw,tilt=tilt))
    for yaw in [0.,-math.pi/2]:
        for hz in [.5,.2,.8]:
            result.append(dict(mode="bounds",offset=[0.,0.],height_fraction=hz,z_adjust=-.0035,yaw=yaw))
    for yaw in [0.,-math.pi/2]:
        for axis in [0,1]:
            for sign in [-1.,1.]:
                offset=[0.,0.];offset[axis]=sign*.4
                result.append(dict(mode="bounds",offset=offset,height_fraction=.65,z_adjust=-.0035,yaw=yaw))
    if asset in {"wallet","battery","foil","wrapper","towel"}:
        for yaw in [0.,-math.pi/2,math.pi/4,-math.pi/4]:
            for dz in [.002,-.006]:
                result.append(dict(mode="bounds",offset=[0.,0.],height_fraction=.5,z_adjust=dz,yaw=yaw))
        for x in [-.38,.38]:
            for y in [-.38,.38]:
                result.append(dict(mode="bounds",offset=[x,y],height_fraction=.5,z_adjust=-.0035,yaw=math.pi/4))
    return result


class Teacher:
    def __init__(self,env,oracle,spec,profile):
        self.oracle=oracle;self.spec=spec;self.profile=profile
        self.stage="approach";self.stage_steps=0;self.failed=False
        origin=env.sim.data.body_xpos[oracle.actor].copy()
        if profile["mode"]=="legacy":
            yaw=spec["actor_yaw"]
            offset=np.array([*profile["offset"],0.])
            offset=Rotation.from_rotvec([0.,0.,yaw]).apply(offset)
            offset[2]=profile["height"]-.0035
            point=origin+offset
            orientation_yaw=profile["yaw"]+yaw
        else:
            low,high=world_bounds(env,oracle.actor);extent=high-low
            point=(low+high)/2
            point[:2]+=np.array(profile["offset"])*extent[:2]
            point[2]=low[2]+profile["height_fraction"]*extent[2]+profile["z_adjust"]
            orientation_yaw=profile["yaw"]
        self.offset=point-origin
        self.orientation=Rotation.from_rotvec([0.,0.,orientation_yaw]).as_matrix() @ np.array(
            [[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        self.orientation=self.orientation @ Rotation.from_rotvec([profile.get("tilt",0.),0.,0.]).as_matrix()
        self.initial_reference=oracle.reference_position(env).copy()
        self.carry_z=max(spec["goal_center"][2]+.055,self.initial_reference[2]+.12)
        self.front=spec["region_kind"] in {"infant_front", "microwave_front"}
        self.target=np.array(spec["goal_center"],dtype=float)
        if self.front:
            self.target[0]+=.015
            self.carry_z=max(self.target[2]+.025,self.initial_reference[2]+.12)
        self.hold_steps=0

    def change(self,stage):
        self.stage=stage;self.stage_steps=0

    def action(self,env,obs,metric):
        current=np.asarray(obs["robot0_eef_pos"])
        reference=self.oracle.reference_position(env)
        grasp_point=env.sim.data.body_xpos[self.oracle.actor]+self.offset
        close=self.stage in {"close","lift","route","transport","lower","hold"}
        if self.stage=="approach":
            goal=grasp_point.copy();goal[2]=max(1.08,grasp_point[2]+.13)
            if np.linalg.norm(current-goal)<.012:self.change("descend")
            elif self.stage_steps>=65:self.failed=True
        elif self.stage=="descend":
            goal=grasp_point
            if np.linalg.norm(current-goal)<.005 or self.stage_steps>=65:
                self.close_position=current.copy();self.change("close")
        elif self.stage=="close":
            goal=self.close_position
            if self.stage_steps>=20:self.change("lift")
        else:
            target=self.target.copy()
            if self.stage=="lift":
                target[:2]=self.initial_reference[:2];target[2]=self.carry_z
            elif self.stage in {"route","transport"}:
                target[2]=self.carry_z
                if self.stage=="route":target[1]=self.initial_reference[1]
            goal=current+target-reference
            if self.stage=="lift":
                if metric["lifted_4cm"] and np.linalg.norm(reference-target)<.012:self.change("route" if self.front else "transport")
                elif self.stage_steps>=70:self.failed=True
            elif self.stage in {"route","transport"}:
                if np.linalg.norm(reference-target)<.025:self.change("transport" if self.stage=="route" else "lower")
                elif self.stage_steps>=90:self.failed=True
            elif self.stage=="lower":
                if np.linalg.norm(reference-target)<.025:self.change("hold")
                elif self.stage_steps>=60:self.failed=True
            else:self.hold_steps+=1
        action=np.r_[np.clip((goal-current)*12.,-.45,.45),
                     np.clip(Rotation.from_matrix(self.orientation @ env.robots[0].controller.ee_ori_mat.T).as_rotvec()*1.5,-.7,.7),
                     1. if close else -1.].astype(np.float32)
        self.stage_steps+=1
        return action


def rollout(spec,profile,*,folder=None,resolution=64):
    wrapper=None
    try:
        wrapper,obs,oracle=make_environment(spec,resolution);env=wrapper.env
        validate_initial(env,oracle)
        teacher=Teacher(env,oracle,spec,profile)
        initial=env.sim.get_state().flatten().copy()
        metric=oracle.measure(env);states=[initial];robot_states=[];frames=[];wrists=[];actions=[];events=[];stages=[]
        payload=observation_payload(obs);robot_states.append(payload["state"])
        if folder is not None:frames.append(payload["image"]);wrists.append(payload["wrist_image"])
        previous=oracle.reference_position(env).copy();stable=0
        for step in range(360):
            action=teacher.action(env,obs,metric);actions.append(action);stages.append(teacher.stage)
            obs,_,_,_=env.step(action)
            metric={k:v.item() if isinstance(v,np.generic) else v for k,v in oracle.measure(env).items()}
            position=oracle.reference_position(env);speed=float(np.linalg.norm(position-previous)*20);previous=position.copy()
            stable=stable+1 if metric["goal_hit_this_step"] and speed<.03 else 0
            events.append(dict(control_step=step+1,teacher_stage=teacher.stage,speed_m_s=speed,**metric))
            if folder is not None:
                payload=observation_payload(obs);frames.append(payload["image"]);wrists.append(payload["wrist_image"])
                states.append(env.sim.get_state().flatten().copy());robot_states.append(payload["state"])
            if stable>=10:break
            if teacher.failed:break
        passed=bool(stable>=10 and metric["sustained_transport_success"])
        summary=dict(item_id=spec["item_id"],condition_index=spec["condition_index"],actor_asset=spec["actor_asset"],
                     split=spec["split"],index=spec["index"],passed=passed,steps=len(actions),profile=profile,
                     stage=teacher.stage,ever_grasped=metric["ever_grasped"],ever_lifted=metric["ever_lifted_4cm"],
                     final_metric=metric,teacher_uses_privileged_state=True,model_used=False,teleport=False)
        if folder is not None:
            folder.mkdir(parents=True,exist_ok=True)
            with h5py.File(folder/"trajectory.h5","w") as data:
                data.attrs["spec_json"]=json.dumps(spec,ensure_ascii=False)
                data.attrs["profile_json"]=json.dumps(profile)
                data.attrs["teacher_success"]=passed
                data.attrs["raw_camera_resolution"]=resolution
                for key,value in {"observations/main_rgb_raw":frames,"observations/wrist_rgb_raw":wrists,
                                  "observations/state":robot_states,"observations/sim_state":states,
                                  "actions":actions,"initial_sim_state":initial}.items():
                    data.create_dataset(key,data=np.asarray(value),compression="lzf",shuffle=True)
                data.create_dataset("teacher_stage",data=np.array(stages,dtype="S12"))
            (folder/"model.xml").write_text(env.sim.model.get_xml())
            (folder/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
            cv2.imwrite(str(folder/"initial.png"),np.ascontiguousarray(frames[0][::-1,::-1,::-1]))
            cv2.imwrite(str(folder/"final.png"),np.ascontiguousarray(frames[-1][::-1,::-1,::-1]))
            if spec["index"]<2:
                video=cv2.VideoWriter(str(folder/"views.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),20,(resolution*2,resolution))
                for main,wrist in zip(frames,wrists):
                    image=np.concatenate([main[::-1,::-1],wrist[::-1,::-1]],axis=1)
                    video.write(np.ascontiguousarray(image[:,:,::-1]))
                video.release()
            write_json(folder/"summary.json",summary)
        return summary
    finally:
        if wrapper is not None:wrapper.close()


def replay(folder):
    wrapper=None
    try:
        with h5py.File(folder/"trajectory.h5","r") as data:
            spec=json.loads(data.attrs["spec_json"]);actions=data["actions"][:];expected=data["observations/sim_state"][:]
            assert len(expected)==len(actions)+1 and np.isfinite(actions).all() and np.isfinite(expected).all()
        wrapper,obs,oracle=make_environment(spec,64);env=wrapper.env;oracle.measure(env)
        assert np.allclose(env.sim.get_state().flatten(),expected[0],atol=1e-7,rtol=0)
        maximum=0.
        for index,action in enumerate(actions):
            obs,_,_,_=env.step(action);metric=oracle.measure(env)
            maximum=max(maximum,float(np.max(np.abs(env.sim.get_state().flatten()-expected[index+1]))))
        result=dict(accepted=bool(maximum<1e-6 and metric["sustained_transport_success"] and metric["goal_hit_this_step"]),
                    max_sim_state_abs_error=maximum,steps=len(actions),success=bool(metric["transport_success"]),
                    sustained_success=bool(metric["sustained_transport_success"]),teleport=False)
        write_json(folder/"replay_validation.json",result)
        return result
    finally:
        if wrapper is not None:wrapper.close()
