"""Privileged physical controller for the declared v5 motor proxies."""

import json
import hashlib
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from expansion_v5_motor import assets, body_bounds, make_environment, sha, target_surface
from rollout_common import observation_payload, write_json


def candidates(spec):
    kind = spec["actor_kind"]
    extent = assets()[spec["actor_asset"]]["extent"]
    primary = -math.pi/2 if extent[0]>extent[1] else 0.
    rows = []
    def add(offset=(0,0),height=.5,yaw=primary,tilt=0.,dz=-.002,tilt_axis="x",side_entry=False):
        row = dict(offset=list(offset),height_fraction=height,yaw=yaw,tilt=tilt,z_adjust=dz)
        if tilt_axis!="x":row["tilt_axis"]=tilt_axis
        if side_entry:row["side_entry"]=True
        if row not in rows:
            rows.append(row)
    if kind=="compass":
        for offset in [(-.38,0),(-.28,0),(-.43,0)]:
            for dz in [.004,.008,0.]:add(offset,.6,0.,dz=dz)
    if (spec.get("flat_side_pinch") or (min(extent[:2])>.08 and extent[2]<.06)) and kind!="compass":
        # Panda fingers close along local x. Rotate about local y to pinch a
        # horizontal sheet vertically, entering from its edge rather than above.
        for tilt in [1.3,1.5,1.1]:
            for offset,yaw,sign in [((.45,0),math.pi/2,1),((-.45,0),math.pi/2,-1),
                                    ((0,.45),0.,-1),((0,-.45),0.,1)]:
                add(offset,.5,yaw,sign*tilt,dz=0.,tilt_axis="y",side_entry=True)
    if kind in {"point_tool","peeler"}:
        for height in [.5,.7,.3]:
            for dz in [.006,.010,.002]:
                for yaw in [primary,primary+math.pi/2]:
                    add(height=height,yaw=yaw,dz=dz)
    if extent[2]<.025 and kind not in {"mesh:knife","blade","saw","cloth","paper","compass"}:
        for height in [.5,.8]:
            for dz in [.004,.008,0.]:
                for yaw in [primary,primary+math.pi/2]:add(height=height,yaw=yaw,dz=dz)
    if kind == "mesh:hand":
        for height in [.7,.5,.9]:
            for offset in [(-.3,-.40),(-.4,-.30),(.3,-.40),(.4,-.30)]:
                for yaw in [0.,math.pi/2]:
                    add(offset,height,yaw)
        # Fingers and wrist sit below the palm's global bounding-box center.
        for height in [.12,.23,.35]:
            for offset in [(.34,.30),(-.34,.30),(.34,-.30),(-.34,-.30),(0,.40),(0,-.40)]:
                for yaw in [0.,math.pi/2]:
                    add(offset,height,yaw)
    if kind in {"bowl","cup","box_container","tray","vented_container"}:
        for height in [.78,.55]:
            for offset,yaw in [((.46,0),0.),((-.46,0),0.),((0,.46),math.pi/2),((0,-.46),math.pi/2)]:
                for tilt in [0.,-.35,.35]:
                    add(offset,height,yaw,tilt)
    if kind in {"mesh:knife","saw"}:
        for offset in [(0,.34),(0,.25),(0,.40)]:
            for height in [.5,.7,.3]:
                add(offset,height,0.)
    if kind=="blade":
        for height in [.85,.7,.95]:
            for yaw in [0.,math.pi/2]:add(height=height,yaw=yaw,dz=.002)
    if kind == "bottle":
        for height in [.82,.65,.38,.50]:
            add(height=height,yaw=0.)
            add(height=height,yaw=-math.pi/2)
    if kind in {"mesh:pan","mesh:kettle","iron","power_tool","handheld_hot_plate"}:
        for offset in [(0,0),(.33,0),(-.33,0),(0,.33),(0,-.33)]:
            for height in [.75,.45]:
                add(offset,height)
    for height in [.5,.3,.7]:
        for yaw in [primary,primary+math.pi/2]:
            add(height=height,yaw=yaw)
    for offset in [(.38,0),(-.38,0),(0,.38),(0,-.38),(.30,.30),(-.30,-.30)]:
        for yaw in [primary,primary+math.pi/2]:
            add(offset,.6,yaw)
    for tilt in [.6,-.6,1.1,-1.1]:
        for offset in [(0,0),(.38,0),(-.38,0),(0,.38),(0,-.38)]:
            for yaw in [primary,primary+math.pi/2]:
                add(offset,.5,yaw,tilt)
    return rows


class Teacher:
    def __init__(self,env,oracle,spec,profile):
        self.oracle,self.spec,self.profile = oracle,spec,profile
        self.stage,self.stage_steps = "approach",0
        self.failed = False
        low,high = body_bounds(env,oracle.actor)
        point = (low+high)/2
        point[:2] += np.array(profile["offset"])*(high-low)[:2]
        point[2] = low[2]+profile["height_fraction"]*(high[2]-low[2])+profile["z_adjust"]
        rotation = env.sim.data.body_xmat[oracle.actor].reshape(3,3)
        self.local_grasp = rotation.T @ (point-oracle.position(env))
        self.grasp_position=point.copy()
        heading=math.atan2(rotation[1,0],rotation[0,0])
        self.orientation = Rotation.from_rotvec([0.,0.,profile["yaw"]+heading]).as_matrix() @ np.array(
            [[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        self.upright_orientation=self.orientation.copy()
        tilt_vector=np.zeros(3)
        tilt_vector[1 if profile.get("tilt_axis")=="y" else 0]=profile["tilt"]
        self.orientation = self.orientation @ Rotation.from_rotvec(tilt_vector).as_matrix()
        self.pregrasp=point-self.orientation[:,2]*.075 if profile.get("side_entry") else point.copy()
        self.reorient=profile.get("reorient_after_lift",abs(profile["tilt"])>.1)
        if spec["skill"]=="contact_proxy" and spec["actor_kind"] not in {"mesh:knife","blade","saw"}:
            offset=np.asarray(profile["offset"])
            contact=(low+high)/2;contact[2]=low[2]
            if max(abs(offset))>=.25:
                axis=int(np.argmax(abs(offset)))
                contact[axis]-=np.sign(offset[axis])*.35*(high[axis]-low[axis])
            elif spec["actor_kind"] in {"point_tool","power_tool","peeler"}:
                axis=int(np.argmax((high-low)[:2]))
                contact[axis]+=.40*(high[axis]-low[axis])
            oracle.working_point=rotation.T@(contact-oracle.position(env))
        self.target=oracle.goal.copy()
        target_prop=next(p for p in spec["props"] if p["role"]=="target")
        if spec["skill"]=="contact_proxy" and target_prop["kind"]=="mesh:hand":
            self.target=target_surface(env,oracle.target,oracle.target_geoms,xy=self.target[:2]+[0,.04])
        self.initial_reference = self.reference(env).copy()
        self.carry_z = max(self.initial_reference[2]+.15,self.target[2]+.09)
        self.close_position = None

    def reference(self,env):
        if self.spec["skill"] == "contact_proxy":
            return self.oracle.work_position(env)
        return self.oracle.position(env)

    def change(self,stage):
        self.stage,self.stage_steps = stage,0

    def action(self,env,obs,metric):
        current = np.asarray(obs["robot0_eef_pos"])
        point = self.grasp_position
        desired_orientation = self.orientation
        if self.reorient and self.stage in {"orient","transport","lower","hold","tilt"}:
            desired_orientation=self.upright_orientation
        close = self.stage not in {"approach","pregrasp","descend"}
        reference = self.reference(env)
        if self.stage == "approach":
            goal = self.pregrasp.copy()
            goal[2] = max(1.18,goal[2]+.18)
            angle=np.linalg.norm(Rotation.from_matrix(self.orientation @ env.robots[0].controller.ee_ori_mat.T).as_rotvec())
            if np.linalg.norm(current-goal)<.012 and angle<.12:
                self.change("pregrasp" if self.profile.get("side_entry") else "descend")
            elif self.stage_steps>=80:
                self.failed=True
        elif self.stage == "pregrasp":
            goal=self.pregrasp
            if np.linalg.norm(current-goal)<.006:
                self.change("descend")
            elif self.stage_steps>=65:
                self.failed=True
        elif self.stage == "descend":
            goal = point
            if np.linalg.norm(current-goal)<.0045 or self.stage_steps>=65:
                self.close_position=current.copy()
                self.change("close")
        elif self.stage == "close":
            goal=self.close_position
            if self.stage_steps>=20:
                self.change("lift")
        else:
            target=self.target.copy()
            if self.stage == "lift":
                target[:2]=self.initial_reference[:2]
                target[2]=self.carry_z
                if metric["ever_lifted"] and np.linalg.norm(reference-target)<.015:
                    self.change("orient" if self.reorient else "transport")
                elif self.stage_steps>=70:
                    self.failed=True
            elif self.stage == "orient":
                target[:2]=self.initial_reference[:2]
                target[2]=self.carry_z
                angle=np.linalg.norm(Rotation.from_matrix(desired_orientation @ env.robots[0].controller.ee_ori_mat.T).as_rotvec())
                if angle<.12 and np.linalg.norm(reference-target)<.020:
                    self.change("transport")
                elif self.stage_steps>=95:
                    self.failed=True
            elif self.stage == "transport":
                target[2]=self.carry_z
                if np.linalg.norm(reference-target)<.020:
                    self.change("lower")
                elif self.stage_steps>=90:
                    self.failed=True
            elif self.stage == "lower":
                if self.spec["skill"] == "contact_proxy":
                    target[2] -= min(.035,max(0,self.stage_steps-20)*.0007)
                    if metric["actor_target_contact"]:
                        self.contact_target=reference.copy()
                        self.change("hold")
                elif np.linalg.norm(reference-target)<.025:
                    self.change("tilt" if self.spec["skill"]=="pour_proxy" else "hold")
                if self.stage_steps>=100:
                    self.failed=True
            elif self.stage == "tilt":
                angle=min(math.radians(70),self.stage_steps*.025)
                desired_orientation=Rotation.from_rotvec([angle,0,0]).as_matrix() @ self.orientation
                if self.stage_steps>=95:
                    self.failed=True
            elif self.stage == "hold":
                if hasattr(self,"contact_target"):
                    if not metric["actor_target_contact"]:
                        self.contact_target[2]-=.0005
                    target=self.contact_target
                if self.stage_steps>=50:
                    self.failed=True
            goal=current+target-reference
        orientation_gain,orientation_limit=(.6,.25) if self.stage=="orient" else (1.5,.7)
        action=np.r_[np.clip((goal-current)*12.,-.5,.5),
            np.clip(Rotation.from_matrix(desired_orientation @ env.robots[0].controller.ee_ori_mat.T).as_rotvec()*orientation_gain,-orientation_limit,orientation_limit),
            1. if close else -1.].astype(np.float32)
        self.stage_steps+=1
        return action


def rollout(spec,profile,folder=None,resolution=0,horizon=460):
    wrapper=None
    try:
        wrapper,obs,oracle=make_environment(spec,resolution)
        env=wrapper.env
        teacher=Teacher(env,oracle,spec,profile)
        initial=env.sim.get_state().flatten().copy()
        metric=oracle.measure(env)
        states=[initial]
        robot_states=[];frames=[];wrists=[];actions=[];events=[];stages=[]
        if folder is not None:
            payload=observation_payload(obs)
            robot_states.append(payload["state"]);frames.append(payload["image"]);wrists.append(payload["wrist_image"])
        unstable=False
        for step in range(horizon):
            action=teacher.action(env,obs,metric)
            actions.append(action);stages.append(teacher.stage)
            previous_time=env.sim.data.time
            obs,_,_,_=env.step(action)
            unstable = (env.sim.data.time<=previous_time or not np.isfinite(env.sim.data.qvel).all()
                        or float(np.max(np.abs(env.sim.data.qvel)))>100.)
            metric=oracle.measure(env)
            if unstable:
                metric.update(proxy_success=False,stable_steps=0,physics_unstable=True)
                teacher.failed=True
            events.append(dict(control_step=step+1,teacher_stage=teacher.stage,**metric))
            if folder is not None:
                payload=observation_payload(obs)
                frames.append(payload["image"]);wrists.append(payload["wrist_image"]);robot_states.append(payload["state"])
                states.append(env.sim.get_state().flatten().copy())
            if metric["stable_steps"]>=10 or teacher.failed:
                break
        passed=bool(not unstable and metric["proxy_success"] and metric["stable_steps"]>=6)
        result=dict(item_id=spec["item_id"],condition_index=spec["condition_index"],passed=passed,
            steps=len(actions),stage=teacher.stage,skill=spec["skill"],profile=profile,
            final_metric=metric,initial_actor_position=oracle.initial.tolist(),goal=oracle.goal.tolist(),
            all_actions_physical=True,teleport_during_recording=False,model_used=False,physics_stable=not unstable)
        result["action_sha256"]=hashlib.sha256(np.asarray(actions,dtype=np.float32).tobytes()).hexdigest()
        result["model_xml_sha256"]=hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest()
        if folder is not None:
            folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
            with h5py.File(folder/"trajectory.h5","w") as data:
                data.attrs["spec_json"]=json.dumps(spec,ensure_ascii=False)
                data.attrs["profile_json"]=json.dumps(profile)
                data.attrs["teacher_success"]=passed
                data.attrs["alignment"]="observations[t] -> actions[t] -> observations[t+1]"
                for key,value in {"observations/main_rgb_raw":frames,"observations/wrist_rgb_raw":wrists,
                    "observations/state":robot_states,"observations/sim_state":states,"actions":actions}.items():
                    array=np.asarray(value)
                    data.create_dataset(key,data=array,compression="lzf",shuffle=True,chunks=(1,*array.shape[1:]))
                data.create_dataset("teacher_stage",data=np.array(stages,dtype="S16"))
            (folder/"model.xml").write_text(env.sim.model.get_xml())
            (folder/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
            for filename,index in [("initial.png",0),("final.png",-1)]:
                cv2.imwrite(str(folder/filename),np.ascontiguousarray(frames[index][::-1,::-1,::-1]))
            if spec["split"]=="calibration" or spec["index"]<2:
                writer=cv2.VideoWriter(str(folder/"views.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),20,(resolution*2,resolution))
                for main,wrist in zip(frames,wrists):
                    writer.write(np.ascontiguousarray(np.concatenate([main[::-1,::-1],wrist[::-1,::-1]],axis=1)[:,:,::-1]))
                writer.release()
            write_json(folder/"summary.json",result)
        return result
    finally:
        if wrapper is not None:
            wrapper.close()


def replay(folder):
    folder=Path(folder)
    with h5py.File(folder/"trajectory.h5") as data:
        spec=json.loads(data.attrs["spec_json"])
        actions=data["actions"][:]
        expected=data["observations/sim_state"][:]
    wrapper,obs,oracle=make_environment(spec,0)
    try:
        env=wrapper.env
        error=float(np.max(np.abs(env.sim.get_state().flatten()-expected[0])))
        oracle.measure(env)
        for i,action in enumerate(actions):
            env.step(action)
            metric=oracle.measure(env)
            error=max(error,float(np.max(np.abs(env.sim.get_state().flatten()-expected[i+1]))))
        result=dict(accepted=bool(error<1e-6 and metric["proxy_success"] and metric["stable_steps"]>=6),
            max_sim_state_abs_error=error,steps=len(actions),final_metric=metric,
            trajectory_sha256=sha(folder/"trajectory.h5"),independent_environment=True)
        write_json(folder/"replay_validation.json",result)
        return result
    finally:
        wrapper.close()
