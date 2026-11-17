"""Evaluate the benign curriculum with a versioned, compatible pour-coordinate fix."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

import benign_adaptation_v1 as b
import evaluate_expansion_v5_lora as evaluation
from rollout_common import write_json

GenerationOracle=b.BenignOracle


class EvaluationOracle(GenerationOracle):
    """Keep generation code immutable; rename only the shadowing stroke scalar."""

    def measure(self,env):
        metric=b.OriginalOracle.measure(self,env)
        # This is the generation oracle's method with the scalar rename below.
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
            stroke_coordinate=float(point[self.spec["stroke_axis"]])
            self.stroke_low=min(self.stroke_low,stroke_coordinate);self.stroke_high=max(self.stroke_high,stroke_coordinate)
        span=max(0.,self.stroke_high-self.stroke_low) if np.isfinite(self.stroke_low) else 0.
        motor_skill=self.spec["motor_skill"]
        released=not metric["grasped"] and np.min(np.abs(env.sim.data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes]))>.020
        if motor_skill=="pour":
            hit=held and np.linalg.norm(position[:2]-self.goal[:2])<.065 and abs(position[2]-self.goal[2])<.08 and metric["tilt_rad"]>=math.radians(50)
        elif motor_skill=="stroke":hit=held and contact and span>=.030 and self.contact_steps>=16
        elif motor_skill=="press":hit=held and contact and self.contact_steps>=14
        else:
            low,high=b.motor.body_bounds(env,self.actor)
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


# This substitution is local to evaluation processes. Dataset generation,
# replay signatures, demonstrations and training manifests remain unchanged.
b.BenignOracle=EvaluationOracle

b.configure()
evaluation.DATA=b.DATA
evaluation.make_environment=b.make_environment
OriginalEpisode=evaluation.Episode


class Episode(OriginalEpisode):
    def packet(self):
        packet=super().packet()
        packet["image_orientation"]="native180" if self.native else "upright"
        return packet

    def save_image(self,name):
        if self.native:return super().save_image(name)
        cv2.imwrite(str(self.folder/name),np.ascontiguousarray(self.obs["agentview_image"][::-1,:,::-1]))

    def frame(self):
        if self.native:return super().frame()
        image=np.concatenate([self.obs["agentview_image"][::-1],self.obs["robot0_eye_in_hand_image"][::-1]],axis=1)
        self.video.write(np.ascontiguousarray(image[:,:,::-1]))


evaluation.Episode=Episode


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out",type=Path)
    args,_=parser.parse_known_args()
    if args.out:
        args.out.mkdir(parents=True,exist_ok=True)
        path=args.out/"evaluator_implementation.json"
        sources={Path(__file__).name:b.sha(Path(__file__)),"benign_adaptation_v1.py":b.sha(Path(b.__file__))}
        fingerprint=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
        if path.exists():assert json.loads(path.read_text())["fingerprint"]==fingerprint,"Evaluator changed"
        else:
            legacy=list((args.out/"episodes").glob("*/summary.json"))
            write_json(path,dict(fingerprint=fingerprint,sources=sources,
                correction="pour coordinate stays a 3-vector after actor-target contact; stroke scalar renamed",
                success_predicates_and_thresholds_unchanged=True,dataset_generation_code_unchanged=True,
                compatible_pre_fix_completed_episodes=len(legacy),native_evaluator_unchanged=True))
    evaluation.main()


if __name__=="__main__":main()
