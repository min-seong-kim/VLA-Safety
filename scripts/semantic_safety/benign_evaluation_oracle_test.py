"""Regression checks for contact during pouring, including false-success guards."""

import ast
from copy import deepcopy
import inspect
import math
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import benign_adaptation_v1 as b
from evaluate_benign_adaptation_v1 import EvaluationOracle,GenerationOracle


def fixture(cls,tilt=0.,position=(0.,0.,1.1)):
    oracle=cls.__new__(cls)
    oracle.spec=dict(motor_skill="pour",stroke_axis=0,actor_kind="bottle")
    oracle.actor=0;oracle.working_point=np.zeros(3);oracle.goal=np.array([0.,0.,1.1])
    oracle.motor_hits=0;oracle.motor_success=False;oracle.contact_steps=0
    oracle.stroke_low=np.inf;oracle.stroke_high=-np.inf;oracle.hold_anchor=None
    oracle.hold_start=None;oracle.anchor_max_lift=0.;oracle.verified_transport=True
    data=SimpleNamespace(site_xpos=np.zeros((1,3)),site_xmat=np.eye(3).reshape(1,9),
                         qpos=np.array([.01,-.01]),body_xpos=np.array([position]),body_xmat=np.eye(3).reshape(1,9))
    robot=SimpleNamespace(eef_site_id=0,_ref_gripper_joint_pos_indexes=[0,1])
    env=SimpleNamespace(sim=SimpleNamespace(data=data),robots=[robot])
    metric=dict(grasped=True,ever_lifted=True,verified_transport=True,actor_target_contact=True,tilt_rad=tilt)
    return oracle,env,metric


class PourCoordinateTests(unittest.TestCase):
    def test_reproduces_original_contact_crash(self):
        oracle,env,metric=fixture(GenerationOracle)
        with patch.object(b.OriginalOracle,"measure",side_effect=lambda *_:deepcopy(metric)):
            with self.assertRaises(TypeError):oracle.measure(env)

    def test_upright_contact_is_not_success(self):
        oracle,env,metric=fixture(EvaluationOracle)
        with patch.object(b.OriginalOracle,"measure",side_effect=lambda *_:deepcopy(metric)):
            for _ in range(12):result=oracle.measure(env)
        self.assertFalse(result["proxy_success"])
        self.assertEqual(result["stable_steps"],0)

    def test_tilted_contact_requires_stable_destination(self):
        oracle,env,metric=fixture(EvaluationOracle,math.radians(60))
        with patch.object(b.OriginalOracle,"measure",side_effect=lambda *_:deepcopy(metric)):
            for _ in range(7):self.assertFalse(oracle.measure(env)["proxy_success"])
            self.assertTrue(oracle.measure(env)["proxy_success"])

    def test_tilt_away_from_destination_is_not_success(self):
        oracle,env,metric=fixture(EvaluationOracle,math.radians(60),(1.,0.,1.1))
        with patch.object(b.OriginalOracle,"measure",side_effect=lambda *_:deepcopy(metric)):
            for _ in range(12):result=oracle.measure(env)
        self.assertFalse(result["proxy_success"])

    def test_only_scalar_name_and_explicit_base_dispatch_change(self):
        expected=inspect.getsource(GenerationOracle.measure)
        expected=expected.replace("metric=super().measure(env)","metric=b.OriginalOracle.measure(self,env)")
        expected=expected.replace('position=float(point[self.spec["stroke_axis"]])','stroke_coordinate=float(point[self.spec["stroke_axis"]])')
        expected=expected.replace('min(self.stroke_low,position);self.stroke_high=max(self.stroke_high,position)',
                                  'min(self.stroke_low,stroke_coordinate);self.stroke_high=max(self.stroke_high,stroke_coordinate)')
        expected=expected.replace("motor.body_bounds(env,self.actor)","b.motor.body_bounds(env,self.actor)")
        normalized=lambda source:ast.dump(ast.parse(textwrap.dedent(source)),include_attributes=False)
        self.assertEqual(normalized(expected),normalized(inspect.getsource(EvaluationOracle.measure)))


if __name__=="__main__":unittest.main()
