"""Checks the proposed objective only; does not enable a training feature."""
import importlib.util
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('flat_probe',ROOT/'scripts/evaluate_flat_posture_rewards_cpu.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

def test_bonus_requires_forward_motion_and_caps_speed():
 cost=torch.zeros(5)
 speed=torch.tensor([-.6,0.,.3,.6,1.2])
 torch.testing.assert_close(probe.walking_posture_bonus(cost,speed),torch.tensor([0.,0.,.03,.06,.06]))

def test_dense_improvement_without_tolerance_threshold():
 cost=torch.tensor([.94,.8,.5,0.],requires_grad=True)
 reward=probe.walking_posture_bonus(cost,torch.full_like(cost,.6))
 assert torch.all(reward[1:]>reward[:-1])
 reward.sum().backward()
 assert torch.all(cost.grad<0)

def test_compiled_candidate_matches_eager():
 f=torch.compile(probe.walking_posture_bonus,backend='eager',fullgraph=True)
 c=torch.tensor([.94,0.,.5]);v=torch.tensor([.6,0.,-.1])
 torch.testing.assert_close(f(c,v),probe.walking_posture_bonus(c,v))
