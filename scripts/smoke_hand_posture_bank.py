#!/usr/bin/env python3
"""Load the complete bank on CPU and step eight test worlds once; no learner."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import torch
from test_mjlab_task import _CPUSimulation
from cat_mjlab.scene_bank import SceneBank
from cat_mjlab.collision import CollisionChecker
from cat_mjlab.config import wholebody_config
from cat_mjlab.task import CATTask

def main():
    torch.set_num_threads(2);start=time.monotonic()
    root=ROOT/'data/furniture/cat_hand_posture_v1_20260919';manifest=root/'manifest.json'
    collision=root.with_name(root.name+'_collision')/'manifest.json';resets=root.with_name(root.name+'_resets')/'manifest.json'
    bank=SceneBank(manifest,device='cpu',collision_manifest=collision,reset_manifest=resets)
    sim=_CPUSimulation(8);checker=CollisionChecker(sim.model,collision,field_manifest=manifest,device='cpu')
    config=wholebody_config(hand_contrast=True,hand_curriculum_success_threshold=.35,collision_manifest=collision,reset_manifest=resets)
    task=CATTask(sim,bank,config,collision=checker,seed=13)
    indices=torch.tensor([0,64,2338,2342,2346,2350,2354,2358]);task.reset(scene_ids=indices)
    before=task.scene_ids.clone();roles=task.contrast['role'].clone()
    assert bank.hand_scene_kind[indices].tolist()==[0,0,1,1,1,2,2,2]
    result=task.step(torch.zeros(8,29))
    assert torch.isfinite(result['reward']).all()
    assert all(torch.isfinite(x).all() for x in result['obs'].values())
    assert task.checkpoint_selection_roles==(1,)
    assert roles.tolist()==[-1,-1,1,1,1,1,1,1]
    assert result['metrics']['hand_scene_kind'].tolist()==[0,0,1,1,1,2,2,2]
    state=task.state_dict();task.load_state_dict(state)
    report=dict(backend='native CPU MuJoCo test bridge plus production SceneBank/CATTask/CollisionChecker',training_launched=False,gpu_used=False,
        scenes=bank.count,forced_indices=before.tolist(),roles=roles.tolist(),state_roundtrip=True,
        observations={k:list(v.shape) for k,v in result['obs'].items()},finite_rewards=result['reward'].tolist(),
        checkpoint_selection_roles=list(task.checkpoint_selection_roles),elapsed_seconds=time.monotonic()-start)
    (root/'runtime-smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
