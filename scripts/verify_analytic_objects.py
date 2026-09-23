#!/usr/bin/env python3
"""No training/rollout: production observation demonstration and query benchmark."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
import numpy as np
import torch
from cat_mjlab.analytic_objects import AnalyticObjects


def guard():
    free = shutil.disk_usage(ROOT).free
    print(f'Disk free: {free / 1024**3:.3f} GiB', flush=True)
    if free < 1024**3:
        raise RuntimeError('Stopping: less than 1 GiB free')
    return free


def objects(b, m, device='cpu'):
    start = torch.zeros((b,m,3), device=device)
    start[..., 0] = .7
    end = start.clone(); end[..., 0] = .2
    return AnalyticObjects(start=start, end=end,
        rotations=torch.eye(3, device=device).expand(b,m,3,3).clone(),
        sizes=torch.full((b,m,3),.06,device=device),
        kinds=(torch.arange(m,device=device)%3).expand(b,-1).clone(),
        valid=torch.ones((b,m),dtype=torch.bool,device=device),
        speed=torch.full((b,m),.1,device=device), hold=torch.ones((b,m),device=device))


def demonstrate():
    from test_mjlab_task import _CPUSimulation, _tiny_bank
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.observation_contract import CONTRACT
    cfg=wholebody_config();cfg['noise_config']['level']=0.
    cfg['randomize_initial_episode_steps']=False
    engine=objects(1,1)
    task=CATTask(_CPUSimulation(1),_tiny_bank(1),cfg,seed=17,analytic_objects=engine)
    positions=task._poses(task.all_ids).clone()
    hand=positions[0,5]
    radius=float(task.hand_radii[0])
    # Approach from the front: .35 -> .05 m surface clearance at .1 m/s.
    engine.state['start'][0,0]=hand+hand.new_tensor([radius+.06+.35,0,0])
    engine.state['end'][0,0]=hand+hand.new_tensor([radius+.06+.05,0,0])
    index=CONTRACT['actor_features'].index('pf.hands.df.0.distance')
    rows=[]
    qpos=task.data.qpos.clone()
    for t in (0.,1.,2.,3.,4.,5.,7.):
        task.info['step'][:]=round(t/task.dt)
        task.data.time[:]=t
        gf,bf,sdf,_=task._fields(positions,task.data.qpos[:,:2],task.all_ids)
        # Hold the robot and commands stationary; no simulator step is taken.
        task.info.update(gf=gf*0,bf=bf,sdf=sdf,gf_delay=gf*0,bf_delay=bf,sdf_delay=sdf)
        task.info['command'].zero_();task.info['command_delay'].zero_()
        task._observe(task.all_ids,torch.zeros((1,2),dtype=torch.bool))
        rows.append(dict(time_s=t,actor_hand_distance_m=float(task.obs['state'][0,index]),
            hand_bf=task.obs['state'][0,172:175].tolist(),
            true_hand_distance_m=float(sdf[0,5,0])))
    np.testing.assert_allclose([r['actor_hand_distance_m'] for r in rows],
                               [.35,.25,.15,.05,.05,.15,.35],atol=2e-6)
    assert torch.equal(task.data.qpos,qpos)
    # Actual serialized object specification restores with the task state.
    saved=deepcopy(task.state_dict())
    engine.state['end']+=10
    task.load_state_dict(saved)
    torch.testing.assert_close(task.analytic_objects.state['end'],saved['analytic_object_state']['end'])
    return dict(passed=True,feature=CONTRACT['actor_features'][index],actor_index=index,
                hand_radius_m=radius,robot_qpos_max_change=0.,physics_steps=0,
                scope='CATTask._fields -> CATTask._observe -> native 222 actor; no bank.sample replacement',samples=rows)


def benchmark(device):
    """Two 11-point and two 2-point hybrid passes, true plus actor per tick."""
    results=[]
    for m in (1,2,4,8):
        engine=objects(30720,m,device)
        ids=torch.arange(30720,device=device);clock=torch.ones(30720,device=device)
        tensors=[]
        for q in (11,2):
            p=torch.zeros((30720,q,3),device=device)
            tensors.append((p,torch.zeros_like(p),torch.zeros_like(p),torch.ones((30720,q,1),device=device)))
        def query():
            for p,gf,bf,sdf in tensors:
                for _ in range(2):
                    result=engine.merge(p,ids,clock,gf,bf,sdf,.10345 if p.shape[1]==11 else .05)
            return result
        for _ in range(3):query()
        sync=lambda:torch.cuda.synchronize() if device=='cuda' else None
        sync()
        samples=[]
        if device=='cuda':
            torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
        else:base=None
        for _ in range(10):
            start=time.perf_counter();query();sync();samples.append((time.perf_counter()-start)*1000)
        peak=(torch.cuda.max_memory_allocated()-base) if base is not None else None
        # Profile live tensor allocation high-water mark on CPU, not process RSS
        # or sums of allocations; paired frees decrement the live counter.
        cpu_peak=None
        if device=='cpu':
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU],profile_memory=True) as prof:
                query()
            events=sorted(prof.profiler.kineto_results.events(),key=lambda e:e.start_ns())
            live=0;cpu_peak=0
            for event in events:
                if event.name()=='[memory]':
                    live+=event.nbytes();cpu_peak=max(cpu_peak,live)
        state_bytes=sum(v.numel()*v.element_size() for v in engine.state.values())
        results.append(dict(objects=m,envs=30720,device=device,threads=torch.get_num_threads(),
            median_ms=float(np.median(samples)),min_ms=min(samples),max_ms=max(samples),
            persistent_object_bytes=state_bytes,incremental_peak_tensor_bytes=cpu_peak if device=='cpu' else peak,
            scope='4 eager hybrid queries (11+11+2+2 points), clock/poses, normals, min and guidance; excludes static lookup, forearm capsules, safety guard and physics'))
        print(json.dumps(results[-1]),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'docs/assets/reactive-analytic-20260922')
    args=parser.parse_args();before=guard()
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    demo=demonstrate()  # Stop immediately if production observation fails.
    print(json.dumps(demo,indent=2),flush=True)
    guard()
    measured=benchmark('cuda' if torch.cuda.is_available() else 'cpu')
    report=dict(perception=demo,benchmark=measured,cuda_available=torch.cuda.is_available(),
                gpu_max_affordable_objects=None,training_ready=False,
                disk_free_before_bytes=before,disk_free_after_bytes=guard())
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'perception-and-cost.json').write_text(json.dumps(report,indent=2)+'\n')
    guard()


if __name__=='__main__':main()
