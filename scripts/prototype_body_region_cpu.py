#!/usr/bin/env python3
"""Measurement-only pelvis-frame ablation. No bank/controller integration or training.

Run with .venv-mjlab/bin/python scripts/prototype_body_region_cpu.py.
Uses the existing frozen snapshot; writes only small artifacts in its own directory.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import copy
import hashlib
import json
import shutil
import time
import numpy as np
import torch
import mujoco
from scipy.optimize import least_squares
from prototype_residual_reference_cpu import ProbeTask, Reference
from diagnose_arm_hold_cpu import (ROOT, T, subset, _CPUSimulation, CollisionChecker,
                                  ActorCritic, LearnerConfig, gaussian_parameters)
from cat_ppo.furniture.grippers import hand_sphere

OUT = ROOT / 'outputs/body_region_cpu'


def rotation(q):
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, np.asarray(q[3:7], dtype=float))
    return mat.reshape(3, 3)


class Region:
    def __init__(self, task, j):
        self.q0 = task.data.qpos[j].numpy().astype(float).copy()
        self.R0 = rotation(self.q0)
        t = task.contrast['route_tangent'][j].numpy().astype(float)
        self.route = np.array([[t[0], -t[1], 0], [t[1], t[0], 0], [0, 0, 1]])
        lo = task.contrast['hand_regions_min'][j, 0].numpy().astype(float)
        hi = task.contrast['hand_regions_max'][j, 0].numpy().astype(float)
        self.center, self.half = (lo + hi) / 2, (hi - lo) / 2
        self.local_center = (self.center @ self.route.T - [0, 0, self.q0[2]]) @ self.R0
        self.local_axes = self.R0.T @ self.route

    def box(self, q, mode):
        if mode == 'absolute':
            return self.center @ self.route.T + [q[0], q[1], 0], self.route
        R = rotation(q)
        return self.local_center @ R.T + q[:3], R @ self.local_axes

    def distance(self, hands, q, mode):
        center, axes = self.box(q, mode)
        delta = (hands - center) @ axes
        excess = np.maximum(np.abs(delta) - self.half, 0)
        return np.linalg.norm(excess, axis=-1), excess


class BodyReference(Reference):
    def __init__(self, task, regions):
        super().__init__(task)
        self.regions = regions

    def solve(self, task, j):
        started = time.perf_counter()
        q = task.data.qpos[j].numpy().astype(float).copy()
        world, axes = self.regions[j].box(q, 'body')
        d = self.data
        def fun(x):
            d.qpos[:] = q
            d.qpos[22:] = x
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            return (d.site_xpos[self.sites] - world).ravel()
        def jac(x):
            fun(x)
            rows = []
            for site in self.sites:
                jp = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, d, jp, None, site)
                rows.append(jp[:, 21:])
            return np.concatenate(rows)
        x = np.clip(self.warm.get(j, q[22:]), self.lo + 1e-9, self.hi - 1e-9)
        result = least_squares(fun, x, jac=jac, bounds=(self.lo, self.hi), max_nfev=12,
                               ftol=1e-5, xtol=1e-5, gtol=1e-5)
        self.warm[j] = result.x
        fun(result.x)
        error = self.regions[j].distance(d.site_xpos[self.sites], q, 'body')[0].max()
        self.times.append(time.perf_counter() - started)
        return result.x, error


class RecordingSimulation(_CPUSimulation):
    recording = False
    def step(self):
        super().step()
        if self.recording:
            self.root_samples.append(self.data.qpos[:, :7].numpy().copy())


def decompose(model, data, sites, region, q, target):
    """Exact vector chain, with endpoint FK (unlike legacy 2-ms-old sites).

    h-target = tracking + pelvis + waist + reference-at-reset residual.
    Pelvis includes height and orientation, but excludes root XY translation.
    Contributions can cancel; their magnitudes are NOT additive error shares.
    """
    def fk(pose):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        return data.site_xpos[sites].copy()
    actual = fk(q)
    refq = q.copy(); refq[22:] = target
    reference = fk(refq)
    frozen = refq.copy(); frozen[2:7] = region.q0[2:7]
    no_pelvis = fk(frozen)
    frozen[19:22] = region.q0[19:22]
    no_waist = fk(frozen)
    center, _ = region.box(q, 'absolute')
    parts = np.stack((actual-reference, reference-no_pelvis,
                      no_pelvis-no_waist, no_waist-center))
    np.testing.assert_allclose(parts.sum(0), actual-center, atol=1e-12)
    actual_frozen = q.copy(); actual_frozen[2:7] = region.q0[2:7]
    no_body_actual = fk(actual_frozen)
    height = np.zeros((2, 3)); height[:, 2] = q[2]-region.q0[2]
    return dict(parts=parts, actual=actual, reference=reference,
                no_pelvis_actual=no_body_actual, height=height,
                orientation=parts[1]-height)


def safety(regions, scenes, samples, ends, dt):
    """Analytic support of rotated boxes + 50-mm ball + enclosing finger sphere.

    Check every 2-ms observed pelvis pose, including terminating control step.
    Lateral support bounds sweep any longitudinal placement through each cabinet
    pair; paired actual root cross-track and centered-root values are separate.
    """
    radius = max(hand_sphere(s)['radius'] for s in ('left', 'right'))
    rows = []
    for j, (region, scene) in enumerate(zip(regions, scenes)):
        roots = samples[:ends[j]*10, j]
        route_start = np.array(scene['route'][0])
        n = region.route[:2, 1]
        width = min(m['width_m'] for m in scene['hand_contrast']['modules'])
        values = []
        for k, root in enumerate(roots):
            center, axes = region.box(root, 'body')
            relative = center - np.r_[root[:2], 0]
            yc = relative[:, :2] @ n
            yh = region.half @ np.abs(axes.T @ np.r_[n, 0])
            cross = (root[:2] - route_start) @ n
            actual = width/2 - np.max(np.abs(yc+cross)+yh) - .05-radius
            centered = width/2 - np.max(np.abs(yc)+yh) - .05-radius
            fixed = width/2 - np.max(np.abs(region.center[:, 1]+cross)+region.half[:, 1])-.05-radius
            floor = np.min(center[:, 2]-region.half @ np.abs(axes[2]))-.05-radius
            values.append([actual, centered, fixed, floor, cross])
        a = np.asarray(values)
        worst = int(a[:, 0].argmin())
        rows.append(dict(samples=len(roots),body_lateral_actual_m=float(a[:,0].min()),
                         body_lateral_centered_m=float(a[:,1].min()),
                         absolute_lateral_actual_m=float(a[:,2].min()),
                         absolute_lateral_centered_m=float(width/2-np.max(np.abs(region.center[:,1])+region.half[:,1])-.05-radius),
                         floor_clearance_m=float(a[:,3].min()),cross_track_range_m=[float(a[:,4].min()),float(a[:,4].max())],
                         root_z_range_m=[float(roots[:,2].min()),float(roots[:,2].max())],
                         worst_time_s=(worst+1)*.002,worst_root=roots[worst].tolist(),
                         worst_box_center=region.box(roots[worst], 'body')[0].tolist(),
                         lateral_nonpositive_samples=int((a[:,0]<=0).sum())))
    return rows


def run_batch(indices, snap, manifest, checker, mode):
    bank, records, scenes = subset(manifest, indices)
    cfg = copy.deepcopy(snap['contract']['environment_config'])
    cfg['wholebody']['body_collision']['bank_manifest'] = 'data/furniture/cat_flat_hand_balance_v6_20260922_collision/manifest.json'
    cfg.update(hand_raised_reset_fraction=0., randomize_initial_episode_steps=False,
               upper_gravity_compensation=True,
               upper_action_scales={'left_shoulder_pitch_joint':1., 'right_shoulder_pitch_joint':1.})
    sim = RecordingSimulation(4)
    class Mapped:
        proposal_path = checker.proposal_path
        def __call__(self, ids, data):
            return checker(T(indices, torch.long)[ids], data)
    cached = json.loads((ROOT/'outputs/arm_hold_diagnosis_cpu/static_hold.json').read_text())
    poses = {r['scene_id']:r['qpos'] for r in cached['scenes']}
    pool = T([[poses[r['scene_id']]] for r in records], torch.float32)
    bank.reset_pool = pool
    task = ProbeTask(sim, bank, cfg, collision=Mapped(), seed=71)
    task.raised_reset_pool = pool; task.raised_reset_eligible = torch.ones(4, dtype=torch.bool)
    task.raised_reset_fraction = 1.
    policy = ActorCritic(LearnerConfig(**snap['config'])).eval()
    policy.load_state_dict(snap['model'])
    task.generator.manual_seed(1729); task.capture=False
    task.reset(scene_ids=torch.arange(4)); task.capture=True
    task.telemetry={}; task.ledger={}
    regions = [Region(task,j) for j in range(4)]
    reference = Reference(task) if mode=='absolute' else BodyReference(task, regions)
    active = np.ones(4, bool); ends=np.zeros(4, int)
    farthest=task.frame()['progress'].copy()
    stats=[]; traces=[]
    for j, idx in enumerate(indices):
        zone=scenes[j]['hand_contrast']['zones'][0]
        stats.append(dict(index=idx,scene=records[j]['scene_id'],mode=mode,start_progress=float(farthest[j]),
                          core_start=zone['start_m']+zone['fade_m'],core_end=zone['end_m']-zone['fade_m'],
                          continuous_steps=0,longest_steps=0,streak=0,first_bad=None,compliant_distance=0.))
    fkdata=mujoco.MjData(sim.model)
    sim.root_samples=[]; sim.recording=True
    for step in range(1,201):
        mu,_=gaussian_parameters(policy.logits(task.obs['state'],0)); action=mu.tanh()
        targets=np.zeros((4,14)); ik=np.zeros(4)
        for j in np.flatnonzero(active):
            targets[j],ik[j]=reference.solve(task,j)
            action[j,15:]=T(np.clip((targets[j]-task.nominal[15:].numpy())/task.upper_action_scales[3:].numpy(),-1,1),torch.float32)
        task.step(action); f=task.captured
        for j in np.flatnonzero(active):
            q=f['qpos'][j].astype(float); region=regions[j]
            # Preserve legacy endpoint metric sampling for comparable compliance.
            err=float(region.distance(f['hand_world'][j],q,mode)[0].max())
            if mode=='absolute':
                np.testing.assert_allclose(err,f['error'][j],atol=1e-6)
                err=float(f['error'][j])
            s=stats[j]; p=float(f['progress'][j]); good=err<=.05
            advance=max(0.,min(p,s['core_end'])-max(float(farthest[j]),s['core_start']))
            if good:s['compliant_distance']+=advance
            farthest[j]=max(farthest[j],p)
            s['streak']=s['streak']+1 if good else 0
            s['longest_steps']=max(s['longest_steps'],s['streak'])
            if not good and s['first_bad'] is None:s['first_bad']=step
            if s['first_bad'] is None:s['continuous_steps']=step
            s.update(steps=step,final_progress=p,fall=bool(f['fall'][j]),obstacle=bool(f['obstacle'][j]),
                     done=bool(f['done'][j]),final_error_m=err,crossed=p>=s['core_end'])
            dec=decompose(sim.model,fkdata,reference.sites,region,q,targets[j])
            traces.append(dict(index=indices[j],step=step,error=err,q=q,target=targets[j],ik=ik[j],
                               progress=p,legacy_hands=f['hand_world'][j],**dec))
            if s['done'] or s['crossed']:active[j]=False;ends[j]=step
        if step%25==0:
            print(mode,indices,'step',step,'active',active.tolist(),flush=True)
        if not active.any():break
    sim.recording=False;ends[ends==0]=step
    samples=np.asarray(sim.root_samples)
    saf=safety(regions,scenes,samples,ends,task.dt)
    for j,s in enumerate(stats):
        s.update(continuous_seconds=s['continuous_steps']*task.dt,
                 compliant_fraction_full_core=s['compliant_distance']/(s['core_end']-s['core_start']),
                 safety=saf[j],full_compliant_crossing=bool(s['crossed'] and s['first_bad'] is None))
    # Small compressed raw evidence: all valid endpoint states and 2-ms roots.
    np.savez_compressed(OUT/f'{mode}_{indices[0]}.npz',
                        **{k:np.asarray([r[k] for r in traces]) for k in traces[0]},
                        roots=samples,ends=ends,indices=indices,
                        initial_q=np.array([r.q0 for r in regions]))
    print('RESULT',mode,[(s['index'],s['continuous_steps'],round(s['compliant_fraction_full_core'],4),s['crossed']) for s in stats],flush=True)
    return stats


def main():
    assert shutil.disk_usage(ROOT).free>2*1024**3, 'Stop: under 2 GiB disk reserve'
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    snapshot=ROOT/'outputs/residual_reference_cpu/policy_snapshot.pt'
    snap=torch.load(snapshot,map_location='cpu',weights_only=True)
    manifest=ROOT/'data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json'
    full=json.loads(manifest.read_text())
    indices=[i for i,r in enumerate(full['scenes']) if r.get('source',{}).get('hand_contrast',{}).get('role')=='forward_protected']
    assert len(indices)==12
    model=_CPUSimulation(1).model
    checker=CollisionChecker(model,ROOT/'data/furniture/cat_flat_hand_balance_v6_20260922_collision/manifest.json',field_manifest=manifest,device='cpu')
    report=dict(checkpoint_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),episodes=[])
    started=time.monotonic()
    # First batch is exactly the historical four-scene ordering/RNG shape.
    for offset in range(3):
        batch=[indices[k+offset] for k in (0,3,6,9)]
        for mode in ('absolute','body'):
            assert shutil.disk_usage(ROOT).free>2*1024**3
            report['episodes'].extend(run_batch(batch,snap,manifest,checker,mode))
            report['elapsed_s']=time.monotonic()-started
            (OUT/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    assert not torch.cuda.is_initialized()


if __name__=='__main__':main()
