#!/usr/bin/env python3
"""Prescribed-target/FK reward ledger, CPU only; no checkpoint or policy rollout.

Compare a nominal pose held at an active zone with a slew-limited raise and
100 subsequent hold steps. Costs use actual CAT kernels/scales, not scale-name
interpretation. The logs do not contain observed joint trajectories, so this
is a reproducible counterfactual, not a reconstruction of the pilot's motion.
"""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from cat_mjlab import constants, task_math
from cat_mjlab.model import assemble_training_xml
from cat_ppo.furniture.control import JOINT_NAMES, posture_actions
from cat_mjlab.navigation import hand_contrast_context, contrast_reward_terms
from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
from cat_ppo.furniture.generalist_fields import scene_directory


def audit(run_path, bank_path):
    config = json.loads(Path(run_path).read_text())['contract']['environment_config']
    manifest = json.loads(Path(bank_path).read_text())
    record = next(r for r in manifest['scenes'] if r.get('source', {}).get('hand_contrast', {}).get('role') == 'forward_protected')
    scene = json.loads((scene_directory(manifest, Path(bank_path).resolve(), record)/'scene.json').read_text())
    model = mujoco.MjModel.from_xml_string(assemble_training_xml()); data = mujoco.MjData(model)
    meta = {k: torch.from_numpy(v) for k, v in pack_hand_contrast([scene]).items()}
    context = hand_contrast_context(meta, torch.tensor([.65]), torch.tensor([[1., 0.]]))
    # This historical static-root ledger measures the original prescribed raise,
    # not the geometry-derived reset (which may also change root and legs).
    preset = np.array(constants.DEFAULT_QPOS, copy=True)
    actions = posture_actions(np.zeros(len(JOINT_NAMES)), JOINT_NAMES, 'raised', upper_scale=.8)
    for name, action in zip(JOINT_NAMES, actions):
        preset[model.joint(name).qposadr[0]] += .95*.8*action
    nominal = torch.tensor(constants.DEFAULT_QPOS[19:]); target = torch.tensor(preset[19:])
    def region(q):
        data.qpos[:] = constants.DEFAULT_QPOS; data.qpos[19:] = q.numpy(); mujoco.mj_forward(model, data)
        hands = torch.tensor(data.site_xpos[[model.site(n).id for n in constants.HAND_SITES]], dtype=torch.float32)[None]
        rewards, metrics = contrast_reward_terms(context, hands, torch.zeros(1, 2),
            torch.tensor([[1., 0., 0.]]), torch.tensor([[1., 0., 0.]]),
            region_scale=config['hand_contrast_region_scale'], hand_good_distance=config['hand_contrast_metric_tolerance'])
        return float(rewards['wholebody_hand_contrast_region'][0]), float(metrics['hand_contrast_hand_good'][0])
    baseline, _ = region(nominal); rows = []
    for rate in (2., 1.):
        previous = nominal.clone(); older = nominal.clone(); trajectory = []
        while not torch.equal(previous, target):
            current = previous+(target-previous).clamp(-rate*.02, rate*.02)
            trajectory.append(current); previous = current
        motion_steps = len(trajectory)
        trajectory += [target]*100
        previous = nominal; older = nominal
        sums = {k: 0. for k in ('velocity', 'acceleration', 'waist_and_posture', 'region_benefit', 'hold_benefit', 'discounted_cost', 'discounted_benefit')}
        for step, current in enumerate(trajectory):
            costs, _ = task_math.upper_stability_terms(current[None], previous[None], older[None], nominal,
                torch.ones(1, 2), torch.ones(1, 2), dt=.02, hand_protection=True,
                velocity_scale=config['upper_velocity_cost_scale'], acceleration_scale=config['upper_acceleration_cost_scale'],
                waist_weight=config['upper_cost_waist_weight'], arm_velocity_weight=config['upper_arm_velocity_weight'],
                arm_acceleration_weight=config['upper_arm_acceleration_weight'], contrast_arm_active=torch.ones(1, 2, dtype=torch.bool))
            total = 0.
            for name, key in [('velocity', 'wholebody_upper_target_velocity'), ('acceleration', 'wholebody_upper_target_acceleration'),
                              ('waist_and_posture', 'wholebody_upper_clear_posture')]:
                value = -float(costs[key][0])*config['reward_config']['scales'][key]*.02
                sums[name] += value; total += value
            value, _ = region(current)
            benefit = (baseline-value)*-config['reward_config']['scales']['wholebody_hand_contrast_region']*.02
            sums['region_benefit'] += benefit
            if step >= motion_steps: sums['hold_benefit'] += benefit
            sums['discounted_cost'] += .98**step*total; sums['discounted_benefit'] += .98**step*benefit
            older, previous = previous, current
        rows.append(dict(rate_rad_s=rate, motion_steps=motion_steps, hold_steps=100, **sums,
                         total_motion_cost=sums['velocity']+sums['acceleration']+sums['waist_and_posture']))
    return dict(method='Prescribed filtered targets + CPU FK; static root, full zone weight; pre-clipping reward units',
                observed_joint_trajectory_available=False, nominal_region_cost=baseline, raised_region_cost=region(target)[0],
                raised_compliance=region(target)[1], comparisons=rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-json', type=Path, required=True)
    parser.add_argument('--bank-manifest', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(audit(args.run_json, args.bank_manifest), indent=2))
