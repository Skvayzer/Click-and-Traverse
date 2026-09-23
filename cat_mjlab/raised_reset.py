"""CPU generation and certification of a small, immutable raised reset pool.

The existing fallback reset bank is never edited. New float32 poses receive
fresh FK/body-proxy, root-route, stored-field and hand-objective checks. This
is a static reset certificate, not a claim of dynamic walking feasibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from . import constants
from .collision import CollisionChecker
from .navigation import route_context, hand_contrast_context, contrast_reward_terms


def arm_joint_names(model):
    return [model.joint(i).name for i in range(1, model.njnt)
            if any(part in model.joint(i).name for part in ('shoulder', 'elbow', 'wrist'))]


def commandable_limits(model, upper_action_scale=.8):
    """Soft physical bounds intersected with the absolute upper-body controller.

    Leg actions are incremental, so their reachable set is the physical range,
    not nominal +/- the per-step leg increment.
    """
    upper_action_scale = np.asarray(upper_action_scale)
    if upper_action_scale.shape not in ((), (17,)) or not np.isfinite(upper_action_scale).all() or (upper_action_scale <= 0).any():
        raise ValueError('upper_action_scale must be finite and positive')
    limits = np.asarray(model.jnt_range[1:])
    center = limits.mean(-1); half = .475 * np.diff(limits, axis=-1)[:, 0]
    lo, hi = center-half, center+half
    nominal = np.asarray(constants.DEFAULT_QPOS[7:])
    lo[12:] = np.maximum(lo[12:], nominal[12:]-upper_action_scale)
    hi[12:] = np.minimum(hi[12:], nominal[12:]+upper_action_scale)
    return lo, hi


def raised_pose(model, zone, *, upper_action_scale=.8):
    """Solve active bank boxes using production FK, preserving planted feet.

    Root height and sagittal displacement allow a crouch when lowered boxes
    cannot be reached from the nominal stance. Legs and arms obey the same
    95% soft limits as certification, intersected with commandable upper-body
    targets; waist and root orientation stay nominal.
    XY translation is replaced by the route placement after solving.
    """
    from scipy.optimize import least_squares
    q = np.array(constants.DEFAULT_QPOS, dtype=np.float64, copy=True)
    q[:2] = 0
    q[3:7] = [1, 0, 0, 0]
    joints = [model.joint(i) for i in range(1, model.njnt)
              if model.joint(i).name in arm_joint_names(model)
              or any(part in model.joint(i).name for part in ('hip', 'knee', 'ankle'))]
    addresses = np.array([int(j.qposadr[0]) for j in joints])
    lo, hi = commandable_limits(model, upper_action_scale)
    joint_lo, joint_hi = lo[addresses-7], hi[addresses-7]
    addresses = np.r_[0, 2, addresses]
    lower_bounds = np.r_[-.3, .55, joint_lo]
    upper_bounds = np.r_[.3, q[2], joint_hi]
    initial = np.clip(q[addresses], lower_bounds, upper_bounds)
    data = mujoco.MjData(model)
    sites = [model.site(name).id for name in constants.HAND_SITES]
    feet = [model.site(name).id for name in constants.FEET_SITES]
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    feet_position = data.site_xpos[feet].copy()
    feet_rotation = data.site_xmat[feet].copy()
    closest = None
    for index, valid in enumerate(zone['region_valid']):
        if not valid:
            continue
        lower = np.asarray(zone['hand_regions_min'][index])
        upper = np.asarray(zone['hand_regions_max'][index])
        target = (lower + upper) / 2
        def residual(values):
            q[addresses] = values
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            return np.r_[(data.site_xpos[sites] - np.r_[q[:2], 0.] - target).ravel(),
                         (data.site_xpos[feet] - feet_position).ravel(),
                         (data.site_xmat[feet] - feet_rotation).ravel()]
        result = least_squares(residual, initial, bounds=(lower_bounds, upper_bounds),
                               max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        residual(result.x)
        local = data.site_xpos[sites] - np.r_[q[:2], 0.]
        distance = np.linalg.norm(np.maximum(lower-local, 0)+np.maximum(local-upper, 0), axis=-1)
        if closest is None or distance.max() < closest[0]:
            closest = (float(distance.max()), local.copy(), distance.copy())
        if (np.all(data.site_xpos[sites] - np.r_[q[:2], 0.] > lower)
                and np.all(data.site_xpos[sites] - np.r_[q[:2], 0.] < upper)
                and np.max(np.abs(data.site_xpos[feet] - feet_position)) < 1e-5
                and np.max(np.abs(data.site_xmat[feet] - feet_rotation)) < 1e-5):
            return q.astype(np.float32)
    if closest is None:
        raise ValueError('Raised reset compliance certificate requires a valid hand region')
    _, local, distance = closest
    raise ValueError(f'Raised reset fails zero-cost hand compliance certificate: commandable IK found no interior pose; best local hands={local.tolist()}, distances_m={distance.tolist()} (local numerical search, not a global infeasibility proof)')


def certify_scene(model, scene, record, directory, *, region_scale=.15, tolerance=.05, upper_action_scale=.8):
    """Four early first-zone poses; fail closed if any requested pose is invalid."""
    from cat_ppo.furniture.room_navigation import pack_room_scenes
    from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
    from cat_ppo.furniture.grippers import hand_sphere
    from .fields import sample_ragged_field
    tensor = lambda x: torch.as_tensor(np.asarray(x).copy())
    zones = scene['hand_contrast']['zones']
    zone = zones[0]
    route = np.asarray(scene['route'], dtype=float)
    delta = route[1]-route[0]
    length = np.linalg.norm(delta)
    # Stay on segment zero so the unchanged ordered-navigation reset is valid.
    start = zone['start_m'] + zone['fade_m'] + .02
    end = min(start+.12, zone['end_m']-zone['fade_m']-.02, length-.02)
    if end <= start or not all(zone['hand_active']):
        raise ValueError('No early first-zone plateau for raised resets')
    qpos = np.repeat(raised_pose(model, zone, upper_action_scale=upper_action_scale)[None], 4, axis=0)
    qpos[:, :2] = route[0]+np.linspace(start, end, 4)[:, None]*delta/length
    yaw = np.arctan2(delta[1], delta[0])
    qpos[:, 3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    lo, hi = commandable_limits(model, upper_action_scale)
    if np.any(qpos[:, 7:] < lo-1e-6) or np.any(qpos[:, 7:] > hi+1e-6):
        raise ValueError('Raised reset violates commandable joint limits')
    d = mujoco.MjData(model)
    rows = {key: [] for key in ('xpos', 'xmat', 'site_xpos')}
    for q in qpos:
        d.qpos[:] = q; d.qvel[:] = 0
        mujoco.mj_forward(model, d)
        for key in rows: rows[key].append(getattr(d, key).copy())
    data = SimpleNamespace(**{key: tensor(np.array(value, np.float32)) for key, value in rows.items()})
    rooms = {k: tensor(v).repeat_interleave(4, 0) for k, v in pack_room_scenes([scene]).items()}
    xy = tensor(qpos[:, :2])
    nav = route_context(xy, xy, torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.bool),
                        rooms['route'], rooms['route_count'], rooms['obstacles'], rooms['obstacle_count'],
                        radius=rooms['navigation_radius'])
    meta = {k: tensor(v).repeat_interleave(4, 0) for k, v in pack_hand_contrast([scene]).items()}
    context = hand_contrast_context(meta, nav['progress_m'], nav['tangent'])
    hands = data.site_xpos[:, [model.site(n).id for n in constants.HAND_SITES]]
    forward = torch.cat((nav['tangent'], torch.zeros(4, 1)), -1)
    reward, metrics = contrast_reward_terms(context, hands, xy, forward, forward,
                                            region_scale=region_scale, hand_good_distance=tolerance)
    if (nav['violation'].any() or not context['core_active'].all()
            or not context['hand_active'].all() or not metrics['hand_contrast_hand_good'].all()
            or reward['wholebody_hand_contrast_region'].max() > 0):
        raise ValueError('Raised reset fails root route or zero-cost hand compliance certificate')
    # Check every native field-query site and both elbows against the real stored SDF.
    names = ['head', 'imu_in_pelvis', 'imu_in_torso', *constants.FEET_SITES, *constants.HAND_SITES,
             *constants.KNEE_SITES, *constants.SHOULDER_SITES, 'left_elbow_probe', 'right_elbow_probe']
    positions = data.site_xpos[:, [model.site(n).id for n in names]]
    sdf = np.load(Path(directory)/'sdf.npy', mmap_mode='r', allow_pickle=False)
    distances = sample_ragged_field(tensor(sdf.reshape(-1, 1)), positions,
        origin=torch.tensor(record['origin']).repeat(4, 1), dx=torch.full((4,), record['dx']),
        shape=torch.tensor(record['shape']).repeat(4, 1), offset=torch.zeros(4, dtype=torch.long))[:, :, 0]
    distances[:, 5:7] -= torch.tensor([hand_sphere(s)['radius'] for s in ('left', 'right')])
    distances[:, -2:] -= .05
    if not torch.isfinite(distances).all() or distances.min() <= 0:
        raise ValueError('Raised reset violates native field clearance')
    return qpos, data, float(distances.min())


def build_raised_reset_pool(model, manifest, bank_path, collision_path, *, proposal_path=None,
                            region_scale=.15, tolerance=.05, upper_action_scale=.8):
    from cat_ppo.furniture.generalist_fields import scene_directory
    kwargs = {} if proposal_path is None else {'proposal_path': proposal_path}
    checker = CollisionChecker(model, collision_path, field_manifest=bank_path, device='cpu', **kwargs)
    pool = np.zeros((len(manifest['scenes']), 4, model.nq), dtype=np.float32)
    eligible = np.zeros(len(pool), dtype=bool)
    minimum_field = float('inf')
    scene_certificates = []
    for index, record in enumerate(manifest['scenes']):
        if record.get('source', {}).get('hand_contrast', {}).get('role') != 'forward_protected':
            continue
        directory = scene_directory(manifest, Path(bank_path), record)
        path = directory/'scene.json'
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['scene_sha256']:
            raise ValueError('Raised reset scene hash mismatch')
        scene = json.loads(path.read_text())
        if scene['hand_contrast']['role'] != 'forward_protected':
            raise ValueError('Raised reset role mismatch')
        qpos, data, margin = certify_scene(model, scene, record, directory,
                                          region_scale=region_scale, tolerance=tolerance, upper_action_scale=upper_action_scale)
        if checker(torch.full((len(qpos),), index, dtype=torch.long), data).any():
            raise ValueError(f'Raised reset body collision: {record["scene_id"]}')
        hands = data.site_xpos[:, [model.site(n).id for n in constants.HAND_SITES]].numpy()
        tangent = np.diff(np.asarray(scene['route'])[:2], axis=0)[0]
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        delta = hands[:, :, :2] - qpos[:, None, :2]
        local = np.stack((delta @ tangent, delta @ normal, hands[:, :, 2]), axis=-1)
        zone = scene['hand_contrast']['zones'][0]
        margins = [np.minimum(local - np.asarray(lo), np.asarray(hi) - local).min()
                   for lo, hi, valid in zip(zone['hand_regions_min'], zone['hand_regions_max'], zone['region_valid']) if valid]
        scene_certificates.append(dict(scene_id=record['scene_id'], compliance_fraction=1.,
            hand_region_cost_max=0., minimum_box_interior_margin_m=float(max(margins)),
            minimum_native_field_clearance_m=margin, body_collision_count=0, root_route_valid=True,
            hand_positions_route_frame_m=local.tolist(),
            maximum_upper_action=float(np.max(np.abs((qpos[:,19:]-np.asarray(constants.DEFAULT_QPOS[19:]))/upper_action_scale))),
            root_height_m=float(qpos[0, 2]),
            joint_targets_rad={model.joint(i).name: float(qpos[0, model.joint(i).qposadr[0]])
                               for i in range(1, model.njnt)},
            arm_joint_targets_rad={name: float(qpos[0, model.joint(name).qposadr[0]])
                                   for name in arm_joint_names(model)}))
        pool[index] = qpos; eligible[index] = True
        minimum_field = min(minimum_field, margin)
    if not eligible.any():
        raise ValueError('Raised resets require certified forward-protected scenes')
    certificate = dict(schema='cat-raised-reset-v1', scenes=int(eligible.sum()), poses=int(eligible.sum()*4),
        qpos_sha256=hashlib.sha256(pool.tobytes()).hexdigest(),
        minimum_native_field_clearance_m=minimum_field, body_collision_count=0,
        hand_region_cost_max=0., compliance_fraction=1., solver="commandable-production-FK-IK", upper_action_scale=np.asarray(upper_action_scale).tolist(),
        commandable_upper_targets=True, per_scene=scene_certificates,
        dynamic_feasibility_validated=False)
    return pool, eligible, certificate
