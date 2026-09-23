"""Opt-in upper PD gravity feedforward and joint-specific target authority."""
import math
import numpy as np
import torch
from cat_ppo.furniture.control import JOINT_NAMES


def action_scales(config):
    values = np.full(17, float(config.get('upper_action_scale', .8)))
    for name, value in config.get('upper_action_scales', {}).items():
        if name not in JOINT_NAMES[12:] or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid upper action scale: {name}={value}')
        values[list(JOINT_NAMES[12:]).index(name)] = value
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Upper action scales must be finite and positive')
    return values


def configure(config, args):
    config['upper_gravity_compensation'] = bool(getattr(args, 'upper_gravity_compensation', False))
    config['protected_hand_sdf_margin'] = bool(getattr(args, 'protected_hand_sdf_margin', False))
    config['lateral_corridor'] = bool(getattr(args, 'lateral_corridor', False))
    if config['lateral_corridor']:
        scales = config['reward_config']['scales']
        # Clearance corridors do not depend on the retired heading objective.
        if scales.get('tracking_root_field', 0) <= 0:
            raise ValueError('Corridor requires active forward tracking reward')
        scales.update(wholebody_lateral_corridor=-3., wholebody_corridor_arm=-3.)
    else:
        for key in ('wholebody_lateral_corridor', 'wholebody_corridor_arm'):
            config['reward_config']['scales'].pop(key, None)
    overrides = {}
    for entry in getattr(args, 'upper_action_scale', None) or []:
        name, value = entry.split('=', 1)
        overrides[name] = float(value)
    config['upper_action_scales'] = overrides
    action_scales(config)


class UpperGravity:
    """Negative gravitational generalized force from current body kinematics.

    Sum descendant COM moments about each hinge. Includes attached fixed bodies,
    excludes velocity/Coriolis/contact forces, compensates waist and both arms.
    """
    def __init__(self, model, device):
        t = lambda x: torch.as_tensor(np.asarray(x).copy(), dtype=torch.float32, device=device)
        joints = [model.joint(n).id for n in JOINT_NAMES[12:]]
        self.bodies = torch.as_tensor(model.jnt_bodyid[joints].copy(), device=device).long()
        weights = np.zeros((17, model.nbody))
        for j, body in enumerate(model.jnt_bodyid[joints]):
            for child in range(1, model.nbody):
                ancestor = child
                while ancestor and ancestor != body:
                    ancestor = model.body_parentid[ancestor]
                if ancestor == body:
                    weights[j, child] = model.body_mass[child]
        self.weights = t(weights)
        self.mass = self.weights.sum(-1)
        self.ipos, self.anchor, self.axis = map(t, (model.body_ipos, model.jnt_pos[joints], model.jnt_axis[joints]))
        self.gravity = t(model.opt.gravity)

    def __call__(self, data):
        rot = data.xmat.reshape(len(data.xpos), -1, 3, 3)
        com = data.xpos + torch.einsum('nbij,bj->nbi', rot, self.ipos)
        joint_rot = rot[:, self.bodies]
        anchor = data.xpos[:, self.bodies] + torch.einsum('nkij,kj->nki', joint_rot, self.anchor)
        axis = torch.einsum('nkij,kj->nki', joint_rot, self.axis)
        moment_arm = torch.einsum('kb,nbj->nkj', self.weights, com) - self.mass[None, :, None]*anchor
        return -(torch.linalg.cross(moment_arm, self.gravity.expand_as(moment_arm))*axis).sum(-1)
