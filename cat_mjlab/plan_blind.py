"""Timed hand-height occlusions for a frozen-policy traversal benchmark."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from .analytic_objects import merge_fields
from .collision import PROPOSAL, REGIONS
from .model import assemble_training_xml
from .planner.data import quat_from_euler
from .sim import CATSimulation

BOX_HALF = (.027, .027, .085)
OBSTACLE = 'plan_blind_obstacle'
HIDDEN = (0., 0., -20.)
LEAD = .7


def obstacle_xml(hand_body_collision=True):
    """Expose the box to the same approved body proxies used for bank contacts."""
    root = ET.fromstring(assemble_training_xml(hand_body_collision=hand_body_collision))
    bodies = {b.get('name'): b for b in root.findall('.//body')}
    world = root.find('worldbody')
    body = ET.SubElement(world, 'body', name=OBSTACLE, mocap='true', pos='0 0 -20')
    numbers = lambda x: ' '.join(map(str, x))
    ET.SubElement(body, 'geom', name=OBSTACLE, type='box', size=numbers(BOX_HALF),
                  rgba='.87 .16 .33 1', contype='0', conaffinity='0')
    contact = root.find('contact')
    for index, shape in enumerate(json.loads(Path(PROPOSAL).read_text())['shapes']):
        geom = f"blind_proxy_{shape['group']}_{index}"
        common = dict(name=geom, type=shape['kind'], density='0', contype='0', conaffinity='0', group='5')
        if shape['kind'] == 'capsule':
            common.update(fromto=numbers(np.asarray(shape['endpoints']).ravel()), size=str(shape['radius']))
        elif shape['kind'] == 'sphere':
            common.update(pos=numbers(shape['center']), size=str(shape['radius']))
        else:
            common.update(pos=numbers(shape['center']), size=numbers(shape['half_size']), quat=numbers(shape['quat']))
        ET.SubElement(bodies[shape['body_name']], 'geom', **common)
        ET.SubElement(contact, 'pair', geom1=OBSTACLE, geom2=geom, condim='1')
    return ET.tostring(root, encoding='unicode')


def box_distance(points, center, half):
    gap = (points - center).abs() - half
    return torch.linalg.vector_norm(gap.clamp_min(0.), dim=-1) + gap.amax(-1).clamp_max(0.)


class BlindObstacle:
    """One immutable world-coordinate schedule plus an episode-assignment mask."""
    def __init__(self, guide, variant, device):
        tensor = lambda key: torch.as_tensor(guide[key], device=device)
        self.end = tensor('end')
        self.start = tensor('start') if variant == 'moving' else self.end
        self.reveal = tensor('reveal')
        self.enabled = tensor('valid').clone()
        self.valid = self.enabled.clone()
        self.half = self.end.new_tensor(BOX_HALF)
        self.hidden = self.end.new_tensor(HIDDEN)

    def locate(self, ids, time):
        elapsed = time - self.reveal[ids]
        fraction = (elapsed / LEAD).clamp(0., 1.)
        center = torch.lerp(self.start[ids], self.end[ids], fraction[..., None])
        return center, self.enabled[ids] & (elapsed >= 0.)

    def merge(self, points, ids, time, gf, bf, sdf, radii):
        center, valid = self.locate(ids, time)
        delta = points - center[:, None]
        edges = delta.abs() - self.half
        outside = edges.clamp_min(0.)
        norm = torch.linalg.vector_norm(outside, dim=-1, keepdim=True)
        sign = torch.where(delta >= 0., 1., -1.)
        inside = torch.nn.functional.one_hot(edges.argmax(-1), 3).to(points.dtype)
        normal = sign * torch.where(norm > 0., outside / norm.clamp_min(1e-9), inside)
        dist = norm + edges.amax(-1, keepdim=True).clamp_max(0.)
        dist = torch.where(valid[:, None, None], dist, torch.inf)
        return merge_fields(gf, bf, sdf, dist, normal, radii)


class BlindSimulation(CATSimulation):
    def __init__(self, num_envs, *, device, nconmax, njmax, model=None):
        # The optional hand/self contact extension is the same as in create_task.
        model = mujoco.MjModel.from_xml_string(obstacle_xml(hand_body_collision=model is not None))
        super().__init__(num_envs, device=device, nconmax=nconmax, njmax=njmax, model=model)
        self.obstacle = None
        self.ids = torch.arange(num_envs, device=device)
        self.geom = model.geom(OBSTACLE).id
        self.mocap = int(model.body_mocapid[model.body(OBSTACLE).id])
        regions = [REGIONS.index(name.split('_')[2]) if name.startswith('blind_proxy_') else -1
                   for name in (model.geom(i).name for i in range(model.ngeom))]
        self.regions = torch.tensor(regions, device=device)
        self.hits = torch.zeros(num_envs, len(REGIONS), device=device, dtype=torch.bool)
        self.revealed = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def step(self):
        if self.obstacle is not None:
            center, active = self.obstacle.locate(self.ids, self.data.time)
            self.data.mocap_pos[:, self.mocap] = torch.where(active[:, None], center, self.obstacle.hidden)
            self.revealed.logical_or_(active)
        super().step()
        if self.obstacle is not None:
            raw = self.backend.wp_data
            contact = raw.contact
            first, second = self.wp.to_torch(contact.geom).long().unbind(-1)
            other = torch.where(first == self.geom, second, first).clamp(0, len(self.regions) - 1)
            region = self.regions[other]
            worlds = self.wp.to_torch(contact.worldid).long().clamp(0, self.num_envs - 1)
            hit = ((first == self.geom) | (second == self.geom)) & (region >= 0)
            hit &= ((self._contact_index < self.wp.to_torch(raw.nacon)[0]) &
                    (self.wp.to_torch(contact.dist) < 0.) & active[worlds])
            flat = worlds * len(REGIONS) + region.clamp_min(0)
            counts = torch.zeros(self.num_envs * len(REGIONS), dtype=torch.int32, device=self.device)
            counts.scatter_reduce_(0, flat, hit.int(), reduce='amax', include_self=True)
            self.hits.logical_or_(counts.reshape_as(self.hits).bool())


def plan_poses(tracker, ids):
    quat = quat_from_euler(tracker.plan_roll[ids], tracker.plan_pitch[ids], tracker.plan_yaw[ids])
    return torch.cat((tracker.plan_xy[ids], tracker.plan_z[ids, :, None], quat,
                      tracker.plan_legs[ids], tracker.plan_upper[ids]), -1)


@torch.no_grad()
def candidate_placements(task, ids, variant):
    """Use the same initial chosen chunk for every tested checkpoint.

    Prefer a box close to the chosen hand envelope while it clears the room,
    the robot at appearance, and the rest of the surrogate body at intercept.
    """
    device = task.device
    dtype = task.data.qpos.dtype
    tensor = lambda data: torch.tensor(data, device=device, dtype=dtype)
    cover = task.online_planner.scorer.spheres
    radii = cover.radii
    hands = cover.hand_mask.nonzero().flatten()
    rows = []
    grid = tensor(list(product((-1., 0., 1.), repeat=3))) * tensor(BOX_HALF)
    angles = tensor([0., .7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978])
    frames, sides, offsets, bearings = zip(*product((14, 16, 18, 20), (0, 1), (.09, .11, .13, .15), range(len(angles))))
    frame = torch.tensor(frames, device=device)
    side = torch.tensor(sides, device=device)
    bearing = torch.tensor(bearings, device=device)
    away = tensor(offsets)
    lead_frames = round(LEAD / .1)
    fractions = tensor([0., .5, 1.]) if variant == 'moving' else tensor([1.])
    for fs in ids.split(16):
        poses = plan_poses(task.tracker, fs)
        points = cover(poses)
        yaw = task.tracker.plan_yaw[fs][:, frame] + angles[bearing][None]
        outward = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
        end = points[:, frame, hands[side]] + outward * away[None, :, None]
        start = end + .15 * outward if variant == 'moving' else end
        path = torch.lerp(start[:, :, None], end[:, :, None], fractions[None, None, :, None])
        probes = path[:, :, :, None] + grid[None, None, None]
        sdf = task.bank.sample('sdf', probes.reshape(len(fs), -1, 3), task.scene_ids[fs]).reshape(probes.shape[:-1])
        static_clear = (sdf.amin((-1, -2)) > .004) & (probes[..., 2].amin((-1, -2)) > .10)
        reveal_gap = box_distance(points[:, frame-lead_frames], start[:, :, None], tensor(BOX_HALF)) - radii
        target_gap = box_distance(points[:, frame], end[:, :, None], tensor(BOX_HALF)) - radii
        other = torch.ones((len(frame), len(radii)), device=device, dtype=torch.bool)
        other[torch.arange(len(frame), device=device), hands[side]] = False
        unrelated_gap = torch.where(other[None], target_gap, torch.inf).amin(-1)
        palm_gap = target_gap.gather(-1, hands[side][None, :, None].expand(len(fs), -1, -1)).squeeze(-1)
        allowed = static_clear & (reveal_gap.amin(-1) > .008) & (unrelated_gap > .005)
        # Reward meaningful hand exposure and preferred side, with a later reveal on ties.
        prefer_side = (side[None] == (fs[:, None] % 2)).float()
        score = -(palm_gap + .018).abs() + .005*prefer_side + .0002*frame
        chosen = torch.where(allowed, score, -torch.inf).argmax(-1)
        r = torch.arange(len(fs), device=device)
        rows.append(dict(start=start[r, chosen], end=end[r, chosen], reveal=(frame[chosen]-lead_frames)*.1,
                         valid=allowed.any(-1), hand=side[chosen], planned_hand_gap=palm_gap[r, chosen]))
    return {key: torch.cat([row[key] for row in rows]).cpu().numpy() for key in rows[0]}


class BlindTrace:
    """Record fixed paired worlds from the same evaluated episodes for the video."""
    def __init__(self, task, obstacle, selected, names, meta, checkpoint):
        self.task = task
        self.obstacle = obstacle
        self.ids = torch.as_tensor(selected, device=task.device)
        if not len(self.ids):
            raise ValueError('The blind guide has no recordable exposure')
        self.end = np.full(len(self.ids), -1, dtype=int)
        self.values = []
        self.contacts = []
        self.band = float((task.reflex or {}).get('hand_band', 0.))
        self.meta = dict(meta, scenes=names[selected].tolist(), rows=selected.tolist(),
                         checkpoint=str(checkpoint), dt=task.dt, band=self.band)

    def record(self):
        if len(self.values) >= 750 or (self.end >= 0).all():
            return
        task, ids = self.task, self.ids
        center, active = self.obstacle.locate(ids, task.data.time[ids])
        on = task.tracker.valid[ids]
        released = on[:, None] & (self.band > 0.) & (task.info['sdf'][ids, 5:7, 0] < self.band)
        self.values.append(dict(qpos=task.data.qpos[ids].cpu().numpy(), plan=plan_poses(task.tracker, ids).cpu().numpy(),
                                plan_t=task.tracker.plan_t[ids].cpu().numpy(), plan_active=on.cpu().numpy(),
                                released=released.cpu().numpy(), position=center.cpu().numpy(), active=active.cpu().numpy()))

    def finish(self, done, hits):
        if len(self.contacts) == len(self.values):
            return
        self.contacts.append(hits[self.ids, 5].cpu().numpy())
        new_end = done[self.ids].cpu().numpy() & (self.end < 0)
        self.end[new_end] = len(self.values) - 1

    def save(self, path):
        self.meta['end'] = np.where(self.end < 0, len(self.values) - 1, self.end).tolist()
        data = {key: np.stack([row[key] for row in self.values]) for key in self.values[0]}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(self.meta), hand_contact=np.stack(self.contacts), **data)
