"""Immutable shared fields and small scene metadata for the Torch task.

Only the existing CPU manifest/geometry validators are reused. GPU task code
does not create JAX arrays or exchange tensors with JAX.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .fields import sample_ragged_field


class SceneBank:
    def __init__(self,manifest,*,device='cuda',reset_manifest=None,collision_manifest=None,
                 proposal_path=None,verify_files=True,hand_protection=True):
        from cat_ppo.furniture.generalist_fields import load_generalist_manifest,sampling_groups
        from cat_ppo.furniture.room_navigation import pack_room_scenes
        from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
        from cat_ppo.furniture.contrastive_bank import contrastive_roles
        from cat_ppo.furniture.hand_curriculum import curriculum_levels,navigation_scene_groups
        self.path=Path(manifest).resolve();self.device=torch.device(device)
        self.manifest=load_generalist_manifest(self.path,verify_files=verify_files)
        records=self.manifest['scenes'];self.count=len(records)
        self.expanded=self.manifest['schema']=='cat-generalist-field-bank-v2'
        tensor=lambda x,dtype=None:torch.as_tensor(np.asarray(x),device=self.device,dtype=dtype)
        sizes=[int(np.prod(s['shape'])) for s in records]
        offsets=np.cumsum([0]+sizes[:-1]).astype(np.int64)
        self.fields={}
        for name,channels in (('sdf',1),('bf',3),('gf',3)):
            values=np.empty((sum(sizes),channels),dtype=np.float32)
            for scene,offset,size in zip(records,offsets,sizes):
                source=np.load(self.path.parent/scene['path']/f'{name}.npy',allow_pickle=False,mmap_mode='r')
                values[offset:offset+size]=source.reshape(-1,channels)
            self.fields[name]=tensor(values)
            del values
        self.offsets=tensor(offsets,torch.long)
        self.shapes=tensor([s['shape'] for s in records],torch.long)
        self.origins=tensor([s['origin'] for s in records],torch.float32)
        self.dxs=tensor([s['dx'] for s in records],torch.float32)
        self.starts=tensor([s['start'] for s in records],torch.float32)
        self.reset_xy_scale=tensor([s['reset_xy_scale'] for s in records],torch.float32)
        self.reset_yaws=tensor([s['reset_yaw'] for s in records],torch.float32)
        self.goals=tensor([s['goal'] for s in records],torch.float32)
        self.is_cat=tensor([s.get('task_kind','cat' if s['family']=='original_cat' else 'room')=='cat' for s in records])
        self.reset_is_cat=tensor([s.get('reset_mode','cat' if s['family']=='original_cat' else 'room')=='cat' for s in records])
        self.crossed_is_plane=tensor([s.get('crossed_mode','x_plane' if s['family']=='original_cat' else 'goal_radius')=='x_plane' for s in records])
        self.episode_lengths=tensor([s.get('episode_length',1000 if s['family']=='original_cat' else 4000) for s in records],torch.long)
        self.weights=tensor([s['sampling_weight'] for s in records],torch.float32)
        rooms=[]
        for record in records:
            room=record.get('task_kind','cat' if record['family']=='original_cat' else 'room')=='room'
            if not room:rooms.append(None);continue
            source=record.get('source',{})
            if (source.get('occupancy')!='conservative-voxel-cell-OBB-intersection-v1'
                    or source.get('room_navigation')!='ordered-certified-route-v1'):
                raise ValueError('Room fields predate the conservative geometry/path fix')
            scene=json.loads((self.path.parent/record['path']/'scene.json').read_text())
            from cat_ppo.furniture.room_geometry import root_cylinder_segment_clearance
            from cat_ppo.furniture.room_navigation import scene_navigation_radius
            route=np.asarray(scene['route'])
            if np.any(root_cylinder_segment_clearance(route[:-1],route[1:],scene['boxes'],radius=scene_navigation_radius(scene))<=0):
                raise ValueError(f"Uncertified root route: {record['scene_id']}")
            rooms.append(scene)
        self.rooms={k:tensor(v) for k,v in pack_room_scenes(rooms).items()}
        self.contrast={k:tensor(v) for k,v in pack_hand_contrast(rooms).items()}
        self.has_contrast=any(s is not None and s.get('hand_contrast') is not None for s in rooms)
        role_values=contrastive_roles(self.manifest)
        self.roles=None if role_values is None else tensor(role_values,torch.long)
        level_values=curriculum_levels(self.manifest,enabled=hand_protection)
        self.levels=None if level_values is None else tensor(level_values,torch.long)
        self.navigation_groups=tensor(navigation_scene_groups(self.manifest),torch.long)
        if self.expanded:
            groups,masses=sampling_groups(self.manifest)
            self.groups=tensor(groups,torch.long);self.group_masses=tensor(masses)
        else:self.groups=self.group_masses=None
        self.reset_pool=None
        self.reset_manifest_path=None if reset_manifest is None else Path(reset_manifest).resolve()
        if reset_manifest is not None:
            from .collision import PROPOSAL
            proposal=Path(proposal_path or PROPOSAL)
            reset_path=Path(reset_manifest).resolve()
            meta=json.loads(reset_path.read_text())
            digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            if collision_manifest is None or meta['collision_bank_sha256']!=digest(collision_manifest) or meta['proxy_sha256']!=digest(proposal):
                raise ValueError('Certified reset pool geometry/hash mismatch')
            pool_path=reset_path.parent/meta['file']
            if digest(pool_path)!=meta['sha256']:raise ValueError('Certified reset pool bytes changed')
            pool=np.load(pool_path,allow_pickle=False)
            if pool.ndim!=3 or pool.shape[0]!=self.count or not np.isfinite(pool).all() or pool.shape[1]<1:
                raise ValueError('Invalid collision-clear reset pool')
            self.reset_pool=tensor(pool,torch.float32)
            for scene in rooms:
                if scene is not None and scene.get('hand_contrast') is not None:
                    if scene['hand_contrast']['certificate']['body_proxy_sha256']!=meta['proxy_sha256']:
                        raise ValueError('Passage certificate and approved body proxies differ')
        if self.has_contrast and self.reset_pool is None:
            raise ValueError('Contrastive scenes require the certified full-body reset bank')

    def sample(self,name,positions,scene_ids):
        return sample_ragged_field(self.fields[name],positions,origin=self.origins[scene_ids],
            dx=self.dxs[scene_ids],shape=self.shapes[scene_ids],offset=self.offsets[scene_ids])

    def probabilities(self,weights=None,stage=None):
        weights=self.weights if weights is None else weights
        if self.roles is not None:
            total=torch.zeros(4,device=self.device).scatter_add_(0,self.roles,weights)
            return .25*weights/total[self.roles].clamp_min(1e-20)
        if self.levels is not None:
            hand=self.levels>=0;active=(~hand)|(self.levels<=stage)
            subgroup=self.groups*2+hand.long()
            selected=torch.where(active,weights,0.)
            totals=torch.zeros(8,device=self.device).scatter_add_(0,subgroup,selected)
            mass=self.group_masses[self.groups]*torch.where(self.groups>=2,.5,1.)
            return selected/totals[subgroup].clamp_min(1e-20)*mass
        if self.groups is not None:
            totals=torch.zeros(4,device=self.device).scatter_add_(0,self.groups,weights)
            return weights/totals[self.groups].clamp_min(1e-20)*self.group_masses[self.groups]
        return weights/weights.sum()
