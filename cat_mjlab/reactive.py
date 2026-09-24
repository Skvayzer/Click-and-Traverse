"""Opt-in standing threats. Immutable bank references; object-only safety control.

The guard changes only object motion, never robot state: object translation is
bounded by a 1-Lipschitz clearance before each ordinary physics substep. Robot
motion into an object follows the existing collision termination path.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .analytic_objects import AnalyticObjects, primitive_distance_normal, merge_fields
from .collision import PROPOSAL, compile_proposal, transform_primitives


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BodyEnvelope:
    """Conservative cover of every approved body proxy, with region ownership.

    Boxes use 27 subcell circumspheres, capsules use nine covering spheres (inflated
    by half the sample spacing); original hand enclosing spheres are unchanged.
    Conservative contact uses the same six fault regions as the static bank.
    """
    def __init__(self, model, device):
        proposal=json.loads(PROPOSAL.read_text())
        self.compiled=compile_proposal(proposal,model,device)
        centers=[];radii=[];owners=[];arms=[[],[]]
        self.arm_compiled=[]
        for side in ('left','right'):
            sub=dict(proposal,shapes=[s for s in proposal['shapes'] if s['group']=='arms' and s['id'].startswith(side)])
            self.arm_compiled.append(compile_proposal(sub,model,device))
        for i,s in enumerate(proposal['shapes']):
            if s['kind']=='capsule':
                ep=torch.tensor(s['endpoints'],dtype=torch.float32)
                # Every capsule point is within r + spacing/2 of a sample.
                for t in torch.linspace(0,1,9):
                    centers.append((ep[0]+t*(ep[1]-ep[0])).tolist())
                    radii.append(s['radius']+float(torch.linalg.vector_norm(ep[1]-ep[0]))/16)
                    owners.append(i)
            elif s['kind']=='box':
                import itertools
                half=torch.tensor(s['half_size'])/3
                rotation=self.compiled['local_rotations'][i].cpu()
                for cell in itertools.product((-2.,0.,2.),repeat=3):
                    center=torch.tensor(s['center'])+rotation@(half*torch.tensor(cell))
                    centers.append(center.tolist());owners.append(i)
                    radii.append(float(torch.linalg.vector_norm(half)))
            else:
                centers.append(s['center']);owners.append(i);radii.append(s['radius'])
        self.local=torch.tensor(centers,device=device)
        self.radii=torch.tensor(radii,device=device)
        self.owners=torch.tensor(owners,device=device)
        self.bodies=self.compiled['body_ids'][self.owners]
        self.regions=self.compiled['region_mask'][:,self.owners]
        self.hand_mask=self.regions[5]

    def spheres(self,data,ids):
        rot=data.xmat.reshape(len(data.xpos),-1,3,3)[ids][:,self.bodies]
        return data.xpos[ids][:,self.bodies]+torch.einsum('bsij,sj->bsi',rot,self.local)

    def arms(self,data,ids):
        from types import SimpleNamespace
        selected=SimpleNamespace(xpos=data.xpos[ids],xmat=data.xmat[ids])
        return [(transform_primitives(selected.xpos,selected.xmat,c)['endpoints'],c['radii']) for c in self.arm_compiled]


def capsule_query(endpoints, radii, centers, rotations, sizes, kinds, valid):
    """Continuous capsule/primitive clearance; no elbow-only sampling.

    Minimize analytic convex point SDF along each capsule axis. Ternary brackets
    bound the numerical error by axis_length * (2/3)^20 / 2 (<0.04 mm for
    a 25 cm capsule). Subtract that bound: returned clearance is conservative.
    This bounded scalar solver is not a voxel or a sampled-capsule approximation.
    """
    b,q=endpoints.shape[:2];m=centers.shape[1]
    a=endpoints[:,:,0];delta=endpoints[:,:,1]-a
    lo=a.new_zeros((b,q,m));hi=torch.ones_like(lo)
    def evaluate(t):
        # Pair each axis/object, flatten object dimension into batch.
        p=a[:,:,None]+t[...,None]*delta[:,:,None]
        p=p.permute(0,2,1,3).reshape(b*m,q,3)
        d,n=primitive_distance_normal(p,centers.reshape(b*m,1,3),rotations.reshape(b*m,1,3,3),sizes.reshape(b*m,1,3),kinds.reshape(b*m,1))
        return d[:,:,0].reshape(b,m,q).transpose(1,2),n[:,:,0].reshape(b,m,q,3).transpose(1,2)
    for _ in range(20):
        t1=(2*lo+hi)/3;t2=(lo+2*hi)/3
        d1,_=evaluate(t1);d2,_=evaluate(t2)
        choose=d1<=d2
        hi=torch.where(choose,t2,hi);lo=torch.where(choose,lo,t1)
    d,n=evaluate((lo+hi)/2)
    error=torch.linalg.vector_norm(delta,dim=-1)[...,None]*(hi-lo)/2
    d=d-error-radii[None,:,None]
    d=torch.where(valid[:,None],d,torch.inf)
    value,index=d.flatten(1).min(-1)
    normal=n.flatten(1,2).gather(1,index[:,None,None].expand(-1,1,3)).squeeze(1)
    return value[:,None],normal


class StandingObjects(AnalyticObjects):
    repeat_approaches = True

    def __init__(self, *, bank, num_envs, model, device, seed=731):
        self.manifest=json.loads(Path(bank).read_text());self.path=Path(bank).resolve()
        validate_bank(self.manifest)
        self.geometry=BodyEnvelope(model,device)
        self.device=torch.device(device)
        self.rows=self.manifest['scenes'];b=num_envs;m=2
        tensor=lambda x:torch.tensor(x,device=device,dtype=torch.float32)
        self.table={key:tensor([r[key] for r in self.rows]) for key in ('qpos','start','end','sizes','rotations','speed','hold','clearance')}
        self.table['kinds']=torch.tensor([r['kinds'] for r in self.rows],device=device,dtype=torch.long)
        self.table['valid']=torch.tensor([r['valid'] for r in self.rows],device=device,dtype=torch.bool)
        codes={'danger':0,'anticipation':1,'negative':2}
        self.table['bucket']=torch.tensor([codes.get(r.get('bucket'),0) for r in self.rows],device=device,dtype=torch.long)
        self.pelvis=model.body('pelvis').id
        kwargs={k:v[:1].expand(b,*v.shape[1:]).clone() for k,v in self.table.items() if k not in ('qpos','clearance','bucket')}
        super().__init__(**kwargs)
        self.state.update(position=self.state['start'].clone(),clock=torch.zeros(b,device=device),
            retreating=torch.zeros((b,m),dtype=torch.bool,device=device),
            active=torch.zeros(b,dtype=torch.bool,device=device),scene=torch.zeros(b,dtype=torch.long,device=device),
            clearance=torch.zeros((b,m),device=device),velocity=torch.zeros((b,m,3),device=device))
        # Per-approach bookkeeping for event success (see advance) and the walking variant.
        self.state.update(bucket=torch.zeros(b,dtype=torch.long,device=device),walking=torch.zeros(b,dtype=torch.bool,device=device),
            spot=torch.zeros((b,2),device=device),hand_arm=torch.zeros((b,2,3),device=device),target_hand=torch.zeros(b,dtype=torch.long,device=device),
            approach_ray=torch.zeros((b,3),device=device),approach_gap=torch.full((b,),torch.inf,device=device),
            approach_retreat=torch.zeros(b,device=device),approach_disp=torch.zeros(b,device=device),
            approach_contact=torch.zeros(b,dtype=torch.bool,device=device),event_finished=torch.zeros(b,dtype=torch.bool,device=device),
            event_success=torch.zeros(b,dtype=torch.bool,device=device),event_retreat=torch.zeros(b,device=device),
            event_bucket=torch.zeros(b,dtype=torch.long,device=device))
        # Runtime design knobs (the runner sets them from CLI flags). nominal_reset: start from the
        # ordinary reset pose and aim the first object at the live hand instead of applying the
        # row's certified pose. mixed_buckets: every re-arm re-draws a row's object parameters.
        self.nominal_reset=True;self.mixed_buckets=True;self.walking_fraction=0.;self.pause_range=(.5,2.);self.episode_length=800
        self.hand_velocity=None
        self.sampling_generator=torch.Generator(device=device).manual_seed(seed)
        self.state['sampling_rng']=self.sampling_generator.get_state().to(device)
        self.state['valid'].zero_()
        self.flat_scene=self.manifest['background_scene_index']
        self.episode_steps=self.manifest['episode_steps']
        weights=tensor([r.get('sampling_weight',1.) for r in self.rows])
        self.row_cdf=weights.cumsum(0)/weights.sum()
        self.force_rows=None
        self.last_robot_contacts=torch.zeros(b,dtype=torch.bool,device=device)
        self.last_object_contacts=self.last_robot_contacts.clone()

    def sample(self,ids,scene_ids):
        self.sampling_generator.set_state(self.state['sampling_rng'].cpu())
        random=torch.rand((len(ids),3),generator=self.sampling_generator,device=self.device)
        self.state['sampling_rng']=self.sampling_generator.get_state().to(self.device)
        return self.choose(ids,random,scene_ids)

    def choose(self,ids,random,scene_ids):
        # Share of resets that become a standing-object episode. .25 was hard-coded and sat
        # outside the experience solver; the runner now sets it from realized episode lengths.
        active=random[:,0]<float(getattr(self,'reactive_fraction',.25))
        rows=torch.searchsorted(self.row_cdf,random[:,1]).clamp_max(len(self.rows)-1)
        if self.force_rows is not None:
            active=torch.ones_like(active);rows=self.force_rows[ids]
        self.state['active'][ids]=active;self.state['scene'][ids]=rows
        self.state['bucket'][ids]=self.table['bucket'][rows]
        walking=active&(random[:,2]<float(self.walking_fraction)) if random.shape[1]>2 else torch.zeros_like(active)
        self.state['walking'][ids]=walking
        for k in ('start','end','sizes','rotations','kinds','valid','speed','hold','clearance'):
            self.state[k][ids]=self.table[k][rows]
        self.state['valid'][ids]&=active[:,None]
        self.state['position'][ids]=self.state['start'][ids]
        self.state['velocity'][ids]=0;self.state['clock'][ids]=0;self.state['retreating'][ids]=False
        return torch.where(active,self.flat_scene,scene_ids),active,self.table['qpos'][rows]

    def centers(self,ids,time=None):return self.state['position'][ids]

    def distances(self,points,ids):
        s=self.state
        d,n=primitive_distance_normal(points,s['position'][ids],s['rotations'][ids],s['sizes'][ids],s['kinds'][ids])
        return torch.where(s['valid'][ids,None],d,torch.inf),n

    def advance(self,data,dt):
        s=self.state;ids=s['active'].nonzero().flatten()
        self.last_object_contacts.zero_()
        if not len(ids):return ids,None,None
        p=self.geometry.spheres(data,ids)
        d,_=self.distances(p,ids);gaps=d-self.geometry.radii[None,:,None]
        # Original hand sphere radii are the approved 0.103446... m envelope.
        hand=gaps[:,self.geometry.hand_mask].amin(1)
        full=gaps.amin(1)
        length=torch.linalg.vector_norm(s['end'][ids]-s['start'][ids],dim=-1)
        duration=length/s['speed'][ids]
        retreat=s['clock'][ids,None]>=duration+s['hold'][ids]
        goal=torch.where(retreat[...,None],s['start'][ids],s['end'][ids])
        delta=goal-s['position'][ids];remaining=torch.linalg.vector_norm(delta,dim=-1)
        direction=delta/remaining.clamp_min(1e-12)[...,None]
        # Robot intrusion latches retreat along the authored path, never chasing.
        blocked=hand<s['clearance'][ids]-1e-6
        retreat=retreat|blocked|s['retreating'][ids]
        s['retreating'][ids]=retreat
        goal=torch.where(retreat[...,None],s['start'][ids],s['end'][ids])
        delta=goal-s['position'][ids];remaining=torch.linalg.vector_norm(delta,dim=-1)
        direction=delta/remaining.clamp_min(1e-12)[...,None]
        # Positive clearance / Lipschitz bounds certify the entire object segment.
        # When inside the hand envelope, only hand-distance-increasing motion is
        # permitted, otherwise pause. Full-body and floor margins always apply.
        step=torch.minimum(s['speed'][ids]*dt,remaining)
        budget=(full-.002).clamp_min(0)
        step=torch.minimum(step,budget)
        trial=s['position'][ids]+step[...,None]*direction
        td,_=primitive_distance_normal(p,trial,s['rotations'][ids],s['sizes'][ids],s['kinds'][ids])
        th=(td-self.geometry.radii[None,:,None])[:,self.geometry.hand_mask].amin(1)
        approach_budget=(hand-s['clearance'][ids]).clamp_min(0)
        step=torch.where(retreat,torch.where(th>=hand,step,0.),torch.minimum(step,approach_budget))
        # Rotations are fixed. Exact primitive support in world Z.
        r=s['rotations'][ids];size=s['sizes'][ids];k=s['kinds'][ids]
        support=torch.where(k==0,size[...,0],torch.where(k==1,(r[...,2,:].abs()*size).sum(-1),
            size[...,0]*torch.linalg.vector_norm(r[...,2,:2],dim=-1)+size[...,1]*r[...,2,2].abs()))
        floor_budget=torch.where(direction[...,2]<0,(s['position'][ids,:,2]-support-.002)/(-direction[...,2]).clamp_min(1e-12),torch.inf)
        step=torch.minimum(step,floor_budget.clamp_min(0))*s['valid'][ids]
        step=step*(s['clock'][ids]>=0.)[:,None]   # a re-armed object waits out its pause at its start
        motion=step[...,None]*direction
        s['position'][ids]+=motion;s['velocity'][ids]=motion/dt;s['clock'][ids]+=dt
        # Approach bookkeeping: closest the object got, how far the targeted hand retreated along
        # the approach axis RELATIVE TO THE ROOT, how far the root strayed, any robot contact.
        root=data.xpos[ids,self.pelvis];rows_=torch.arange(len(ids),device=self.device)
        s['approach_gap'][ids]=torch.minimum(s['approach_gap'][ids],hand.amin(-1))
        rel=p[:,self.geometry.hand_mask][rows_,s['target_hand'][ids]]-root-s['hand_arm'][ids][rows_,s['target_hand'][ids]]
        s['approach_retreat'][ids]=torch.maximum(s['approach_retreat'][ids],(rel*s['approach_ray'][ids]).sum(-1))
        s['approach_disp'][ids]=torch.maximum(s['approach_disp'][ids],torch.linalg.vector_norm(root[:,:2]-s['spot'][ids],dim=-1))
        s['approach_contact'][ids]|=self.last_robot_contacts[ids]
        if self.repeat_approaches:
            # Home = every VALID primitive back at its start. The unused second primitive never
            # moves, so an unmasked minimum declared the cycle over the instant retreat began and
            # objects teleported to their next start without ever retreating.
            away=torch.linalg.vector_norm(s['position'][ids]-s['start'][ids],dim=-1)
            home=torch.where(s['valid'][ids],away,torch.zeros_like(away)).amax(-1)
            finished=s['retreating'][ids].all(-1)&(home<=.01)&(s['clock'][ids]>0.)
            if bool(finished.any()):
                f=ids[finished];bucket=s['bucket'][f];retreat=s['approach_retreat'][f]
                # An approach is handled when the robot stayed on its spot (walking excepted), never
                # touched the object, and moved the hand >= 5 cm away for a threatening object --
                # or did NOT flinch for a 'negative' one that was going to pass by.
                stayed=(s['approach_disp'][f]<=.3)|s['walking'][f];reacted=retreat>=.05
                success=stayed&~s['approach_contact'][f]&torch.where(bucket==2,~reacted,reacted)
                s['event_finished'][f]=True;s['event_success'][f]=success;s['event_retreat'][f]=retreat;s['event_bucket'][f]=bucket
                self.rearm(data,f,self.sampling_generator)
        # Store start-of-robot-motion clearance after safe object movement.
        d,_=self.distances(p,ids)
        after_gap=d-self.geometry.radii[None,:,None]
        self.last_object_contacts[ids]=(full.amin(-1)>0)&(after_gap.amin((1,2))<=0)
        return ids,p,after_gap

    def rearm(self, data, ids, generator, redraw=True):
        """Re-aim a finished object at where the hand IS, and send it in again.

        An authored approach fires once and then the object parks at its start for the
        rest of the episode: measured, an object is in motion for 3.49 s of an 80 s
        episode, a 4.4% duty cycle. The other 95.6% is a robot standing on an empty plane
        with nothing to react to, which is most of what half the batch was buying.

        The new path is computed from the CURRENT hand position rather than the authored
        pose. Re-using the authored start/end would recreate exactly the defect this bank
        was rebuilt to fix -- objects flying at a hand height the policy had long since
        left, 0.306 m away, arriving at empty air. Safety is not weakened by computing the
        path at runtime: the per-substep guard in advance() bounds every motion by the
        live full-body gap, which is what has kept object-initiated contact at zero.
        """
        if not len(ids):
            return
        state = self.state
        if redraw and self.mixed_buckets and self.force_rows is None:
            # A forced row (recordings, approval checks) keeps its authored object parameters.
            self.redraw(ids, generator)
        centers = self.geometry.spheres(data, ids)
        hands = centers[:, self.geometry.hand_mask]
        pick = torch.randint(hands.shape[1], (len(ids),), device=self.device, generator=generator)
        hand = hands[torch.arange(len(ids), device=self.device), pick]
        actual = hand
        gap = state['clearance'][ids, 0]
        radius = float(self.geometry.radii[self.geometry.hand_mask].max())
        if self.hand_velocity is not None:
            # Walking variant: aim at where the hand WILL be when the object arrives.
            v = self.hand_velocity[ids][torch.arange(len(ids), device=self.device), pick]
            lead = (.335 + gap + radius) / state['speed'][ids, 0].clamp_min(1e-3)
            hand = hand + v * (lead * state['walking'][ids].float())[:, None]
        # Approach from a side where the HAND is the first thing the object meets: with the
        # hands hanging at the thighs, a ray from the body side parks the object against the
        # leg and any leg motion ends the episode as a body collision. Try several rays and
        # keep the one whose end point is clearest of the non-hand body spheres.
        others = centers[:, ~self.geometry.hand_mask]; other_r = self.geometry.radii[~self.geometry.hand_mask]
        size = state['sizes'][ids, 0].max(-1).values
        best_clear = torch.full((len(ids),), -torch.inf, device=self.device); ray = None; end = None
        for _ in range(8):
            azimuth = torch.rand(len(ids), device=self.device, generator=generator) * 2 * torch.pi
            elevation = torch.deg2rad(torch.rand(len(ids), device=self.device, generator=generator) * 100. - 35.)
            candidate = torch.stack((elevation.cos() * azimuth.cos(), elevation.cos() * azimuth.sin(), elevation.sin()), -1)
            candidate_end = hand + candidate * (gap + radius)[:, None]
            clear = (torch.linalg.vector_norm(candidate_end[:, None] - others, dim=-1) - other_r[None]).amin(-1) - size
            better = clear > best_clear
            best_clear = torch.where(better, clear, best_clear)
            ray = candidate if ray is None else torch.where(better[:, None], candidate, ray)
            end = candidate_end if end is None else torch.where(better[:, None], candidate_end, end)
        start = end + ray * (.22 + torch.rand(len(ids), device=self.device, generator=generator) * .23)[:, None]
        support = state['sizes'][ids, 0].max(-1).values
        floor = support + .003
        start = torch.cat((start[:, :2], start[:, 2:].clamp_min(floor[:, None])), -1)
        end = torch.cat((end[:, :2], end[:, 2:].clamp_min(floor[:, None])), -1)
        state['start'][ids, 0] = start
        state['end'][ids, 0] = end
        state['position'][ids, 0] = start
        state['velocity'][ids] = 0.
        state['retreating'][ids] = False
        root = data.xpos[ids, self.pelvis]
        state['spot'][ids] = root[:, :2]
        state['hand_arm'][ids] = hands - root[:, None]
        state['target_hand'][ids] = pick
        state['approach_ray'][ids] = -ray
        state['approach_gap'][ids] = torch.inf; state['approach_retreat'][ids] = 0.; state['approach_disp'][ids] = 0.
        state['approach_contact'][ids] = False
        low, high = self.pause_range
        state['clock'][ids] = -(low + torch.rand(len(ids), device=self.device, generator=generator) * (high - low))

    def redraw(self, ids, generator):
        """Re-sample a bank row's object parameters (shape, size, speed, hold, clearance, bucket)."""
        u = torch.rand(len(ids), device=self.device, generator=generator)
        rows = torch.searchsorted(self.row_cdf, u).clamp_max(len(self.rows) - 1)
        s = self.state; s['scene'][ids] = rows; s['bucket'][ids] = self.table['bucket'][rows]
        for k in ('sizes', 'rotations', 'kinds', 'valid', 'speed', 'hold', 'clearance'):
            s[k][ids] = self.table[k][rows]
        s['valid'][ids] &= s['active'][ids, None]

    def begin_step(self, hand_velocity=None):
        """Clear per-step event flags and take the current hand velocities (for the walking lead)."""
        self.state['event_finished'].zero_(); self.state['event_success'].zero_()
        self.hand_velocity = hand_velocity

    def contacts(self,data,ids,before,before_gap):
        regions=torch.zeros((len(self.state['active']),6),dtype=torch.bool,device=self.device)
        robot=regions[:,0].clone()
        if not len(ids):return regions,robot
        after=self.geometry.spheres(data,ids)
        d,_=self.distances(after,ids)
        gap=d-self.geometry.radii[None,:,None]
        # Endpoint envelope contact is an ordinary fault, not a rejected step.
        # Do not label an inconclusive swept bound as an actual contact.
        hit=(gap<=0).any(-1)
        regions[ids]=(hit[:,None]&self.geometry.regions[None]).any(-1)
        robot[ids]=regions[ids].any(-1)&(before_gap.amin((1,2))>0)
        self.last_robot_contacts=robot
        return regions,robot

    def arm_fields(self,data,ids,gf,bf,sdf):
        s=self.state
        active=s['active'][ids];local=active.nonzero().flatten()
        if not len(local):return gf,bf,sdf
        selected=ids[local]
        ds=[];ns=[]
        for endpoints,radii in self.geometry.arms(data,selected):
            d,n=capsule_query(endpoints,radii,s['position'][selected],s['rotations'][selected],s['sizes'][selected],s['kinds'][selected],s['valid'][selected])
            ds.append(d);ns.append(n)
        distance=torch.stack(ds,1);normal=torch.stack(ns,1)
        result=merge_fields(gf[local],bf[local],sdf[local],distance,normal,0.)
        out=[x.clone() for x in (gf,bf,sdf)]
        for dst,src in zip(out,result):dst[local]=src
        return tuple(out)


def validate_bank(manifest):
    if manifest.get('schema')!='cat-reactive-standing-v1' or manifest.get('reactive_mass')!=.25:
        raise ValueError('Invalid reactive bank')
    for key in ('base_bank','collision_bank','reset_bank','proxy'):
        pin=manifest[key]
        if digest(pin['path'])!=pin['sha256']:raise ValueError(f'Reactive source pin changed: {key}')
    if not manifest['scenes']:raise ValueError('Empty reactive bank')
    return manifest


def load_standing_task(manifest, sim, config, *, seed=0):
    """Explicit opt-in composite bank loader; never used by legacy scene banks.

    The original sampler supplies 75% of resets with its original distribution.
    Reactive resets use a pinned flat background with empty standing fields.
    No fields/banks/checkpoints are copied or written by this loader.
    """
    from .scene_bank import SceneBank
    from .collision import CollisionChecker
    from .task import CATTask
    meta=validate_bank(json.loads(Path(manifest).read_text()))
    bank=SceneBank(meta['base_bank']['path'], device=sim.data.qpos.device,
        reset_manifest=meta['reset_bank']['path'],collision_manifest=meta['collision_bank']['path'])
    objects=StandingObjects(bank=manifest,num_envs=len(sim.data.qpos),model=sim.model,device=sim.data.qpos.device)
    collision=CollisionChecker(sim.model,meta['collision_bank']['path'],field_manifest=meta['base_bank']['path'],device=sim.data.qpos.device)
    from copy import deepcopy
    config=deepcopy(config)
    config['wholebody_hand_contrast']=bank.has_contrast  # Metadata plumbing, not reward weights.
    return CATTask(sim,bank,config,collision=collision,seed=seed,analytic_objects=objects)
