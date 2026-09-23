"""Batched Torch port of CAT's certified ordered room guidance.

The leading dimension is the world batch. All metadata has already been
selected by scene ID; no host/device synchronization occurs in these kernels.
"""
from __future__ import annotations

import math
import torch


def _local(points, obstacles):
    # points [B,...,2], obstacles [B,M,6].
    extra = points.ndim - 2
    obs = obstacles.reshape(obstacles.shape[0], *((1,) * extra), *obstacles.shape[1:])
    delta = points.unsqueeze(-2) - obs[..., :2]
    x, y = delta.unbind(-1)
    c, s = obs[..., 4], obs[..., 5]
    return torch.stack((c*x+s*y, -s*x+c*y), -1), obs


def _reduce(distances, obstacle_count, radius):
    extra = distances.ndim - 2
    counts = obstacle_count.reshape(-1, *((1,) * (extra + 1)))
    mask = torch.arange(distances.shape[-1], device=distances.device) < counts
    r = radius.reshape(-1, *((1,) * (extra + 1)))
    return torch.where(mask, distances-r, torch.inf).amin(-1)


def root_clearance(points, obstacles, obstacle_count, *, radius):
    local, obs = _local(points, obstacles)
    q = local.abs() - obs[..., 2:4]
    distance = torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.amax(-1).clamp_max(0)
    return _reduce(distance, obstacle_count, radius)


def swept_root_clearance(first, last, obstacles, obstacle_count, *, radius):
    first, last = torch.broadcast_tensors(first, last)
    a, obs = _local(first, obstacles)
    b, _ = _local(last, obstacles)
    half = obs[..., 2:4]
    delta = b-a
    moving = delta != 0
    divisor = torch.where(moving, delta, 1.)
    low, high = (-half-a)/divisor, (half-a)/divisor
    enter = torch.where(moving, torch.minimum(low, high), -torch.inf).amax(-1).clamp_min(0)
    leave = torch.where(moving, torch.maximum(low, high), torch.inf).amin(-1).clamp_max(1)
    intersects = (moving | (a.abs() <= half)).all(-1) & (enter <= leave)
    distance = torch.minimum(torch.linalg.vector_norm((a.abs()-half).clamp_min(0), dim=-1),
                             torch.linalg.vector_norm((b.abs()-half).clamp_min(0), dim=-1))
    denominator = (delta*delta).sum(-1).clamp_min(1e-20)
    for sx, sy in ((-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)):
        signs = half.new_tensor((sx,sy))
        offset = half*signs-a
        fraction = ((offset*delta).sum(-1)/denominator).clamp(0,1)
        distance = torch.minimum(distance, torch.linalg.vector_norm(offset-fraction.unsqueeze(-1)*delta,dim=-1))
    return _reduce(torch.where(intersects,0.,distance),obstacle_count,radius)


def _pick(values, index):
    return values[torch.arange(values.shape[0],device=values.device),index.long()]


def _segment_target(root, route, segment, lookahead):
    a,b = _pick(route,segment),_pick(route,segment+1)
    delta = b-a
    length = torch.linalg.vector_norm(delta,dim=-1).clamp_min(1e-7)
    progress = ((root-a)*delta).sum(-1)/length
    progress = torch.minimum(progress.clamp_min(0),length)
    projection = a+progress.unsqueeze(-1)*delta/length.unsqueeze(-1)
    target = a+torch.minimum(progress+lookahead,length).unsqueeze(-1)*delta/length.unsqueeze(-1)
    return target,projection,b


def route_context(root_xy, previous_xy, segment, sticky_violation, route, route_count,
                  obstacles, obstacle_count, *, radius, lookahead=.4,
                  waypoint_radius=.15, goal_radius=.20, speed=.6):
    root,previous = root_xy,previous_xy
    last_segment = (route_count-2).clamp_min(0)
    segment = torch.minimum(segment.long().clamp_min(0),last_segment)
    next_segment = torch.minimum(segment+1,last_segment)
    _,_,corner = _segment_target(root,route,segment,lookahead)
    next_target,_,_ = _segment_target(root,route,next_segment,lookahead)
    next_clearance = swept_root_clearance(root,next_target,obstacles,obstacle_count,radius=radius)
    advance = ((segment < last_segment) & (torch.linalg.vector_norm(root-corner,dim=-1) <= waypoint_radius)
               & (next_clearance >= 2e-6))
    segment = torch.where(advance,next_segment,segment)
    carrot,projection,_ = _segment_target(root,route,segment,lookahead)
    delta = route[:,1:]-route[:,:-1]
    lengths = torch.linalg.vector_norm(delta,dim=-1)
    tangent = _pick(delta,segment)/_pick(lengths,segment).clamp_min(1e-7).unsqueeze(-1)
    offsets = torch.where(torch.arange(lengths.shape[1],device=root.device)[None] < segment[:,None],lengths,0.).sum(-1)
    progress = offsets+torch.linalg.vector_norm(projection-_pick(route,segment),dim=-1)
    targets = torch.stack((carrot,projection),1)
    margins = swept_root_clearance(root[:,None],targets,obstacles,obstacle_count,radius=radius)
    visible,recovery = margins[:,0]>=2e-6,margins[:,1]>=2e-6
    target = torch.where(visible[:,None],carrot,torch.where(recovery[:,None],projection,root))
    valid = route_count >= 2
    finite = torch.isfinite(root).all(-1) & torch.isfinite(previous).all(-1)
    root_margin = root_clearance(root,obstacles,obstacle_count,radius=radius)
    sweep_margin = swept_root_clearance(previous,root,obstacles,obstacle_count,radius=radius)
    swept_violation = (sweep_margin < -2e-6) | ~finite
    violation = sticky_violation | swept_violation
    goal = _pick(route,(route_count-1).clamp_min(0))
    goal_distance = torch.linalg.vector_norm(root-goal,dim=-1)
    complete = valid & (segment==last_segment) & (goal_distance<=goal_radius) & ~violation
    displacement = target-root
    target_distance = torch.linalg.vector_norm(displacement,dim=-1)
    blocked = ~valid | ~finite | (~visible & (~recovery | (target_distance<=1e-7)))
    active = ~blocked & ~complete & ~violation
    direction = torch.where(active[:,None],displacement/target_distance.clamp_min(1e-7)[:,None],0.)
    taper = torch.where(segment==last_segment,(goal_distance/.30).clamp(0,1),1.)
    guidance = torch.cat((direction*(speed*taper[:,None]),root.new_zeros((len(root),1))),-1)
    return dict(segment=segment,target=target,direction=direction,guidance=guidance,
                progress_m=progress,tangent=tangent,root_clearance=root_margin,swept_clearance=sweep_margin,
                swept_violation=swept_violation,violation=violation,route_complete=complete,blocked=blocked,
                cross_track=torch.linalg.vector_norm(root-projection,dim=-1),root_xy=root.clone())


def hand_contrast_context(metadata, progress, tangent, *, approach_distance=0.):
    inside = metadata['zone_valid'] & (progress[:,None]>=metadata['start_m']) & (progress[:,None]<metadata['end_m'])
    enabled = metadata['enabled'] & inside.any(-1)
    index = inside.to(torch.int64).argmax(-1)
    start,end,fade = (_pick(metadata[k],index) for k in ('start_m','end_m','fade_m'))
    ramp = (torch.minimum(progress-start,end-progress)/fade.clamp_min(1e-8)).clamp(0,1)
    weight = torch.where(enabled,torch.where(fade>0,ramp,1.),0.)
    tangent = tangent/torch.linalg.vector_norm(tangent,dim=-1,keepdim=True).clamp_min(1e-8)
    context = dict(enabled=enabled,role=metadata['role'],zone_index=index,phase_weight=weight,
                core_active=enabled & (weight>=1.-1e-6),
                required_forward_zones=metadata['enabled'][:,None]&metadata['zone_valid']&(metadata['forward_weight']>0),
                required_hand_zones=metadata['enabled'][:,None]&metadata['zone_valid']&metadata['hand_active'].any(-1),
                forward_weight=_pick(metadata['forward_weight'],index)*weight,
                hand_active=_pick(metadata['hand_active'],index)&enabled[:,None],
                hand_regions_min=_pick(metadata['hand_regions_min'],index),
                hand_regions_max=_pick(metadata['hand_regions_max'],index),
                region_valid=_pick(metadata['region_valid'],index)&enabled[:,None],route_tangent=tangent)
    if 'heading_override' in metadata:
        override = _pick(metadata['heading_override'],index) & enabled
        context.update(heading_override=override,
            heading_target_rad=_pick(metadata['heading_target_rad'],index),
            heading_axis=_pick(metadata['heading_axis'],index),
            heading_weight=torch.where(override,
                _pick(metadata['heading_weight'],index)*weight.square()*(3-2*weight),
                context['forward_weight']))
    if approach_distance > 0:
        # Only forward-protected scenes. Never borrow a target across the
        # preceding zone; heading and compliance retain their original timing.
        previous_end = torch.cat((torch.zeros_like(metadata['end_m'][:,:1]),
                                  metadata['end_m'][:,:-1]), -1)
        early_start = torch.maximum(metadata['start_m']-approach_distance, previous_end)
        early_start = torch.where((metadata['role']==1)[:,None], early_start, metadata['start_m'])
        early = dict(metadata, start_m=early_start,
                     fade_m=metadata['fade_m']+metadata['start_m']-early_start)
        reward_context = hand_contrast_context(early, progress, tangent)
        # Preserve the original outgoing fade (the extension is incoming only).
        idx = reward_context['zone_index']
        end, fade = (_pick(metadata[k],idx) for k in ('end_m','fade_m'))
        incoming_fade = _pick(early['fade_m'],idx)
        incoming = torch.where(incoming_fade>0,
            ((progress-_pick(early_start,idx))/incoming_fade.clamp_min(1e-8)).clamp(0,1), 1.)
        outgoing = torch.where(fade>0, ((end-progress)/fade.clamp_min(1e-8)).clamp(0,1), 1.)
        reward_context['phase_weight'] = torch.where(reward_context['enabled'], torch.minimum(incoming,outgoing), 0.)
        reward_context['phase_weight'] = torch.where(metadata['role']==1,
            reward_context['phase_weight'], context['phase_weight'])
        for key in ('enabled','phase_weight','hand_active','region_valid','hand_regions_min','hand_regions_max'):
            context['reward_'+key] = reward_context[key]
    return context


def contrast_reward_terms(context,hands_world,root_xy,pelvis_forward,torso_forward,
                          *,region_scale=.15,hand_good_distance=.05):
    tangent=context['route_tangent']
    headings=torch.stack((pelvis_forward[:,:2],torso_forward[:,:2]),1)
    cosine=((headings*tangent[:,None]).sum(-1)/torch.linalg.vector_norm(headings,dim=-1).clamp_min(1e-8)).clamp(-1,1)
    heading=(1.-cosine).mean(-1)*context['forward_weight']
    heading_good=(cosine>=math.cos(math.radians(15))).all(-1)
    if 'heading_override' in context:
        angle=context['heading_target_rad']
        normal=torch.stack((-tangent[:,1],tangent[:,0]),-1)
        target=angle.cos()[:,None]*tangent+angle.sin()[:,None]*normal
        alignment=((headings*target[:,None]).sum(-1)/torch.linalg.vector_norm(headings,dim=-1).clamp_min(1e-8)).clamp(-1,1)
        alignment=torch.where(context['heading_axis'][:,None],alignment.abs(),alignment)
        heading=torch.where(context['heading_override'],
            (1-alignment).mean(-1)*context['heading_weight'],heading)
        heading_good=torch.where(context['heading_override'],
            (alignment>=math.cos(math.radians(15))).all(-1),heading_good)
    delta=hands_world[:,:,:2]-root_xy[:,None]
    normal=torch.stack((-tangent[:,1],tangent[:,0]),-1)
    local=torch.stack(((delta*tangent[:,None]).sum(-1),(delta*normal[:,None]).sum(-1),hands_world[:,:,2]),-1)
    distance=(context['hand_regions_min']-local[:,None]).clamp_min(0)+(local[:,None]-context['hand_regions_max']).clamp_min(0)
    squared=distance.square().sum(-1)
    active=context['hand_active'].float()
    per_region=(squared/(squared+region_scale**2)*active[:,None]).sum(-1)/active.sum(-1).clamp_min(1)[:,None]
    cost=torch.where(context['region_valid'],per_region,2.).amin(-1)
    any_active=context['hand_active'].any(-1)&context['region_valid'].any(-1)&context['enabled']
    hand=torch.where(any_active,cost*context['phase_weight'],0.)
    inside=((squared<=hand_good_distance**2+1e-10)|~context['hand_active'][:,None]).all(-1)
    good=~any_active|(inside&context['region_valid']).any(-1)
    if 'reward_enabled' in context:
        reward_context = {key: value for key,value in context.items() if not key.startswith('reward_')}
        reward_context.update({key[7:]: value for key,value in context.items() if key.startswith('reward_')})
        shaped,_ = contrast_reward_terms(reward_context,hands_world,root_xy,pelvis_forward,torso_forward,
                                         region_scale=region_scale,hand_good_distance=hand_good_distance)
        hand = shaped['wholebody_hand_contrast_region']
    return {'wholebody_hand_contrast_heading':heading,'wholebody_hand_contrast_region':hand},dict(
        hand_contrast_heading_cost=heading,hand_contrast_region_cost=hand,
        hand_contrast_heading_good=heading_good.float(),
        hand_contrast_hand_good=good.float(),hand_contrast_forward_weight=context['forward_weight'],
        hand_contrast_hand_active=any_active.float())
