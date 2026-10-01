"""Pinned CAT configuration plus unchanged whole-body task additions."""
from __future__ import annotations

from .observation_contract import ACTOR_SIZE, CRITIC_SIZE

import copy
import math
from pathlib import Path


def hand_curriculum_threshold(config, manifest=None):
    """Run override, then bank metadata, then the historical 60% gate."""
    marker=(manifest or {}).get('hand_protection_curriculum', {})
    default=marker.get('clean_goal_success_threshold', .6) if isinstance(marker, dict) else .6
    value=config.get('hand_curriculum_success_threshold', default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0<value<=1:
        raise ValueError('hand_curriculum_success_threshold must be finite and in (0, 1]')
    return float(value)


def wholebody_config(base_config=None,*,bank_manifest=None,stabilization=True,
                     hand_protection=True,hand_contrast=False,collision_manifest=None,
                     reset_manifest=None,proposal_path=None,hand_curriculum_success_threshold=None,
                     hand_contrast_region_weight=None,hand_contrast_heading_weight=None,
                     hand_contrast_approach_distance=None,hand_raised_reset_fraction=None,
                     hand_reward_soft_floor=None,disable_hand_contrast=None,
                     hand_clearance_weight=None,arm_clearance_weight=None,hand_clearance_target=None,
                     hand_clearance_anticipation=None,hand_clearance_near_weight=None,
                     tracking_root_field_weight=None,standing_gf_bonus=None,
                     reactive_hand_guidance=None,handsdf_weight=None,heading_align_weight=None,
                     upright_weight=None,stand_tall_weight=None,torso_rate_weight=None,self_clearance_weight=None,
                     upper_posture_weight=None,upper_home_shoulder_pitch=None,terminate_on_hand_self_contact=None,
                     stand_still_weight=None,standing_requires_stillness=None,sdf_rate_obs=None,reactive_event_weight=None,
                     spot_hold_weight=None,standing_stillness_speed=None,route_recovery_speed=None,route_lost_seconds=None,goal_hold_seconds=None,
                     action_rate_weight=None,joint_acc_weight=None,hand_body_contact_weight=None,
                     heading_hand_probes=None,head_guidance_near_sdf=None,teleop_fraction=None,
                     teleop_body_commands=None,tracking_yaw_weight=None,teleop_heading_commands=None,
                     heading_track_weight=None,upright_roll=None,body_motion_weight=None,style_free_near_obstacles=None):
    from cat_ppo.furniture.generalist_config import released_config
    from .collision import PROPOSAL
    config=copy.deepcopy(released_config()['env_config'] if base_config is None else base_config)
    if hand_curriculum_success_threshold is not None:
        config['hand_curriculum_success_threshold']=hand_curriculum_success_threshold
    hand_curriculum_threshold(config)
    width = config.get('hand_reward_soft_floor', 0.) if hand_reward_soft_floor is None else hand_reward_soft_floor
    if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or not 0 <= width < 1:
        raise ValueError('hand_reward_soft_floor must be finite and in [0, 1) reward/step')
    if width and not hand_contrast:
        raise ValueError('Hand reward soft floor requires a contrastive bank')
    config['hand_reward_soft_floor'] = float(width)
    approach = config.get('hand_contrast_approach_distance', 0.) if hand_contrast_approach_distance is None else hand_contrast_approach_distance
    if isinstance(approach, bool) or not isinstance(approach, (int, float)) or not math.isfinite(approach) or approach < 0:
        raise ValueError('hand_contrast_approach_distance must be finite and >= 0')
    if approach and not hand_contrast:
        raise ValueError('Hand approach shaping requires a contrastive bank')
    config['hand_contrast_approach_distance'] = float(approach)
    fraction = config.get('hand_raised_reset_fraction', 0.) if hand_raised_reset_fraction is None else hand_raised_reset_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('hand_raised_reset_fraction must be finite and in [0, 1]')
    if fraction and not hand_contrast:
        raise ValueError('Raised resets require a contrastive bank')
    config['hand_raised_reset_fraction'] = float(fraction)
    if hand_protection and not stabilization:raise ValueError('Hand protection requires upper stabilization')
    if hand_contrast and not stabilization:raise ValueError('Contrastive guidance requires upper stabilization')
    config.update(compatibility_mode=False,wholebody_stabilization=stabilization,
        wholebody_hand_protection=hand_protection,wholebody_hand_contrast=hand_contrast,
        wholebody_first_outcome_metrics=True,hand_contrast_region_scale=.15,hand_contrast_metric_tolerance=.05,
        upper_action_scale=.8,upper_target_rate=2.,hand_protection_enabled=True,terminate_on_elbow_collision=True,
        hand_clearance_margin=.12,arm_clearance_margin=.08,clutter_episode_length=4000,
        upper_velocity_cost_scale=2.,upper_acceleration_cost_scale=20.,upper_posture_clearance_taper=.12,
        upper_cost_waist_weight=4.,hand_protection_target_clearance=.04,hand_protection_anticipation_distance=.20,
        hand_protection_near_weight=.8,hand_protection_posture_taper=.10,upper_arm_velocity_weight=.5,
        upper_arm_acceleration_weight=.1,num_act=29,num_obs=ACTOR_SIZE,num_pri=CRITIC_SIZE)
    config['wholebody']={'body_collision':dict(enabled=collision_manifest is not None,
        bank_manifest='' if collision_manifest is None else str(Path(collision_manifest).resolve()),
        reset_manifest='' if reset_manifest is None else str(Path(reset_manifest).resolve()),
        proposal=str(Path(proposal_path or PROPOSAL).resolve()),event_penalty=1.)}
    if bank_manifest is not None:
        config['pf_config']['bank_manifest']=str(Path(bank_manifest).resolve())
        config['pf_config']['sampling_alpha']=1.
        config['pf_config']['sampling_ema_decay']=.95
    scales=config['reward_config']['scales']
    if tracking_root_field_weight is not None:
        value=tracking_root_field_weight
        if isinstance(value,bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('tracking_root_field_weight must be finite and positive')
        scales['tracking_root_field']=float(value)
    def _signed(name,value,*,positive):
        if isinstance(value,bool) or not math.isfinite(value) or (value<0 if positive else value>0):
            raise ValueError(f'{name} must be finite and {">= 0" if positive else "<= 0"}')
        return float(value)
    # Posture (A): bonuses for a straight back and full height off-crouch, a cost on torso
    # angular rate; see task_math.posture_terms. Weight 0 removes the term.
    for name,value,positive in (('upright',upright_weight,True),('stand_tall',stand_tall_weight,True),('torso_rate',torso_rate_weight,False)):
        if value is not None:
            weight=_signed(name+'_weight',value,positive=positive)
            if weight==0:scales.pop(name,None)
            else:scales[name]=weight
    # Fingers (B): hand envelope vs own leg capsules; home pose for the posture prior.
    if self_clearance_weight is not None:
        weight=_signed('self_clearance_weight',self_clearance_weight,positive=False)
        if weight==0:scales.pop('self_clearance',None)
        else:scales['self_clearance']=weight
    if upper_home_shoulder_pitch is not None:
        config['upper_home_shoulder_pitch']=float(upper_home_shoulder_pitch)
    if terminate_on_hand_self_contact is not None:
        config['terminate_on_hand_self_contact']=bool(terminate_on_hand_self_contact)
    if sdf_rate_obs is not None:
        config['sdf_rate_obs']=bool(sdf_rate_obs)
    if reactive_event_weight is not None:
        weight=_signed('reactive_event_weight',reactive_event_weight,positive=True)
        if weight==0:scales.pop('reactive_event',None)
        else:scales['reactive_event']=weight
    # Standing scenes: price body motion at zero command and pay the standing bonus only when still.
    if stand_still_weight is not None:
        weight=_signed('stand_still_weight',stand_still_weight,positive=False)
        if weight==0:scales.pop('stand_still',None)
        else:scales['stand_still']=weight
    if standing_requires_stillness is not None:
        config['standing_requires_stillness_speed']=(float(standing_stillness_speed) if standing_stillness_speed else .05) if standing_requires_stillness else None
    # Smoothness. action_rate: first+second differences of the policy action (a correct rate term;
    # the legacy smoothness_action only sees the first difference). joint_acc_weight overrides the
    # legacy smoothness_joint scale (0.01 qd^2 + qdd^2).
    if action_rate_weight is not None:
        weight=_signed('action_rate_weight',action_rate_weight,positive=False)
        if weight==0:scales.pop('action_rate',None)
        else:scales['action_rate']=weight
    # Hands vs torso/head/other arm/other hand: physical contact pairs (model.hand_body_pairs) and a
    # contact-only cost per touching pair. 0 keeps the physics pairs but drops the cost.
    if hand_body_contact_weight is not None:
        weight=_signed('hand_body_contact_weight',hand_body_contact_weight,positive=False)
        config['hand_body_contact']=True
        if weight==0:scales.pop('hand_body_contact',None)
        else:scales['hand_body_contact']=weight
    if joint_acc_weight is not None:
        scales['smoothness_joint']=_signed('joint_acc_weight',joint_acc_weight,positive=False)
    # Head guidance pulls down (and opens the crouch gate) only with geometry within this distance
    # of the head; see CATTask._gate_head_pull. 0 restores the original behaviour.
    if head_guidance_near_sdf is not None:
        near=float(head_guidance_near_sdf)
        if not math.isfinite(near) or near<0:raise ValueError('head_guidance_near_sdf must be finite and >= 0')
        if near==0:config.pop('head_guidance_near_sdf',None)
        else:config['head_guidance_near_sdf']=near
    # Joystick training: a fraction of room episodes follow random commands (see
    # CATTask._teleop_resample) instead of the route; 0 disables.
    if teleop_fraction is not None:
        f=float(teleop_fraction)
        if not 0<=f<=1:raise ValueError('teleop_fraction must lie in [0, 1]')
        if f==0:config.pop('teleop',None)
        else:config['teleop']=dict(fraction=f,speed_min=.3,speed_max=.8,p_stop=.15,p_toward=.5,
                                   hold_min=1.5,hold_max=4.,episode_seconds=20.)
    # Joystick interface: body-frame (forward, sideways, turn rate) commands in the teleop episodes,
    # the turn rate kept in the actor observation, a turn-rate tracking reward, no forward-facing
    # bonus in those episodes.
    if teleop_body_commands:
        if 'teleop' not in config:raise ValueError('teleop_body_commands requires teleop_fraction')
        config['teleop'].update(mode='body',vx_min=-.5,vx_max=.8,vy_max=.3,yaw_max=1.,p_turn=.2,p_toward=.3)
        config['yaw_command']=True
        scales['tracking_yaw']=float(tracking_yaw_weight if tracking_yaw_weight is not None else 1.)
    elif tracking_yaw_weight is not None:
        raise ValueError('tracking_yaw_weight requires teleop_body_commands')
    # Hands first, body second: sideways lean counts in upright, moving off the (obstacle-safe) command
    # line costs more, and the style prior does not judge the arms while a hand is near an obstacle.
    if upright_roll is not None:
        if upright_roll:config['upright_roll']=True
        else:config.pop('upright_roll',None)
    if body_motion_weight is not None:
        scales['body_motion']=_signed('body_motion_weight',body_motion_weight,positive=False)
    if style_free_near_obstacles is not None:
        if style_free_near_obstacles:config['style_free_near_obstacles']=True
        else:config.pop('style_free_near_obstacles',None)
    if goal_hold_seconds is not None:
        hold=float(goal_hold_seconds)
        if not math.isfinite(hold) or hold<0:raise ValueError('goal_hold_seconds must be finite and >= 0')
        if hold==0:config.pop('goal_hold_seconds',None)
        else:config['goal_hold_seconds']=hold
    # Rooms: when the route is out of sight, follow the stored geodesic goal field instead of
    # standing still, and end the episode as a failure if the route stays lost.
    if route_recovery_speed is not None:
        speed=float(route_recovery_speed);lost=float(route_lost_seconds if route_lost_seconds is not None else 5.)
        if not (math.isfinite(speed) and math.isfinite(lost)) or speed<0 or lost<=0:
            raise ValueError('route_recovery_speed must be >= 0 and route_lost_seconds > 0')
        if speed==0:config.pop('route_recovery',None)
        else:config['route_recovery']=dict(speed=speed,lost_seconds=lost)
    elif route_lost_seconds is not None:
        raise ValueError('route_lost_seconds requires route_recovery_speed')
    if spot_hold_weight is not None:
        weight=_signed('spot_hold_weight',spot_hold_weight,positive=False)
        if weight==0:scales.pop('spot_hold',None)
        else:scales['spot_hold']=weight
    if heading_align_weight is not None:
        value=heading_align_weight
        if isinstance(value,bool) or not math.isfinite(value) or value<0:
            raise ValueError('heading_align_weight must be finite and >= 0')
        if value==0:
            # Explicit off: drop the term entirely rather than computing a zero-weighted one.
            scales.pop('heading_align',None);config.pop('heading_align',None)
        else:
            # Forward-facing bonus gated on counterfactual shoulder clearance; see
            # task_math.heading_align_reward. half_width = G1 shoulder joint (0.100) + arm capsule.
            # Gate closed at <=0.05 m of shoulder clearance (corridor <=0.42 m), open at >=0.15 m (>=0.62 m).
            scales['heading_align']=float(value)
            config['heading_align']=dict(weight=float(value),half_width=.16,lookahead=.30,margins=[.05,.15])
    # Hand-height probes for the heading gate (and the style gates that reuse it): the shoulder probes
    # sit ~0.4 m above a 0.71 m tabletop and saw a 0.43 m table gap as open, so the robot was paid to
    # face forward into a gap its hands could not pass. Probes where the hands would hang facing
    # forward (0.22 m lateral, current palm height), minus the hand radius; gate = min of both.
    if heading_hand_probes is not None:
        if heading_hand_probes and 'heading_align' not in config:
            raise ValueError('heading_hand_probes requires heading_align_weight')
        if 'heading_align' in config:
            if heading_hand_probes:config['heading_align']['hand_probe']=dict(half_width=.22,radius=.10)
            else:config['heading_align'].pop('hand_probe',None)
    # Joystick interface v8: one command for route and joystick worlds -- a world-fixed walking velocity
    # and a reference heading whose error reaches the policy gated by clearance (see
    # CATTask._heading_command); velocity reward toward the obstacle-safe target; heading reward gated.
    if teleop_heading_commands:
        if 'teleop' not in config:raise ValueError('teleop_heading_commands requires teleop_fraction')
        if 'heading_align' not in config:raise ValueError('teleop_heading_commands requires heading_align_weight')
        config['teleop'].update(mode='heading',vx_min=-.5,vx_max=.8,vy_max=.3,yaw_max=1.,p_turn=.2,p_toward=.3)
        config['heading_command']=True;config['safe_tracking']=True
        if heading_track_weight is not None:scales['heading_align']=float(heading_track_weight)
    elif heading_track_weight is not None:
        raise ValueError('heading_track_weight requires teleop_heading_commands')
    scales.update(wholebody_hand_clearance=-.5 if hand_protection else -5.,wholebody_arm_clearance=-2.)
    if standing_gf_bonus is not None:
        if isinstance(standing_gf_bonus,bool) or not math.isfinite(standing_gf_bonus) or standing_gf_bonus<0:
            raise ValueError('standing_gf_bonus must be finite and nonnegative')
        config['standing_gf_bonus']=float(standing_gf_bonus)
    if reactive_hand_guidance is not None:
        config['reactive_hand_guidance']=bool(reactive_hand_guidance)
    if handsdf_weight is not None:
        # handsdf and wholebody_hand_clearance are both functions of sdf[:,5:7]; measured,
        # the legacy +1 handsdf term carried 1.8x-5.4x more near-field gradient than the
        # -20 clearance term, so tuning the clearance weight moved less than a third of
        # the real signal. Zero one of them so the remaining weight means something.
        if isinstance(handsdf_weight,bool) or not math.isfinite(handsdf_weight):
            raise ValueError('handsdf_weight must be finite')
        scales['handsdf']=float(handsdf_weight)
    arm_weight = config.get('arm_clearance_weight_override') if arm_clearance_weight is None else arm_clearance_weight
    if arm_weight is not None:
        if isinstance(arm_weight, bool) or not math.isfinite(arm_weight) or arm_weight > 0:
            raise ValueError('arm_clearance_weight must be finite and <= 0')
        scales['wholebody_arm_clearance'] = float(arm_weight)
        config['arm_clearance_weight_override'] = float(arm_weight)
    if stabilization:scales.update(wholebody_upper_target_velocity=-.05,wholebody_upper_target_acceleration=-.02,wholebody_upper_clear_posture=-.05)
    if upper_posture_weight is not None:
        scales['wholebody_upper_clear_posture']=_signed('upper_posture_weight',upper_posture_weight,positive=False)
    # Retired objective: accept legacy override arguments as no-ops.
    scales.update(wholebody_hand_contrast_region=0., wholebody_hand_contrast_heading=0.)
    from .clearance_objective import configure
    inherited=config.get('clearance_objective_parameters', {})
    configure(config, disable_hand_contrast=disable_hand_contrast,
        hand_clearance_weight=inherited.get('weight') if hand_clearance_weight is None else hand_clearance_weight,
        hand_clearance_target=inherited.get('target') if hand_clearance_target is None else hand_clearance_target,
        hand_clearance_anticipation=inherited.get('anticipation') if hand_clearance_anticipation is None else hand_clearance_anticipation,
        hand_clearance_near_weight=inherited.get('near_weight') if hand_clearance_near_weight is None else hand_clearance_near_weight)
    return config
