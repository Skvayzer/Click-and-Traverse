"""Pinned CAT configuration plus unchanged whole-body task additions."""
from __future__ import annotations

from .observation_contract import ACTOR_SIZE, CRITIC_SIZE

import copy
import json
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


def _portable(path):
    """A repository file as a path relative to the repository root, so the recorded configuration (part of the resume
    contract) does not depend on where the code runs: jobs run a private copy under /tmp/slurm-<job id>/code, whose
    absolute path changed on every resubmission and refused every resume (E1, E2 on 2026-11-16). Only recorded --
    the collision checker resolves its own proposal path."""
    from .collision import ROOT
    path=Path(path).resolve()
    try:
        return str(path.relative_to(Path(ROOT).resolve()))
    except ValueError:
        return str(path)


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
                     heading_track_weight=None,upright_roll=None,body_motion_weight=None,style_free_near_obstacles=None,
                     keyboard_v10=None,hand_clearance_tight=None,teleop_progress_weight=None,teleop_stall_weight=None,
                     teleop_cat_fraction=None,heading_lookaheads=None,heading_blocking_probes=None,sideways_bonus_weight=None,
                     side_gap_curriculum=None,teleop_side_prob=None,hand_tight_walls_only=None,side_gap_tumbling=None,
                     sideways_bonus_progress=None,contact_penalty=None,stall_weight=None,goal_progress_weight=None,
                     goal_bonus_weight=None,timeout_failure=None,reward_group_room=None,commanded_blocking_map=None,
                     hand_tight_cat_only=None,sgs=None,robustness=None,upper_target_average=None,
                     reactive_stationary=None,reactive_power_weight=None,reactive_feet_hold_weight=None,
                     reactive_hand_clearance=None,reactive_retreat_success=None,table_facing_weight=None,
                     table_leg_power_weight=None,table_progress=None,table_hands_up_weight=None,
                     table_hands_up_band=None,table_center_weight=None,table_zone_facing_weight=None,passage_progress=None,
                     table_power_weight=None,global_power_weight=None,hand_tight_passages=None,passage_stall_factor=None,
                     passage_soft_contact=None,passage_stall_gate=None,route_energy=None,operator_speed=None,
                     operator_rest_starts=None,operator_v2=None,operator_v3=None,operator_v4=None,operator_v5=None,operator_v6=None,overspeed_weight=None,goal_seed_cache=None,goal_ground=None,tracker=None):
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
        proposal=_portable(proposal_path or PROPOSAL),event_penalty=1.)}
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
    if reactive_stationary is not None:
        # Stationary standing against approaching objects: [feet_m, air_steps(, grace_steps(, root_m))]; task._termination.
        # The standing spot radius (spot_hold, event 'stayed') stays reactive_spot_radius (0.3 m) in every variant.
        feet_m,air_steps=float(reactive_stationary[0]),int(reactive_stationary[1])
        grace=int(reactive_stationary[2]) if len(reactive_stationary)>2 else 30
        root_m=float(reactive_stationary[3]) if len(reactive_stationary)>3 else .2
        if not (0<feet_m<=.2 and 0<=air_steps<=50 and 0<=grace<=200 and 0<root_m<=.5):raise ValueError('invalid reactive_stationary')
        config['reactive_stationary']=dict(feet_m=feet_m,air_steps=air_steps,grace_steps=grace,root_m=root_m,
                                           push_recover_steps=25,settle_wait_steps=25)
    if reactive_power_weight is not None:
        weight=_signed('reactive_power_weight',reactive_power_weight,positive=False)
        if weight==0:scales.pop('reactive_power',None)
        else:scales['reactive_power']=weight
    if reactive_hand_clearance is not None:
        # Standing episodes only: [target, anticipation] of the hand clearance pressure (default .04-.09 / .20 m).
        target,anticipation=float(reactive_hand_clearance[0]),float(reactive_hand_clearance[1])
        if not 0<target<anticipation<=.6:raise ValueError('reactive_hand_clearance needs 0 < target < anticipation <= 0.6')
        config['reactive_hand_clearance']=dict(target=target,anticipation=anticipation)
    if reactive_retreat_success is not None:
        if not 0<float(reactive_retreat_success)<=.4:raise ValueError('reactive_retreat_success must be in (0, 0.4] m')
        config['reactive_retreat_success_m']=float(reactive_retreat_success)
    if table_hands_up_weight is not None:
        weight=_signed('table_hands_up_weight',table_hands_up_weight,positive=True)
        if weight==0:scales.pop('table_hands_up',None)
        else:scales['table_hands_up']=weight
    if table_hands_up_band is not None:
        low,high=float(table_hands_up_band[0]),float(table_hands_up_band[1])
        if not .70<low<high<=1.2:raise ValueError('table_hands_up_band needs 0.70 < low < high <= 1.2 (palm z, m)')
        config['table_hands_up_band']=(low,high)
    if table_progress or passage_progress:
        # Table-edge passages keep the CAT weights of the progress terms the room group zeroes (task._rewards).
        config['table_progress_terms']=['stall','goal_progress','goal_bonus']
    if passage_progress:
        # ... and so do the other room passages (hand-contrast passages, width scaffolds).
        config['progress_passage_kinds']=['contrastive-hand-passage','width-curriculum-scaffold']
    for name,key,weight in (('table_facing_weight','table_facing',table_facing_weight),
                            ('table_leg_power_weight','table_leg_power',table_leg_power_weight),
                            ('table_center_weight','table_center',table_center_weight),
                            ('table_zone_facing_weight','table_zone_facing',table_zone_facing_weight),
                            ('table_power_weight','table_power',table_power_weight),
                            ('global_power_weight','global_power',global_power_weight)):
        if weight is not None:
            # Table-edge passages only (task._rewards): face the route while walking; price leg + waist power;
            # between a table pair, stay on the route line and face along it.
            weight=_signed(name,weight,positive=False)
            if weight==0:scales.pop(key,None)
            else:scales[key]=weight
    if reactive_feet_hold_weight is not None:
        weight=_signed('reactive_feet_hold_weight',reactive_feet_hold_weight,positive=False)
        if weight==0:scales.pop('reactive_feet_hold',None)
        else:scales['reactive_feet_hold']=weight
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
    # Keyboard v10 (docs/obsidian/Style-Prior-Pipeline.md 7i): only what blocks the body band (0.25-0.95 m)
    # repels -- beams and hurdles no longer zero the velocity target or release the heading; the safe target
    # applies to joystick worlds only; a heading deadband (0.20 on / 0.08 off rad) so a robot with no command
    # stands instead of marking time; keyboard practice in CAT scenes, mostly THROUGH the obstacle along the
    # stored goal field; stop practice doubled and a stop holds the current facing.
    if keyboard_v10:
        if not config.get('heading_command'):raise ValueError('keyboard_v10 requires teleop_heading_commands')
        config.update(blocking_map=True,heading_deadband=(.20,.08),safe_tracking_teleop_only=True)
        config['teleop'].update(cat_scenes=True,p_stop=.30,p_through=.30,p_through_cat=.70)
    # v11: refusing a passable keyboard command must cost (see CATTask._rewards, teleop_progress/stall).
    for name,value,positive in (('teleop_progress',teleop_progress_weight,True),('teleop_stall',teleop_stall_weight,False)):
        if value is not None:
            if not config.get('safe_tracking'):raise ValueError(name+'_weight requires teleop_heading_commands')
            scales[name]=_signed(name+'_weight',value,positive=positive)
    if teleop_cat_fraction is not None:
        if not (config.get('teleop') or {}).get('cat_scenes'):raise ValueError('teleop_cat_fraction requires keyboard_v10')
        if not 0<=teleop_cat_fraction<=1:raise ValueError('teleop_cat_fraction must be in [0, 1]')
        config['teleop']['cat_fraction']=float(teleop_cat_fraction)
    # Side-gap curriculum (docs 7k): the forward-facing bonus releases earlier (probes at several lookaheads,
    # on the 2-D blocking footprint), a bonus for being sideways where forward does not fit, width levels with
    # sideways starts, open-ground sidesteps.
    if heading_lookaheads is not None or heading_blocking_probes:
        if 'heading_align' not in config:raise ValueError('heading_lookaheads/heading_blocking_probes require heading_align_weight')
        if heading_lookaheads is not None:config['heading_align']['lookaheads']=[float(v) for v in heading_lookaheads]
        if heading_blocking_probes:config['heading_align']['blocking']=True;config['blocking_map']=True
    if sideways_bonus_weight is not None:
        if 'heading_align' not in config:raise ValueError('sideways_bonus_weight requires heading_align_weight')
        scales['sideways_bonus']=_signed('sideways_bonus_weight',sideways_bonus_weight,positive=True)
    if side_gap_curriculum:
        config['side_gap_curriculum']=dict(thresholds=[.50,.42,.36,0.],advance=.30,window=300,sideways_starts=[.5,.35,.2,.1])
    if sideways_bonus_progress is not None:
        if 'sideways_bonus' not in scales:raise ValueError('sideways_bonus_progress requires sideways_bonus_weight')
        if not sideways_bonus_progress>0:raise ValueError('sideways_bonus_progress must be > 0 (m/s for the full bonus)')
        config['sideways_bonus_progress']=float(sideways_bonus_progress)
    # Anti-refusal (reports/RL refusal versus attempt incentives.md)
    for name,value,positive in (('stall',stall_weight,False),('goal_progress',goal_progress_weight,True),('goal_bonus',goal_bonus_weight,True)):
        if value is not None:scales[name]=_signed(name+'_weight',value,positive=positive)
    if timeout_failure:config['timeout_failure']=True
    if contact_penalty is not None:
        if not contact_penalty>=0:raise ValueError('contact_penalty must be >= 0')
        config['contact_penalty']=float(contact_penalty)
    if side_gap_tumbling:
        if 'side_gap_curriculum' not in config:raise ValueError('side_gap_tumbling requires side_gap_curriculum')
        config['side_gap_curriculum']['tumbling']=True
    if teleop_side_prob is not None:
        if (config.get('teleop') or {}).get('mode') not in ('body','heading'):raise ValueError('teleop_side_prob requires body/heading teleop')
        config['teleop']['p_side']=float(teleop_side_prob)
    if hand_clearance_tight is not None:
        low,high,floor=(float(v) for v in hand_clearance_tight)
        if not (0<=low<high and 0<floor<=1):raise ValueError('hand_clearance_tight: 0 <= low < high, 0 < floor <= 1')
        config.update(blocking_map=True,hand_clearance_tight=dict(low=low,high=high,floor=floor,walls_only=bool(hand_tight_walls_only)))
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
    if commanded_blocking_map is not None:
        config['blocking_map_commanded']=commanded_blocking_map=='on' if isinstance(commanded_blocking_map,str) else bool(commanded_blocking_map)
    if hand_tight_cat_only:
        if not config.get('hand_clearance_tight'):raise ValueError('hand_tight_cat_only requires hand_clearance_tight')
        config['hand_clearance_tight']['cat_only']=True
    if hand_tight_passages:
        if not config.get('hand_clearance_tight'):raise ValueError('hand_tight_passages requires hand_clearance_tight')
        config['hand_clearance_tight']['passages']=True
    if operator_speed is not None:
        # Slow traversal (task.py, operator speed): fields give the direction, the operator the speed. [min, max] m/s
        # log-uniform per episode, a slower speed near obstacles in 60% of episodes, re-drawn every 4-10 s (p 0.3);
        # stops only where the projected field command leaves < 0.05 m/s, never latched; clearance costs per metre,
        # tracking / progress / stall relative to the command; episodes slower than 0.4 m/s free of the anchor.
        lo,hi=(float(x) for x in operator_speed)
        if not 0<lo<=hi<=1.:
            raise ValueError(f"operator_speed needs 0 < min <= max <= 1 m/s, got {operator_speed}")
        fixed=lo==hi
        config['operator_speed']=dict(min=lo,max=hi,nominal=.525,cutout=.05,near_sdf=.35,near_share=0. if fixed else .6,
            resample=None if fixed else [4.,10.,.3],length_scale_max=.525/.1,anchor_free_below=.4,stall_fraction=.2,
            rest_start_share=float(operator_rest_starts or 0.))
        if operator_v2:
            # V2: 35% of episodes near nominal speed (anchored), rest starts inside only where a sideways body fits and
            # in the nominal pose.
            config['operator_speed'].update(nominal_share=0. if fixed else .35,rest_fix=True)
        if operator_v3:
            # V3 = V2 + (1) headgf / handsgf / feetgf as velocity along the field relative to the operator speed (capped
            # at 1) instead of its cosine: a slow, swaying velocity read as misaligned near obstacles, where these terms
            # act, so only a fast pass scored (near-obstacle speed 4x a 0.1 m/s command in E2 / V1); (2) tracking and
            # overspeed on the 0.4 s mean pelvis velocity; (3) overspeed inside the clip, above 1.25x the command
            # + 0.05 m/s, relative to max(command, 0.2). V2's (instantaneous, 1.2x, outside the clip) charged the gait's
            # own sway, so walking slowly paid less than standing and ending the episode on contact paid most.
            config['operator_speed'].update(nominal_share=0. if fixed else .35,rest_fix=True,gf_along=True,smooth_tau=.4,
                                            overspeed_inside=True,overspeed_ratio=1.25,overspeed_slack=.05,overspeed_floor=.2)
        if operator_v4:
            # V4 = V3 + (1) clearance and SDF costs at the larger of the commanded and the actual mean speed (per metre
            # of travel when faster than told: crossing a gap quickly no longer costs less), (2) side gaps and room
            # passages free of the anchor (E2 u300, which lunges through them at ~1 m/s whatever it is told).
            if not operator_v3:raise ValueError('operator_v4 builds on operator_v3')
            config['operator_speed'].update(clearance_per_metre=True,gap_anchor_free=True)
        if operator_v5:
            # V5 = V4 + the goal bonus scaled by 1 - (share of the last ~2 s with the 0.4 s mean speed above 1.25x the
            # command + 0.05 m/s) / 0.25: discounted at 0.98 per step, the bonus paid ~20x more for arriving 1 s out than
            # 4 s out, and every CAT / side-gap goal plane lies 0.4-1.4 m past its obstacle -- a sprint through the gap.
            if not operator_v4:raise ValueError('operator_v5 builds on operator_v4')
            if not overspeed_weight:raise ValueError('operator_v5 needs the overspeed limit (--overspeed-weight)')
            config['operator_speed'].update(goal_compliance=.25)
        if operator_v6:
            # V6 = V5 + the guidance terms' constant for points past the goal line (4 each, x > 1.5 m in CAT scenes: ~+16/s
            # for being across, ~0 just before) scaled by the same speed-compliance factor, and every goal-line (CAT) scene
            # anchor-free (E2 sprints for the line in all of them).
            if not operator_v5:raise ValueError('operator_v6 builds on operator_v5')
            config['operator_speed'].update(crossed_gate=True,cat_anchor_free=True)
    if goal_seed_cache is not None:
        # Ground-projected goal: column-seed guidance from a cache pinned to the bank (cat_mjlab/goal_seed.py).
        meta=json.loads((Path(goal_seed_cache)/'cache.json').read_text())
        config['goal_seed']=dict(mode='column',cache=str(goal_seed_cache),cache_sha256=meta['cache_sha256'],
                                 seed_radius_m=meta['seed_radius_m'],column_z_m=meta['column_z_m'])
    if goal_ground:
        # Success at the goal on the ground instead of past the x = 1.5 m line (0.35-0.40 m before the goal point, where
        # the policies learned to stop): a body point counts as across within cross_radius of the goal xy; the operator
        # command stops within arrive m of the goal (released beyond release m), as the robot executor does; CAT-type
        # episodes end hold_seconds after success (truncation), as route worlds do.
        if not goal_seed_cache:raise ValueError('goal_ground needs the ground-projected goal (goal_seed_cache)')
        config['goal_ground']=dict(cross_radius=.30,arrive=.25,release=.45,hold_seconds=2.)
    if tracker:
        # Tracker v1 (cat_mjlab/tracking.py): a time-indexed reference (path with a waiting clock, optional heading,
        # pelvis height and upper-body targets) in both observations, tracking terms, and the command taken from the
        # reference while one is active. The potential-field inputs stay: one network tracks when it has a reference
        # and reacts when it has none (source 'none'). Needs the operator speed (the field source plays the guidance
        # path at that speed).
        if not operator_speed:raise ValueError('tracker needs operator_speed')
        spec=dict(tracker) if isinstance(tracker,dict) else {}
        mix=spec.get('mix') or (('field',.5),('random',.3),('none',.2))
        config['tracker']=dict(mix=[[str(a),float(b)] for a,b in mix],sigma=float(spec.get('sigma',.15)))
        # Planner with a field reflex: planner v1 in the loop for the 'plan' source, stand-still plans for a share of the
        # reactive standing worlds, and the reflex rule on the tracking terms (task._rewards).
        names={str(a) for a,_ in mix}
        if 'plan' in names and not spec.get('planner'):raise ValueError("tracker source 'plan' needs a planner checkpoint")
        if spec.get('planner'):
            planner=dict(spec['planner']) if isinstance(spec['planner'],dict) else dict(path=str(spec['planner']))
            if not planner.get('path'):raise ValueError('tracker planner needs a path')
            config['tracker']['planner']=dict(path=str(planner['path']),samples=int(planner.get('samples',4)),
                steps=int(planner.get('steps',6)),guidance=float(planner.get('guidance',2.)),horizon=float(planner.get('horizon',2.)),
                interval=[float(x) for x in planner.get('interval',(.3,1.))],clear_cap=float(planner.get('clear_cap',.10)),
                side_penalty=float(planner.get('side_penalty',.02)),switch_penalty=float(planner.get('switch_penalty',.1)))
        if spec.get('hold_plan') is not None:
            config['tracker']['hold_plan']=float(spec['hold_plan'])
            if not 0.<=config['tracker']['hold_plan']<=1.:raise ValueError('tracker hold_plan must be in [0, 1]')
            if config['tracker']['hold_plan']>0 and not spec.get('planner'):raise ValueError('tracker hold_plan needs a planner')
        if spec.get('reflex'):
            reflex=dict(spec['reflex']) if isinstance(spec['reflex'],dict) else {}
            config['tracker']['reflex']=dict(hand_band=float(reflex.get('hand_band',.15)),body_band=float(reflex.get('body_band',.35)),
                                             deadband=float(reflex.get('deadband',.20)))
        for name,value in dict(track_pos=4.,track_lead=-8.,track_heading=1.,track_height=1.,track_upper=1.).items():
            scales[name]=float((spec.get('scales') or {}).get(name,value))
    if overspeed_weight is not None:
        if not operator_speed:raise ValueError('overspeed_weight needs operator_speed')
        weight=_signed('overspeed_weight',overspeed_weight,positive=False)
        if weight==0:scales.pop('overspeed',None)
        else:scales['overspeed']=weight
    if route_energy is not None:
        # Energy per metre in place of the per-second table/global price (task._rewards): [weight, q_J_per_m,
        # cruise_mps, table_factor]; weight < 0 is the price per joule, as global_power's.
        weight,q,cap,table_factor=(float(x) for x in route_energy)
        weight=_signed('route_energy_weight',weight,positive=False)
        if not (q>0 and cap>0 and table_factor>0):
            raise ValueError(f"route_energy needs q, cruise speed and table factor > 0, got {route_energy}")
        if global_power_weight is not None or table_power_weight is not None:
            raise ValueError("route_energy replaces global_power and table_power; leave those unset")
        # A fine-tune inherits the student's scales (E2: global_power, table_power): the per-metre price replaces them.
        scales.pop('global_power',None);scales.pop('table_power',None)
        scales['route_energy']=weight
        config['route_energy']=dict(q=q,cap=cap,table_factor=table_factor)
    if passage_stall_gate:
        # Room passages: no positive reward while stalled under a go command (task._rewards).
        config['passage_stall_gate']=True
    if passage_soft_contact is not None:
        if not float(passage_soft_contact)>0:raise ValueError('passage_soft_contact must be > 0')
        if contact_penalty is not None:raise ValueError('passage_soft_contact and contact_penalty are exclusive')
        config['passage_soft_contact']=float(passage_soft_contact)
    if passage_stall_factor is not None:
        if not config.get('progress_passage_kinds'):raise ValueError('passage_stall_factor requires passage_progress')
        if not 1<=float(passage_stall_factor)<=20:raise ValueError('passage_stall_factor must be in [1, 20]')
        config['passage_stall_factor']=float(passage_stall_factor)
    if sgs:
        # Success-Guided Sampling (task._init_sgs): paper defaults for locomotion are history 100, target .66,
        # kappa 5, temperature 2, epsilon 1e-8.
        sgs=dict(sgs)
        for key,low in (('slots',1),('history',1)):
            if int(sgs[key])<low:raise ValueError(f'sgs {key} must be >= {low}')
        if not 0<float(sgs['target'])<1:raise ValueError('sgs target must be in (0, 1)')
        if float(sgs['kappa'])<0 or float(sgs['temperature'])<=0 or float(sgs['epsilon'])<=0:raise ValueError('invalid sgs kernel')
        if not 0<=float(sgs.get('uniform_mix',0.))<=1:raise ValueError('sgs uniform_mix must be in [0, 1]')
        config['sgs']=sgs
    if upper_target_average:
        config['upper_target_average']=True
    if robustness:
        # Sim-to-real robustness (task: _pose_estimate, latency in the PD loop, per-world friction/mass): actor-side
        # localisation error (constant mount error + jitter + update period) and observation-to-motor latency.
        rb=dict(robustness)
        if not 0<=float(rb.get('action_latency_ms',0.))<=40:raise ValueError('action latency must be within two control periods (0-40 ms)')
        if rb.get('com') is not None and any(not 0<=float(v)<=.2 for v in rb['com']):raise ValueError('com offsets must be in [0, 0.2] m')
        if not 0<=float(rb.get('torque_offset',0.))<=.5:raise ValueError('torque offset must be in [0, 0.5] of the torque limit')
        lo,hi=(int(v) for v in rb.get('pose_period_steps',[5,5]))
        if not 1<=lo<=hi:raise ValueError('invalid pose update period range')
        for key in ('pose_noise','mount'):
            if any(float(v)<0 for v in rb.get(key,[0.,0.,0.])):raise ValueError(key+' must be non-negative')
        for key in ('friction','mass'):
            if rb.get(key) is not None:
                lo,hi=(float(v) for v in rb[key])
                if not 0<lo<=hi:raise ValueError(key+' range must be positive and ordered')
        config['robustness']=rb
    if reward_group_room:
        # Per scene group reward weights (task._rewards): ROOM worlds use these, CAT worlds the global scales.
        room={}
        for item in reward_group_room:
            term,value=item.split('=',1) if isinstance(item,str) else item
            if term not in scales:raise ValueError(f'reward_group_room: unknown reward term {term}')
            value=float(value)
            if not math.isfinite(value):raise ValueError('reward_group_room scales must be finite')
            room[term]=value
        config['reward_group_scales']=dict(room=room)
    return config
