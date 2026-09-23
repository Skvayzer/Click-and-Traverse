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
                     tracking_root_field_weight=None):
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
    scales.update(wholebody_hand_clearance=-.5 if hand_protection else -5.,wholebody_arm_clearance=-2.)
    arm_weight = config.get('arm_clearance_weight_override') if arm_clearance_weight is None else arm_clearance_weight
    if arm_weight is not None:
        if isinstance(arm_weight, bool) or not math.isfinite(arm_weight) or arm_weight > 0:
            raise ValueError('arm_clearance_weight must be finite and <= 0')
        scales['wholebody_arm_clearance'] = float(arm_weight)
        config['arm_clearance_weight_override'] = float(arm_weight)
    if stabilization:scales.update(wholebody_upper_target_velocity=-.05,wholebody_upper_target_acceleration=-.02,wholebody_upper_clear_posture=-.05)
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
