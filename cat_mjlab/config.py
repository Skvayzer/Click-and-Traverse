"""Pinned CAT configuration plus unchanged whole-body task additions."""
from __future__ import annotations

import copy
from pathlib import Path


def wholebody_config(base_config=None,*,bank_manifest=None,stabilization=True,
                     hand_protection=True,hand_contrast=False,collision_manifest=None,
                     reset_manifest=None,proposal_path=None):
    from cat_ppo.furniture.generalist_config import released_config
    from .collision import PROPOSAL
    config=copy.deepcopy(released_config()['env_config'] if base_config is None else base_config)
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
        upper_arm_acceleration_weight=.1,num_act=29,num_obs=222,num_pri=310)
    config['wholebody']={'body_collision':dict(enabled=collision_manifest is not None,
        bank_manifest='' if collision_manifest is None else str(Path(collision_manifest).resolve()),
        reset_manifest='' if reset_manifest is None else str(Path(reset_manifest).resolve()),
        proposal=str(Path(proposal_path or PROPOSAL).resolve()),event_penalty=1.)}
    if bank_manifest is not None:
        config['pf_config']['bank_manifest']=str(Path(bank_manifest).resolve())
        config['pf_config']['sampling_alpha']=1.
        config['pf_config']['sampling_ema_decay']=.95
    scales=config['reward_config']['scales']
    scales.update(wholebody_hand_clearance=-.5 if hand_protection else -5.,wholebody_arm_clearance=-2.)
    if stabilization:scales.update(wholebody_upper_target_velocity=-.05,wholebody_upper_target_acceleration=-.02,wholebody_upper_clear_posture=-.05)
    if hand_contrast:scales.update(wholebody_hand_contrast_heading=-1.,wholebody_hand_contrast_region=-1.)
    return config
