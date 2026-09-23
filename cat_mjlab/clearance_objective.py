"""Clearance-primary objective; authored box rewards are permanently retired."""
import math


def configure(config, *, disable_hand_contrast=None, hand_clearance_weight=None,
              hand_clearance_target=None, hand_clearance_anticipation=None,
              hand_clearance_near_weight=None):
    # Kept as a compatibility key; old configs/False cannot revive box rewards.
    config['disable_hand_contrast'] = True
    scales = config['reward_config']['scales']
    if hand_clearance_weight is not None:
        if isinstance(hand_clearance_weight, bool) or not math.isfinite(hand_clearance_weight) or hand_clearance_weight > 0:
            raise ValueError('hand_clearance_weight must be finite and <= 0')
        scales['wholebody_hand_clearance'] = float(hand_clearance_weight)
    names = ('target_clearance', 'anticipation_distance', 'near_weight')
    for name, value in zip(names, (hand_clearance_target, hand_clearance_anticipation, hand_clearance_near_weight)):
        if value is not None:
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError('hand_clearance parameters must be finite numbers')
            config['hand_protection_'+name] = float(value)
    target, anticipation, near = (config['hand_protection_'+name] for name in names)
    if not 0 < target < anticipation or not 0 <= near <= 1:
        raise ValueError('hand_clearance requires 0 < target < anticipation and 0 <= near_weight <= 1')
    if any(v is not None for v in (hand_clearance_weight, hand_clearance_target, hand_clearance_anticipation, hand_clearance_near_weight)):
        config['clearance_objective_parameters'] = dict(weight=scales['wholebody_hand_clearance'],
            target=target, anticipation=anticipation, near_weight=near)
    if config.get('disable_hand_contrast', False):
        if config.get('hand_tolerance_curriculum') or config.get('hand_speed_curriculum'):
            raise ValueError('Box-based speed/tolerance curricula cannot be used with clearance primary')
        for name in ('wholebody_hand_contrast_region', 'wholebody_hand_contrast_heading'):
            if name in scales:
                scales[name] = 0.


def clearance_selection(metrics, history):
    """Cumulative assigned S trials, plus rolling CAT and flat locomotion retention.

    No box compliance. Pending assignments remain in the S denominator. This is
    changing-policy training telemetry, never frozen-checkpoint acceptance.
    """
    key = 'training/passage_acceptance_leader_'
    if not metrics.get(key+'assigned_count', 0) or not metrics.get(key+'clearance_observed_s', 0):
        return None
    if not metrics.get(key+'pass_count', 0) and not metrics.get(key+'fail_count', 0):
        return None
    required = ('success/cat_goal_success_rate', 'selection/flat_walking')
    if any(name not in history for name in required):
        return None
    return .6*metrics[key+'success_rate'] + .2*history[required[0]] + .2*history[required[1]]
