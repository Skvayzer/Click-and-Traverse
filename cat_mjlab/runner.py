"""Continuous training-only mjlab runner with one run and bounded checkpoints.

No evaluators or retention/rollback mechanisms are constructed. Flat-balance
models prioritize protected-core compliance alongside flat walking, smooth
progress and CAT retention success;
other contrastive models use role-balanced leader training success. Remaining
banks use the rollout reward criterion. One overwritten
resume.pt stores learner/Adam, task, simulator and PyTorch RNG state. Changing
physics backend starts a new W&B lineage even when weights/Adam are converted.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, fields, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time
import uuid

import torch

from .checkpoint_arrays import read_array_archive, sampling_state_from_archive
from .conversion import load_array_archive, extract_native_leader
from .learning import Learner, LearnerConfig


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_device_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_device_tree(item, device) for item in value)
    return value


def atomic_torch_save(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Checkpoint destination cannot be a symlink")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(_cpu_tree(value), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


ROLE_NAMES = ("open", "forward_protected", "narrow_passage", "posture_transition")
BODY_PARTS = ("feet", "legs", "trunk", "head", "arms", "hands")


def selection_weights(values):
    values = tuple(float(v) for v in values)
    if len(values) != 4 or any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(sum(values), 1.):
        raise ValueError('Selection requires four finite nonnegative weights summing to 1')
    return values


class SuccessWindow:
    """First-outcome counts, never virtual relabeled SAPG transitions."""
    def __init__(self, maxlen=100):
        self.navigation = deque(maxlen=maxlen)
        self.contrast = deque(maxlen=maxlen)
        self.roles = deque(maxlen=maxlen)
        self.dense = deque(maxlen=maxlen)

    def append(self, navigation, contrast, roles=None, dense=None):
        self.dense.append({key: value.detach().cpu().tolist() for key, value in (dense or {}).items()})
        self.navigation.append(navigation.detach().cpu().tolist())
        self.contrast.append(contrast.detach().cpu().tolist())
        self.roles.append((torch.zeros(4, 2, dtype=torch.long) if roles is None else roles).detach().cpu().tolist())

    def role_balanced_score(self, required_roles=(0, 1, 2, 3)):
        """Equal role weights, only after each role has resolved leader episodes."""
        if not self.roles:
            return None
        counts = torch.tensor(list(self.roles), dtype=torch.int64).sum(dim=0)
        counts = counts[list(required_roles)]
        if bool((counts[:, 0] == 0).any()):
            return None
        return float((counts[:, 1].double() / counts[:, 0]).mean())

    def metrics(self, *, contrastive=False):
        result = {"training/checkpoint_selection_window_length": len(self.roles)}
        if self.roles:
            counts = torch.tensor(list(self.roles)).sum(0)
            for name, (resolved, success) in zip(ROLE_NAMES, counts.tolist()):
                result[f"training/leader_{name}_resolved_count"] = resolved
                result[f"training/leader_{name}_success_count"] = success
        for key in {key for row in self.dense for key in row}:
            numerator, denominator = torch.tensor([row[key] for row in self.dense if key in row], dtype=torch.float64).sum(0).tolist()
            result[key + "_sample_count"] = denominator
            if denominator:
                result[key] = numerator / denominator
        for rows, names in ((self.navigation, ("cat_goal", "ordinary_clutter_goal", "hand_protection_goal", "goal")),
                            (self.contrast, ("forward_protected", "narrow_passage", "posture_transition"))):
            if not rows:
                continue
            counts = torch.tensor(list(rows), dtype=torch.int64).sum(dim=0)
            for name, (resolved, success) in zip(names, counts.tolist()):
                if resolved:
                    result[f"success/{name}_success_rate"] = success / resolved
                result[f"training/{name}_resolved_count"] = resolved
        return result

    def legacy_flat_balance_score(self):
        """Monotone bounded composite; absent evidence is not a zero outcome."""
        metrics = self.metrics()
        keys = ("selection/flat_walking_compliant", "selection/flat_soft_progress",
                "success/cat_goal_success_rate")
        if any(key not in metrics for key in keys):
            return None
        return sum(weight * metrics[key] for weight, key in zip((.5, .25, .25), keys))

    def flat_balance_score(self, weights=(.6, .2, .1, .1)):
        """Protected core posture takes priority; missing evidence defers selection."""
        weights = selection_weights(weights)
        metrics = self.metrics()
        keys = ('selection/protected_core_compliant', 'selection/flat_walking_compliant',
                'selection/flat_soft_progress', 'success/cat_goal_success_rate')
        if any(key not in metrics for key, weight in zip(keys, weights) if weight):
            return None
        return sum(weight * metrics.get(key, 0.) for key, weight in zip(keys, weights))

    def state_dict(self):
        return dict(maxlen=self.navigation.maxlen, navigation=list(self.navigation),
                    contrast=list(self.contrast), roles=list(self.roles), dense=list(self.dense))

    def load_state_dict(self, state):
        if state["maxlen"] != self.navigation.maxlen:
            raise ValueError("Success window configuration differs")
        self.navigation.clear(); self.navigation.extend(state["navigation"])
        self.contrast.clear(); self.contrast.extend(state["contrast"])
        self.roles.clear(); self.roles.extend(state["roles"])
        # Old snapshots have no dense history; do not invent zero observations.
        self.dense.clear(); self.dense.extend(state.get("dense", [{} for _ in self.roles]))


class Logger:
    def __init__(self, directory, *, mode, project, entity, record, resume=False):
        self.directory = Path(directory)
        self.run = None
        self.path = self.directory / "wandb.json"
        if resume:
            self.identity = json.loads(self.path.read_text())
            if any(self.identity[key] != value for key, value in (("mode", mode), ("project", project), ("entity", entity))):
                raise ValueError("W&B destination differs from the saved run")
        else:
            self.identity = dict(id=uuid.uuid4().hex[:8], mode=mode, project=project, entity=entity,
                                 initialized=False, last_global_step=-1)
            atomic_json(self.path, self.identity)
        if mode != "disabled":
            import wandb
            self.run = wandb.init(project=project, entity=entity or None, id=self.identity["id"], mode=mode,
                resume=("must" if resume and self.identity["initialized"] else "allow" if resume else "never"),
                name=self.directory.name, dir=str(self.directory), config=record)
            self.run.define_metric("global_step")
            self.run.define_metric("*", step_metric="global_step")
            self.run.define_metric("success/*", step_metric="global_step", summary="max")
            self.identity.update(initialized=True, url=self.run.url)
            atomic_json(self.path, self.identity)

    def log(self, step, values):
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise FloatingPointError("Refusing nonfinite training metrics")
        # A crash can leave logs ahead of the last durable update. Resume the
        # same identity, suppress stale steps until learner progress catches up.
        if int(step) < self.identity.get("last_global_step", -1):
            return
        event = dict(global_step=int(step), **{key: float(value) for key, value in values.items()})
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if self.run is not None:
            self.run.log(event)
        self.identity["last_global_step"] = int(step)
        atomic_json(self.path, self.identity)
        print(json.dumps(event, sort_keys=True), flush=True)

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


@contextmanager
def stop_requests(directory):
    requested = [False]
    def handler(signum, frame):
        del signum, frame
        requested[0] = True
    previous = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield lambda: requested[0] or (Path(directory) / "STOP").exists()
    finally:
        for sig, value in previous.items():
            signal.signal(sig, value)


def collect_rollout(task, learner, *, unroll_length, trajectories, policy_ids):
    """Collect native [trajectory,time] chunks with bounded preallocated storage.

    The old wrapper exposes autoreset observations to the learner; preserve that
    choice rather than substituting the separately available terminal_obs.
    """
    n = task.num_envs
    if trajectories % n:
        raise ValueError("Trajectory count must be divisible by parallel environments")
    config = learner.config
    shape = (trajectories, unroll_length)
    rollout = {key: torch.empty((*shape, width), device=learner.device) for key, width in (
        ("state", config.actor_obs), ("privileged_state", config.critic_obs),
        ("next_state", config.actor_obs), ("next_privileged_state", config.critic_obs),
        ("raw_action", config.action_size))}
    rollout.update({key: torch.empty(shape, device=learner.device) for key in ("log_prob", "reward", "discount", "truncation")})
    rollout["policy_id"] = torch.empty(shape, dtype=torch.long, device=learner.device)
    if getattr(task,'balance_settings',None) is not None:
        rollout['task_bucket']=torch.empty(shape,dtype=torch.long,device=learner.device)
    before_nav = task.navigation_counts.clone()
    before_contrast = task.contrast_counts.clone()
    before_roles = task.role_counts.clone()
    completed = torch.zeros((), device=learner.device)
    sum_return = completed.clone(); sum_length = completed.clone(); collisions = completed.clone()
    std_sum = completed.clone()
    leg_std = completed.clone(); upper_std = completed.clone()
    outcomes = {}
    dense = {}
    response_counts = {}
    reactive_counts = {'steps': 0., 'active': 0., 'robot_hit': 0., 'object_hit': 0.,
                       'active_gap_sum': 0., 'active_gap_min': float('inf'), 'inside_anticipation': 0.}
    # Per-term reward totals. The task computes every component each step and the loop
    # below used to discard them, so a run could report 867 metrics and not one of them
    # said which term the policy was actually being paid by.
    reward_totals, reward_steps = {}, 0.
    # [attempted, resolved, successful, timed_out, hand_self_contact, fell, length_sum] per scene type.
    bucket_of_scene = torch.as_tensor(_scene_buckets(task.bank.manifest), device=task.device)
    bucket_stats = torch.zeros((len(SCENE_BUCKETS), 7), dtype=torch.float64, device=task.device)
    reactive_bucket = SCENE_BUCKETS.index('reactive_standing')
    # Realized experience per sampling group: steps every step, lengths at episode end.
    sampling_ids = getattr(task.bank, 'sampling_ids', None)
    n_groups = int(task.bank.sampling_masses.numel()) if sampling_ids is not None else 0
    group_steps = torch.zeros(max(n_groups, 1), dtype=torch.float64, device=task.device)
    group_length_sum = torch.zeros_like(group_steps); group_length_count = torch.zeros_like(group_steps)
    reactive_steps = torch.zeros((), dtype=torch.float64, device=task.device)
    objects_state = getattr(getattr(task, 'reactive_objects', None), 'state', None)
    last_active = objects_state['active'].clone() if objects_state is not None else None
    reactive_length_sum = torch.zeros_like(reactive_steps); reactive_length_count = torch.zeros_like(reactive_steps)
    from .balance import BalanceMetrics
    balance_metrics=BalanceMetrics()
    acceptance = getattr(task, 'acceptance_observer', None)
    if acceptance is None and hasattr(getattr(task, 'bank', None), 'acceptance'):
        from .acceptance import TrainingAcceptance
        acceptance = task.acceptance_observer = TrainingAcceptance()
    for begin in range(0, trajectories, n):
        section = slice(begin, begin + n)
        for t in range(unroll_length):
            if acceptance is not None:
                acceptance.before_step(task, policy_ids)
            # task.step mutates task.obs; persist current observations first.
            rollout["state"][section, t].copy_(task.obs["state"])
            rollout["privileged_state"][section, t].copy_(task.obs["privileged_state"])
            acted = learner.act(task.obs, policy_ids=policy_ids)
            if 'task_bucket' in rollout:
                rollout['task_bucket'][section,t].copy_(task.bank.sampling_ids[task.scene_ids])
            transition = task.step(acted["action"])
            if acceptance is not None:
                acceptance.after_step(task, transition)
            rollout["next_state"][section, t].copy_(transition["obs"]["state"])
            rollout["next_privileged_state"][section, t].copy_(transition["obs"]["privileged_state"])
            for key in ("raw_action", "log_prob", "policy_id"):
                rollout[key][section, t].copy_(acted[key])
            rollout["reward"][section, t].copy_(transition["reward"])
            rollout["discount"][section, t].copy_((~transition["done"]).float())
            rollout["truncation"][section, t].copy_(transition["truncated"].float())
            done = transition["done"]
            metrics = transition["metrics"]
            for key, value in metrics.items():
                if key.startswith('response_split/') and not key.endswith('active_count'):
                    name = key.split('/',1)[1]
                    response_counts[name] = response_counts.get(name, 0) + value.sum()
            bucket = bucket_of_scene[metrics['scene_ids']]
            if 'reactive/active' in metrics:
                # Reactive scenes borrow a flat background scene id, so they are
                # indistinguishable by scene_id alone and would be miscounted as flat_balance.
                # Autoreset re-samples the flag before metrics are read, so an episode that
                # just ENDED must be classified by last step's flag, not this step's.
                active_now = metrics['reactive/active'].bool()
                was_active = torch.where(done, last_active, active_now) if last_active is not None else active_now
                last_active = active_now
                bucket = torch.where(was_active, reactive_bucket, bucket)
            # Counted on different events on purpose: a verdict can land hundreds of steps
            # before the episode ends, so terminations and resolutions are not the same
            # population and must not be divided by one another within an update.
            ended = done.bool()
            verdict = metrics['resolved'].bool()
            undecided = ended & ~metrics['outcome_counted'].bool() & ~verdict
            touched = ended & metrics.get('episode/hand_self_contact', torch.zeros_like(done)).bool()
            fell = ended & metrics.get('episode/fall', torch.zeros_like(done)).bool()
            for column, flag in enumerate((ended, verdict, metrics['successful'].bool(), undecided, touched, fell)):
                bucket_stats[:, column].scatter_add_(0, bucket, flag.double())
            bucket_stats[:, 6].scatter_add_(0, bucket, torch.where(done, metrics['episode_length'], 0.).double())
            for key, value in metrics.items():
                if key.startswith('reward/'):
                    reward_totals[key] = reward_totals.get(key, 0.) + value.sum()
            reward_steps += float(done.numel())
            if 'reactive/active' in metrics:
                active = metrics['reactive/active'].bool()
                reactive_counts['steps'] += float(active.numel())
                reactive_counts['active'] += float(active.sum())
                reactive_counts['robot_hit'] += float((metrics['reactive/robot_initiated_contact'] & active).sum())
                reactive_counts['object_hit'] += float((metrics['reactive/object_initiated_contact'] & active).sum())
                # Episode-level: did the robot leave its spot (>0.3 m) instead of moving the hand?
                ended_reactive = done & was_active
                reactive_counts['ended'] = reactive_counts.get('ended', 0.) + float(ended_reactive.sum())
                reactive_counts['left_spot'] = reactive_counts.get('left_spot', 0.) + float((ended_reactive & metrics.get('episode/reactive_left_spot', torch.zeros_like(done)).bool()).sum())
                reactive_counts['displacement_sum'] = reactive_counts.get('displacement_sum', 0.) + float((metrics['reactive/root_displacement_m'] * active).sum())
                if active.any() and 'acceptance/hand_clearance' in metrics:
                    gap = metrics['acceptance/hand_clearance'].amin(-1)[active]
                    if 'reactive/target_clearance' in metrics:
                        margin = gap - metrics['reactive/target_clearance'][active]
                        reactive_counts['margin_sum'] = reactive_counts.get('margin_sum', 0.) + float(margin.sum())
                        reactive_counts['margin_min'] = min(reactive_counts.get('margin_min', float('inf')),
                                                            float(margin.amin()))
                    reactive_counts['active_gap_sum'] += float(gap.sum())
                    reactive_counts['active_gap_min'] = min(reactive_counts['active_gap_min'], float(gap.amin()))
                    reactive_counts['inside_anticipation'] += float((gap < .20).sum())
            completed += done.sum()
            sum_return += torch.where(done, metrics["episode_return"], 0.).sum()
            sum_length += torch.where(done, metrics["episode_length"], 0.).sum()
            if n_groups:
                # Reactive envs borrow the flat background scene id; count them on their own,
                # not against the flat group.
                goal = ~was_active if 'reactive/active' in metrics else torch.ones_like(done, dtype=torch.bool)
                group = sampling_ids[metrics["scene_ids"]]
                ended = torch.where(done, metrics["episode_length"], 0.).double()
                group_steps.scatter_add_(0, group, goal.double())
                group_length_sum.scatter_add_(0, group, ended * goal)
                group_length_count.scatter_add_(0, group, (done & goal).double())
                reactive_steps += (~goal).sum(); reactive_length_sum += (ended * ~goal).sum()
                reactive_length_count += (done & ~goal).sum()
            collisions += (done & metrics.get("episode/body_collision", torch.zeros_like(done))).sum()
            std_sum += acted["action_std"].sum()
            leg_std += acted["action_std"][..., :12].sum()
            upper_std += acted["action_std"][..., 12:].sum()
            leader = acted["policy_id"] == 0
            balance_metrics.append(metrics,done,leader,getattr(task,"dt",.02))
            if 'protected_core' in metrics:
                core = leader & metrics['protected_core']
                for key, value in (
                    ('selection/protected_core_compliant', metrics.get('corridor_selection_good', metrics['hand_contrast_hand_good'])),
                    ('training/protected_core_reward_floor_clipping_fraction', metrics['reward_floor_clipped'])):
                    pair = torch.stack((torch.where(core, value, 0.).sum(), core.sum()))
                    dense[key] = dense.get(key, torch.zeros_like(pair)) + pair
                for metric in ('reward_pre_floor_negative', 'reward_soft_floor_lift', 'reward_floor_slope', 'corridor_arm_clearance', 'corridor_pose_good', 'corridor_selection_good'):
                    if metric in metrics:
                        pair = torch.stack((torch.where(core, metrics[metric], 0.).sum(), core.sum()))
                        key = 'training/protected_core_' + metric
                        dense[key] = dense.get(key, torch.zeros_like(pair)) + pair
            masks = {"": torch.ones_like(done), "leader_": leader}
            if "contrast_role" in metrics:
                masks.update({f"leader_{name}_": leader & (metrics["contrast_role"] == role)
                              for role, name in enumerate(ROLE_NAMES)})
                if 'hand_raised_seeded' in metrics:
                    protected=leader & (metrics['contrast_role']==1)
                    seeded=metrics['hand_raised_seeded']
                    masks.update(leader_forward_protected_seeded_=protected&seeded,
                        leader_forward_protected_unseeded_=protected&~seeded,
                        leader_forward_protected_seeded_after100_=protected&seeded&(metrics['episode_length']>=100))
                if "hand_scene_kind" in metrics:
                    masks.update({f"leader_{name}_":leader & (metrics['hand_scene_kind']==kind)
                                  for kind,name in ((1,'table'),(2,'shelf'))})
            for prefix, mask in masks.items():
                counts = torch.stack(((done & mask).sum(),
                    (done & mask & metrics.get("episode/body_collision", torch.zeros_like(done))).sum(),
                    (transition["truncated"] & mask).sum(),
                    *((done & mask & metrics.get("episode/body_collision_" + part,
                        torch.zeros_like(done))).sum() for part in BODY_PARTS)))
                outcomes[prefix] = outcomes.get(prefix, torch.zeros_like(counts)) + counts
                if not prefix or "contrast_role" not in metrics:
                    continue
                active = mask & metrics["contrast_unresolved"]
                for name, value, eligible in (
                    ("heading_compliance_fraction", metrics["hand_contrast_heading_good"], metrics["contrast_heading_active"]),
                    ("hand_region_compliance_fraction", metrics["hand_contrast_hand_good"], metrics["contrast_hand_active"]),
                    ("mean_heading_cost", metrics["hand_contrast_heading_cost"], metrics["contrast_heading_active"]),
                    ("mean_region_cost", metrics["hand_contrast_region_cost"], metrics["contrast_hand_active"])):
                    selected = active & eligible
                    pair = torch.stack((torch.where(selected, value, 0.).sum(), selected.sum()))
                    key = f"training/{prefix}hand_contrast_{name}"
                    dense[key] = dense.get(key, torch.zeros_like(pair)) + pair
                pair = torch.stack((torch.where(active, metrics["contrast_zone_seen"], 0).sum(),
                                    torch.where(active, metrics["contrast_zone_required"], 0).sum()))
                key = f"training/{prefix}hand_contrast_zone_reached_progress_ratio"
                dense[key] = dense.get(key, torch.zeros_like(pair)) + pair
    info = dict(navigation_counts=task.navigation_counts - before_nav,
                contrast_counts=task.contrast_counts - before_contrast,
                role_counts=task.role_counts - before_roles,
                action_std=std_sum / (trajectories * unroll_length * config.action_size))
    info["metrics"] = {"training/completed_episode_count": float(completed), "training/action_std": float(info["action_std"])}
    info["dense"] = dense
    info['metrics'].update(balance_metrics.result())
    from .response_split import summaries
    info['metrics'].update({'training/response_split/'+k:v for k,v in
        summaries({k:float(v) for k,v in response_counts.items()}).items()})
    stats = bucket_stats.cpu().numpy()
    for index, name in enumerate(SCENE_BUCKETS):
        ended, resolved, successful, undecided, touched, fell, length_sum = (float(x) for x in stats[index])
        decided = resolved + undecided
        if not (ended or resolved):
            continue
        # success_rate keeps the legacy semantics (of episodes that DECIDED); completion_rate
        # is the honest one (of every episode started), since 37-43% decide nothing and are
        # silently dropped from the legacy denominator.
        info['metrics'][f'scene/{name}/episodes_ended'] = ended
        info['metrics'][f'scene/{name}/resolved'] = resolved
        info['metrics'][f'scene/{name}/hand_self_contact_rate'] = touched / ended if ended else 0.
        info['metrics'][f'scene/{name}/fall_rate'] = fell / ended if ended else 0.
        info['metrics'][f'scene/{name}/mean_episode_length'] = length_sum / ended if ended else 0.
        if name in GOALLESS_BUCKETS:
            # Standing scenes have no goal, so a success rate would be a constant 0 that
            # reads as failure. Their outcomes live under reactive/ and balance/ instead.
            continue
        info['metrics'][f'scene/{name}/success_rate'] = successful / resolved if resolved else 0.
        # Denominator is every episode that finished, whether or not it reached a verdict,
        # so an episode that ran out the clock counts against success instead of vanishing.
        info['metrics'][f'scene/{name}/completion_rate'] = successful / decided if decided else 0.
        info['metrics'][f'scene/{name}/undecided_rate'] = undecided / decided if decided else 0.
    if reward_steps:
        # Emitted twice: raw per-step magnitude for debugging, and a share of the total
        # positive/negative mass so it is obvious at a glance what dominates the reward.
        per_step = {k: float(v) / reward_steps for k, v in reward_totals.items()}
        positive = sum(v for v in per_step.values() if v > 0) or 1.
        negative = -sum(v for v in per_step.values() if v < 0) or 1.
        for key, value in sorted(per_step.items()):
            term = key.split('/', 1)[1]
            info['metrics']['reward_term/' + term] = value
            info['metrics']['reward_share/' + term] = value / (positive if value > 0 else negative)
        info['metrics']['reward_term_total/positive_per_step'] = positive
        info['metrics']['reward_term_total/negative_per_step'] = -negative
        info['metrics']['reward_term_total/net_per_step'] = positive - negative
    if reactive_counts['steps']:
        active = reactive_counts['active']
        info['metrics'].update({
            'reactive/active_fraction': active / reactive_counts['steps'],
            'reactive/active_env_steps': active,
            'reactive/robot_initiated_contact_rate': reactive_counts['robot_hit'] / max(active, 1.),
            'reactive/left_spot_rate': reactive_counts.get('left_spot', 0.) / max(reactive_counts.get('ended', 0.), 1.),
            'reactive/root_displacement_mean_m': reactive_counts.get('displacement_sum', 0.) / max(active, 1.),
            'reactive/object_initiated_contact_rate': reactive_counts['object_hit'] / max(active, 1.),
            'reactive/mean_hand_gap_m': reactive_counts['active_gap_sum'] / max(active, 1.),
            'reactive/min_hand_gap_m': (reactive_counts['active_gap_min']
                                        if reactive_counts['active_gap_min'] != float('inf') else float('nan')),
            'reactive/inside_anticipation_fraction': reactive_counts['inside_anticipation'] / max(active, 1.),
            # Positive means the hand ended up further from the object than the scene placed it,
            # i.e. the robot actively retreated. Zero means it did nothing. This is the priority-#1
            # number; inside_anticipation_fraction is only a validity check that objects arrive.
            'reactive/avoidance_margin_mean_m': reactive_counts.get('margin_sum', 0.) / max(active, 1.),
            'reactive/avoidance_margin_min_m': (reactive_counts.get('margin_min', float('inf'))
                                                if reactive_counts.get('margin_min', float('inf')) != float('inf')
                                                else float('nan'))})
    if 'response_split/active_count' in metrics:
        info['metrics']['training/response_split/pending_event_count'] = float(metrics['response_split/active_count'].sum())
    if acceptance is not None:
        info['metrics'].update(acceptance.metrics())
    if balance_metrics.counts is not None:
        counts = balance_metrics.counts
        dense['selection/flat_walking'] = torch.stack((counts[1], counts[0]))
        dense['selection/flat_walking_compliant'] = torch.stack((counts[2], counts[0]))
        # Remove reward scale/dt: bounded proximity times forward-speed factor.
        scale = task.balance_reward['bonus_scale'] * task.dt
        dense['selection/flat_soft_progress'] = torch.stack((counts[10] / scale, counts[0]))
    samples = trajectories * unroll_length
    info["metrics"]["training/action_std_legs"] = float(leg_std / (samples * min(12, config.action_size)))
    if config.action_size > 12:
        info["metrics"]["training/action_std_upper_body"] = float(upper_std / (samples * (config.action_size - 12)))
    for prefix, counts in outcomes.items():
        total, collision, timeout, *body_parts = counts.tolist()
        for name, value in (("completed_episode_count", total), ("body_collision_count", collision), ("timeout_count", timeout)):
            info["metrics"][f"training/{prefix}{name}"] = value
        if total:
            info["metrics"][f"training/{prefix}body_collision_rate"] = collision / total
            info["metrics"][f"training/{prefix}timeout_rate"] = timeout / total
        for part, count in zip(BODY_PARTS, body_parts):
            info["metrics"][f"training/{prefix}body_collision_{part}_count"] = count
            if total:
                info["metrics"][f"training/{prefix}body_collision_{part}_rate"] = count / total
    if bool(completed > 0):
        info["metrics"].update({"episode/return": float(sum_return / completed),
                               "episode/length": float(sum_length / completed),
                               "training/body_collision_rate": float(collisions / completed)})
    if getattr(getattr(task,'bank',None),'width_curriculum',None) is not None:
        info['metrics']['training/width_curriculum_stage']=int(task.curriculum_stage)
        for rung,(completed_count,success_count) in enumerate(zip(task.curriculum_completed.tolist(),task.curriculum_goals.tolist())):
            prefix=f'training/leader_narrow_rung_{rung}'
            info['metrics'][prefix+'_resolved_count']=completed_count
            info['metrics'][prefix+'_success_count']=success_count
            if completed_count:info['metrics'][prefix+'_success_rate']=success_count/completed_count
    elif getattr(getattr(task,'bank',None),'levels',None) is not None:
        info['metrics']['hand_curriculum/stage']=int(task.curriculum_stage)
        for level,(completed_count,success_count) in enumerate(zip(task.curriculum_completed.tolist(),task.curriculum_goals.tolist())):
            prefix=f'hand_curriculum/level_{level}'
            info['metrics'][prefix+'_resolved_count']=completed_count
            info['metrics'][prefix+'_clean_goal_count']=success_count
            if completed_count:info['metrics'][prefix+'_clean_goal_rate']=success_count/completed_count
    if hasattr(task, 'tolerance_state'):
        from .tolerance_curriculum import metrics
        info['metrics'].update(metrics(task.tolerance_state))
    if hasattr(task, 'speed_state'):
        from .speed_curriculum import metrics
        info['metrics'].update(metrics(task.speed_state))
    if n_groups:
        objects = getattr(task, 'reactive_objects', None)
        reactive = None if objects is None else (float(reactive_steps), float(reactive_length_sum), float(reactive_length_count), objects)
        _adapt_experience_masses(task.bank, group_steps, group_length_sum, group_length_count, info, reactive)
    return rollout, info


def learner_config_from_archive(metadata, *, algorithm=None):
    source = metadata.get("contract", {})
    algorithm = algorithm or ("sapg" if source.get("sapg") else "ppo")
    accepted = {field.name for field in fields(LearnerConfig)}
    kwargs = {key: value for key, value in source.items() if key in accepted}
    kwargs.update(source.get("sapg", {}))
    kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    kwargs["algorithm"] = algorithm
    return LearnerConfig(**kwargs)


def environment_config_from_archive(metadata):
    source = metadata.get("contract", {})
    recorded = source.get("metadata", {}).get("config", {})
    config = recorded.get("env_config")
    if config is None:
        config = json.loads((ROOT / "configs/cat_generalist_released.json").read_text())["env_config"]
    randomize = source.get("randomize_initial_episode_steps",
                           recorded.get("policy_config", {}).get("randomize_initial_episode_steps", True))
    return dict(config, randomize_initial_episode_steps=randomize)


GOALLESS_BUCKETS = ('reactive_standing', 'flat_balance')
SCENE_BUCKETS = ('procedural_cat', 'original_cat', 'published_cat',
                 'clutter_dense', 'clutter_pilot', 'clutter_legacy',
                 'furniture_dense', 'furniture_pilot', 'furniture_legacy',
                 'narrow_passage', 'protected_passage', 'transition_passage',
                 'open_passage', 'flat_balance', 'reactive_standing')


def _scene_buckets(manifest):
    """One reporting bucket per scene TYPE, not per sampling group.

    The three navigation groups collapse 2290 CAT scenes into a single number and mix
    clutter rooms with furniture rooms, so "what is actually failing" was unanswerable
    from the logs: a flat cat_goal could be procedural improving while the 37 authored
    scenes rot, and nobody could see it. These buckets are for reading, not sampling.
    """
    import numpy as _np
    ids = []
    for scene in manifest['scenes']:
        family = scene['family']
        source = scene.get('source', {}) or {}
        role = (source.get('hand_contrast') or {}).get('role')
        difficulty = source.get('difficulty')
        if scene['scene_id'] == 'flat-balance-walk-v1':
            name = 'flat_balance'
        elif role:
            name = {'narrow': 'narrow_passage', 'forward_protected': 'protected_passage',
                    'transition': 'transition_passage', 'open': 'open_passage'}.get(role, 'open_passage')
        elif family in ('generic_clutter', 'furniture'):
            stem = 'clutter' if family == 'generic_clutter' else 'furniture'
            name = f'{stem}_{difficulty}' if difficulty in ('dense', 'pilot') else f'{stem}_legacy'
        else:
            name = family
        ids.append(SCENE_BUCKETS.index(name))
    return _np.asarray(ids, dtype=_np.int64)


def _solve_experience_masses(shares, lengths):
    """Reset mass per group from a target share of TRANSITIONS: share / mean episode length."""
    resolved = [float(share) / max(float(length), 1.) for share, length in zip(shares, lengths)]
    total = sum(resolved)
    if not total > 0:
        raise ValueError('Experience masses must sum to a positive value')
    return [r / total for r in resolved]


def _adapt_experience_masses(bank, step_count, length_sum, length_count, info, reactive=None):
    """Fold this update's realized lengths into the bank's estimate and periodically re-solve
    the reset masses, so each group's share of transitions tracks its target instead of
    drifting with episode length. Groups with under 20 finished episodes since the last
    re-solve keep their previous estimate. Always logs the realized share so the drift is
    visible whether or not re-solving is enabled."""
    targets = getattr(bank, 'experience_targets', None)
    if not targets:
        return
    steps = step_count.cpu().tolist(); sums = length_sum.cpu().tolist(); counts = length_count.cpu().tolist()
    r_target = getattr(bank, 'experience_reactive_target', None)
    r_steps = 0.
    if reactive is not None:
        r_steps, r_sum, r_count, objects = reactive
        bank.experience_reactive_sum += r_sum; bank.experience_reactive_count += r_count
        info['metrics']['balance/reactive_experience_share'] = r_steps / ((sum(steps) + r_steps) or 1.)
        if r_target is not None:
            info['metrics']['balance/reactive_experience_target'] = r_target
    total = (sum(steps) + r_steps) or 1.
    for index, target in enumerate(targets):
        bank.experience_length_sum[index] += sums[index]
        bank.experience_length_count[index] += counts[index]
        info['metrics'][f'balance/group{index}_experience_share'] = steps[index] / total
        info['metrics'][f'balance/group{index}_experience_target'] = target
    every = getattr(bank, 'experience_rebalance_every', 0)
    bank.experience_updates += 1
    if not every or bank.experience_updates % every:
        return
    for index in range(len(targets)):
        if bank.experience_length_count[index] >= 20:
            realized = bank.experience_length_sum[index] / bank.experience_length_count[index]
            bank.experience_lengths[index] = .7 * bank.experience_lengths[index] + .3 * realized
        bank.experience_length_sum[index] = bank.experience_length_count[index] = 0.
    if reactive is not None and r_target is not None and bank.experience_reactive_count >= 20:
        realized = bank.experience_reactive_sum / bank.experience_reactive_count
        previous = bank.experience_reactive_length
        bank.experience_reactive_length = realized if previous is None else .7 * previous + .3 * realized
        bank.experience_reactive_sum = bank.experience_reactive_count = 0.
    if reactive is not None and r_target is not None and bank.experience_reactive_length:
        # One categorical over groups + reactive, then split: the reactive coin is flipped
        # first, so group masses are conditional on the coin coming up "goal episode".
        scale = 1. - float(r_target)
        joint = _solve_experience_masses([t * scale for t in targets] + [float(r_target)],
                                         bank.experience_lengths + [bank.experience_reactive_length])
        fraction = joint[-1]
        resolved = [m / max(1. - fraction, 1e-9) for m in joint[:-1]]
        objects.reactive_fraction = fraction
        info['metrics']['balance/reactive_reset_fraction'] = fraction
        info['metrics']['balance/reactive_episode_length'] = bank.experience_reactive_length
    else:
        resolved = _solve_experience_masses(targets, bank.experience_lengths)
    bank.sampling_masses = torch.as_tensor(resolved, dtype=bank.sampling_masses.dtype,
                                           device=bank.sampling_masses.device)
    for index, mass in enumerate(resolved):
        info['metrics'][f'balance/group{index}_episode_length'] = bank.experience_lengths[index]
        info['metrics'][f'balance/group{index}_reset_mass'] = mass


def _rebalance_bank(bank, args):
    """Apply sampling and episode-length overrides after the bank is built.

    These are deliberately runtime overrides rather than edits to MASSES or to the scene
    records: both are pinned by manifest hashes that checkpoints depend on. The defaults
    they correct are severe. Group 5 is a SINGLE empty scene holding 0.225 of reset mass,
    which measured out at 37.8% of all experience, while the 461 real clutter rooms shared
    0.03375 between them. Separately, CAT scenes run 1000 steps against 4000 for rooms, so
    CAT families received 22.7% of experience against 60% of declared mass.
    """
    import numpy as _np
    masses = getattr(args, 'sampling_masses', None)
    if masses:
        if bank.sampling_masses is None:
            raise ValueError('This bank has no grouped sampler to rebalance')
        if len(masses) != len(bank.sampling_masses):
            raise ValueError(f'Expected {len(bank.sampling_masses)} sampling masses, got {len(masses)}')
        total = float(sum(masses))
        if not total > 0:
            raise ValueError('Sampling masses must sum to a positive value')
        bank.sampling_masses = torch.as_tensor([m / total for m in masses],
                                               dtype=bank.sampling_masses.dtype,
                                               device=bank.sampling_masses.device)
    if getattr(args, 'narrow_sampling_group', None) is not None and bank.sampling_ids is not None:
        # The 50 continuously-sampled narrow scenes were emitted into generic_clutter and so
        # drew from a 0.034 budget shared with 461 rooms, 11x less per scene than the 12
        # fixed-width originals they were meant to supersede.
        target = int(args.narrow_sampling_group)
        narrow = torch.as_tensor([bool('-narrow-' in r['scene_id']) for r in bank.manifest['scenes']],
                                 device=bank.sampling_ids.device)
        bank.sampling_ids = torch.where(narrow, target, bank.sampling_ids)
    experience = getattr(args, 'experience_masses', None)
    if experience:
        # Solve reset mass from a target share of EXPERIENCE. Sampling masses govern resets,
        # but a group's share of transitions is mass * episode_length, so groups with short
        # episodes are silently starved: CAT ran 1000 steps against 4000 for rooms and drew
        # 22.7% of experience on 60% of declared mass. Lengthening CAT episodes would have
        # "fixed" the ratio by paying CAT to idle past a goal that does not end the episode,
        # so correct the mass instead and leave every task's horizon alone. Doing it here
        # also makes the drift structural rather than a constant someone has to re-tune.
        if bank.sampling_masses is None or bank.sampling_ids is None:
            raise ValueError('This bank has no grouped sampler to rebalance')
        if len(experience) != len(bank.sampling_masses):
            raise ValueError(f'Expected {len(bank.sampling_masses)} experience masses, got {len(experience)}')
        groups = bank.sampling_ids.detach().cpu().numpy()
        lengths = bank.episode_lengths.detach().cpu().numpy().astype(float)
        per_group = [float(lengths[groups == i].mean()) if (groups == i).any() else 1. for i in range(len(experience))]
        bank.sampling_masses = torch.as_tensor(_solve_experience_masses(experience, per_group),
                                               dtype=bank.sampling_masses.dtype,
                                               device=bank.sampling_masses.device)
        # The configured horizon is only a first guess at how long episodes really run. Real
        # episodes end early on collisions and goals, and how early moves with the policy:
        # on the 2026-09-24 run CAT slid from 66% to 37% of samples in 43 updates against a
        # 59% target while rooms stopped falling and ran to their full 4000 steps. So keep
        # the targets and a per-group length estimate on the bank and let collect_rollout
        # re-solve from REALIZED lengths (see _adapt_experience_masses).
        bank.experience_targets = [float(s) for s in experience]
        bank.experience_lengths = per_group
        bank.experience_rebalance_every = int(getattr(args, 'experience_rebalance_every', None) or 0)
        bank.experience_updates = 0
        bank.experience_length_sum = [0.] * len(experience)
        bank.experience_length_count = [0.] * len(experience)
        # Reactive standing episodes are drawn BEFORE the group sampler (a fixed .25 of resets
        # in reactive.py) and outlive every goal episode, so they grew from 26% to 64% of all
        # env-steps over 66 updates on the 2026-09-24 run. Solve their reset fraction too.
        bank.experience_reactive_target = getattr(args, 'experience_reactive_share', None)
        bank.experience_reactive_length = None
        bank.experience_reactive_sum = bank.experience_reactive_count = 0.
    length = getattr(args, 'cat_episode_length', None)
    if length:
        cat = torch.as_tensor([r.get('task_kind') == 'cat' for r in bank.manifest['scenes']],
                              device=bank.episode_lengths.device)
        bank.episode_lengths = torch.where(cat, int(length), bank.episode_lengths)


def create_task(args, *, environment_config=None):
    from cat_ppo.furniture.contrast_preflight import contrast_preflight
    contrast_preflight(args.bank_manifest,
        require_hand_contrast=True if getattr(args, "require_hand_contrast", False) else None)
    from .sim import CATSimulation
    from .scene_bank import SceneBank
    from .collision import CollisionChecker
    from .config import wholebody_config
    from .task import CATTask
    sim = CATSimulation(args.num_envs, device=args.device, nconmax=args.nconmax, njmax=args.njmax)
    bank = SceneBank(args.bank_manifest, device=args.device, reset_manifest=args.body_collision_resets,
                     collision_manifest=args.body_collision_bank, passage_rewards=getattr(args, "passage_rewards", None))
    collision = CollisionChecker(sim.model, args.body_collision_bank, field_manifest=args.bank_manifest, device=args.device)
    _rebalance_bank(bank, args)
    config = wholebody_config(environment_config, stabilization=True, hand_protection=True, hand_contrast=bank.has_contrast,
        hand_curriculum_success_threshold=getattr(args, 'hand_curriculum_success_threshold', None),
        disable_hand_contrast=getattr(args, 'disable_hand_contrast', None),
        hand_clearance_weight=getattr(args, 'hand_clearance_weight', None),
        arm_clearance_weight=getattr(args, 'arm_clearance_weight', None),
        tracking_root_field_weight=getattr(args, 'tracking_root_field_weight', None),
        heading_align_weight=getattr(args, 'heading_align_weight', None),
        upright_weight=getattr(args, 'upright_weight', None), stand_tall_weight=getattr(args, 'stand_tall_weight', None),
        torso_rate_weight=getattr(args, 'torso_rate_weight', None), self_clearance_weight=getattr(args, 'self_clearance_weight', None),
        upper_posture_weight=getattr(args, 'upper_posture_weight', None), upper_home_shoulder_pitch=getattr(args, 'upper_home_shoulder_pitch', None),
        terminate_on_hand_self_contact=getattr(args, 'terminate_on_hand_self_contact', None),
        stand_still_weight=getattr(args, 'stand_still_weight', None), standing_requires_stillness=getattr(args, 'standing_requires_stillness', None),
        standing_gf_bonus=getattr(args, 'standing_gf_bonus', None),
        reactive_hand_guidance=getattr(args, 'reactive_hand_guidance', None),
        handsdf_weight=getattr(args, 'handsdf_weight', None),
        hand_clearance_target=getattr(args, 'hand_clearance_target', None),
        hand_clearance_anticipation=getattr(args, 'hand_clearance_anticipation', None),
        hand_clearance_near_weight=getattr(args, 'hand_clearance_near_weight', None),
        hand_contrast_region_weight=getattr(args, 'hand_contrast_region_weight', None),
        hand_reward_soft_floor=getattr(args, 'hand_reward_soft_floor', None),
        hand_contrast_heading_weight=getattr(args, 'hand_contrast_heading_weight', None),
        hand_contrast_approach_distance=getattr(args, 'hand_contrast_approach_distance', None),
        hand_raised_reset_fraction=getattr(args, 'hand_raised_reset_fraction', None) or 0.)
    if config.get('disable_hand_contrast',False) and (getattr(args,'hand_speed_curriculum',False) or getattr(args,'hand_tolerance_curriculum',False)):
        raise ValueError('Box-based speed/tolerance curricula cannot be used with clearance primary')
    from .upper_control import configure as configure_upper
    configure_upper(config, args)
    from .tolerance_curriculum import configure
    configure(config, enabled=getattr(args, 'hand_tolerance_curriculum', False),
              window=getattr(args, 'hand_tolerance_curriculum_window', 256))
    from .speed_curriculum import configure as configure_speed
    configure_speed(config, enabled=getattr(args, 'hand_speed_curriculum', False),
        window=getattr(args, 'hand_speed_curriculum_window', 256),
        threshold=getattr(args, 'hand_speed_curriculum_success_threshold', .8),
        demote_threshold=getattr(args, 'hand_speed_curriculum_demote_threshold', .4))
    config["wholebody"]["body_collision"].update(enabled=True, bank_manifest=str(args.body_collision_bank),
                                                 reset_manifest=str(args.body_collision_resets))
    from .balance import reward_overrides
    config['checkpoint_selection_weights'] = selection_weights(getattr(args, 'checkpoint_selection_weights', None) or config.get('checkpoint_selection_weights', (.6,.2,.1,.1)))
    resolved=reward_overrides(bank.balance_settings,config.get('flat_balance_reward'),
        bonus_scale=getattr(args,'flat_bonus_scale',None),region_scale=getattr(args,'flat_region_scale',None))
    if resolved is not None:config['flat_balance_reward']=resolved
    objects = None
    reactive_manifest = getattr(args, 'reactive_bank', None)
    if reactive_manifest is not None:
        from .reactive import StandingObjects, validate_bank
        meta = validate_bank(json.loads(Path(reactive_manifest).read_text()))
        pinned = Path(meta['base_bank']['path']).resolve()
        if pinned != Path(args.bank_manifest).resolve():
            raise ValueError(f'Reactive bank pins base bank {pinned}, but --bank-manifest is '
                             f'{Path(args.bank_manifest).resolve()}; the approaching-object scenes would '
                             f'reference a different scene distribution than training loads')
        objects = StandingObjects(bank=str(reactive_manifest), num_envs=len(sim.data.qpos),
                                  model=sim.model, device=sim.data.qpos.device)
    task = CATTask(sim, bank, config, collision=collision, seed=args.seed, analytic_objects=objects)
    if args.compile_task:
        task.enable_compilation()
    return task, sim, config


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity():
    paths = sorted((ROOT / "cat_mjlab").glob("*.py")) + [ROOT / "train_cat_mjlab.py",
        ROOT / "cat_ppo/furniture/control.py"]
    hashes = {str(path.relative_to(ROOT)): _file_hash(path) for path in paths if path.is_file()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _versions():
    result = {}
    for name in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def native_initialization(path):
    """Memory-map old native snapshots; extract only config and model weights.

    Never restore Adam, counters, sampler, simulator, RNG or W&B lineage here.
    Exact continuation remains exclusively the strict --resume path.
    """
    snapshot = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    from .checkpoint_upgrade import validate_upgrade
    validate_upgrade(snapshot)
    from cat_ppo.furniture.control import mjlab_observation_contract
    if snapshot.get('observation_contract', mjlab_observation_contract()) != mjlab_observation_contract():
        raise ValueError('Native checkpoint observation contract differs; use --from-scratch, no conversion is performed')
    if snapshot.get("schema") == "cat-mjlab-runtime-v1":
        source = snapshot["learner"]
    elif snapshot.get("schema") == "cat-mjlab-best-v1":
        source = snapshot
    else:
        raise ValueError("Expected a native mjlab runtime or best checkpoint")
    return source["config"], source["model"], snapshot["contract"]["environment_config"]


def run(args):
    directory = Path(args.run_dir).resolve()
    if args.max_updates < 0 or args.checkpoint_interval_updates < 1:
        raise ValueError("Invalid update/checkpoint interval")
    if args.command == "verify" and (args.max_updates < 1 or args.wandb_mode != "disabled"):
        raise ValueError("Verification must be bounded and W&B disabled")
    if args.resume and args.fresh_optimizer:
        raise ValueError("--fresh-optimizer applies only to fresh initialization, not --resume")
    native_path = getattr(args, "checkpoint_native", None)
    from_scratch = getattr(args, "from_scratch", False)
    initial_std = getattr(args, "init_action_std", None)
    if not args.resume and (native_path is not None or initial_std is not None) and not args.fresh_optimizer:
        raise ValueError("Native warm start / sigma reset requires --fresh-optimizer")
    if (directory / "STOP").exists():
        raise ValueError("STOP exists; retire it deliberately before resuming")
    if directory.is_symlink():
        raise ValueError("Run directory cannot be a symlink")
    if not args.resume:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Run directory is not empty; use --resume for a complete native resume")
        directory.mkdir(parents=True, exist_ok=True)
    elif not directory.is_dir():
        raise ValueError("Resume directory does not exist")
    native_weights = None
    if from_scratch:
        config = LearnerConfig(algorithm=args.algorithm or 'ppo',
                               num_policies=6 if args.algorithm == 'sapg' else 1)
        metadata, archive_arrays, source, config_source = {}, [], {}, None
    elif native_path is not None:
        source_config, native_weights, config_source = native_initialization(native_path)
        config = LearnerConfig(**source_config)
        if args.algorithm is not None:
            config = replace(config, algorithm=args.algorithm)
        if source_config['algorithm'] == 'sapg' and config.algorithm == 'ppo':
            native_weights = extract_native_leader(native_weights, source_config)
            config = replace(config, num_policies=1)
        metadata, archive_arrays, source = {}, [], {}
    else:
        metadata, archive_arrays = read_array_archive(args.checkpoint_npz)
        config = learner_config_from_archive(metadata, algorithm=args.algorithm)
        source = metadata.get("contract", {})
        config_source = environment_config_from_archive(metadata)
    from .observation_contract import ACTOR_SIZE, CRITIC_SIZE
    if (config.actor_obs, config.critic_obs, config.action_size) != (ACTOR_SIZE, CRITIC_SIZE, 29):
        raise ValueError('Native training requires actor 222 / critic 310 and 29 actions. '
                         'Use --from-scratch; obsolete observation contracts are not converted.')
    overrides = {name: getattr(args, name) for name in ("max_action_std", "discounting", "num_minibatches")
                 if getattr(args, name, None) is not None}
    if overrides.get("max_action_std") == 0:
        overrides["max_action_std"] = None
    config = replace(config, **overrides)
    batch_size = args.batch_size if args.batch_size is not None else source.get("batch_size", args.num_envs // config.num_minibatches)
    unroll = args.unroll_length if args.unroll_length is not None else source.get("unroll_length", 32)
    trajectories = batch_size * config.num_minibatches
    if min(batch_size, unroll, args.num_envs) <= 0:
        raise ValueError("batch_size, unroll_length and num_envs must be positive")
    if trajectories % args.num_envs:
        raise ValueError(f"batch_size*num_minibatches ({batch_size}*{config.num_minibatches}={trajectories}) "
                         f"must be divisible by num_envs ({args.num_envs})")
    if config.algorithm == "sapg" and (args.num_envs % config.num_policies or batch_size % config.num_policies):
        raise ValueError(f"SAPG num_envs ({args.num_envs}) and batch_size ({batch_size}) "
                         f"must be divisible by num_policies ({config.num_policies})")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    learner = Learner(config, device=args.device)
    if native_weights is not None:
        if any(not bool(torch.isfinite(value).all()) for value in native_weights.values()):
            raise ValueError("Native source has nonfinite model weights")
        learner.model.load_state_dict(native_weights, strict=True)
        migration = dict(kind="native_weights_only", optimizer="fresh", exact_runtime_resume=False,
                         sampling="fresh", counters="zero", environment="fresh")
        if source_config['algorithm'] == 'sapg' and config.algorithm == 'ppo':
            migration.update(selected_policy_id=0, conditioning="folded_into_actor_and_critic_biases",
                             followers="discarded", actor_and_critic_preserved=True)
        del native_weights
    elif from_scratch:
        migration = dict(kind='from_scratch', optimizer='fresh', sampling='fresh',
                         counters='zero', environment='fresh', exact_runtime_resume=False)
    else:
        migration = load_array_archive(learner, args.checkpoint_npz, restore_optimizer=not args.fresh_optimizer)
    if not args.resume and initial_std is not None:
        learner.reset_action_std(initial_std)
        migration["initial_action_std"] = initial_std
    task, sim, environment_config = create_task(args, environment_config=config_source)
    task_sizes = tuple(task.obs[key].shape[-1] for key in ('state', 'privileged_state'))
    if task_sizes != (config.actor_obs, config.critic_obs):
        raise ValueError(f'Policy observation sizes {(config.actor_obs, config.critic_obs)} '
                         f'differ from task {task_sizes}; no implicit checkpoint expansion')
    policies = config.num_policies if config.algorithm == "sapg" else 1
    policy_ids = torch.arange(policies, device=learner.device).repeat_interleave(args.num_envs // policies)
    task.set_policy_ids(policy_ids)
    if not args.resume and native_path is None and not from_scratch:
        sampling, report = sampling_state_from_archive(metadata, archive_arrays, _file_hash(args.bank_manifest))
        migration["sampling"] = report
        if sampling is not None:
            task.restore_sampling_state(sampling, resample=True)
    del archive_arrays
    contract = dict(learner_config=asdict(config), num_envs=args.num_envs, batch_size=batch_size,
        unroll_length=unroll, seed=args.seed, nconmax=args.nconmax, njmax=args.njmax,
        compile_task=args.compile_task,
        bank_sha256=_file_hash(args.bank_manifest), collision_sha256=_file_hash(args.body_collision_bank),
        resets_sha256=_file_hash(args.body_collision_resets), source_sha256=_source_identity(), versions=_versions(),
        environment_config=environment_config,
        checkpoint_archive_sha256=None if from_scratch else _file_hash(native_path or args.checkpoint_npz))
    if from_scratch:
        contract['initial_checkpoint_format'] = 'from_scratch'
    if native_path is not None:
        contract["initial_checkpoint_format"] = "native_weights_only"
    if initial_std is not None:
        contract["initial_action_std"] = initial_std
    if getattr(args, "passage_rewards", None) is not None:
        contract["passage_rewards_sha256"] = _file_hash(args.passage_rewards)
    contract = json.loads(json.dumps(contract))  # tuples -> JSON lists consistently
    selection = "role-balanced leader training success" if task.hand_contrast else "leader mean physical rollout reward"
    hand_upgrade = getattr(task, 'checkpoint_selection_roles', (0, 1, 2, 3)) == (1,)
    if hand_upgrade:
        selection = "leader posture-qualified hand-task success"
    if getattr(task,"balance_settings",None) is not None:
        selection = "protected core / flat walking / flat progress / CAT retention: " + str(environment_config["checkpoint_selection_weights"])
    if environment_config.get('disable_hand_contrast', False):
        selection = "0.6 assigned clearance S + 0.2 CAT clean goal + 0.2 flat walking (no box metrics)"
    record = dict(schema="cat-mjlab-run-v1", contract=contract, migration=migration,
                  backend="mjlab/MuJoCo Warp", evaluation=False, retention=False,
                  best_selection=selection, success="leader-only first outcomes; 100-update rolling window",
                  best_score_semantics="100-update training history; published weights are post-update; no checkpoint evaluation",
                  adaptive_sampling=("legacy hand curriculum: all-policy raw clean goals; native survival for retention"
                                     if hand_upgrade else "all physical policies; posture-qualified successes for protected/transition roles"),
                  physics_timestep=.002, control_timestep=.02,
                  state_resume="full state restoration; GPU Warp does not guarantee bitwise deterministic trajectories")
    if getattr(task,"balance_settings",None) is not None:
        record['best_score_semantics']='100-update count-weighted history; missing protected-core/flat/retention evidence defers selection; post-update weights; no evaluation'
        record['adaptive_sampling']='Fixed six-bucket reset masses; within-bucket survival adaptation for retention, clean-goal adaptation for narrow; no curriculum'
    if hasattr(task, 'tolerance_state'):
        record['hand_tolerance_curriculum'] = environment_config['hand_tolerance_curriculum']
        record['adaptive_sampling'] = ('Fixed bank reset bucket masses; tolerance-qualified protected/transition '
            'within-bucket adaptation; unseeded leader rolling gate per role; strict 5 cm reported success')
    if hasattr(task, 'speed_state'):
        record['hand_speed_curriculum'] = environment_config['hand_speed_curriculum']
        record['adaptive_sampling'] = ('Fixed bank reset masses; strict posture-qualified within-bucket '
            'adaptation; speed curriculum controls active hand-zone commands only')
        record['hand_speed_semantics'] = ('Local first-core-zone trials; strict 5 cm; unseeded leader '
            'per-role advancement/demotion; bounded zero hold then required release progress; not passage acceptance')
    if environment_config.get('disable_hand_contrast', False):
        record['best_score_semantics'] = 'Cumulative assigned clearance S, pending included; 100-update CAT/flat locomotion; changing-policy training, not checkpoint evaluation'
        record['adaptive_sampling'] = 'Clean goals with episode minimum hand clearance >= .04 for protected/transition; no box compliance'
    record['metric_population'] = 'single policy, all environments' if config.algorithm == 'ppo' else 'leader-only outcomes; all-policy aggregate episodes'
    from .acceptance import SCHEMA, V_MIN, CROSSING_EDGES, TrainingAcceptance
    record['passage_acceptance'] = dict(schema='cat-clearance-acceptance-v1' if environment_config.get('disable_hand_contrast',False) else SCHEMA, v_min_m_s=V_MIN,
        speed_semantics='design choice, not a validated requirement',
        geometry='full declared zone gates; module width root corridor; swept world XY',
        deadline='remaining physical wrapper horizon frozen before first action',
        assignments='every observed reset before action; seeded/non-upstream diagnostic only',
        population='cumulative training assignments, changing policy; not checkpoint evaluation',
        crossing_histogram_edges_s=CROSSING_EDGES,
        selection=bool(environment_config.get('disable_hand_contrast', False)))
    if config.algorithm == 'ppo':
        record['success'] = 'all-environment first outcomes; 100-update rolling window'
    window = SuccessWindow()
    best_score = None
    previous_walltime = 0.
    if args.resume:
        old_record = json.loads((directory / "run.json").read_text())
        if old_record["contract"] != contract:
            raise ValueError("Native mjlab resume contract differs from saved run")
        snapshot = torch.load(directory / "resume.pt", map_location="cpu", weights_only=True)
        if snapshot.get("schema") != "cat-mjlab-runtime-v1" or snapshot["contract"] != contract:
            raise ValueError("Native runtime contract differs")
        learner.load_state_dict(snapshot["learner"])
        sim.load_state_dict(_device_tree(snapshot["simulation"], learner.device))
        task.load_state_dict(_device_tree(snapshot["task"], learner.device))
        if snapshot.get('passage_acceptance_reset_regions') is not None:
            task.acceptance_reset_regions = snapshot['passage_acceptance_reset_regions'].to(learner.device)
        if snapshot.get('passage_acceptance') is not None:
            task.acceptance_observer = TrainingAcceptance()
            task.acceptance_observer.load_state_dict(_device_tree(snapshot['passage_acceptance'], learner.device))
        torch.set_rng_state(snapshot["rng_cpu"])
        if snapshot["rng_cuda"]:
            torch.cuda.set_rng_state_all(snapshot["rng_cuda"])
        window.load_state_dict(snapshot["success_window"])
        best_score, previous_walltime = snapshot["best_score"], snapshot["walltime"]
        # Best-model publication can be newer than the less frequent full
        # runtime snapshot. Preserve its selection score across crash recovery.
        if (directory / "best.pt").exists():
            incumbent = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            if (incumbent.get("schema") != "cat-mjlab-best-v1" or incumbent.get("contract") != contract
                    or not math.isfinite(incumbent["score"])):
                raise ValueError("Existing best checkpoint differs from the native run contract")
            best_score = incumbent["score"] if best_score is None else max(best_score, incumbent["score"])
        record = old_record
    else:
        atomic_json(directory / "run.json", record)
    logger = Logger(directory, mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity,
                    record=record, resume=args.resume)
    started = time.monotonic()
    local_updates = 0
    status = "running"

    def save_runtime():
        snapshot = dict(schema="cat-mjlab-runtime-v1", contract=contract, learner=learner.state_dict(),
            simulation=sim.state_dict(), task=task.state_dict(), rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            success_window=window.state_dict(), best_score=best_score,
            walltime=previous_walltime + time.monotonic() - started)
        observer = getattr(task, 'acceptance_observer', None)
        snapshot['passage_acceptance'] = None if observer is None else observer.state_dict()
        snapshot['passage_acceptance_reset_regions'] = getattr(task, 'acceptance_reset_regions', None)
        atomic_torch_save(directory / "resume.pt", snapshot)
        atomic_json(directory / "status.json", dict(phase=status, env_steps=learner.env_steps,
            updates=learner.updates, best_score=best_score, resume="resume.pt", walltime=snapshot["walltime"]))

    try:
        if not args.resume:
            save_runtime()
        with stop_requests(directory) as should_stop:
            while not should_stop() and (args.max_updates == 0 or local_updates < args.max_updates):
                began = time.monotonic()
                if learner.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(learner.device)
                rollout, collected = collect_rollout(task, learner, unroll_length=unroll,
                                                     trajectories=trajectories, policy_ids=policy_ids)
                sim.capacity_report()  # Latched overflow must stop before any gradient update.
                metrics = learner.update(rollout)
                del rollout
                local_updates += 1
                window.append(collected["navigation_counts"], collected["contrast_counts"], collected["role_counts"], collected["dense"])
                elapsed = time.monotonic() - began
                required_roles = getattr(task, 'checkpoint_selection_roles', (0, 1, 2, 3))
                score = window.role_balanced_score(required_roles) if task.hand_contrast else metrics["rollout_reward_mean"]
                if getattr(task,'balance_settings',None) is not None:
                    score=window.flat_balance_score(environment_config['checkpoint_selection_weights'])
                if environment_config.get('disable_hand_contrast', False):
                    from .clearance_objective import clearance_selection
                    score = clearance_selection(collected['metrics'], window.metrics())
                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    atomic_torch_save(directory / "best.pt", dict(schema="cat-mjlab-best-v1",
                        model=learner.model.state_dict(), config=asdict(config), contract=contract, step=learner.env_steps,
                        score=score, selected_policy_id=0, selection=selection,
                        score_semantics=record["best_score_semantics"],
                        observation_contract=task.contract))
                values = {"learner/" + key: value for key, value in metrics.items()
                          if key in ("total_loss", "policy_loss", "v_loss", "entropy_loss") or key.startswith("diagnostics/")}
                legacy_score = window.legacy_flat_balance_score()
                if legacy_score is not None:
                    values['training/legacy_flat_checkpoint_selection_score'] = legacy_score
                values.update(window.metrics(contrastive=task.hand_contrast)); values.update(collected["metrics"])
                if getattr(task,'balance_settings',None) is not None:
                    for old,new in (('success/hand_protection_goal_success_rate','success/narrow_replay_clean_goal_success_rate'),
                                    ('training/hand_protection_goal_resolved_count','training/narrow_replay_clean_goal_resolved_count')):
                        if old in values:values[new]=values.pop(old)
                values.update({"training/rollout_reward_mean": metrics["rollout_reward_mean"],
                    "performance/control_steps_per_second": metrics["physical_transitions"] / elapsed,
                    "performance/update_seconds": elapsed})
                if score is not None:
                    values["training/checkpoint_selection_score"] = score
                if best_score is not None:
                    values['training/best_checkpoint_selection_score'] = best_score
                values['training/metric_population_envs'] = args.num_envs // policies
                values['training/physical_transitions_per_update'] = metrics['physical_transitions']
                values['training/optimizer_transitions_per_epoch'] = metrics['optimizer_transitions']
                if config.algorithm == 'ppo':
                    values = {key.replace('on_policy_leader/', 'on_policy/').replace('leader_', 'policy_'): value
                              for key, value in values.items()}
                    values = {('success/policy_' + key.removeprefix('success/') if key.startswith('success/') else key): value
                              for key, value in values.items()}
                # Plain-language aliases. The canonical keys stay for continuity, but nothing
                # in a name like success/policy_narrow_replay_clean_goal_success_rate tells you
                # it is objective #2, so the panel below is ordered by the owner's priorities.
                ALIASES = (
                    ('success/policy_narrow_replay_clean_goal_success_rate', 'progress/p2_narrow_passage_success'),
                    ('success/policy_ordinary_clutter_goal_success_rate', 'progress/clutter_room_success'),
                    ('success/policy_cat_goal_success_rate', 'progress/p3_cat_navigation_success'),
                    ('success/policy_goal_success_rate', 'progress/any_goal_success'),
                    ('success/policy_forward_protected_success_rate', 'progress/protected_zone_success'),
                    ('success/policy_narrow_passage_success_rate', 'progress/narrow_zone_success'),
                    ('success/policy_posture_transition_success_rate', 'progress/posture_transition_success'),
                    ('balance/policy_fall_rate_per_episode', 'progress/falls_per_episode'),
                    ('reactive/object_initiated_contact_rate', 'progress/p1_object_touched_robot'),
                    ('reactive/inside_anticipation_fraction', 'progress/p1_hand_near_object_fraction'),
                    ('reactive/mean_hand_gap_m', 'progress/p1_mean_hand_gap_m'),
                    ('reactive/avoidance_margin_mean_m', 'progress/p1_avoidance_margin_m'),
                    ('reactive/avoidance_margin_min_m', 'progress/p1_avoidance_margin_worst_m'),
                    ('reactive/active_fraction', 'health/reactive_experience_share'),
                    ('training/timeout_rate', 'health/episodes_timed_out'),
                    ('performance/update_seconds', 'health/seconds_per_update'),
                )
                for source, alias in ALIASES:
                    if source in values:
                        values[alias] = values[source]
                if learner.device.type == "cuda":
                    free, total = torch.cuda.mem_get_info(learner.device)
                    values["performance/device_vram_used_gib"] = (total - free) / 2**30
                    values['performance/device_vram_free_gib'] = free / 2**30
                    values['performance/device_vram_total_gib'] = total / 2**30
                    values['performance/torch_peak_allocated_gib'] = torch.cuda.max_memory_allocated(learner.device) / 2**30
                    values['performance/torch_peak_reserved_gib'] = torch.cuda.max_memory_reserved(learner.device) / 2**30
                logger.log(learner.env_steps, values)
                if local_updates % args.checkpoint_interval_updates == 0:
                    save_runtime()
            status = "stopped_by_request" if should_stop() else "bounded_verification_complete" if args.command == "verify" else "completed_requested_updates"
        save_runtime()
    except BaseException as error:
        # Preserve the last committed snapshot: a partially failed update is not
        # a valid replacement for its finite pre-update learner/environment.
        status = "failed"
        atomic_json(directory / "failure.json", dict(error=type(error).__name__, message=str(error),
            observed_steps=learner.env_steps, local_updates=local_updates, durable_checkpoint="resume.pt"))
        logger.finish(exit_code=1)
        raise
    else:
        logger.finish()
    return dict(status=status, run_dir=str(directory), env_steps=learner.env_steps, updates=local_updates, best_score=best_score)
