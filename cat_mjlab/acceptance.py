"""Observational passage acceptance, independent of rewards and task outcomes.

Trials are assigned at reset, before the first action. All assigned upstream
trials stay in the denominator, including failures and pending trials. No RNG,
policy inference, reward, termination, curriculum or checkpoint selection here.
"""
from __future__ import annotations

import torch

SCHEMA = 'cat-passage-acceptance-v2-corridor'
# Explicit DESIGN CHOICE, not a validated finger-protection requirement.
V_MIN = 0.2  # m/s; each crossing must take <= geometric zone length / V_MIN.
FAILURES = ('non_entry', 'bypass', 'timeout', 'fall', 'body_collision', 'hand_collision', 'other')
# Fixed histogram edges in seconds. Includes crossings from failed trials.
CROSSING_EDGES = (0.5, 1., 2., 4., 8., 16., 32.)
ROLES = ('open', 'forward_protected', 'narrow_passage', 'posture_transition')


def pack_geometry(rooms):
    """Freeze world gates/corridors from supported straight contrastive scenes.

    Full shaping-zone lengths (not just cabinet depths) are required. The root
    must stay within each module's physical opening for the entire zone. Body
    extent is checked by the existing collision signals, not a new detector.
    Unsupported geometry fails closed at bank load, never silently approximates
    a bent route with progress or its endpoint chord.
    """
    import numpy as np
    n, z = len(rooms), 6
    result = dict(origin=np.zeros((n, 2)), tangent=np.tile([1., 0.], (n, 1)),
                  start=np.zeros((n, z)), end=np.ones((n, z)),
                  half_width=np.ones((n, z)), valid=np.zeros((n, z), dtype=bool), protection=np.zeros(n, dtype=bool),
                  lower=np.full(n, -1.), upper=np.ones(n), bound=np.ones(n))
    for i, room in enumerate(rooms):
        if room is None or not room.get('hand_contrast'):
            continue
        contrast = room['hand_contrast']
        if contrast.get('role') == 'forward_protected' and contrast.get('geometry_family', 'cabinet') != 'table_edges':
            # Table-edge passages have no cabinet faces; their hand protection is judged by
            # goal success and hand clearance, not the lateral-corridor acceptance gate.
            from .lateral_corridor import scene_faces, MARGIN
            from cat_ppo.furniture.grippers import hand_sphere
            lower, upper = scene_faces(room)
            bound = min(-lower, upper)-max(hand_sphere(side)['radius'] for side in ('left', 'right'))-MARGIN
            if bound <= 0: raise ValueError('Empty hand corridor')
            for key, value in dict(protection=True, lower=lower, upper=upper, bound=bound).items():
                result[key][i] = value
        route = np.asarray(room['route'], dtype=float)
        zones, modules = contrast['zones'], contrast['modules']
        if route.shape != (2, 2) or not 0 < len(zones) <= z or len(zones) != len(modules):
            raise ValueError('Acceptance requires a straight route and one module per zone')
        delta = route[1] - route[0]
        length = np.linalg.norm(delta)
        if not np.isfinite(route).all() or length <= 0:
            raise ValueError('Invalid acceptance route')
        result['origin'][i], result['tangent'][i] = route[0], delta / length
        previous = -float('inf')
        for j, (zone, module) in enumerate(zip(zones, modules)):
            a, b, width = zone['start_m'], zone['end_m'], module['width_m']
            if not np.isfinite([a, b, width]).all() or not 0 <= a < b <= length + 1e-6 or a < previous or width <= 0:
                raise ValueError('Invalid/overlapping acceptance zones')
            previous = b
            for key, value in (('start', a), ('end', b), ('half_width', width / 2), ('valid', True)):
                result[key][i, j] = value
    return result


class PassageGate:
    """Vector state machine; caller supplies fixed trials BEFORE their rollouts.

    Coordinates are root world XY at every control step; segments between
    samples are checked against fixed rectangles. ``done`` is a physical end,
    not a rollout-chunk boundary. A caller must continue pending trials through
    their assigned deadlines; stopping observation leaves them PENDING.
    Failure flags overlap (e.g. fall plus non-entry); never condition on success.
    """
    def __init__(self, geometry, xy, deadline, *, seeded=None, initial_body=None, initial_hand=None):
        self.geometry = {k: v.detach().clone() for k, v in geometry.items()}
        self.previous = xy.detach().clone()
        from .response_split import FIELDS
        self.response_counts = {k:xy.new_zeros(len(xy), dtype=torch.float64) for k in FIELDS}
        n, z = geometry['valid'].shape
        self.time = xy.new_zeros(n, dtype=torch.float64)
        self.deadline = torch.as_tensor(deadline, device=xy.device, dtype=torch.float64).expand(n).clone()
        if bool((~torch.isfinite(self.deadline) | (self.deadline <= 0)).any()):
            raise ValueError('Every trial needs a positive finite deadline')
        self.entered = torch.zeros((n, z), dtype=torch.bool, device=xy.device)
        self.exited = self.entered.clone()
        self.entry_time = xy.new_zeros((n, z), dtype=torch.float64)
        self.crossing_time = xy.new_zeros((n, z), dtype=torch.float64)
        self.finished = torch.zeros(n, dtype=torch.bool, device=xy.device)
        self.passed = self.finished.clone()
        self.failures = torch.zeros((n, len(FAILURES)), dtype=torch.bool, device=xy.device)
        self.hand = self.finished.clone()
        self.eligible = geometry['valid'].any(-1)
        along, _ = self.coordinates(xy)
        upstream = along < geometry['start'][:, 0]
        seeded = torch.zeros_like(upstream) if seeded is None else seeded.bool()
        self.diagnostic = seeded | ~upstream
        self.failures[:, 1] = self.eligible & ~upstream
        if initial_body is not None:
            self.failures[:, 4] |= initial_body.bool()
        if initial_hand is not None:
            self.hand |= initial_hand.bool()
            self.failures[:, 5] |= self.hand

    def coordinates(self, xy):
        offset = xy - self.geometry['origin']
        tangent = self.geometry['tangent']
        return (offset * tangent).sum(-1), offset[:, 1] * tangent[:, 0] - offset[:, 0] * tangent[:, 1]

    @staticmethod
    def outside_corridor(x0, y0, x1, y1, a, b, width):
        """Intersect the XY segment with a fixed corridor rectangle's slab."""
        dx = x1-x0
        safe_dx = torch.where(dx.abs() > 1e-12, dx, torch.ones_like(dx))
        u, v = (a-x0)/safe_dx, (b-x0)/safe_dx
        lo, hi = torch.minimum(u, v).clamp(0, 1), torch.maximum(u, v).clamp(0, 1)
        overlap = (torch.maximum(x0, x1) >= a) & (torch.minimum(x0, x1) <= b)
        left, right = y0+lo*(y1-y0), y0+hi*(y1-y0)
        left = torch.where(dx.abs() > 1e-12, left, y0)
        right = torch.where(dx.abs() > 1e-12, right, y1)
        return overlap & ((left.abs() > width) | (right.abs() > width))

    @torch.no_grad()
    def advance(self, xy, time, *, clean_goal, done, hand_collision, body_collision, fall, other=None, protection_good=None, response_split=None):
        now = torch.as_tensor(time, device=xy.device, dtype=torch.float64).expand_as(self.time)
        if bool((~torch.isfinite(now) | (now <= self.time)).any()):
            raise ValueError('Observation times must strictly increase')
        active = self.eligible & ~self.finished
        if response_split is not None:
            for key in self.response_counts:
                self.response_counts[key] += response_split[key].double()*active
        x0, y0 = self.coordinates(self.previous)
        x1, y1 = self.coordinates(xy)
        dx = x1 - x0
        finite = torch.isfinite(xy).all(-1)
        self.failures[:, 6] |= active & ~finite
        # Signals are reset-to-completion latches from the existing task detector.
        self.hand |= active & hand_collision
        for index, flag in ((3, fall), (4, body_collision), (5, hand_collision)):
            self.failures[:, index] |= active & flag
        if other is not None:
            self.failures[:, 6] |= active & other
        for j in range(self.entered.shape[1]):
            valid = active & self.geometry['valid'][:, j]
            a, b, w = (self.geometry[k][:, j] for k in ('start', 'end', 'half_width'))
            # Fixed rectangular zones plus fixed connecting corridors. The
            # narrower adjacent opening defines each inter-zone corridor; this
            # is an explicit conservative root-path design choice. It cannot
            # be enlarged based on a policy's observed path.
            if j:
                gap_width = torch.minimum(w, self.geometry['half_width'][:, j-1])
                self.failures[:, 1] |= valid & self.outside_corridor(
                    x0, y0, x1, y1, self.geometry['end'][:, j-1], a, gap_width)
            safe_dx = torch.where(dx.abs() > 1e-12, dx, torch.ones_like(dx))
            u, v = (a - x0) / safe_dx, (b - x0) / safe_dx
            # Missing protected-pose evidence fails closed, including exit steps.
            protected = self.geometry.get('protection', torch.zeros_like(active))
            overlap = (torch.maximum(x0, x1) >= a) & (torch.minimum(x0, x1) <= b)
            good = torch.zeros_like(active) if protection_good is None else protection_good.bool()
            self.failures[:, 6] |= valid & protected & overlap & ~good
            bad_corridor = self.outside_corridor(x0, y0, x1, y1, a, b, w)
            self.failures[:, 1] |= valid & bad_corridor
            entry_y = y0 + u.clamp(0, 1) * (y1-y0)
            entry = valid & ~self.entered[:, j] & (x0 < a) & (x1 >= a) & (entry_y.abs() <= w)
            ordered = torch.ones_like(entry) if j == 0 else self.exited[:, j-1]
            self.failures[:, 1] |= entry & ~ordered
            entry_t = self.time + u.clamp(0, 1) * (now-self.time)
            self.entry_time[:, j] = torch.where(entry, entry_t, self.entry_time[:, j])
            self.entered[:, j] |= entry
            exit_gate = valid & ~self.exited[:, j] & (x0 < b) & (x1 >= b)
            self.failures[:, 1] |= exit_gate & ~self.entered[:, j]
            exit_y = y0 + v.clamp(0, 1) * (y1-y0)
            crossed = exit_gate & self.entered[:, j] & (exit_y.abs() <= w)
            exit_t = self.time + v.clamp(0, 1) * (now-self.time)
            duration = exit_t - self.entry_time[:, j]
            self.crossing_time[:, j] = torch.where(crossed, duration, self.crossing_time[:, j])
            self.exited[:, j] |= crossed
            over_budget = self.entered[:, j] & torch.where(self.exited[:, j], self.crossing_time[:, j], now-self.entry_time[:, j]).gt((b-a)/V_MIN + 1e-6)
            self.failures[:, 2] |= valid & over_budget
            # Exiting back through entry is not a completed ordered passage.
            self.failures[:, 1] |= valid & self.entered[:, j] & (x1 < a)
        all_exited = (self.exited | ~self.geometry['valid']).all(-1)
        late = now > self.deadline + 1e-6
        deadline = now >= self.deadline - 1e-6
        self.failures[:, 2] |= active & (late | (deadline & ~clean_goal))
        self.failures[:, 1] |= active & clean_goal & ~all_exited
        finish = active & (done | clean_goal | deadline)
        self.failures[:, 0] |= finish & ~(self.entered | ~self.geometry['valid']).all(-1)
        self.failures[:, 6] |= finish & ~clean_goal & ~self.failures.any(-1)
        self.passed |= finish & clean_goal & all_exited & ~self.failures.any(-1)
        self.finished |= finish
        self.previous.copy_(xy)
        self.time.copy_(now)

    def counts(self):
        """Per-assignment sufficient statistics, including pending denominator."""
        failed = self.failures.any(-1)
        values = dict(assigned_count=self.eligible, pass_count=self.passed,
                      fail_count=failed & self.eligible,
                      pending_count=~self.finished & ~failed & self.eligible,
                      unfinished_count=~self.finished & self.eligible,
                      hand_collision_count=self.hand & self.eligible,
                      zone_entry_count=self.entered.any(-1) & self.eligible,
                      entered_zone_count=self.entered.sum(-1),
                      required_zone_count=self.geometry['valid'].sum(-1),
                      crossing_count=self.exited.sum(-1),
                      crossing_time_sum_s=(self.crossing_time*self.exited).sum(-1))
        for j, name in enumerate(FAILURES):
            values['failure_'+name+'_count'] = self.failures[:, j] & self.eligible
        # Exhaustive histogram; non-crossings are accounted for above, never
        # silently treated as fast crossings. No survivor-only distribution.
        lower = -float('inf')
        for upper in (*CROSSING_EDGES, float('inf')):
            name = 'inf' if upper == float('inf') else str(upper).replace('.', 'p')
            values['crossing_time_bin_le_'+name+'_s_count'] = (self.exited & (self.crossing_time > lower) & (self.crossing_time <= upper)).sum(-1)
            lower = upper
        values.update({'response_split_'+k:v for k,v in self.response_counts.items()})
        return {key: value.double() * self.eligible for key, value in values.items()}


class ClearancePassageGate(PassageGate):
    """S geometry/deadlines with clearance evidence instead of prescribed poses.

    Fixed acceptance thresholds (.20/.04 m), independent of reward tuning.
    Exposure uses either-hand minimum over reset-to-first-outcome, dt weighted.
    Missing/nonfinite samples fail closed. Contacts retain the production latch.
    """
    def __init__(self, *args, initial_clearance=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.clearance_min = torch.full_like(self.time, torch.inf)
        self.clearance_observed = torch.zeros_like(self.time)
        self.clearance_inside20 = torch.zeros_like(self.time)
        self.clearance_below4 = torch.zeros_like(self.time)
        self.clearance_missing = torch.zeros_like(self.time)
        if initial_clearance is None:
            raise ValueError('Clearance acceptance requires hand clearance measured at reset')
        if initial_clearance is not None:
            d = initial_clearance.reshape(len(self.time), 2).amin(-1)
            self.clearance_min = torch.where(torch.isfinite(d), d.double(), self.clearance_min)
            self.failures[:, 6] |= self.eligible & (~torch.isfinite(d) | (d < .04))

    def advance(self, xy, time, *, hand_clearance=None, protection_good=None, **kwargs):
        now = torch.as_tensor(time, device=xy.device, dtype=torch.float64).expand_as(self.time)
        if bool((~torch.isfinite(now) | (now <= self.time)).any()):
            raise ValueError('Observation times must strictly increase')
        active = self.eligible & ~self.finished
        d = torch.full_like(self.time, torch.nan) if hand_clearance is None else hand_clearance.reshape(len(self.time), 2).amin(-1)
        finite = torch.isfinite(d)
        duration = torch.minimum(now, self.deadline)-torch.minimum(self.time, self.deadline)
        dt = torch.where(active, duration, 0.)
        self.clearance_observed += dt*finite
        self.clearance_missing += dt*~finite
        self.clearance_inside20 += dt*(finite & (d < .20))
        self.clearance_below4 += dt*(finite & (d < .04))
        self.clearance_min = torch.minimum(self.clearance_min, torch.where(active & finite, d, torch.inf))
        good = finite & (d >= .04)
        self.failures[:, 6] |= active & ~good
        super().advance(xy, time, protection_good=good, **kwargs)

    def counts(self):
        result = super().counts()
        observed = torch.isfinite(self.clearance_min) & self.eligible
        extra = dict(clearance_observed_s=self.clearance_observed,
            clearance_missing_s=self.clearance_missing,
            clearance_inside_20cm_s=self.clearance_inside20,
            clearance_below_4cm_s=self.clearance_below4,
            clearance_trial_minimum_sum_m=torch.where(observed, self.clearance_min, 0.),
            clearance_observed_trial_count=observed.double())
        result.update({k: v*self.eligible for k,v in extra.items()})
        return result


class TrainingAcceptance:
    """Persistent training telemetry, not a frozen-checkpoint evaluation.

    New assignments follow the existing reset sampler (unchanged). Geometry,
    cohorts and remaining wrapper deadline are frozen before any action. A
    resumed legacy mid-episode cannot be observed from reset and is excluded
    until its next reset, with an explicit counter. State is checkpointed.
    """
    def __init__(self):
        self.gate = None
        self.totals = None
        self.count_names = []
        self.masks = {}
        self.previous_counts = {}
        self.registered = None
        self.unobserved = 0

    def before_step(self, task, policy_ids):
        fresh = task.info['step'] == 0
        if self.registered is None:
            self.registered = torch.zeros_like(fresh)
            self.unobserved = int((~fresh).sum())
        new = fresh & ~self.registered
        if not bool(new.any()):
            return
        geometry = {k: v[task.scene_ids] for k, v in task.bank.acceptance.items()}
        seeded = task.info.get('hand_raised_seeded', torch.zeros_like(fresh))
        reset_regions = getattr(task, 'acceptance_reset_regions', torch.zeros((task.num_envs, 6), dtype=torch.bool, device=fresh.device))
        gate_type = ClearancePassageGate if getattr(task, 'config', {}).get('disable_hand_contrast', False) else PassageGate
        initial = {'initial_clearance': task.info['sdf'][:,5:7,0]} if gate_type is ClearancePassageGate else {}
        candidate = gate_type(geometry, task.data.qpos[:, :2],
            (task.bank.episode_lengths[task.scene_ids]-task.info['wrapper_steps']).clamp_min(1).double()*task.dt,
            seeded=seeded, initial_body=reset_regions.any(-1), initial_hand=reset_regions[:, 5], **initial)
        candidate.eligible &= new
        if self.gate is None:
            self.gate = candidate
        else:
            for key, value in vars(candidate).items():
                target = getattr(self.gate, key)
                if isinstance(value, dict):
                    for k, v in value.items(): target[k][new] = v[new]
                else:
                    target[new] = value[new]
        roles = task.bank.contrast['role'][task.scene_ids]
        leader = policy_ids == 0
        groups = {'': torch.ones_like(fresh), 'leader_': leader}
        groups.update({f'leader_{name}_': leader & (roles == j) for j, name in enumerate(ROLES)})
        masks = {}
        for prefix, mask in groups.items():
            masks[prefix] = mask & ~candidate.diagnostic
            masks[prefix+'seeded_inside_'] = mask & candidate.diagnostic
        for key, value in masks.items():
            if key not in self.masks: self.masks[key] = torch.zeros_like(value)
            self.masks[key][new] = value[new]
        counts = self.gate.counts()
        for key, value in counts.items():
            if key not in self.previous_counts: self.previous_counts[key] = torch.zeros_like(value)
            self.previous_counts[key][new] = 0
        self.registered |= new
        self._accumulate()

    def _accumulate(self):
        if isinstance(self.gate, ClearancePassageGate):
            if not hasattr(self, 'minimum_clearance'):self.minimum_clearance = {}
            for prefix, mask in self.masks.items():
                observed = mask & self.gate.eligible & torch.isfinite(self.gate.clearance_min)
                minimum = torch.where(observed,self.gate.clearance_min,torch.inf).min()
                self.minimum_clearance[prefix] = torch.minimum(self.minimum_clearance.get(prefix, minimum), minimum)
        counts = self.gate.counts()
        self.count_names = list(counts)
        delta = torch.stack([counts[k]-self.previous_counts[k] for k in self.count_names], -1)
        masks = torch.stack(list(self.masks.values())).double()
        increment = masks @ delta
        if self.totals is None:
            self.totals = torch.zeros_like(increment)
        self.totals += increment
        self.previous_counts = {k: v.clone() for k, v in counts.items()}

    def after_step(self, task, transition):
        if self.gate is None: return
        m = transition['metrics']
        zero = torch.zeros_like(transition['done'])
        flag = lambda name: m.get('episode/'+name, zero)
        clearance = {'hand_clearance': m.get('acceptance/hand_clearance')} if isinstance(self.gate, ClearancePassageGate) else {}
        from .response_split import FIELDS
        response = ({k:m['response_split/'+k] for k in FIELDS}
                    if 'response_split/event_count' in m else None)
        self.gate.advance(m['acceptance/root_xy'], self.gate.time + task.dt,
            response_split=response,
            clean_goal=flag('goal_reached'), done=transition['done'],
            protection_good=m.get('acceptance/protection_good'),
            hand_collision=flag('body_collision_hands') | flag('hand_violation'),
            body_collision=flag('body_collision'), fall=flag('fall'),
            other=flag('obstacle') | flag('self_contact') | flag('numerical') | flag('outside_bounds') | flag('elbow_violation'), **clearance)
        self._accumulate()
        self.registered &= ~transition['done']

    def metrics(self):
        result = {'training/passage_acceptance_unobserved_initial_count': self.unobserved}
        for prefix, minimum in getattr(self, 'minimum_clearance', {}).items():
            if torch.isfinite(minimum):
                result['training/passage_acceptance_'+prefix+'minimum_hand_clearance_m'] = float(minimum)
        if self.totals is None:
            return result
        totals = self.totals.detach().cpu().tolist()
        for prefix, row in zip(self.masks, totals):
            counts = dict(zip(self.count_names, row))
            for key, value in counts.items():
                result['training/passage_acceptance_'+prefix+key] = value
            from .response_split import summaries
            response = {k.removeprefix('response_split_'):v for k,v in counts.items() if k.startswith('response_split_')}
            result.update({'training/passage_acceptance_'+prefix+'response_split_'+k:v
                           for k,v in summaries(response).items()})
            get = lambda key: counts.get(key, 0)
            n = get('assigned_count')
            if n:
                base = 'training/passage_acceptance_'+prefix
                result[base+'success_rate'] = get('pass_count') / n
                result[base+'hand_collision_incidence'] = get('hand_collision_count') / n
                result[base+'trial_entry_rate'] = get('zone_entry_count') / n
                result[base+'zone_entry_rate'] = get('entered_zone_count') / get('required_zone_count')
                result[base+'complete'] = float(get('unfinished_count') == 0)
                for failure in FAILURES:
                    result[base+'failure_'+failure+'_rate'] = get('failure_'+failure+'_count') / n
            seconds = get('clearance_observed_s')
            if seconds:
                base = 'training/passage_acceptance_'+prefix
                result[base+'clearance_inside_20cm_fraction'] = get('clearance_inside_20cm_s')/seconds
                result[base+'clearance_below_4cm_fraction'] = get('clearance_below_4cm_s')/seconds
            observed = get('clearance_observed_trial_count')
            if observed:
                result['training/passage_acceptance_'+prefix+'clearance_trial_minimum_mean_m'] = get('clearance_trial_minimum_sum_m')/observed
            count = get('crossing_count')
            if count:
                result['training/passage_acceptance_'+prefix+'crossing_time_mean_s'] = get('crossing_time_sum_s') / count
        return result

    def state_dict(self):
        state = dict(vars(self))
        state['gate'] = None if self.gate is None else vars(self.gate)
        return state

    def load_state_dict(self, state):
        state = dict(state)
        gate = state.pop('gate')
        self.__dict__.update(state)
        self.gate = None
        if gate is not None:
            self.gate = object.__new__(ClearancePassageGate if 'clearance_min' in gate else PassageGate)
            self.gate.__dict__.update(gate)
            if not hasattr(self.gate, 'response_counts'):
                from .response_split import FIELDS
                self.gate.response_counts = {k:torch.zeros_like(self.gate.time) for k in FIELDS}
                self.previous_counts.update({'response_split_'+k:v.clone() for k,v in self.gate.response_counts.items()})
                names = list(self.gate.counts())
                if self.totals is not None:
                    # Clearance counts follow super().counts(); insert by name,
                    # not at the end, or legacy clearance totals shift columns.
                    upgraded = self.totals.new_zeros((len(self.masks),len(names)))
                    for column, name in enumerate(self.count_names):
                        upgraded[:,names.index(name)] = self.totals[:,column]
                    self.totals = upgraded
                self.count_names = names


def evaluate_assigned_trials(assignments, rollout, *, clearance_primary=False):
    """CPU reference evaluator for a preregistered, finite frozen-policy cohort.

    ``assignments`` is fully materialized and validated BEFORE calling rollout.
    Each row contains trial_id, room, start_xy, deadline_s, cohort (upstream or
    seeded_inside), reset_body_collision, reset_hand_collision. ``rollout(row)``
    yields records with time_s, xy and all six advance flags. This adapter does
    not load/train a policy or invent missing observations. A caller supplying
    a policy must freeze it and its seeds outside this instrumentation module.

    Every assignment is consumed until clean goal, physical end or its deadline,
    including trials that have already violated a crossing/corridor constraint.
    An exhausted stream leaves PENDING evidence visible; complete=False forbids
    reporting this as a finished evaluation. No successful-traversal filtering.
    """
    assignments = list(assignments)
    ids, trials = set(), []
    for assignment in assignments:
        row = dict(assignment)
        trial_id = row['trial_id']
        if trial_id in ids: raise ValueError('Duplicate assigned trial ID')
        ids.add(trial_id)
        if row['cohort'] not in ('upstream', 'seeded_inside'):
            raise ValueError('Unknown trial cohort')
        geom = {k: torch.as_tensor(v) for k, v in pack_geometry([row['room']]).items()}
        gate_type = ClearancePassageGate if clearance_primary else PassageGate
        initial = {'initial_clearance': torch.tensor([row['initial_hand_clearance']])} if clearance_primary else {}
        gate = gate_type(geom, torch.tensor([row['start_xy']], dtype=torch.float64), row['deadline_s'],
            seeded=torch.tensor([row['cohort'] == 'seeded_inside']),
            initial_body=torch.tensor([row['reset_body_collision']]),
            initial_hand=torch.tensor([row['reset_hand_collision']]), **initial)
        if not bool(gate.eligible[0]): raise ValueError('Assigned trial has no required passages')
        if row['cohort'] == 'upstream' and bool(gate.diagnostic[0]):
            raise ValueError('Main-cohort trial must start upstream')
        trials.append((row, gate))
    result = dict(schema='cat-clearance-acceptance-v1' if clearance_primary else SCHEMA, trials={}, cohorts={})
    for row, gate in trials:
        response_state = None
        for event in rollout(row):
            event = dict(event)
            if all(k in event for k in ('hand_xyz','root_xyz','hand_normals','hand_clearance')):
                from .response_split import initial, advance
                root = torch.tensor([event['root_xyz']], dtype=torch.float64)
                if response_state is None: response_state = initial(root)
                response = advance(response_state, torch.tensor([event['hand_xyz']], dtype=torch.float64),
                    root, torch.tensor([event['hand_clearance']], dtype=torch.float64),
                    torch.tensor([event['hand_normals']], dtype=torch.float64),
                    torch.tensor([event['done'] or event['clean_goal'] or event['time_s'] >= row['deadline_s']]))
                event['response_split'] = {k:float(v[0]) for k,v in response.items()}
            clearance = {'hand_clearance': None if 'hand_clearance' not in event else torch.tensor([event['hand_clearance']])} if clearance_primary else {}
            gate.advance(torch.tensor([event['xy']], dtype=torch.float64), event['time_s'],
                response_split=None if 'response_split' not in event else
                    {k:torch.tensor([v]) for k,v in event['response_split'].items()},
                protection_good=None if 'protection_good' not in event else torch.tensor([event['protection_good']]), **clearance,
                **{key: torch.tensor([event[key]], dtype=torch.bool) for key in
                   ('clean_goal', 'done', 'hand_collision', 'body_collision', 'fall', 'other')})
            if bool(gate.finished[0]): break
        counts = {k: float(v[0]) for k, v in gate.counts().items()}
        state = 'PASS' if bool(gate.passed[0]) else 'FAIL' if bool(gate.failures.any()) else 'PENDING'
        result['trials'][row['trial_id']] = dict(status=state, finished=bool(gate.finished[0]), **counts)
        if clearance_primary:
            minimum = float(gate.clearance_min[0])
            result['trials'][row['trial_id']]['minimum_hand_clearance_m'] = minimum if torch.isfinite(gate.clearance_min[0]) else None
        total = result['cohorts'].setdefault(row['cohort'], {})
        for key, value in counts.items(): total[key] = total.get(key, 0) + value
    for counts in result['cohorts'].values():
        counts['success_rate'] = counts['pass_count'] / counts['assigned_count']
        counts['hand_collision_incidence'] = counts['hand_collision_count'] / counts['assigned_count']
        counts['zone_entry_rate'] = counts['entered_zone_count'] / counts['required_zone_count']
        counts['complete'] = counts['unfinished_count'] == 0
        if clearance_primary and counts['clearance_observed_s']:
            counts['inside_20cm_fraction'] = counts['clearance_inside_20cm_s']/counts['clearance_observed_s']
            counts['below_4cm_fraction'] = counts['clearance_below_4cm_s']/counts['clearance_observed_s']
    if clearance_primary:
        for cohort, counts in result['cohorts'].items():
            minima = [result['trials'][row['trial_id']]['minimum_hand_clearance_m'] for row, _ in trials if row['cohort']==cohort]
            minima = [v for v in minima if v is not None]
            counts['minimum_hand_clearance_m'] = min(minima) if minima else None
    result['complete'] = all(c['complete'] for c in result['cohorts'].values())
    return result
