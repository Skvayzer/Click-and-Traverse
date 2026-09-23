"""CPU-only preregistered trials and adversarial passage traces."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from cat_mjlab.acceptance import PassageGate, TrainingAcceptance, pack_geometry, FAILURES


def geometry(n=1, zones=((1., 2.),), width=1.):
    rooms = [dict(route=[[0., 0.], [10., 0.]], hand_contrast=dict(
        zones=[dict(start_m=a, end_m=b) for a, b in zones],
        modules=[dict(width_m=width) for _ in zones]))] * n
    return {k: torch.as_tensor(v) for k, v in pack_geometry(rooms).items()}


def gate(*, n=1, zones=((1., 2.),), deadline=20., xy=None, **kwargs):
    return PassageGate(geometry(n, zones), torch.zeros(n, 2) if xy is None else torch.tensor(xy),
                       deadline, **kwargs)


def step(g, x, time, y=0., **flags):
    def flag(name):
        value = flags.get(name, False)
        return torch.as_tensor(value, dtype=torch.bool).expand(len(g.time))
    xy = torch.tensor([x, y], dtype=torch.float32).expand(len(g.time), 2)
    g.advance(xy, time, **{name: flag(name) for name in
        ('clean_goal', 'done', 'hand_collision', 'body_collision', 'fall', 'other')})


def status(g):
    c = g.counts()
    return ('PASS' if c['pass_count'][0] else 'FAIL' if c['fail_count'][0] else 'PENDING')


def test_legitimate_safe_traversal_passes_all_ordered_zones():
    g = gate(zones=((1., 2.), (3., 4.)))
    for x in range(1, 6): step(g, x, x, clean_goal=x == 5)
    assert status(g) == 'PASS'
    assert g.counts()['crossing_count'].item() == 2
    assert g.counts()['crossing_time_sum_s'].item() == 2
    assert not g.hand.any()


@pytest.mark.parametrize('strategy,category', [
    ('refuse_entry', 'non_entry'), ('crawl', 'timeout'),
    ('lateral_bypass', 'bypass'), ('fall_first', 'fall'),
    ('body_collision', 'body_collision'), ('hand_collision', 'hand_collision'),
    ('goal_without_passage', 'bypass'), ('other_failure_first', 'other'),
])
def test_anti_gaming_fails(strategy, category):
    g = gate()
    if strategy == 'refuse_entry': step(g, 0., 20., done=True)
    elif strategy == 'crawl':
        step(g, 1., 1.)
        step(g, 1.5, 7.)
        step(g, 2.1, 8.)
        step(g, 3., 9., clean_goal=True)
    elif strategy == 'lateral_bypass':
        step(g, .5, 1., y=2.)
        step(g, 2.5, 2., y=2.)
        step(g, 3., 3., clean_goal=True)
    elif strategy == 'fall_first': step(g, .1, 1., fall=True, done=True)
    elif strategy == 'other_failure_first': step(g, .1, 1., other=True, done=True)
    elif strategy == 'goal_without_passage': step(g, .1, 1., clean_goal=True)
    else: step(g, .1, 1., done=True, **{strategy: True})
    assert status(g) == 'FAIL'
    assert g.counts()['failure_'+category+'_count'].item() == 1
    assert g.counts()['assigned_count'].item() == 1
    assert g.counts()['pass_count'].item() == 0


def test_pending_never_pass_and_all_assigned_denominator():
    g = gate(n=3)
    g.advance(torch.tensor([[3., 0.], [0., 0.], [0., 0.]]), 3.,
              clean_goal=torch.tensor([True, False, False]),
              done=torch.tensor([False, True, False]),
              hand_collision=torch.tensor([False, True, False]),
              body_collision=torch.tensor([False, True, False]), fall=torch.zeros(3, dtype=torch.bool))
    c = g.counts()
    assert c['pass_count'].sum() / c['assigned_count'].sum() == pytest.approx(1/3)
    assert c['pending_count'].sum() == 1
    assert c['hand_collision_count'].sum() / c['assigned_count'].sum() == pytest.approx(1/3)
    assert c['failure_non_entry_count'].sum() == 1
    step(g, 0., 20., done=True)
    assert g.counts()['pending_count'].sum() == 0
    assert g.counts()['pass_count'].sum() == 1


def test_swept_gate_interpolation_and_crossing_budget_boundary():
    for duration, expected in ((5., 'PASS'), (5.01, 'FAIL')):
        g = gate()
        step(g, 1., 1.)
        step(g, 2., 1.+duration)
        step(g, 3., 2.+duration, clean_goal=True)
        assert status(g) == expected
        assert g.crossing_time[0, 0] == pytest.approx(duration)
    g = gate(zones=((1., 2.), (3., 4.)))
    step(g, 5., 5., clean_goal=True)
    assert status(g) == 'PASS'  # Crossed both fixed gates between samples.
    assert g.entry_time[0, :2].tolist() == pytest.approx([1., 3.])


def test_overall_deadline_goal_boundary_and_late_goal():
    for time, expected in ((5., 'PASS'), (5.1, 'FAIL')):
        g = gate(deadline=5.)
        step(g, 2.5, 2.5)
        step(g, 3., time, clean_goal=True)
        assert status(g) == expected


@pytest.mark.parametrize('points', [
    [(1.2, 0., 1.), (1.2, 1., 2.), (3., 0., 3.)],  # Lateral while stationary in progress.
    [(1.2, 0., 1.), (.5, 0., 2.), (3., 0., 3.)],   # Back out through entry.
    [(3., 2., 3.)],                                # Jump over slab outside gate.
])
def test_corridor_escape_backtracking_and_swept_bypass(points):
    g = gate()
    for x, y, time in points: step(g, x, time, y, clean_goal=time == points[-1][2])
    assert status(g) == 'FAIL'
    assert g.failures[0, FAILURES.index('bypass')]


def test_reset_collision_and_collision_on_goal_cannot_pass():
    for kwargs in ({'initial_hand': torch.tensor([True])}, {'initial_body': torch.tensor([True])}, {}):
        g = gate(**kwargs)
        step(g, 3., 3., clean_goal=True, hand_collision=not kwargs)
        assert status(g) == 'FAIL'


def test_collision_after_completion_does_not_retroactively_fail():
    g = gate()
    step(g, 3., 3., clean_goal=True)
    step(g, 4., 4., hand_collision=True, body_collision=True, done=True)
    assert status(g) == 'PASS'
    assert g.counts()['hand_collision_count'].item() == 0


def test_seeded_and_inside_starts_are_separate_cohort():
    g = gate(n=3, xy=[[0., 0.], [1.5, 0.], [0., 0.]], seeded=torch.tensor([False, False, True]))
    assert g.diagnostic.tolist() == [False, True, True]


def test_geometry_rotation_and_validation():
    room = dict(route=[[10., 10.], [10., 20.]], hand_contrast=dict(
        zones=[dict(start_m=1., end_m=2.)], modules=[dict(width_m=1.)]))
    geom = {k: torch.as_tensor(v) for k, v in pack_geometry([room]).items()}
    g = PassageGate(geom, torch.tensor([[10., 10.]]), 20.)
    g.advance(torch.tensor([[10., 13.]]), 3., clean_goal=torch.tensor([True]),
              **{k: torch.tensor([False]) for k in ('done', 'hand_collision', 'body_collision', 'fall')})
    assert status(g) == 'PASS'
    room['route'].insert(1, [11., 11.])
    with pytest.raises(ValueError, match='straight'): pack_geometry([room])


def fake_task(n=2):
    return SimpleNamespace(num_envs=n, dt=1., data=SimpleNamespace(qpos=torch.zeros(n, 2)),
        info=dict(step=torch.zeros(n, dtype=torch.long), wrapper_steps=torch.zeros(n, dtype=torch.long)),
        scene_ids=torch.arange(n), bank=SimpleNamespace(acceptance=geometry(n),
            episode_lengths=torch.full((n,), 10), contrast={'role': torch.ones(n, dtype=torch.long)}))


def transition(task, x, *, done=False, goal=False, hand=False):
    task.info['step'] += 1
    task.data.qpos[:, 0] = x
    result = dict(done=torch.full((task.num_envs,), done), metrics={
        'acceptance/root_xy': task.data.qpos.clone(),
        'episode/goal_reached': torch.full((task.num_envs,), goal),
        'episode/body_collision_hands': torch.full((task.num_envs,), hand)})
    if done:  # Physical autoreset before observer sees the transition.
        task.info['step'].zero_(); task.data.qpos.zero_()
    return result


def test_observer_chunks_autoreset_cohorts_and_resume():
    task = fake_task()
    task.info['hand_raised_seeded'] = torch.tensor([False, True])
    policies = torch.tensor([0, 1])
    observer = TrainingAcceptance()
    observer.before_step(task, policies)
    observer.after_step(task, transition(task, 1.))
    metrics = observer.metrics()
    base = 'training/passage_acceptance_'
    assert metrics[base+'assigned_count'] == 1
    assert metrics[base+'seeded_inside_assigned_count'] == 1
    assert metrics[base+'pending_count'] == 1
    restored = TrainingAcceptance()
    restored.load_state_dict(deepcopy(observer.state_dict()))
    restored.before_step(task, policies)  # Chunk boundary is not an assignment.
    restored.after_step(task, transition(task, 3., goal=True, done=True))
    assert restored.metrics()[base+'success_rate'] == 1
    assert restored.metrics()[base+'leader_success_rate'] == 1
    restored.before_step(task, policies)
    assert restored.metrics()[base+'assigned_count'] == 2
    assert restored.metrics()[base+'success_rate'] == .5
    restored.after_step(task, transition(task, .1, hand=True, done=True))
    assert restored.metrics()[base+'hand_collision_incidence'] == .5
    assert restored.metrics()[base+'pending_count'] == 0


def test_legacy_mid_episode_not_invented_and_rng_untouched():
    task = fake_task()
    task.info['step'][:] = 4
    rng = torch.get_rng_state().clone()
    observer = TrainingAcceptance()
    observer.before_step(task, torch.tensor([0, 0]))
    assert observer.gate is None
    assert observer.metrics()['training/passage_acceptance_unobserved_initial_count'] == 2
    assert torch.equal(rng, torch.get_rng_state())


def test_histogram_includes_slow_failed_crossings():
    g = gate()
    step(g, 1., 1.)
    step(g, 2., 8.)
    assert status(g) == 'FAIL'
    assert g.counts()['crossing_time_bin_le_8p0_s_count'].item() == 1


def test_fixed_assigned_evaluator_does_not_drop_failures_or_pending():
    from cat_mjlab.acceptance import evaluate_assigned_trials
    room = dict(route=[[0., 0.], [10., 0.]], hand_contrast=dict(
        zones=[dict(start_m=1., end_m=2.)], modules=[dict(width_m=1.)]))
    assignments = [dict(trial_id=name, room=room, start_xy=[0., 0.], deadline_s=10.,
        cohort='upstream', reset_body_collision=False, reset_hand_collision=False)
        for name in ('safe', 'refusal', 'fall', 'pending')]
    visited = []
    def rollout(row):
        name = row['trial_id']; visited.append(name)
        if name == 'pending': return
        yield dict(time_s=10. if name == 'refusal' else 3., xy=[3. if name == 'safe' else 0., 0.],
                   clean_goal=name == 'safe', done=name == 'fall', fall=name == 'fall',
                   body_collision=False, hand_collision=False, other=False)
    result = evaluate_assigned_trials(assignments, rollout)
    assert visited == ['safe', 'refusal', 'fall', 'pending']
    assert result['cohorts']['upstream']['success_rate'] == .25
    assert not result['complete']
    assert [r['status'] for r in result['trials'].values()] == ['PASS', 'FAIL', 'FAIL', 'PENDING']
    assignments[-1]['start_xy'] = [1.5, 0.]
    visited.clear()
    with pytest.raises(ValueError, match='upstream'):
        evaluate_assigned_trials(assignments, rollout)
    assert not visited  # Validate EVERY trial before ANY rollout.


def test_observation_preserves_rollout_and_rng_exactly():
    from cat_mjlab.runner import collect_rollout
    from cat_mjlab.learning import Learner
    from test_mjlab_runner import FakeTask, tiny_config
    class Task(FakeTask):
        def __init__(self, instrumented):
            super().__init__()
            state = fake_task(6)
            self.bank, self.data, self.info, self.scene_ids = state.bank, state.data, state.info, state.scene_ids
            self.dt = 1.
            if not instrumented: del self.bank.acceptance
        def step(self, action):
            result = super().step(action)
            self.info['step'] += 1
            self.data.qpos[:, 0] += 1.5
            result['metrics']['acceptance/root_xy'] = self.data.qpos.clone()
            result['metrics']['episode/goal_reached'] = result['done'].clone()
            if result['done'].any():
                self.info['step'].zero_(); self.data.qpos.zero_()
            return result
    learner = Learner(tiny_config(), device='cpu')
    rng = torch.get_rng_state().clone()
    observed, info = collect_rollout(Task(True), learner, unroll_length=3, trajectories=6,
                                    policy_ids=torch.tensor([0, 0, 1, 1, 2, 2]))
    observed_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    plain, _ = collect_rollout(Task(False), learner, unroll_length=3, trajectories=6,
                              policy_ids=torch.tensor([0, 0, 1, 1, 2, 2]))
    assert torch.equal(observed_rng, torch.get_rng_state())
    for key in plain: torch.testing.assert_close(plain[key], observed[key], rtol=0, atol=0)
    assert info['metrics']['training/passage_acceptance_assigned_count'] == 12
    assert info['metrics']['training/passage_acceptance_success_rate'] == .5


def test_lateral_escape_between_required_zones_fails():
    g = gate(zones=((1., 2.), (3., 4.)))
    step(g, 2.1, 2.1)
    step(g, 2.5, 3., y=1.)
    step(g, 2.9, 4.)
    step(g, 5., 6., clean_goal=True)
    assert status(g) == 'FAIL'
    assert g.failures[0, FAILURES.index('bypass')]


def test_partial_zone_nonentry_and_bypass_continue_to_deadline():
    g = gate(zones=((1., 2.), (3., 4.)), deadline=10.)
    step(g, 2.1, 2.1)
    step(g, 2.5, 3., y=1.)
    assert not g.finished[0]  # Still collect collisions after a geometric failure.
    step(g, 2.5, 10., y=1., hand_collision=True)
    assert g.finished[0]
    assert g.failures[0, FAILURES.index('non_entry')]
    assert g.hand[0]


def test_real_cpu_physics_observer_does_not_change_task_dynamics():
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.task import CATTask
    from test_mjlab_task import _CPUSimulation
    from cat_mjlab.constants import DEFAULT_QPOS
    from cat_ppo.furniture.room_navigation import pack_room_scenes
    from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
    cfg = wholebody_config(); cfg['randomize_initial_episode_steps'] = False
    class Collision:
        active = False
        def __call__(self, scenes, data):
            result = torch.zeros((len(scenes), 6), dtype=torch.bool)
            if self.active: result[:, 5] = True
            return result
    tasks, detectors = [], []
    for _ in range(2):
        bank = SimpleNamespace(count=2, device=torch.device('cpu'), has_contrast=False, levels=None, roles=None,
            is_cat=torch.ones(2, dtype=torch.bool), reset_is_cat=torch.ones(2, dtype=torch.bool),
            crossed_is_plane=torch.ones(2, dtype=torch.bool), reset_yaws=torch.zeros(2),
            starts=torch.zeros(2, 3), reset_xy_scale=torch.ones(2, 2),
            goals=torch.tensor([[2., 0., .7]]).expand(2, -1), origins=torch.tensor([[-5., -5., -.5]]).expand(2, -1),
            shapes=torch.tensor([[60, 60, 30]]).expand(2, -1), dxs=torch.full((2,), .2),
            episode_lengths=torch.full((2,), 3), navigation_groups=torch.zeros(2, dtype=torch.long),
            reset_pool=torch.tensor(DEFAULT_QPOS)[None, None].expand(2, 1, -1),
            rooms={k: torch.as_tensor(v) for k, v in pack_room_scenes([None, None]).items()},
            contrast={k: torch.as_tensor(v) for k, v in pack_hand_contrast([None, None]).items()},
            acceptance=geometry(2))
        bank.probabilities = lambda weights=None, stage=None: torch.ones(2)/2 if weights is None else weights/weights.sum()
        def sample(name, positions, scene_ids):
            if name == 'sdf': return torch.ones((*positions.shape[:-1], 1))
            result = torch.zeros_like(positions)
            if name == 'gf': result[:, :, 0] = .7
            return result
        bank.sample = sample
        detector = Collision(); detectors.append(detector)
        tasks.append(CATTask(_CPUSimulation(2), bank, deepcopy(cfg), collision=detector, seed=13))
    observed, plain = tasks
    observer = TrainingAcceptance()
    for tick in range(5):
        for detector in detectors: detector.active = tick == 1
        observer.before_step(observed, torch.zeros(2, dtype=torch.long))
        a = observed.step(torch.zeros(2, 29))
        observer.after_step(observed, a)
        b = plain.step(torch.zeros(2, 29))
        for key in ('reward', 'done', 'terminated', 'truncated'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for key in a['obs']: torch.testing.assert_close(a['obs'][key], b['obs'][key], rtol=0, atol=0)
        for key in vars(observed.data):
            torch.testing.assert_close(getattr(observed.data, key), getattr(plain.data, key), rtol=0, atol=0)
        assert torch.equal(observed.generator.get_state(), plain.generator.get_state())
    assert observer.metrics()['training/passage_acceptance_hand_collision_count'] > 0


def test_prepared_pilot_is_bounded_online_and_reads_requested_checkpoint():
    import shlex
    from pathlib import Path
    from train_cat_mjlab import parser
    script = (Path(__file__).resolve().parents[1] / 'configs/pilots/hand_v5_acceptance_50.sh').read_text()
    arguments = script.split('train_cat_mjlab.py ', 1)[1].replace('\\\n', ' ')
    args = parser().parse_args(shlex.split(arguments))
    assert args.max_updates == 50
    assert str(args.checkpoint_native) == 'outputs/cat_hand_feasible_v5_seeded_pilot50_20260921/best.pt'
    assert args.wandb_mode == 'online' and args.wandb_project == 'CAT-wholebody' and args.wandb_entity == 'skvayzer'
    assert args.hand_raised_reset_fraction == .5
    assert 'acceptance_pilot50' in str(args.run_dir)


def test_training_deadline_uses_exact_step_count_not_float32_rounding():
    task = fake_task()
    task.dt = .02
    task.bank.episode_lengths[:] = 3333
    observer = TrainingAcceptance()
    observer.before_step(task, torch.zeros(2, dtype=torch.long))
    assert observer.gate.deadline.tolist() == pytest.approx([66.66, 66.66], abs=1e-10)
