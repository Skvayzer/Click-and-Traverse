"""CPU checks for re-solving reset masses from realized episode lengths."""
import torch
from cat_mjlab.runner import _solve_experience_masses, _adapt_experience_masses


class Bank:
    def __init__(self, targets, lengths, every):
        self.experience_targets = list(targets); self.experience_lengths = list(lengths)
        self.experience_rebalance_every = every; self.experience_updates = 0
        self.experience_length_sum = [0.] * len(targets); self.experience_length_count = [0.] * len(targets)
        self.sampling_masses = torch.as_tensor(_solve_experience_masses(targets, lengths))


def test_solver_equalises_transition_share():
    masses = _solve_experience_masses([.6, .4], [1000., 4000.])
    share = [m * l for m, l in zip(masses, [1000., 4000.])]
    assert abs(share[0] / sum(share) - .6) < 1e-9


def test_resolve_follows_realized_lengths_and_logs_share():
    bank = Bank([.6, .4], [1000., 4000.], every=2)
    before = bank.sampling_masses.clone()
    # CAT episodes really end at 300 steps: on the old masses CAT's share collapses.
    steps = torch.tensor([300. * 60, 4000. * 40]); sums = torch.tensor([300. * 60, 4000. * 40]); counts = torch.tensor([60., 40.])
    info = dict(metrics={})
    _adapt_experience_masses(bank, steps, sums, counts, info)      # update 1: accumulate only
    assert torch.equal(bank.sampling_masses, before)
    assert abs(info['metrics']['balance/group0_experience_share'] - 18000 / 178000) < 1e-9
    _adapt_experience_masses(bank, steps, sums, counts, info)      # update 2: re-solve
    assert bank.experience_lengths[0] < 1000. and bank.sampling_masses[0] > before[0]
    assert bank.experience_length_count == [0., 0.]
    assert 'balance/group0_reset_mass' in info['metrics']


def test_small_groups_keep_their_estimate_and_zero_disables():
    bank = Bank([.5, .5], [1000., 1000.], every=1)
    _adapt_experience_masses(bank, torch.tensor([100., 100.]), torch.tensor([50., 5000.]), torch.tensor([5., 5.]), dict(metrics={}))
    assert bank.experience_lengths == [1000., 1000.]
    off = Bank([.5, .5], [1000., 1000.], every=0); m = off.sampling_masses.clone()
    _adapt_experience_masses(off, torch.tensor([1., 1.]), torch.tensor([10., 4000.]), torch.tensor([30., 30.]), dict(metrics={}))
    assert torch.equal(off.sampling_masses, m)


class Objects:
    reactive_fraction = .25


def test_reactive_fraction_is_solved_jointly_and_scaled_targets_hold():
    bank = Bank([.6, .4], [1000., 4000.], every=1)
    bank.experience_reactive_target = .10; bank.experience_length_sum = [0., 0.]; bank.experience_length_count = [0., 0.]
    bank.experience_reactive_length = None; bank.experience_reactive_sum = bank.experience_reactive_count = 0.
    objects = Objects(); info = dict(metrics={})
    # Goal episodes really last 150 steps, reactive ones 500: on a .25 coin reactive would hold ~53% of steps.
    steps = torch.tensor([150. * 100, 150. * 100]); sums = steps.clone(); counts = torch.tensor([100., 100.])
    _adapt_experience_masses(bank, steps, sums, counts, info, reactive=(500. * 60, 500. * 60, 60., objects))
    p = objects.reactive_fraction
    assert p < .25
    # Steady-state share check: reactive = p*L_r / (p*L_r + (1-p)*sum(m_g*L_g)).
    m = bank.sampling_masses.tolist(); L = bank.experience_lengths
    goal = (1 - p) * sum(mi * li for mi, li in zip(m, L)); react = p * bank.experience_reactive_length
    assert abs(react / (react + goal) - .10) < 1e-6
    share0 = (1 - p) * m[0] * L[0] / (react + goal)
    assert abs(share0 - .6 * .9) < 1e-6
    assert info['metrics']['balance/reactive_reset_fraction'] == p
