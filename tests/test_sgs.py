"""Success-Guided Sampling (arXiv 2610.12465) in CATTask: kernel, group masses, ring buffers, priors."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from cat_mjlab.config import wholebody_config
from cat_mjlab.task import CATTask


def make(S=6, K=2, H=4, mix=0., prior=None, num_envs=8, tmp=None):
    t = SimpleNamespace()
    t.device = torch.device("cpu"); t.num_envs = num_envs
    t.generator = torch.Generator().manual_seed(0)
    t._rand = lambda shape, low=0., high=1.: low + (high - low) * torch.rand(shape, generator=t.generator)
    t.bank = SimpleNamespace(count=S, manifest={"scenes": [{"scene_id": f"s{i}"} for i in range(S)]},
                             weights=torch.ones(S), sampling_ids=torch.tensor([0, 0, 0, 1, 1, 1][:S]),
                             sampling_masses=torch.tensor([.7, .3]))
    files = []
    if prior is not None:
        path = tmp / "prior.json"; path.write_text(json.dumps({"per_scene": prior})); files = [str(path)]
    t.config = {"sgs": dict(slots=K, history=H, target=.66, kappa=5., temperature=2., epsilon=1e-8, uniform_mix=mix,
                            prior_files=files, prior_count=4.)}
    t.episode = {"goal_reached": torch.zeros(num_envs, dtype=torch.bool), "obstacle": torch.zeros(num_envs, dtype=torch.bool)}
    for name in ("_init_sgs", "_sgs_success", "_sgs_probabilities", "_sgs_record"):
        setattr(t, name, getattr(CATTask, name).__get__(t))
    t._init_sgs()
    return t


def kernel(p, t=.66, k=5., T=2., eps=1e-8):
    w = (p + eps) ** (k * t) * (1 - p + eps) ** (k * (1 - t))
    return math.exp(math.log(max(w, eps) + eps) / max(T, 1.))


def test_kernel_and_group_masses(tmp_path):
    prior = {"s0": {"clean": 0.}, "s1": {"clean": .66}, "s2": {"clean": 1.}, "s3": {"clean": .3}, "s4": {"clean": .9}}
    t = make(prior=prior, tmp=tmp_path)
    prob = t._sgs_probabilities()
    assert prob.shape == (12,)
    assert torch.allclose(prob[:6].sum(), torch.tensor(.7)) and torch.allclose(prob[6:].sum(), torch.tensor(.3))
    # within group 0: ratios follow the paper's kernel
    assert torch.allclose(prob[2] / prob[0], torch.tensor(kernel(.66) / kernel(0.)), rtol=1e-4)
    assert prob[2] > prob[4] and prob[10] > prob[8]  # same group: frontier (.66) > mastered (1.0); no prior (= t) > .9
    assert torch.allclose(prob[4], prob[0])        # p = 1 and p = 0 both hit the epsilon floor (paper behaviour)
    assert torch.equal(prob[0], prob[1])           # both slots of a scene share the prior
    assert abs(float(t.sgs_p[10]) - .66) < 1e-6    # s5 has no prior: p = t


def test_uniform_mix_and_zero_weight(tmp_path):
    t = make(mix=.5, prior={"s0": {"clean": 0.}, "s1": {"clean": .66}, "s2": {"clean": .66}}, tmp=tmp_path)
    t.bank.weights[2] = 0.
    prob = t._sgs_probabilities()
    assert float(prob[4]) == 0. and float(prob[5]) == 0.
    assert float(prob[0]) >= .7 * .5 / 4 - 1e-6    # the uniform half reaches a hopeless configuration
    assert torch.allclose(prob[:6].sum(), torch.tensor(.7))


def test_ring_buffer_records_duplicates_and_window(tmp_path):
    t = make(H=4, num_envs=6, prior={}, tmp=tmp_path)
    t.sgs_env_config[:] = torch.tensor([3, 3, 3, 5, -1, 3])
    t.episode["goal_reached"][:] = torch.tensor([True, True, False, True, True, True])
    t.episode["obstacle"][:] = torch.tensor([False, True, False, False, False, False])
    t._sgs_record(torch.ones(6, dtype=torch.bool))
    assert int(t.sgs_count[3]) == 4 and int(t.sgs_count[5]) == 1 and int(t.sgs_count.sum()) == 5
    assert int(t.sgs_hist[3].sum()) == 2            # envs 0 and 5 clean; 1 touched, 2 no goal
    assert abs(float(t._sgs_success()[3]) - .5) < 1e-6 and float(t._sgs_success()[5]) == 1.
    t.sgs_env_config[:] = torch.tensor([3, 3, -1, -1, -1, -1])
    t.episode["goal_reached"][:] = True; t.episode["obstacle"][:] = False
    t._sgs_record(torch.ones(6, dtype=torch.bool))  # window 4: the two oldest outcomes drop out
    assert int(t.sgs_count[3]) == 6 and abs(float(t._sgs_success()[3]) - .75) < 1e-6


def test_prior_fades_with_outcomes(tmp_path):
    t = make(H=100, num_envs=1, prior={"s0": {"clean": 1.}}, tmp=tmp_path)
    t.sgs_env_config[:] = 0
    for _ in range(4):
        t._sgs_record(torch.ones(1, dtype=torch.bool))  # 4 failures (goal not reached)
    assert abs(float(t._sgs_success()[0]) - .5) < 1e-6  # (0 + 4*1) / (4 + 4)


def test_config_validation():
    c = wholebody_config(sgs=dict(slots=4, history=100, target=.66, kappa=5., temperature=2., epsilon=1e-8, uniform_mix=.05))
    assert c["sgs"]["slots"] == 4
    with pytest.raises(ValueError):
        wholebody_config(sgs=dict(slots=4, history=100, target=1.2, kappa=5., temperature=2., epsilon=1e-8))
