"""Probability invariants for thousands of scenes in one learner."""
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np

from cat_ppo.learning.train.pf_utils import SamplePFWrapper, balanced_scene_logits


def test_adaptive_probabilities_keep_family_mass_with_unequal_scene_counts():
    groups = jp.asarray([0]*64 + [1]*2226 + [2]*24 + [3]*24)
    mass = jp.asarray([.2, .4, .25, .15])
    weights = jp.linspace(.001, 1, len(groups))
    probabilities = jax.nn.softmax(jax.jit(balanced_scene_logits)(weights, groups, mass))
    observed = np.bincount(np.asarray(groups), weights=np.asarray(probabilities), minlength=4)
    np.testing.assert_allclose(observed, mass, rtol=2e-6)
    assert np.all(np.asarray(probabilities) > 0)
    assert probabilities[64] < probabilities[1000]


def test_expanded_sampling_update_counts_finished_episodes_and_preserves_groups():
    groups = jp.array([0, 0, 1, 1, 1, 2, 3])
    masses = jp.array([.2, .4, .25, .15])
    batch, scenes = 5, len(groups)
    tile = lambda value: jp.broadcast_to(value, (batch,)+value.shape)
    info = {
        "pf_id": jp.array([0, 0, 2, 5, 6]), "truncation": jp.array([1., 0., 0., 1., 0.]),
        "pf_success_ema": jp.zeros((batch, scenes)), "pf_episode_ema": jp.zeros((batch, scenes)),
        "pf_sampling_logits": jp.zeros((batch, scenes)), "pf_sampling_ema_decay": jp.ones(batch)*.95,
        "pf_sampling_alpha": jp.ones(batch), "pf_sampling_group_ids": tile(groups),
        "pf_sampling_group_masses": tile(masses),
    }
    def update(info, done):
        state, logits = SamplePFWrapper._update_pf_sampling_info(SimpleNamespace(info=info), done)
        return state.info, logits
    updated, logits = jax.jit(update)(info, jp.array([1., 1., 0., 1., 1.]))
    np.testing.assert_array_equal(updated["pf_episode_ema"][0], [2, 0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(updated["pf_success_ema"][0], [1, 0, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(updated["pf_episode_ema"], np.broadcast_to(updated["pf_episode_ema"][0], (batch,scenes)))
    p = np.asarray(jax.nn.softmax(logits))
    np.testing.assert_allclose(np.bincount(np.asarray(groups), weights=p, minlength=4), masses, rtol=1e-6)
    assert p[0] < p[1]  # CAT's timeout-survival weighting still acts within a family.


def test_absent_groups_have_zero_mass_without_nan():
    logits = balanced_scene_logits(jp.array([1., .1, .5]), jp.array([0, 0, 2]), jp.array([.4,0,.6,0]))
    p = np.asarray(jax.nn.softmax(logits))
    assert np.isfinite(p).all()
    np.testing.assert_allclose([p[:2].sum(), p[2]], [.4,.6], rtol=1e-6)
