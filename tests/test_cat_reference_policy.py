"""Action/scene masking, frozen references and exact native PPO continuation."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cat_ppo.learning.policy.ppo.reference_policy import (
    masked_gaussian_reference_kl, normalize_reference_config,
)
from test_cat_learner_continuity import ToyMixedEnv, assert_tree_equal, run_updates


def test_reference_direction_and_selected_leg_marginal_are_analytic():
    current_mean = jnp.zeros((4, 29))
    reference_mean = jnp.zeros_like(current_mean)
    current_std = jnp.full_like(current_mean, 2.)
    reference_std = jnp.ones_like(current_mean)
    value, fraction = masked_gaussian_reference_kl(current_mean, current_std,
        reference_mean, reference_std, jnp.array([0, 1, 0, 1]),
        action_indices=range(12), scene_mask=[True, False])
    assert float(value) == pytest.approx(12 * (np.log(2) + 1 / 8 - .5), rel=1e-6)
    assert float(fraction) == .5
    # Averaging is over selected CAT transitions, so adding room transitions
    # cannot dilute the coefficient applied to the original action marginal.
    all_cat, _ = masked_gaussian_reference_kl(current_mean[:1], current_std[:1],
        reference_mean[:1], reference_std[:1], jnp.array([0]),
        action_indices=range(12), scene_mask=[True, False])
    np.testing.assert_allclose(value, all_cat)


def test_gradient_only_updates_original_leg_outputs_on_cat_rows_and_never_reference():
    current_mean = jnp.ones((4, 29))
    current_std = jnp.ones_like(current_mean)
    reference_mean = jnp.zeros_like(current_mean)
    reference_std = jnp.ones_like(current_mean)
    def objective(cm, cs, rm, rs):
        return masked_gaussian_reference_kl(cm, cs, rm, rs, jnp.array([0, 1, 0, 1]),
            action_indices=range(12), scene_mask=[True, False])[0]
    gradients = jax.grad(objective, argnums=(0, 1, 2, 3))(
        current_mean, current_std, reference_mean, reference_std)
    for gradient in gradients[:2]:
        assert np.any(np.asarray(gradient)[[0, 2], :12] != 0)
        np.testing.assert_array_equal(np.asarray(gradient)[[1, 3]], 0.)
        np.testing.assert_array_equal(np.asarray(gradient)[:, 12:], 0.)
    for gradient in gradients[2:]:
        np.testing.assert_array_equal(gradient, 0.)


def test_room_only_minibatch_has_exact_zero_penalty_and_zero_gradients():
    def objective(mean):
        return masked_gaussian_reference_kl(mean, jnp.ones_like(mean), jnp.zeros_like(mean),
            jnp.ones_like(mean), jnp.array([1, 1]), action_indices=[0], scene_mask=[True, False])[0]
    means = jnp.ones((2, 29))
    assert float(objective(means)) == 0.
    np.testing.assert_array_equal(jax.grad(objective)(means), 0.)


@pytest.mark.parametrize("change", [dict(coefficient=0), dict(coefficient=float("nan")),
    dict(action_indices=[0, 0]), dict(action_indices=[29]), dict(scene_mask=[False, False]),
    dict(scene_mask=[1, 0]), dict(extra=True)])
def test_reference_contract_rejects_ambiguous_or_invalid_configuration(change):
    config = dict(coefficient=.05, action_indices=list(range(12)), scene_mask=[True, False])
    config.update(change)
    with pytest.raises(ValueError):
        normalize_reference_config(config, 29)
    assert normalize_reference_config(None, 29) is None


@pytest.fixture(scope="module")
def reference_training():
    # This fixture deliberately uses tiny one-action networks; production sets
    # action_indices=range(12) on the adapted 29-action actor.
    _, _, initial_result = run_updates(0)
    warmstart = initial_result[1]
    config = dict(coefficient=.05, action_indices=[0], scene_mask=[True, False])
    full, logs, _ = run_updates(3, restore_params=warmstart, restore_value_fn=True,
        reference_kl_config=config, normalize_observations=True, include_initial=True)
    resumed, _, _ = run_updates(1, restore=full[2], reference_kl_config=config,
                                normalize_observations=True)
    return config, full, resumed, logs, warmstart


def test_reference_actor_and_normalizer_stay_frozen_while_current_policy_learns(reference_training):
    config, full, _, logs, warmstart = reference_training
    assert [snapshot["step"] for snapshot in full] == [0, 12, 24, 36]
    for snapshot in full:
        assert snapshot["contract"]["reference_kl"]["coefficient"] == .05
        assert snapshot["contract"]["reference_kl"]["scene_mask"] == config["scene_mask"]
        assert_tree_equal(full[0]["reference_policy"], snapshot["reference_policy"])
    assert any(not np.array_equal(first, last) for first, last in zip(
        full[0]["training_state"]["leaves"], full[-1]["training_state"]["leaves"]))
    assert any("training/reference_kl" in metrics for _, metrics in logs)
    assert any("training/reference_cat_fraction" in metrics for _, metrics in logs)
    from cat_ppo.learning.policy.ppo.train import _runtime_tree_state
    expected = _runtime_tree_state({"policy": warmstart[1], "normalizer": warmstart[0]})
    assert_tree_equal(expected, full[0]["reference_policy"])


def test_exact_resume_restores_original_reference_not_last_student(reference_training):
    _, full, resumed, _, _ = reference_training
    assert resumed[-1]["step"] == full[-1]["step"] == 36
    for key in ("training_state", "env_state", "local_key", "key_envs", "reference_policy"):
        assert_tree_equal(full[-1][key], resumed[-1][key])


def test_resume_rejects_changed_loss_contract_and_missing_reference(reference_training):
    config, full, _, _, _ = reference_training
    with pytest.raises(ValueError, match="configuration differs"):
        run_updates(1, restore=full[1], reference_kl_config=dict(config, coefficient=.1),
                    normalize_observations=True)
    missing = copy.deepcopy(full[1])
    del missing["reference_policy"]
    with pytest.raises(ValueError, match="missing its frozen reference"):
        run_updates(1, restore=missing, reference_kl_config=config, normalize_observations=True)


def test_enabled_reference_requires_warmstart_and_scene_ids():
    config = dict(coefficient=.05, action_indices=[0], scene_mask=[True, False])
    with pytest.raises(ValueError, match="requires a warm-start"):
        run_updates(1, reference_kl_config=config)
    _, _, initial_result = run_updates(0)
    class NoSceneIds(ToyMixedEnv):
        def reset(self, keys):
            state = super().reset(keys)
            state.info.pop("pf_id")
            return state
    with pytest.raises(ValueError, match="pf_id scene attribution"):
        run_updates(1, environment=NoSceneIds(), restore_params=initial_result[1],
                    reference_kl_config=config)
