"""Recovery keeps the saved actor's state-dependent leg sigma in every PPO path."""

import copy
import functools
import hashlib
import json

from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cat_ppo.learning.policy.ppo.wholebody_distribution import (
    COHERENT_EXPLORATION_VERSION, WholeBodyNormalTanhDistribution,
    checkpoint_distribution_config, load_checkpoint_policy, make_ppo_networks,
    pack_leg_noise_reference, unpack_leg_noise_reference,
)


SIZES = {"state": (222,), "privileged_state": (310,)}
BASE = dict(policy_hidden_layer_sizes=(10, 7), value_hidden_layer_sizes=(13,),
            policy_obs_key="state", value_obs_key="privileged_state",
            leg_action_count=12, upper_std_min=.02, upper_std_max=.1,
            upper_entropy_weight=0.)
V2 = dict(exploration_version=COHERENT_EXPLORATION_VERSION, arm_persistence=.95,
          arm_correlation=.8, arm_last_action_indices=list(range(79, 93)),
          arm_correlation_pattern="g1_raise_tuck_v1", arm_innovation_scale=(1 - .95 ** 2) ** .5)


def _fixture(coherent=True):
    kwargs = {**BASE, **(V2 if coherent else {})}
    source_factory = functools.partial(make_ppo_networks, **kwargs)
    source = source_factory(SIZES, 29)
    actor = source.policy_network.init(jax.random.PRNGKey(2))
    head = actor["params"]["hidden_2"]
    head["kernel"] = head["kernel"].at[:, 29:].multiply(.1)
    head["bias"] = head["bias"].at[29:41].set(-1.3).at[41:].set(-3.)
    normalizer = running_statistics.init_state({
        key: specs.Array(shape, jnp.dtype("float32")) for key, shape in SIZES.items()})
    params = (normalizer, actor, source.value_network.init(jax.random.PRNGKey(3)))
    config = checkpoint.network_config(SIZES, 29, False, source_factory)
    reference = pack_leg_noise_reference(actor, config)
    factory = functools.partial(make_ppo_networks, **kwargs, leg_noise_reference=reference)
    recovery = factory(SIZES, 29)
    obs = {key: jax.random.uniform(jax.random.PRNGKey(i), (4, *shape), minval=-.2, maxval=.2)
           for i, (key, shape) in enumerate(SIZES.items())}
    return source, recovery, params, obs, reference, factory, config


@pytest.mark.parametrize("coherent", [False, True])
def test_baseline_all_policy_paths_and_critic_are_exact(coherent):
    source, recovery, params, obs, payload, _, _ = _fixture(coherent)
    expected = source.policy_network.apply(params[0], params[1], obs)
    actual = recovery.policy_network.apply(params[0], params[1], obs)
    np.testing.assert_array_equal(actual, expected)
    # Reference is state dependent, not one scalar or per-joint constant.
    assert np.max(np.std(np.asarray(actual[..., 29:41]), axis=0)) > 1e-4
    key = jax.random.PRNGKey(11)
    for operation in ("sample_no_postprocessing", "entropy"):
        np.testing.assert_array_equal(
            getattr(source.parametric_action_distribution, operation)(expected, key),
            getattr(recovery.parametric_action_distribution, operation)(actual, key))
    raw = source.parametric_action_distribution.sample_no_postprocessing(expected, key)
    np.testing.assert_array_equal(source.parametric_action_distribution.log_prob(expected, raw),
                                  recovery.parametric_action_distribution.log_prob(actual, raw))
    np.testing.assert_array_equal(source.value_network.apply(params[0], params[2], obs),
                                  recovery.value_network.apply(params[0], params[2], obs))
    restored = unpack_leg_noise_reference(json.loads(json.dumps(payload)))
    for expected_leaf, actual_leaf in zip(jax.tree.leaves(params[1]), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(expected_leaf, actual_leaf)
    config = recovery.parametric_action_distribution.config
    assert config["leg_noise_reference"]["sha256"] == payload["sha256"]
    assert WholeBodyNormalTanhDistribution.from_config(config).config == config


@pytest.mark.parametrize("coherent", [False, True])
def test_current_trunk_and_head_updates_cannot_change_leg_noise(coherent):
    source, recovery, params, obs, _, _, _ = _fixture(coherent)
    changed = jax.tree.map(lambda x: x + .07, params[1])
    baseline = recovery.policy_network.apply(params[0], params[1], obs)
    current = jax.jit(recovery.policy_network.apply)(params[0], changed, obs)
    unprotected = source.policy_network.apply(params[0], changed, obs)
    np.testing.assert_allclose(current[..., 29:41], baseline[..., 29:41], atol=1e-7)
    assert np.max(np.abs(np.asarray(unprotected[..., 29:41] - baseline[..., 29:41]))) > .01
    # All means, plus waist/arm scales, remain the current actor's outputs.
    np.testing.assert_allclose(current[..., :29], unprotected[..., :29], atol=1e-7)
    np.testing.assert_allclose(current[..., 41:], unprotected[..., 41:], atol=1e-7)
    assert np.max(np.abs(np.asarray(current[..., :29] - baseline[..., :29]))) > .01
    dist = recovery.parametric_action_distribution
    old_scale, new_scale = dist.create_dist(baseline).scale, dist.create_dist(current).scale
    np.testing.assert_allclose(new_scale[..., :12], old_scale[..., :12], atol=1e-7)
    assert np.max(np.abs(np.asarray(new_scale[..., 15:] - old_scale[..., 15:]))) > .001


def test_fixed_scale_gradient_is_zero_but_leg_mean_and_arm_gradients_survive():
    _, recovery, params, obs, _, _, _ = _fixture()
    apply = lambda actor: recovery.policy_network.apply(params[0], actor, obs)
    scale_grad = jax.grad(lambda actor: jnp.sum(apply(actor)[..., 29:41]))(params[1])
    for leaf in jax.tree.leaves(scale_grad):
        np.testing.assert_array_equal(leaf, 0.)
    dist = recovery.parametric_action_distribution
    raw = dist.sample_no_postprocessing(apply(params[1]), jax.random.PRNGKey(77))
    gradient = jax.grad(lambda actor: -dist.log_prob(apply(actor), raw).mean())(params[1])
    head = gradient["params"]["hidden_2"]
    np.testing.assert_array_equal(head["kernel"][:, 29:41], 0.)
    np.testing.assert_array_equal(head["bias"][29:41], 0.)
    assert np.linalg.norm(head["kernel"][:, :12]) > .01
    assert np.linalg.norm(head["kernel"][:, 15:29]) > .01
    assert np.linalg.norm(head["kernel"][:, 44:58]) > .001
    assert np.linalg.norm(gradient["params"]["hidden_0"]["kernel"]) > .01


@pytest.mark.parametrize("coherent", [False, True])
def test_native_checkpoint_self_contained_roundtrip_after_training(tmp_path, coherent):
    _, recovery, params, obs, payload, factory, _ = _fixture(coherent)
    params = (params[0], jax.tree.map(lambda x: x + .07, params[1]), params[2])
    config = checkpoint.network_config(SIZES, 29, False, factory)
    config.action_distribution = recovery.parametric_action_distribution.config
    checkpoint.save(tmp_path, 1, params, config)
    path = tmp_path / "000000000001"
    saved = json.loads((path / "ppo_network_config.json").read_text())
    assert saved["network_factory_kwargs"]["leg_noise_reference"]["sha256"] == payload["sha256"]
    assert checkpoint_distribution_config(saved) == recovery.parametric_action_distribution.config
    for deterministic in (False, True):
        expected = networks.make_inference_fn(recovery)(params, deterministic=deterministic)(
            obs, jax.random.PRNGKey(87))
        actual = load_checkpoint_policy(path, deterministic=deterministic)(obs, jax.random.PRNGKey(87))
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(a, b)


def test_incompatible_or_corrupted_reference_fails_closed():
    _, network, params, _, payload, factory, source_config = _fixture()
    config = checkpoint.network_config(SIZES, 29, False, factory).to_dict()
    config["action_distribution"] = network.parametric_action_distribution.config
    changed = copy.deepcopy(config)
    changed["network_factory_kwargs"]["leg_noise_reference"]["layers"][0]["bias"]["base64"] = "AAAA"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        checkpoint_distribution_config(changed)
    changed = copy.deepcopy(config)
    changed["action_distribution"]["leg_noise_reference"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="disagrees"):
        checkpoint_distribution_config(changed)
    changed = copy.deepcopy(config)
    changed["network_factory_kwargs"]["leg_noise_reference"] = None
    with pytest.raises(ValueError, match="disagrees"):
        checkpoint_distribution_config(changed)
    changed = copy.deepcopy(config)
    changed["normalize_observations"] = True
    with pytest.raises(ValueError, match="unnormalized"):
        checkpoint_distribution_config(changed)
    normalized_source = source_config.to_dict()
    normalized_source["normalize_observations"] = True
    with pytest.raises(ValueError, match="unnormalized"):
        pack_leg_noise_reference(params[1], normalized_source)
    with pytest.raises(ValueError, match="architecture differs"):
        make_ppo_networks(SIZES, 29, **{**BASE, "policy_hidden_layer_sizes": (12, 7)},
                          **V2, leg_noise_reference=payload)
    changed_payload = copy.deepcopy(payload)
    changed_payload["layers"][0]["kernel"]["shape"][0] += 1
    changed_payload["sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in changed_payload.items() if k != "sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="shape/dtype"):
        unpack_leg_noise_reference(changed_payload)


@pytest.mark.parametrize("coherent", [False, True])
def test_old_checkpoint_configs_and_resaving_fixed_reference_remain_compatible(coherent):
    source, _, params, _, payload, factory, old = _fixture(coherent)
    old = old.to_dict()
    del old["network_factory_kwargs"]["leg_noise_reference"]
    assert checkpoint_distribution_config(old) == source.parametric_action_distribution.config
    assert "leg_noise_reference" not in checkpoint_distribution_config(old)
    newer = checkpoint.network_config(SIZES, 29, False, factory)
    assert pack_leg_noise_reference(jax.tree.map(lambda x: x + 1., params[1]), newer) == payload
