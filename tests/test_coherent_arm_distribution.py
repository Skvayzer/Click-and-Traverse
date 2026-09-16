"""Conditional joint-density, replay, temporal covariance and checkpoint regressions."""
import copy
import functools
import json

from brax.training import distribution, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from cat_ppo.furniture.control import wholebody_observation_contract
from cat_ppo.learning.policy.ppo.wholebody_distribution import (
    COHERENT_DISTRIBUTION_KIND, COHERENT_EXPLORATION_VERSION,
    CoherentArmNormalTanhDistribution, WholeBodyNormalTanhDistribution,
    checkpoint_distribution_config, load_checkpoint_policy, make_ppo_networks,
)


BASE = dict(leg_action_count=12, upper_std_min=.02, upper_std_max=.10,
            upper_entropy_weight=0.)
CONTRACT = wholebody_observation_contract()
V2 = dict(exploration_version=COHERENT_EXPLORATION_VERSION, arm_persistence=.95,
          arm_correlation=.8, arm_correlation_pattern="g1_raise_tuck_v1",
          arm_last_action_indices=[CONTRACT["actor_features"].index("last_action." + name)
                                   for name in CONTRACT["action_names"][15:]])


def _distribution(**overrides):
    return CoherentArmNormalTanhDistribution(29, **{**BASE, **V2, **overrides})


def _params(shape=(3,), sigma=.05):
    loc = jnp.zeros((*shape, 29))
    return jnp.concatenate([loc, jnp.full_like(loc, jnp.log(jnp.expm1(sigma - .001)))], -1)


def _observations(shape=(3,)):
    state = jnp.zeros((*shape, 222)).at[..., jnp.array(V2["arm_last_action_indices"])].set(.2)
    return {"state": state, "privileged_state": jnp.zeros((*shape, 310))}


def test_joint_density_matches_explicit_full_gaussian_and_tanh_jacobian():
    d = _distribution()
    parameters = d.condition_parameters(_params(), _observations()["state"])
    raw = d.sample_no_postprocessing(parameters, jax.random.PRNGKey(17))
    joint = d.create_dist(parameters)
    covariance = np.eye(29)
    covariance[15:, 15:] = d.arm_correlation_matrix
    expected = []
    jac = np.asarray(distribution.TanhBijector().forward_log_det_jacobian(raw))
    for i in range(3):
        sigma = np.asarray(joint.scale[i])
        cov = sigma[:, None] * covariance * sigma[None, :]
        expected.append(multivariate_normal.logpdf(np.asarray(raw[i]),
                        mean=np.asarray(joint.loc[i]), cov=cov) - jac[i].sum())
    np.testing.assert_allclose(d.log_prob(parameters, raw), expected, atol=2e-5, rtol=2e-6)
    # Ignoring correlation really would give the wrong PPO probability.
    independent = WholeBodyNormalTanhDistribution(29).log_prob(parameters, raw)
    assert np.max(np.abs(np.asarray(independent) - expected)) > .1


def test_sampling_preserves_exact_v1_leg_and_waist_draws_and_entropy():
    p = _params((8,))
    old = WholeBodyNormalTanhDistribution(29)
    new = _distribution()
    conditioned = new.condition_parameters(p, _observations((8,))["state"])
    key = jax.random.PRNGKey(91)
    np.testing.assert_array_equal(old.sample_no_postprocessing(p, key)[..., :15],
                                  new.sample_no_postprocessing(conditioned, key)[..., :15])
    np.testing.assert_array_equal(old.entropy(p, key), new.entropy(conditioned, key))
    grad = jax.grad(lambda x: new.entropy(x, key).sum())(conditioned)
    np.testing.assert_array_equal(grad[..., 12:29], 0)
    np.testing.assert_array_equal(grad[..., 41:58], 0)
    # Gaussian leg marginals used by the existing reference KL stay identical.
    np.testing.assert_array_equal(old.create_dist(p).loc[..., :12],
                                  new.create_dist(conditioned).loc[..., :12])
    np.testing.assert_array_equal(old.create_dist(p).scale[..., :12],
                                  new.create_dist(conditioned).scale[..., :12])


def test_nonzero_upper_entropy_uses_joint_log_determinant_and_same_tanh_sample():
    d = _distribution(upper_entropy_weight=.3)
    p = _params()
    key = jax.random.PRNGKey(12)
    joint = d.create_dist(p)
    leg = distribution.NormalTanhDistribution(12)
    leg_p = jnp.concatenate([p[..., :12], p[..., 29:41]], -1)
    raw = joint.sample(jax.random.fold_in(key, 1))
    scale = np.asarray(joint.scale[0])
    sign, logdet = np.linalg.slogdet(d.arm_correlation_matrix)
    assert sign > 0
    gaussian_upper_entropy = np.sum(.5 * np.log(2 * np.pi * np.e * scale[12:] ** 2)) + .5 * logdet
    correction = distribution.TanhBijector().forward_log_det_jacobian(raw)[..., 12:].sum(-1)
    expected = leg.entropy(leg_p, key) + .3 * (gaussian_upper_entropy + correction)
    np.testing.assert_allclose(d.entropy(p, key), expected, atol=2e-6)


def test_replay_and_changed_policy_ratio_use_stored_observation_without_hidden_state():
    d = _distribution()
    obs = _observations((4, 5))["state"]
    obs = obs.at[..., V2["arm_last_action_indices"][0]].set(jnp.arange(20).reshape(4, 5) / 30.)
    p = _params((4, 5))
    old = d.condition_parameters(p, obs)
    raw = d.sample_no_postprocessing(old, jax.random.PRNGKey(20))
    old_logprob = d.log_prob(old, raw)
    replay = jax.jit(lambda params, states, actions:
                     d.log_prob(d.condition_parameters(params, states), actions))
    np.testing.assert_allclose(jnp.exp(replay(p, obs, raw) - old_logprob), 1., atol=1e-5)
    changed = p.at[..., 15].add(.4).at[..., 44].add(.1)
    changed_lp = replay(changed, obs, raw)
    permutation = jnp.array([2, 0, 3, 1])
    np.testing.assert_allclose(replay(changed[permutation], obs[permutation], raw[permutation]),
                               changed_lp[permutation], atol=1e-5)
    assert np.std(np.asarray(jnp.exp(changed_lp - old_logprob))) > .01
    # Finite difference the ACTUAL conditioned density, including (1-rho).
    objective = lambda mean: d.log_prob(d.condition_parameters(p.at[..., 15].set(mean), obs), raw).sum()
    derivative = jax.grad(objective)(.2)
    numerical = (objective(.201) - objective(.199)) / .002
    np.testing.assert_allclose(derivative, numerical, atol=.02, rtol=.02)


def test_temporal_exploration_matches_finite_ar_covariance_without_raising_innovation_bounds():
    d = _distribution()
    batch, steps = 4096, 120
    parameters = _params((batch,))
    observations = jnp.zeros((batch, 222))
    indices = jnp.array(V2["arm_last_action_indices"])
    def step(carry, key):
        previous, obs = carry
        p = d.condition_parameters(parameters, obs)
        current = d.sample_no_postprocessing(p, key)
        obs = obs.at[..., indices].set(jnp.tanh(current[..., 15:]))
        return (current, obs), (previous[..., 15:], current[..., 15:])
    keys = jax.random.split(jax.random.PRNGKey(191), steps)
    (_, _), (previous, current) = jax.jit(lambda: jax.lax.scan(
        step, (jnp.zeros((batch, 29)), observations), keys))()
    prev, last = np.asarray(previous[-1]), np.asarray(current[-1])
    expected_variance = .05 ** 2 * (1 - .95 ** (2 * steps)) / (1 - .95 ** 2)
    empirical = np.cov(last, rowvar=False)
    np.testing.assert_allclose(np.diag(empirical), expected_variance, rtol=.065)
    expected_covariance = expected_variance * d.arm_correlation_matrix
    np.testing.assert_allclose(empirical, expected_covariance, atol=.0016)
    assert .935 < np.corrcoef(prev[:, 0], last[:, 0])[0, 1] < .965
    assert np.isfinite(last).all()
    np.testing.assert_allclose(d.create_dist(parameters).scale[..., 15:], .05, atol=1e-7)


def _factory(normalize=False):
    kwargs = dict(policy_hidden_layer_sizes=(10, 7), value_hidden_layer_sizes=(13,),
                  policy_obs_key="state", value_obs_key="privileged_state")
    sizes = {"state": (222,), "privileged_state": (310,)}
    factory = functools.partial(make_ppo_networks, **kwargs, **BASE, **V2)
    preprocess = running_statistics.normalize if normalize else types.identity_observation_preprocessor
    network = factory(sizes, 29, preprocess_observations_fn=preprocess)
    normalizer = running_statistics.init_state({
        name: specs.Array(size, jnp.dtype("float32")) for name, size in sizes.items()})
    normalizer = normalizer.replace(mean={name: jnp.full(size, .7) for name, size in sizes.items()},
                                    std={name: jnp.full(size, .3) for name, size in sizes.items()})
    params = (normalizer, network.policy_network.init(jax.random.PRNGKey(0)),
              network.value_network.init(jax.random.PRNGKey(1)))
    return factory, network, params, sizes, kwargs, preprocess


@pytest.mark.parametrize("normalize", [False, True])
def test_warmstart_shapes_values_and_checkpoint_roundtrip_are_exact(tmp_path, normalize):
    factory, network, params, sizes, kwargs, preprocess = _factory(normalize)
    v1 = make_ppo_networks(sizes, 29, **kwargs, **BASE, preprocess_observations_fn=preprocess)
    for a, b in zip(jax.tree.leaves(params[1]),
                    jax.tree.leaves(v1.policy_network.init(jax.random.PRNGKey(0)))):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(params[2]),
                    jax.tree.leaves(v1.value_network.init(jax.random.PRNGKey(1)))):
        np.testing.assert_array_equal(a, b)
    obs = _observations()
    native_logits = v1.policy_network.apply(params[0], params[1], obs)
    conditioned = network.policy_network.apply(params[0], params[1], obs)
    expected_arm_means = .05 * native_logits[..., 15:29] + .95 * jnp.arctanh(.2)
    np.testing.assert_allclose(conditioned[..., 15:29], expected_arm_means, atol=1e-7)
    np.testing.assert_array_equal(conditioned[..., :15], native_logits[..., :15])
    np.testing.assert_array_equal(conditioned[..., 29:], native_logits[..., 29:])
    np.testing.assert_array_equal(network.value_network.apply(params[0], params[2], obs),
                                  v1.value_network.apply(params[0], params[2], obs))
    config = checkpoint.network_config(sizes, 29, normalize, factory)
    checkpoint.save(tmp_path, 1, params, config)
    path = tmp_path / "000000000001"
    assert checkpoint_distribution_config(config)["kind"] == COHERENT_DISTRIBUTION_KIND
    restored = WholeBodyNormalTanhDistribution.from_config(json.loads(json.dumps(
        network.parametric_action_distribution.config)))
    assert restored.config == network.parametric_action_distribution.config
    for deterministic in (False, True):
        actual = load_checkpoint_policy(path, deterministic=deterministic)(obs, jax.random.PRNGKey(19))
        expected = networks.make_inference_fn(network)(params, deterministic=deterministic)(
            obs, jax.random.PRNGKey(19))
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(a, b)
    deterministic = networks.make_inference_fn(network)(params, deterministic=True)
    np.testing.assert_array_equal(deterministic(obs, jax.random.PRNGKey(99))[0], jnp.tanh(conditioned[..., :29]))


def test_configuration_rejects_partial_v2_and_mismatched_semantics():
    factory, _, _, sizes, _, _ = _factory()
    config = checkpoint.network_config(sizes, 29, False, factory).to_dict()
    for missing in V2:
        partial = copy.deepcopy(config)
        del partial["network_factory_kwargs"][missing]
        with pytest.raises(ValueError, match="all five"):
            checkpoint_distribution_config(partial)
    for key in V2:
        null = copy.deepcopy(config)
        null["network_factory_kwargs"][key] = None
        with pytest.raises(ValueError, match="null"):
            checkpoint_distribution_config(null)
    changed = copy.deepcopy(config)
    changed["action_distribution"] = _distribution(arm_persistence=.9).config
    with pytest.raises(ValueError, match="disagrees"):
        checkpoint_distribution_config(changed)
    wrong = _distribution().config
    wrong["previous_action_clip"] = .99
    with pytest.raises(ValueError, match="semantics differ"):
        WholeBodyNormalTanhDistribution.from_config(wrong)

    # Brax includes None signature defaults when saving v1 with the new
    # factory. Accept that encoding, but not a claimed v2 with empty settings.
    empty = copy.deepcopy(config)
    for key in V2:
        empty["network_factory_kwargs"][key] = None
    assert checkpoint_distribution_config(empty) == WholeBodyNormalTanhDistribution(29).config
    empty["action_distribution"] = _distribution().config
    with pytest.raises(ValueError, match="disagrees"):
        checkpoint_distribution_config(empty)


@pytest.mark.parametrize("changes", [
    {"arm_persistence": 1.}, {"arm_persistence": -1.}, {"arm_persistence": float("nan")},
    {"arm_correlation": 1.}, {"arm_correlation": -1.},
    {"arm_correlation_pattern": "unknown"}, {"exploration_version": "future"},
    {"arm_last_action_indices": list(range(13))}, {"arm_last_action_indices": [1] * 14},
])
def test_invalid_coherence_configuration_fails(changes):
    with pytest.raises(ValueError):
        _distribution(**changes)


def test_near_saturated_previous_actions_remain_finite_and_reset_history_clears_memory():
    d = _distribution()
    p = _params()
    state = _observations()["state"].at[..., jnp.array(V2["arm_last_action_indices"])].set(1.)
    saturated = d.condition_parameters(p, state)
    assert np.isfinite(np.asarray(saturated)).all()
    assert np.isfinite(np.asarray(d.log_prob(saturated, saturated[..., :29]))).all()
    reset = d.condition_parameters(p, jnp.zeros_like(state))
    np.testing.assert_array_equal(d.mode(reset), jnp.zeros((3, 29)))


def test_correlation_axes_match_fk_verified_g1_raise_and_tuck_directions():
    import mujoco
    from cat_ppo.envs.g1.constants import DEFAULT_QPOS
    from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml
    from cat_ppo.furniture.control import JOINT_NAMES, posture_actions

    assert V2["arm_last_action_indices"] == list(range(79, 93))
    raised = posture_actions(np.zeros(29), JOINT_NAMES, "raised")[15:]
    tucked = posture_actions(np.zeros(29), JOINT_NAMES, "tucked")[15:]
    factors = np.zeros((14, 4))
    for side in range(2):
        rows = slice(side * 7, (side + 1) * 7)
        factors[rows, 2 * side] = raised[rows]
        factors[rows, 2 * side + 1] = tucked[rows]
    lengths = np.linalg.norm(factors, axis=1)
    active = lengths > 0
    factors[active] /= lengths[active, None]
    expected = .8 * np.einsum("ik,jk->ij", factors, factors) + np.diag(1. - .8 * active)
    actual = _distribution().arm_correlation_matrix
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(np.diag(actual), 1., atol=1e-14)
    assert np.linalg.eigvalsh(actual).min() >= .2 - 1e-14

    # Independently verify the directional names against actual training XML.
    # This is a static kinematic check, not a claim of learned dynamic behavior.
    model = mujoco.MjModel.from_xml_string(assemble_training_xml())
    palms = [model.site(f"{side}_palm").id for side in ("left", "right")]
    positions = {}
    for name in ("nominal", "raised", "tucked"):
        data = mujoco.MjData(model)
        data.qpos[:] = DEFAULT_QPOS
        data.qpos[7:] += .8 * posture_actions(np.zeros(29), JOINT_NAMES, name)
        mujoco.mj_forward(model, data)
        positions[name] = data.site_xpos[palms].copy()
    assert np.all(positions["raised"][:, 2] > positions["nominal"][:, 2] + .25)
    assert np.ptp(positions["tucked"][:, 1]) < np.ptp(positions["nominal"][:, 1]) - .10
    # The unit-marginal normalization changes factor magnitudes. Check the
    # actual normalized factors too: the positive raise factor moves inward
    # and upward, while the tuck factor narrows the hand span. Noise remains
    # zero-mean, so neither direction is a scripted posture command.
    for label, column in (("raise", 0), ("tuck", 1)):
        data = mujoco.MjData(model)
        data.qpos[:] = DEFAULT_QPOS
        both_arms = factors[:, column] + factors[:, column + 2]
        data.qpos[22:36] += .8 * .2 * both_arms
        mujoco.mj_forward(model, data)
        positions[label + "_factor"] = data.site_xpos[palms].copy()
        assert np.ptp(positions[label + "_factor"][:, 1]) < np.ptp(positions["nominal"][:, 1]) - .05
    assert np.all(positions["raise_factor"][:, 2] > positions["nominal"][:, 2] + .001)


def test_onnx_export_refuses_conditional_policy_before_writing_output(tmp_path):
    from cat_ppo.furniture.export import export_native_policy, export_selected
    from cat_ppo.furniture.checkpoint import BestCheckpointStore, native_writer

    factory, network, params, sizes, _, _ = _factory()
    output = tmp_path / "policy.onnx"
    output.write_bytes(b"existing export remains intact")
    with pytest.raises(ValueError, match="omit previous-action conditioning"):
        export_native_policy(params, CONTRACT, output,
                             action_distribution_config=network.parametric_action_distribution.config)
    assert output.read_bytes() == b"existing export remains intact"

    run = tmp_path / "run"
    store = BestCheckpointStore(run)
    config = checkpoint.network_config(sizes, 29, False, factory)
    assert store.consider(step=1, metrics={"proxy_score": 1.}, source="training_proxy",
                          write_checkpoint=native_writer(params, config, 1), contract=CONTRACT)
    with pytest.raises(ValueError, match="Use load_checkpoint_policy"):
        export_selected(run)
    assert not (run / "export").exists()
