"""Checkpoint inference and ONNX export retain explicit distribution provenance."""

import copy
import functools
import json

from brax.training import types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cat_ppo.furniture.control import wholebody_observation_contract
from cat_ppo.learning.policy.ppo.wholebody_distribution import (
    WholeBodyNormalTanhDistribution, checkpoint_distribution_config,
    load_checkpoint_policy, make_ppo_networks,
)


def _fixture(bounded=True, normalize=False):
    contract = wholebody_observation_contract()
    sizes = {"state": (len(contract["actor_features"]),),
             "privileged_state": (len(contract["critic_features"]),)}
    kwargs = dict(policy_hidden_layer_sizes=(10, 7), value_hidden_layer_sizes=(13,),
                  policy_obs_key="state", value_obs_key="privileged_state")
    if bounded:
        kwargs.update(leg_action_count=12, upper_std_min=.02, upper_std_max=.1, upper_entropy_weight=0.)
    factory = functools.partial(make_ppo_networks if bounded else networks.make_ppo_networks, **kwargs)
    preprocess = running_statistics.normalize if normalize else types.identity_observation_preprocessor
    network = factory(sizes, 29, preprocess_observations_fn=preprocess)
    normalizer = running_statistics.init_state({
        key: specs.Array(shape, jnp.dtype("float32")) for key, shape in sizes.items()
    }).replace(mean={key: jnp.full(shape, 3.) for key, shape in sizes.items()},
               std={key: jnp.full(shape, .25) for key, shape in sizes.items()})
    params = (normalizer, network.policy_network.init(jax.random.PRNGKey(0)),
              network.value_network.init(jax.random.PRNGKey(1)))
    config = checkpoint.network_config(sizes, 29, normalize, factory)
    return contract, network, params, config


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_checkpoint_roundtrip_preserves_stochastic_actions_and_normalization(tmp_path, bounded, normalize):
    _, network, params, config = _fixture(bounded, normalize)
    checkpoint.save(tmp_path, 1, params, config)
    path = tmp_path / "000000000001"
    observations = {"state": jnp.ones((3, 222)), "privileged_state": jnp.ones((3, 310))}
    for deterministic in (False, True):
        expected = networks.make_inference_fn(network)(params, deterministic=deterministic)
        loaded = load_checkpoint_policy(path, deterministic=deterministic)
        reference = expected(observations, jax.random.PRNGKey(87))
        actual = loaded(observations, jax.random.PRNGKey(87))
        for left, right in zip(jax.tree.leaves(reference), jax.tree.leaves(actual)):
            np.testing.assert_array_equal(left, right)
    if bounded:
        # An unmodified external Brax loader refuses unknown settings instead
        # of silently reconstructing an unbounded Gaussian.
        with pytest.raises(TypeError, match="leg_action_count"):
            checkpoint.load_policy(path)


def test_partial_settings_unknown_versions_and_conflicting_metadata_fail_closed():
    _, _, _, config = _fixture()
    config = config.to_dict()
    assert checkpoint_distribution_config(config) == WholeBodyNormalTanhDistribution(29).config
    partial = copy.deepcopy(config)
    del partial["network_factory_kwargs"]["upper_std_max"]
    with pytest.raises(ValueError, match="all distribution settings"):
        checkpoint_distribution_config(partial)
    conflicting = copy.deepcopy(config)
    conflicting["action_distribution"] = WholeBodyNormalTanhDistribution(29, upper_std_max=.08).config
    with pytest.raises(ValueError, match="disagrees"):
        checkpoint_distribution_config(conflicting)
    malformed = WholeBodyNormalTanhDistribution(29).config
    malformed["kind"] = "future_incompatible_v2"
    with pytest.raises(ValueError, match="Unknown"):
        WholeBodyNormalTanhDistribution.from_config(malformed)
    malformed = WholeBodyNormalTanhDistribution(29).config
    malformed["native_min_std"] = .01
    with pytest.raises(ValueError, match="semantics differ"):
        WholeBodyNormalTanhDistribution.from_config(malformed)


def test_selected_bounded_checkpoint_exports_mean_policy_with_correct_provenance(tmp_path):
    import onnx
    from cat_ppo.furniture.checkpoint import BestCheckpointStore, native_writer
    from cat_ppo.furniture.export import export_selected

    contract, _, params, config = _fixture()
    store = BestCheckpointStore(tmp_path)
    assert store.consider(step=1, metrics={"proxy_score": 1.}, source="training_proxy",
        write_checkpoint=native_writer(params, config, 1), contract=contract)
    result = export_selected(tmp_path)
    assert result["max_abs_error_native_jax"] < 2e-5
    assert result["contract"]["distribution_config"] == WholeBodyNormalTanhDistribution(29).config
    assert result["contract"]["deterministic_output"] == "tanh(mean)"
    assert result["contract"]["stochastic_sampling_embedded"] is False
    props = {entry.key: entry.value for entry in onnx.load(result["path"]).metadata_props}
    encoded = json.loads(props["cat_furniture_contract"])
    assert encoded["distribution_config"] == WholeBodyNormalTanhDistribution(29).config
    assert encoded["normalize_observations"] is False


def test_legacy_noise_diagnostic_refuses_bounded_or_normalized_snapshots():
    from scripts.evaluate_cat_noise_ablation import validate_snapshot_contract

    validate_snapshot_contract({"contract": {"normalize_observations": False}})
    validate_snapshot_contract({"contract": {"distribution": None}})
    with pytest.raises(ValueError, match="Bounded-policy"):
        validate_snapshot_contract({"contract": {"distribution": WholeBodyNormalTanhDistribution(29).config}})
    with pytest.raises(ValueError, match="unnormalized"):
        validate_snapshot_contract({"contract": {"normalize_observations": True}})
