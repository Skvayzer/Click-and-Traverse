"""Compact-input warm-start regressions using the pinned released CAT weights."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cat_ppo.furniture.control import (
    legacy_observation_contract, observation_contract, wholebody_observation_contract,
)
from cat_ppo.furniture.learning import (
    DEFAULT_MANIFEST, adapt_native_params, dense_layers, index_mapping,
    load_native, mlp_hidden_sizes, verify_warmstart_parity,
)


def _network_and_parameters(source, contract):
    import jax
    import jax.numpy as jnp
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import networks

    sizes = {"state": (len(contract["actor_features"]),),
             "privileged_state": (len(contract["critic_features"]),)}
    network = networks.make_ppo_networks(
        sizes, len(contract["action_names"]),
        policy_hidden_layer_sizes=mlp_hidden_sizes(source[1]),
        value_hidden_layer_sizes=mlp_hidden_sizes(source[2]),
        policy_obs_key="state", value_obs_key="privileged_state",
    )
    params = (
        running_statistics.init_state({key: specs.Array(size, jnp.dtype("float32"))
                                       for key, size in sizes.items()}),
        network.policy_network.init(jax.random.PRNGKey(1)),
        network.value_network.init(jax.random.PRNGKey(2)),
    )
    return network, params


@pytest.fixture(scope="module")
def released_expansion():
    # Tests never fetch model weights or depend on mutable remote state.
    native_path = Path(__file__).resolve().parents[1] / "data/furniture/native_generalist_v1"
    if not (native_path / "ppo_network_config.json").is_file():
        pytest.skip("Fetch the pinned public native checkpoint to run release integration")
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    assert manifest["revision"] == "46ce4b57ba0639168d51741b661ff62f7ce6f045"
    assert manifest["checkpoint_path"].endswith("/005033164800")
    for entry in manifest["files"]:
        payload = (native_path / entry["path"]).read_bytes()
        assert len(payload) == entry["size"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    source = load_native(native_path)
    old, compact = legacy_observation_contract(), wholebody_observation_contract()
    network, target = _network_and_parameters(source, compact)
    expanded, report = adapt_native_params(source, target, old, compact)
    return source, old, compact, network, target, expanded, report


def test_compact_contract_reuses_hands_and_has_only_thirteen_cat_field_sites():
    contract = wholebody_observation_contract()
    assert len(contract["actor_features"]) == 222
    assert len(contract["critic_features"]) == 310
    assert len(contract["action_names"]) == 29
    assert contract["field_sample_count"] == 13
    assert contract["prediction_horizons_seconds"] == []
    for key in ("actor_features", "critic_features"):
        features = contract[key]
        assert len(features) == len(set(features))
        assert not any(name.startswith("probe.") for name in features)
        assert not any(token in name for name in features for token in (
            "0.2s", "0.4s", "future", "age_seconds", "unknown", "position_uncertainty",
        ))
        fields = [name.split(".") for name in features if name.startswith("pf.")]
        sites = {(parts[1], parts[3]) for parts in fields}
        assert len(fields) == 13 * 7
        assert len(sites) == 13
        assert {(group, index) for group, index in sites if group == "hands"} == {
            ("hands", "0"), ("hands", "1"),
        }
        assert {(group, index) for group, index in sites if group == "elbows"} == {
            ("elbows", "0"), ("elbows", "1"),
        }


def test_pinned_release_maps_into_compact_actor_and_critic_without_changing_trunks(released_expansion):
    source, old, compact, _, _, expanded, report = released_expansion
    assert mlp_hidden_sizes(expanded[1]) == (512, 256, 128, 64)
    assert mlp_hidden_sizes(expanded[2]) == (1024, 512, 256, 128)
    assert report["source_action_count"] == 12 and report["target_action_count"] == 29
    assert max(verify_warmstart_parity(source, expanded, old, compact).values()) < 2e-6
    for key, slot in (("actor_features", 1), ("critic_features", 2)):
        mapping = index_mapping(old[key], compact[key])
        added = np.setdiff1d(np.arange(len(compact[key])), mapping)
        assert len(added) == 60
        src, dst = source[slot]["params"], expanded[slot]["params"]
        names = dense_layers(source[slot])
        np.testing.assert_array_equal(dst[names[0]]["kernel"][mapping], src[names[0]]["kernel"])
        assert np.count_nonzero(dst[names[0]]["kernel"][added]) == 0
        np.testing.assert_array_equal(dst[names[0]]["bias"], src[names[0]]["bias"])
        for layer in names[1:]:
            if slot == 1 and layer == names[-1]:
                continue  # The actor output expands from 24 to 58 parameters.
            for field in ("kernel", "bias"):
                np.testing.assert_array_equal(dst[layer][field], src[layer][field])

    action_map = index_mapping(old["action_names"], compact["action_names"])
    added_actions = np.setdiff1d(np.arange(29), action_map)
    assert len(added_actions) == 17
    src_head = source[1]["params"][dense_layers(source[1])[-1]]
    head = expanded[1]["params"][dense_layers(expanded[1])[-1]]
    for field in ("kernel", "bias"):
        np.testing.assert_array_equal(head[field][..., action_map], src_head[field][..., :12])
        np.testing.assert_array_equal(head[field][..., 29 + action_map], src_head[field][..., 12:])
    assert np.count_nonzero(head["kernel"][:, added_actions]) == 0
    assert np.count_nonzero(head["kernel"][:, 29 + added_actions]) == 0
    np.testing.assert_array_equal(head["bias"][added_actions], np.zeros(17))
    np.testing.assert_allclose(np.logaddexp(0, head["bias"][29 + added_actions]) + .001,
                               .05, rtol=0, atol=1e-7)


def test_native_brax_forward_preserves_pretrained_outputs_with_compact_inputs(released_expansion):
    import jax
    import jax.numpy as jnp

    source, old, compact, network, _, expanded, _ = released_expansion
    original_network, _ = _network_and_parameters(source, old)
    rng = np.random.default_rng(7)
    original_obs, compact_obs = {}, {}
    for key, field in (("state", "actor_features"), ("privileged_state", "critic_features")):
        original_obs[key] = rng.normal(0, .3, (7, len(old[field]))).astype(np.float32)
        compact_obs[key] = rng.normal(0, .3, (7, len(compact[field]))).astype(np.float32)
        compact_obs[key][:, index_mapping(old[field], compact[field])] = original_obs[key]
    original_obs = jax.tree.map(jnp.asarray, original_obs)
    compact_obs = jax.tree.map(jnp.asarray, compact_obs)
    source, expanded = jax.tree.map(jnp.asarray, (source, expanded))
    with jax.default_matmul_precision("highest"):
        original_actor = original_network.policy_network.apply(source[0], source[1], original_obs)
        compact_actor = network.policy_network.apply(expanded[0], expanded[1], compact_obs)
        original_value = original_network.value_network.apply(source[0], source[2], original_obs)
        compact_value = network.value_network.apply(expanded[0], expanded[2], compact_obs)
    mapping = index_mapping(old["action_names"], compact["action_names"])
    np.testing.assert_allclose(np.asarray(compact_actor)[:, np.r_[mapping, 29 + mapping]],
                               original_actor, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(compact_value, original_value, rtol=2e-5, atol=2e-5)
    added = np.setdiff1d(np.arange(29), mapping)
    np.testing.assert_array_equal(np.asarray(compact_actor)[:, added], np.zeros((7, 17)))
    np.testing.assert_allclose(np.logaddexp(0, np.asarray(compact_actor)[:, 29 + added]) + .001,
                               .05, rtol=0, atol=1e-7)


def test_old_406_input_checkpoint_cannot_silently_drop_its_probe_features(released_expansion):
    source, old, compact, _, target, _, _ = released_expansion
    old_wholebody = observation_contract(29)
    assert len(old_wholebody["actor_features"]) == 406
    assert len(old_wholebody["critic_features"]) == 494
    _, old_target = _network_and_parameters(source, old_wholebody)
    old_expanded, _ = adapt_native_params(source, old_target, old, old_wholebody)
    with pytest.raises(ValueError, match=r"Target contract omits source names:.*probe\."):
        adapt_native_params(old_expanded, target, old_wholebody, compact)
