"""Full-policy/optimizer conversion checks, not leader-only warm-start claims."""
from collections import namedtuple
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp
import optax
from cat_mjlab.conversion import expand_released_params, load_jax_runtime, load_native_params, load_array_archive
from cat_mjlab.learning import Learner, LearnerConfig
from test_mjlab_learning import setup_pair


def runtime_fixture():
    learner, network, params = setup_pair()
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(learning_rate=learner.config.learning_rate))
    state = optimizer.init(params)
    gradients = jax.tree.map(lambda x: jnp.ones_like(x) * .01, params)
    for _ in range(3):
        updates, state = optimizer.update(gradients, state, params)
        params = optax.apply_updates(params, updates)
    state_type = namedtuple("State", "params optimizer_state")
    pairs, _ = jax.tree_util.tree_flatten_with_path(state_type(params, state))
    contract = {key: value for key, value in asdict(learner.config).items() if key in (
        "learning_rate", "entropy_cost", "discounting", "reward_scaling", "clipping_epsilon",
        "gae_lambda", "max_grad_norm", "normalize_advantage", "num_minibatches", "num_updates_per_batch")}
    contract.update(normalize_observations=False, distribution=None, sapg=dict(num_policies=3, embedding_dim=2))
    snapshot = dict(schema="cat-ppo-runtime-v1", step=72, contract=contract,
        training_state=dict(paths=[jax.tree_util.keystr(path) for path, value in pairs],
                            leaves=[np.asarray(value)[None] for path, value in pairs]))
    return learner, network, params, optimizer, state, gradients, snapshot


def test_converted_runtime_preserves_every_policy_and_adam_next_update():
    learner, network, params, optimizer, state, gradients, snapshot = runtime_fixture()
    report = load_jax_runtime(learner, snapshot)
    assert report["adam_update_count"] == 3
    assert report["all_sapg_embeddings_preserved"] and not report["exact_runtime_resume"]
    assert learner.env_steps == 72
    inputs = np.random.default_rng(2).normal(size=(6, 3)).astype(np.float32)
    privileged = np.random.default_rng(4).normal(size=(6, 4)).astype(np.float32)
    ids = np.arange(6) % 3
    with torch.no_grad():
        actor = learner.model.logits(torch.tensor(inputs), torch.tensor(ids))
        critic = learner.model.value(torch.tensor(privileged), torch.tensor(ids))
    observations = {"state": jnp.asarray(inputs), "privileged_state": jnp.asarray(privileged)}
    np.testing.assert_allclose(actor, network.policy_network.apply(None, params.policy, observations, jnp.asarray(ids)), atol=2e-6)
    np.testing.assert_allclose(critic, network.value_network.apply(None, params.value, observations, jnp.asarray(ids),
                              policy_embeddings=params.policy["policy_embeddings"]), atol=2e-6)
    updates, state = optimizer.update(gradients, state, params)
    expected_params = optax.apply_updates(params, updates)
    # Gradients used above have global norm < 1, so no clipping required here.
    for parameter in learner.model.parameters():
        parameter.grad = torch.full_like(parameter, .01)
    learner.optimizer.step()
    expected = Learner(learner.config, device="cpu")
    load_native_params(expected, (None, expected_params.policy, expected_params.value))
    for actual, wanted in zip(learner.model.parameters(), expected.model.parameters()):
        torch.testing.assert_close(actual, wanted, atol=2e-7, rtol=2e-6)


def test_incompatible_migration_fails_before_mutating_model():
    learner, _, _, _, _, _, snapshot = runtime_fixture()
    saved = {key: value.clone() for key, value in learner.model.state_dict().items()}
    snapshot["contract"]["learning_rate"] *= .5
    with pytest.raises(ValueError, match="learning_rate"):
        load_jax_runtime(learner, snapshot)
    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, saved[key], atol=0, rtol=0)


def test_incomplete_follower_or_optimizer_state_rejected():
    learner, _, _, _, _, _, snapshot = runtime_fixture()
    index = next(i for i, path in enumerate(snapshot["training_state"]["paths"])
                 if ".nu.policy" in path and "policy_embeddings" in path)
    snapshot["training_state"]["paths"].pop(index)
    snapshot["training_state"]["leaves"].pop(index)
    with pytest.raises(ValueError, match="Incomplete"):
        load_jax_runtime(learner, snapshot)


def test_ppo_parameter_load_into_sapg_preserves_all_initial_outputs():
    source, _, params = setup_pair("ppo")
    learner, _, _ = setup_pair("sapg")
    report = load_native_params(learner, (None, params.policy, params.value))
    assert report["sapg_embeddings"] == "initialized_with_zero_input_weights"
    state = torch.randn(12, 3)
    privileged = torch.randn(12, 4)
    for i in range(3):
        torch.testing.assert_close(learner.model.logits(state, i), source.model.logits(state), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(learner.model.value(privileged, i), source.model.value(privileged), atol=1e-6, rtol=1e-6)


def test_released_checkpoint_expansion_preserves_named_inputs_actions_and_value():
    from brax.training.agents.ppo.networks import make_ppo_networks
    from cat_ppo.furniture.control import legacy_observation_contract, wholebody_observation_contract
    from cat_ppo.furniture.learning import verify_warmstart_parity
    source_contract, target_contract = legacy_observation_contract(), wholebody_observation_contract()
    network = make_ppo_networks({"state": (162,), "privileged_state": (250,)}, 12,
        policy_hidden_layer_sizes=(7, 5), value_hidden_layer_sizes=(9, 6),
        policy_obs_key="state", value_obs_key="privileged_state")
    normalizer = dict(count=np.array(100., np.float32),
        mean={key: np.zeros(width, np.float32) for key, width in (("state", 162), ("privileged_state", 250))},
        std={key: np.ones(width, np.float32) for key, width in (("state", 162), ("privileged_state", 250))},
        summed_variance={key: np.ones(width, np.float32) * 100 for key, width in (("state", 162), ("privileged_state", 250))})
    params = (normalizer, network.policy_network.init(jax.random.PRNGKey(3)), network.value_network.init(jax.random.PRNGKey(4)))
    expanded, report = expand_released_params(params)
    assert report["source_action_count"] == 12 and report["target_action_count"] == 29
    verify_warmstart_parity(params, expanded, source_contract, target_contract)
    learner = Learner(LearnerConfig(actor_hidden=(7, 5), critic_hidden=(9, 6)), device="cpu")
    load_native_params(learner, expanded)
    with torch.no_grad():
        logits = learner.model.logits(torch.randn(4, 222))
    assert logits.shape == (4, 58)
    torch.testing.assert_close(logits[:, 12:29], torch.zeros(4, 17))
    torch.testing.assert_close(torch.nn.functional.softplus(logits[:, 41:]) + .001,
                              torch.full((4, 17), .05), atol=1e-7, rtol=1e-6)


def test_offline_archive_roundtrip_excludes_environment_and_retains_optimizer(tmp_path):
    import hashlib
    from flax import serialization
    from scripts.export_mjlab_checkpoint import export_runtime
    from cat_mjlab.checkpoint_arrays import read_array_archive
    learner, _, _, _, _, _, snapshot = runtime_fixture()
    snapshot["contract"]["metadata"] = dict(bank_sha256="a" * 64)
    sampler = {"pf_episode_ema": [4., 5.], "pf_success_ema": [2., 3.],
               "pf_sampling_logits": np.log([.4, .6]), "pf_hand_curriculum_stage": 2,
               "pf_hand_curriculum_completed": [100, 80, 60], "pf_hand_curriculum_goals": [70, 60, 40]}
    snapshot["env_state"] = dict(paths=[".data.unneeded"] + [f".info['{key}']" for key in sampler],
        leaves=[np.zeros((1, 3, 100, 100), np.float32)] +
               [np.broadcast_to(value, (1, 3, *np.shape(value))).copy() for value in sampler.values()])
    source = tmp_path / "resume.msgpack"
    source.write_bytes(serialization.msgpack_serialize(snapshot))
    original = source.read_bytes()
    destination = tmp_path / "migrated.npz"
    metadata = export_runtime(source, destination, hashlib.sha256(original).hexdigest())
    assert source.read_bytes() == original
    assert metadata["kind"] == "runtime"
    actual_metadata, arrays = read_array_archive(destination)
    assert all(path.startswith((".params.", ".optimizer_state")) for path in actual_metadata["paths"])
    assert sum(array.nbytes for array in arrays) < len(original)
    report = load_array_archive(learner, destination)
    assert report["adam_update_count"] == 3
    assert report["archive_metadata"]["source_sha256"] == hashlib.sha256(original).hexdigest()
    from cat_mjlab.checkpoint_arrays import sampling_state_from_archive
    actual, sampling_report = sampling_state_from_archive(actual_metadata, arrays, "a" * 64)
    assert sampling_report["status"] == "preserved" and sampling_report["curriculum_stage"] == 2
    for key in sampler:
        np.testing.assert_array_equal(actual[key], sampler[key])
    assert sampling_state_from_archive(actual_metadata, arrays, "b" * 64)[1]["status"] == "new_bank_new_curriculum"
    with pytest.raises(ValueError, match="destination must be new"):
        export_runtime(source, destination)
    with pytest.raises(ValueError, match="checksum"):
        export_runtime(source, tmp_path / "bad.npz", "0" * 64)


def test_sampler_export_rejects_inconsistent_replicas_and_missing_same_bank_state():
    from cat_mjlab.checkpoint_arrays import extract_shared_sampling, sampling_state_from_archive
    bad = dict(env_state=dict(paths=[".info['pf_episode_ema']"],
                             leaves=[np.array([[[1., 2.], [1., 3.]]])]))
    with pytest.raises(ValueError, match="replicas disagree"):
        extract_shared_sampling(bad)
    with pytest.raises(ValueError, match="re-export"):
        sampling_state_from_archive(dict(kind="runtime", contract=dict(metadata=dict(bank_sha256="abc"))), [], "abc")
