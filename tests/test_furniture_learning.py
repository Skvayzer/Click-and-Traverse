import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cat_ppo.furniture.control import legacy_observation_contract, observation_contract
from cat_ppo.furniture.learning import (
    adapt_native_params, dense_layers, index_mapping, predict_numpy,
    verify_warmstart_parity, fetch_native_checkpoint,
)


def parameters(contract, seed=0):
    rng = np.random.default_rng(seed)

    def network(sizes):
        return {"params": {f"hidden_{i}": {
            "kernel": rng.normal(0, .05, (n, m)).astype(np.float32),
            "bias": rng.normal(0, .03, m).astype(np.float32),
        } for i, (n, m) in enumerate(zip(sizes, sizes[1:]))}}

    widths = {"state": len(contract["actor_features"]), "privileged_state": len(contract["critic_features"])}
    means = {key: rng.normal(0, .2, width).astype(np.float32) for key, width in widths.items()}
    stds = {key: rng.uniform(.4, 2, width).astype(np.float32) for key, width in widths.items()}
    stats = {"count": np.asarray(100, dtype=np.float32), "mean": means, "std": stds,
             "summed_variance": {key: value**2 * 100 for key, value in stds.items()}}
    return (stats, network((widths["state"], 10, 7, 2*len(contract["action_names"]))),
            network((widths["privileged_state"], 13, 1)))


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dofs", [12, 23, 29])
def test_native_actor_scale_critic_and_normalizer_mapping(dofs, normalize):
    old, new = legacy_observation_contract(), observation_contract(dofs)
    source, target = parameters(old), parameters(new, 1)
    before = copy.deepcopy(source)
    expanded, report = adapt_native_params(source, target, old, new, new_action_std=.04)
    errors = verify_warmstart_parity(source, expanded, old, new, normalize_observations=normalize)
    assert max(errors.values()) < 2e-6
    assert report["critic_restored"] and report["optimizer"] == "fresh"
    mapping = index_mapping(old["actor_features"], new["actor_features"])
    added = sorted(set(range(len(new["actor_features"]))) - set(mapping))
    assert np.count_nonzero(expanded[1]["params"]["hidden_0"]["kernel"][added]) == 0
    for field in ("mean", "std", "summed_variance"):
        np.testing.assert_array_equal(expanded[0][field]["state"][mapping], source[0][field]["state"])
    new_actions = len(new["action_names"])
    head = expanded[1]["params"][dense_layers(expanded[1])[-1]]
    np.testing.assert_array_equal(head["bias"][:12], source[1]["params"]["hidden_2"]["bias"][:12])
    if new_actions > 12:
        np.testing.assert_array_equal(head["bias"][12:new_actions], 0)
        np.testing.assert_allclose(np.logaddexp(0, head["bias"][new_actions+12:]) + .001, .04, atol=1e-7)
    np.testing.assert_array_equal(source[1]["params"]["hidden_0"]["kernel"], before[1]["params"]["hidden_0"]["kernel"])


def test_permuted_actions_and_noncontiguous_wrist_inputs_are_mapped_by_name():
    old, new = legacy_observation_contract(), observation_contract(29)
    new["action_names"] = new["action_names"][::-1]
    expanded, _ = adapt_native_params(parameters(old), parameters(new), old, new)
    errors = verify_warmstart_parity(parameters(old), expanded, old, new)
    assert max(errors.values()) < 2e-6
    mapping = index_mapping(old["actor_features"], new["actor_features"])
    assert mapping[old["actor_features"].index("joint_position.right_shoulder_pitch_joint")] == new["actor_features"].index("joint_position.right_shoulder_pitch_joint")


def test_existing_whole_body_checkpoint_can_start_next_curriculum_scene():
    contract = observation_contract(29)
    source = parameters(contract)
    expanded, _ = adapt_native_params(source, parameters(contract, 1), contract, contract)
    errors = verify_warmstart_parity(source, expanded, contract, contract, normalize_observations=True)
    assert max(errors.values()) < 2e-6
    for name in dense_layers(source[1]):
        for field in ("kernel", "bias"):
            np.testing.assert_array_equal(source[1]["params"][name][field], expanded[1]["params"][name][field])


def test_malformed_contract_and_head_fail_loudly():
    old, new = legacy_observation_contract(), observation_contract(29)
    broken = copy.deepcopy(new);broken["actor_features"][1] = broken["actor_features"][0]
    with pytest.raises(ValueError, match="unique"):
        adapt_native_params(parameters(old), parameters(new), old, broken)
    broken = parameters(new)
    broken[1]["params"]["hidden_2"]["kernel"] = np.zeros((7, 29), dtype=np.float32)
    broken[1]["params"]["hidden_2"]["bias"] = np.zeros(29, dtype=np.float32)
    with pytest.raises(ValueError, match="mean and raw-scale"):
        adapt_native_params(parameters(old), broken, old, new)


def test_onnx_export_includes_normalizer_and_matches_native_jax(tmp_path):
    import jax
    import jax.numpy as jnp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    import onnx
    from cat_ppo.furniture.export import export_native_policy

    contract = observation_contract(29)
    params = parameters(contract)
    native_params = (running_statistics.RunningStatisticsState(**params[0]), params[1], params[2])
    native_params = jax.tree.map(jnp.asarray, native_params)
    network = networks.make_ppo_networks(
        {"state": (len(contract["actor_features"]),), "privileged_state": (len(contract["critic_features"]),)},
        29, policy_hidden_layer_sizes=(10, 7), value_hidden_layer_sizes=(13,),
        policy_obs_key="state", value_obs_key="privileged_state", preprocess_observations_fn=running_statistics.normalize)
    policy = networks.make_inference_fn(network)(native_params, deterministic=True)
    reference_precisions = []
    def native_inference(obs):
        reference_precisions.append(jax.config.jax_default_matmul_precision)
        return policy(jax.tree.map(jnp.asarray, obs), jax.random.PRNGKey(0))[0]
    with jax.default_matmul_precision("default"):
        result = export_native_policy(params, contract, tmp_path/"policy.onnx", normalize_observations=True,
            native_inference=native_inference)
        assert jax.config.jax_default_matmul_precision == "default"
    assert reference_precisions == ["highest"] * 3
    assert result["max_abs_error_native_jax"] < 2e-5
    model = onnx.load(tmp_path/"policy.onnx")
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    exported = json.loads(metadata["cat_furniture_contract"])
    assert exported["action_names"] == contract["action_names"]
    assert exported["actor_features"] == contract["actor_features"]
    assert exported["normalization_embedded"]
    assert exported["native_reference_matmul_precision"] == "highest"


def test_fetch_refuses_modified_existing_file_without_network(tmp_path):
    import hashlib
    manifest = {"repo_id": "owner/model", "revision": "a"*40, "checkpoint_path": "checkpoint",
                "files": [{"path": "weights.bin", "size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}]}
    path = tmp_path/"manifest.json";path.write_text(json.dumps(manifest))
    destination = tmp_path/"native";destination.mkdir()
    (destination/"weights.bin").write_bytes(b"bad!")
    with pytest.raises(ValueError, match="does not match"):
        fetch_native_checkpoint(destination, path)
    assert (destination/"weights.bin").read_bytes() == b"bad!"


def test_train_cli_budget_and_zero_eval_defaults():
    from train_furniture import parser, _validate
    args = parser().parse_args(["--scene-dir", "scene", "--run-dir", "run", "--steps", "512"])
    assert args.num_evals == 0 and args.wandb_mode == "disabled" and args.seed == 0
    _validate(args)
    args.steps = 513
    with pytest.raises(ValueError, match="silent budget rounding"):
        _validate(args)


def test_pinned_release_parity_and_selected_native_roundtrip(tmp_path):
    """Uses an explicitly fetched local fixture; unit tests never download weights."""
    native_path = Path(__file__).resolve().parents[1] / "data/furniture/native_generalist_v1"
    if not (native_path / "ppo_network_config.json").is_file():
        pytest.skip("Fetch the pinned public native checkpoint to run release integration")
    import functools
    import hashlib
    import jax
    import jax.numpy as jnp
    import onnxruntime as ort
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import checkpoint as brax_checkpoint, networks
    from cat_ppo.furniture.learning import DEFAULT_MANIFEST, load_native, mlp_hidden_sizes
    from cat_ppo.furniture.checkpoint import BestCheckpointStore, native_writer
    from cat_ppo.furniture.export import export_selected

    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    for entry in manifest["files"]:
        payload = (native_path/entry["path"]).read_bytes()
        assert len(payload) == entry["size"] and hashlib.sha256(payload).hexdigest() == entry["sha256"]
    source = load_native(native_path)
    old, new = legacy_observation_contract(), observation_contract(29)
    sizes = {"state": (len(new["actor_features"]),), "privileged_state": (len(new["critic_features"]),)}
    factory = functools.partial(networks.make_ppo_networks,
        policy_hidden_layer_sizes=mlp_hidden_sizes(source[1]), value_hidden_layer_sizes=mlp_hidden_sizes(source[2]),
        policy_obs_key="state", value_obs_key="privileged_state")
    network = factory(sizes, 29)
    target = (running_statistics.init_state({key: specs.Array(size, jnp.dtype("float32")) for key,size in sizes.items()}),
              network.policy_network.init(jax.random.PRNGKey(1)), network.value_network.init(jax.random.PRNGKey(2)))
    expanded, _ = adapt_native_params(source, target, old, new)
    assert max(verify_warmstart_parity(source, expanded, old, new).values()) < 2e-6
    original_onnx = ort.InferenceSession(str(native_path/"policy.onnx"), providers=["CPUExecutionProvider"])
    observations = {"state": np.random.default_rng(7).normal(0,.3,(7,162)).astype(np.float32)}
    np.testing.assert_allclose(original_onnx.run(None, {"obs":observations["state"]})[0],
                               predict_numpy(source, observations), atol=2e-5, rtol=2e-5)
    params = jax.tree.map(jnp.asarray, expanded)
    config = brax_checkpoint.network_config(sizes,29,False,factory)
    store = BestCheckpointStore(tmp_path)
    assert store.consider(step=1,metrics={"proxy_score":1.0},source="training_proxy",
        write_checkpoint=native_writer(params,config,1),contract=new,
        provenance={"test_fixture":True,"not_trained":True})
    restored = load_native(store.selected()["path"])
    assert max(verify_warmstart_parity(params,restored,new,new).values()) < 2e-6
    assert not store.consider(step=2,metrics={"proxy_score":0.0},source="training_proxy",
        write_checkpoint=lambda _:pytest.fail("Worse final candidate must not be written"),contract=new)
    exported = export_selected(tmp_path)
    assert exported["max_abs_error_native_jax"] < 2e-5
    assert store.selected()["step"] == 1
