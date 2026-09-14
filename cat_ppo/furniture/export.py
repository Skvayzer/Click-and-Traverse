"""Export the selected native CAT MLP to ONNX without TensorFlow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from cat_ppo.furniture.learning import dense_layers, feature_names, predict_numpy, _stat_get


def export_native_policy(params, contract, output_path, *, normalize_observations=False,
                         metadata=None, native_inference=None, seed=0):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import onnxruntime as ort

    action_count = len(contract["action_names"])
    observation_count = len(feature_names(contract, "state"))
    names = dense_layers(params[1])
    if np.shape(params[1]["params"][names[0]]["kernel"])[0] != observation_count:
        raise ValueError("Export input shape does not match observation contract")
    if np.shape(params[1]["params"][names[-1]]["bias"])[0] != 2 * action_count:
        raise ValueError("Export action shape does not match mean/scale contract")
    nodes, initializers = [], []
    x = "obs"

    def array(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value), name=name))
        return name

    if normalize_observations:
        mean = np.asarray(_stat_get(params[0], "mean")["state"], dtype=np.float32)
        std = np.asarray(_stat_get(params[0], "std")["state"], dtype=np.float32)
        if mean.shape != (observation_count,) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Invalid export normalizer")
        nodes += [helper.make_node("Sub", [x, array("obs_mean", mean)], ["centered"]),
                  helper.make_node("Div", ["centered", array("obs_std", std)], ["normalized"])]
        x = "normalized"
    for i, name in enumerate(names):
        layer = params[1]["params"][name]
        kernel = array(f"{name}_kernel", np.asarray(layer["kernel"], dtype=np.float32))
        bias = array(f"{name}_bias", np.asarray(layer["bias"], dtype=np.float32))
        nodes += [helper.make_node("MatMul", [x, kernel], [f"{name}_matmul"]),
                  helper.make_node("Add", [f"{name}_matmul", bias], [f"{name}_linear"])]
        x = f"{name}_linear"
        if i != len(names) - 1:
            nodes += [helper.make_node("Sigmoid", [x], [f"{name}_sigmoid"]),
                      helper.make_node("Mul", [x, f"{name}_sigmoid"], [f"{name}_swish"])]
            x = f"{name}_swish"
    indices = array("mean_indices", np.arange(action_count, dtype=np.int64))
    nodes += [helper.make_node("Gather", [x, indices], ["action_mean"], axis=-1),
              helper.make_node("Tanh", ["action_mean"], ["continuous_actions"])]
    graph = helper.make_graph(nodes, "CAT_furniture_native_actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", observation_count])],
        [helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, ["batch", action_count])],
        initializer=initializers)
    model = helper.make_model(graph, producer_name="CAT furniture native exporter",
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    export_contract = {**contract, "input": {"name": "obs", "dtype": "float32", "shape": [None, observation_count]},
        "output": {"name": "continuous_actions", "dtype": "float32", "shape": [None, action_count]},
        "normalize_observations": bool(normalize_observations), "activation": "swish",
        "distribution": "NormalTanhDistribution", "deterministic_output": "tanh(mean)",
        "normalization_embedded": bool(normalize_observations),
        "native_reference_matmul_precision": "highest" if native_inference is not None else None}
    helper.set_model_props(model, {"cat_furniture_contract": json.dumps(export_contract, sort_keys=True),
                                  "checkpoint_provenance": json.dumps(metadata or {}, sort_keys=True)})
    onnx.checker.check_model(model, full_check=True)
    payload = model.SerializeToString()
    session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    errors = []
    native_errors = []
    for batch in (1, 7, 32):
        observations = {key: rng.normal(0, .5, (batch, len(feature_names(contract, key)))).astype(np.float32)
                        for key in ("state", "privileged_state")}
        observations["state"][0] = 0
        expected = predict_numpy(params, observations, normalize_observations=normalize_observations)
        actual = session.run(["continuous_actions"], {"obs": observations["state"]})[0]
        errors.append(float(np.max(np.abs(actual - expected))))
        if not np.allclose(actual, expected, atol=2e-5, rtol=2e-5):
            raise ValueError(f"ONNX/Numpy parity failed: {errors[-1]}")
        if native_inference is not None:
            import jax
            # Default CUDA float32 GEMM can use reduced-precision tensor cores.
            # Verify the same float32 function as ONNX, without relaxing parity.
            with jax.default_matmul_precision("highest"):
                native = np.asarray(native_inference(observations))
            native_errors.append(float(np.max(np.abs(actual - native))))
            if not np.allclose(actual, native, atol=2e-5, rtol=2e-5):
                raise ValueError(f"ONNX/native JAX parity failed: {native_errors[-1]}")
    output_path = Path(output_path)
    if output_path.is_symlink():
        raise ValueError("Refusing symlink export destination")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".onnx-", dir=output_path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"path": str(output_path), "sha256": hashlib.sha256(payload).hexdigest(),
            "max_abs_error_numpy": max(errors), "max_abs_error_native_jax": max(native_errors) if native_errors else None,
            "batch_sizes_checked": [1, 7, 32], "contract": export_contract}


def export_selected(run_dir, output_path=None):
    import jax
    import jax.numpy as jnp
    from brax.training.agents.ppo import checkpoint as brax_checkpoint
    from cat_ppo.furniture.checkpoint import BestCheckpointStore

    selected = BestCheckpointStore(run_dir).selected(verify=True)
    if selected is None:
        raise ValueError("Run has no selected learned checkpoint")
    native = Path(selected["path"])
    contract = json.loads((native / "observation_contract.json").read_text())
    network_config = json.loads((native / "ppo_network_config.json").read_text())
    params = brax_checkpoint.load(native)
    inference = brax_checkpoint.load_policy(native, deterministic=True)

    def apply(observations):
        return inference(jax.tree.map(jnp.asarray, observations), jax.random.PRNGKey(0))[0]

    output_path = Path(output_path or Path(run_dir) / "export" / "policy.onnx")
    if output_path.resolve().is_relative_to(Path(run_dir).resolve() / ".checkpoint-generations"):
        raise ValueError("Derived exports must not modify immutable native checkpoint generations")
    metadata = {"step": selected["step"], "selection_source": selected["selection_source"],
                "score": selected["score"], "native_files": selected["files"],
                "native_network_config": network_config}
    return export_native_policy(params, contract, output_path,
        normalize_observations=network_config["normalize_observations"], metadata=metadata, native_inference=apply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = export_selected(args.run_dir, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "contract"}, indent=2))
