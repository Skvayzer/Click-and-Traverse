"""The isolated replay must preserve runtime arrays and optimizer mathematics."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/replay_sapg_update.py"
SPEC = importlib.util.spec_from_file_location("replay_sapg_update", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def evidence():
    record = dict(config={"algorithm": "sapg"}, bank_sha256="bank", code={"source_sha256": "old"},
                  observation_contract={"actor": 222})
    identity = dict(config=record["config"], bank_sha256="bank", source_sha256="old",
                    contract=record["observation_contract"])
    state = dict(step=100, contract=dict(metadata=identity, num_envs=36864),
                 training_state={"weights": np.ones(2)}, env_state={"position": np.zeros(4)})
    return record, state


def test_explicit_source_migration_preserves_arrays_and_original_metadata():
    old, snapshot = evidence()
    changed = copy.deepcopy(old)
    changed["code"]["source_sha256"] = "fixed"
    result, metadata = replay.runtime_for_source(snapshot, old, changed, allow_source_change=True)
    assert result["training_state"] is snapshot["training_state"]
    assert result["env_state"] is snapshot["env_state"]
    assert snapshot["contract"]["metadata"]["source_sha256"] == "old"
    assert metadata["source_sha256"] == "fixed"
    assert result["contract"]["num_envs"] == 36864


@pytest.mark.parametrize("key", ["config", "bank_sha256", "observation_contract"])
def test_source_migration_cannot_change_training_or_environment_contract(key):
    old, snapshot = evidence()
    changed = copy.deepcopy(old)
    changed[key] = "different"
    with pytest.raises(ValueError, match=key):
        replay.runtime_for_source(snapshot, old, changed, allow_source_change=True)


def test_changed_source_and_mismatched_snapshot_are_rejected():
    old, snapshot = evidence()
    changed = copy.deepcopy(old)
    changed["code"]["source_sha256"] = "fixed"
    with pytest.raises(ValueError, match="source-change"):
        replay.runtime_for_source(snapshot, old, changed, allow_source_change=False)
    snapshot["contract"]["metadata"]["source_sha256"] = "unrelated"
    with pytest.raises(ValueError, match="original run"):
        replay.runtime_for_source(snapshot, old, changed, allow_source_change=True)


def test_diagnostic_gradient_wrapper_preserves_adam_update():
    import jax
    import jax.numpy as jnp
    import optax
    from brax.training import gradients

    def objective(params):
        loss = jnp.sum(params["weights"] ** 2)
        return loss, {"total_loss": loss}

    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4))
    params = {"weights": jnp.array([1., -2., 3.])}
    state = optimizer.init(params)
    original = gradients.gradient_update_fn(objective, optimizer, None, has_aux=True)
    expected = original(params, optimizer_state=state)
    events = []
    restore = replay.install_diagnostics(events)
    try:
        observed = jax.jit(gradients.gradient_update_fn(objective, optimizer, None, has_aux=True))(
            params, optimizer_state=state)
        jax.effects_barrier()
    finally:
        restore()
    for first, second in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(observed)):
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-8)
    assert len(events) == 1
    assert events[0]["adam_update_index"] == 0
    assert events[0]["gradient_nonfinite"] == 0
    assert events[0]["updated_params_nonfinite"] == 0


def test_nonfinite_floats_remain_explicit_in_json_and_state_summary():
    result = replay.clean_json({"a": np.float32(np.nan), "b": float("inf"), "c": 1.})
    assert result == {"a": "nan", "b": "inf", "c": 1.}
    summary = replay.host_finiteness({"x": np.array([1., np.nan]), "count": 1})
    assert summary["nonfinite_elements"] == 1
    assert not summary["finite"]


def completed_report():
    return dict(execution_completed=True, start_step=100, transitions_per_update=10,
                requested_updates=1, final_step=110, initial_training_state={"finite": True},
                final_params={"finite": True}, runtime_unchanged=True,
                boundaries=[dict(step=110, training_state={"finite": True})],
                updates=[dict(step=110, metrics={"training/" + key: .1 for key in
                    ("total_loss", "policy_loss", "v_loss", "entropy_loss")})], diagnostic_events=[])


def test_execution_completed_can_still_fail_numerical_validation():
    report = completed_report()
    assert replay.numerical_assessment(report)["numerical_pass"]
    report["updates"][0]["metrics"]["training/policy_loss"] = "nan"
    result = replay.numerical_assessment(report)
    assert not result["numerical_pass"]
    assert result["requested_end_step"] == 110
    assert any("policy_loss" in reason for reason in result["numerical_failures"])


@pytest.mark.parametrize("kind", ["state", "params", "update", "step", "runtime", "execution"])
def test_incomplete_or_corrupt_replay_never_passes(kind):
    report = completed_report()
    if kind == "state":
        report["boundaries"][0]["training_state"]["finite"] = False
    elif kind == "params":
        report["final_params"]["finite"] = False
    elif kind == "update":
        report["updates"].clear()
    elif kind == "step":
        report["final_step"] = 100
    elif kind == "runtime":
        report["runtime_unchanged"] = False
    else:
        report["execution_completed"] = False
    assert not replay.numerical_assessment(report)["numerical_pass"]


def test_fixed_loss_can_pass_with_nonfinite_comparison_to_old_expression():
    report = completed_report()
    event = dict(kind="minibatch", adam_update_index=3,
                 total_loss=.1, policy_loss=.1, v_loss=.1, entropy_loss=.1,
                 gradient_nonfinite=0, updated_params_nonfinite=0, optimizer_nonfinite=0,
                 replay_legacy_surrogate_nonfinite=1, replay_legacy_q_max_finite="-inf")
    report["diagnostic_events"].append(event)
    assert replay.numerical_assessment(report)["numerical_pass"]
    event["gradient_nonfinite"] = 1
    result = replay.numerical_assessment(report)
    assert not result["numerical_pass"]
    assert any("minibatch 3" in reason for reason in result["numerical_failures"])
