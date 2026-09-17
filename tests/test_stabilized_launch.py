"""Stabilized launcher baseline recovery and gated best-checkpoint integration."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import train_cat_wholebody as launcher
from cat_ppo.furniture.checkpoint import BestCheckpointStore
from cat_ppo.furniture.retention_validation import FIXED_SCENE_IDS, ValidationResult, retention_selection


def mode_summaries(*, cat=.9, clutter=.4, hand=.1):
    record = dict(episode_count=32, cat_goal_success_rate=cat, clutter_goal_success_rate=clutter,
                  clutter_hand_violation_rate=hand, clutter_mean_min_hand_clearance_m=.08,
                  numerical_failure_rate=0., scenes={
                      "hurdle": dict(family="published_cat", episode_count=16, goal_success_rate=cat),
                      "room": dict(family="furniture", episode_count=16, goal_success_rate=clutter),
                  })
    return {mode: deepcopy(record) for mode in ("deterministic", "stochastic")}


def writer(payload, network_config=None):
    def write(path):
        path.mkdir()
        (path / "weights.bin").write_bytes(payload)
        (path / "ppo_network_config.json").write_text(json.dumps(network_config or {}))
    return write


def test_retention_checkpoint_refuses_forgetting_and_preserves_ties(tmp_path):
    store = BestCheckpointStore(tmp_path)
    baseline = mode_summaries()
    calls = []
    def submit(step, modes):
        selection = retention_selection(modes, baseline)
        def write(path):
            calls.append(step)
            writer(str(step).encode())(path)
        return store.consider(step=step, metrics={"selection": selection},
                              source="retention_validation", write_checkpoint=write,
                              contract={"actor_features": ["test"]})
    assert not submit(10, mode_summaries(cat=.2, clutter=1.))
    assert store.selected() is None and calls == []
    assert submit(20, baseline)
    assert not submit(30, baseline)
    assert not submit(40, mode_summaries(cat=.2, clutter=1.))
    assert calls == [20] and store.selected()["step"] == 20
    assert submit(50, mode_summaries(clutter=.6))
    selected = store.selected()
    assert selected["step"] == 50
    assert selected["selection_source"] == "retention_validation"
    assert selected["rules"]["eligible"] is True
    assert len(list(store.generations.glob("candidate-*"))) == 1


def test_prepare_uses_and_records_bounded_factory_without_changing_network_sizes(tmp_path, monkeypatch):
    import jax
    from brax.training.agents.ppo import checkpoint
    from cat_ppo.envs.g1 import env_cat_wholebody
    from cat_ppo.furniture.control import wholebody_observation_contract
    from cat_ppo.learning.policy.ppo.wholebody_distribution import WholeBodyNormalTanhDistribution

    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    args = launcher.parser().parse_args(["validate", "--bank-manifest", str(bank),
                                         "--finetuning", "stabilized"])
    specification = launcher.plan(args)
    contract = wholebody_observation_contract()
    shapes = {"state": (222,), "privileged_state": (310,)}
    class Environment:
        action_size = 29
        field_bank_manifest = {"scenes": [dict(scene_id="cat", family="original_cat",
                                                source={"arrays_unchanged": True})]}
        def __init__(self, config):
            self.config = config
        def observation_contract(self):
            return contract
        def reset(self, key):
            raise AssertionError("shape fixture should replace actual simulator tracing")
    monkeypatch.setattr(env_cat_wholebody, "G1CatWholeBodyEnv", Environment)
    monkeypatch.setattr(env_cat_wholebody, "wholebody_config",
                        lambda config, **kwargs: config)
    monkeypatch.setattr(jax, "eval_shape", lambda *args: SimpleNamespace(
        obs={key: SimpleNamespace(shape=shape) for key, shape in shapes.items()}))
    monkeypatch.setattr(launcher, "code_identity", lambda: dict(git_commit="test", source_sha256="test"))
    environment, factory, target, record = launcher.prepare(args, specification, restore_model=False)
    assert target is None
    network = factory(shapes, environment.action_size)
    assert isinstance(network.parametric_action_distribution, WholeBodyNormalTanhDistribution)
    network_config = checkpoint.network_config(shapes, 29, False, factory)
    kwargs = network_config.to_dict()["network_factory_kwargs"]
    assert kwargs["upper_std_min"] == .02 and kwargs["upper_std_max"] == .1
    assert kwargs["upper_entropy_weight"] == 0
    assert tuple(kwargs["policy_hidden_layer_sizes"]) == (512, 256, 128, 64)
    assert tuple(kwargs["value_hidden_layer_sizes"]) == (1024, 512, 256, 128)
    assert record["config"]["fine_tuning"]["action_distribution"] == {
        key: kwargs[key] for key in ("leg_action_count", "upper_std_min", "upper_std_max", "upper_entropy_weight")}


@pytest.fixture
def stabilized_run(tmp_path, monkeypatch):
    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    args = launcher.parser().parse_args(["run", "--bank-manifest", str(bank),
        "--run-dir", str(tmp_path / "run"), "--wandb-mode", "disabled", "--finetuning", "stabilized"])
    spec = launcher.plan(args)
    events = dict(preparations=[], evaluations=[], learner_calls=[], fail_before_runtime=False)
    factory = object()
    def prepare(args, specification, *, restore_model=True):
        events["preparations"].append(restore_model)
        record = deepcopy(specification)
        record.update(bank_sha256="bank", code={"git_commit": "commit", "source_sha256": "source"},
                      observation_contract={"actor_features": ["a"], "critic_features": ["b"],
                                            "action_names": ["joint"]},
                      warmstart={"source": "released"} if restore_model else None)
        env = SimpleNamespace(field_bank_manifest={"scenes": [dict(family="original_cat", task_kind="cat")]})
        return env, factory, "initial" if restore_model else None, record

    class Validator:
        def __init__(self, environment, supplied_factory, *, seeds, scene_ids):
            assert supplied_factory is factory and list(seeds) == list(range(16))
            assert tuple(scene_ids) == FIXED_SCENE_IDS
        def evaluate(self, params, *, step, baseline=None):
            events["evaluations"].append(dict(params=params, step=step, baseline=deepcopy(baseline)))
            modes = mode_summaries(clutter=.4 if step == 0 else .6)
            selection = retention_selection(modes, baseline) if baseline is not None else None
            return ValidationResult(step, {"validation/goal_success_rate": .7}, modes, {}, selection,
                                    {"stopping": "first clean goal, native failure, or native horizon"})

    def learner(**kwargs):
        events["learner_calls"].append(kwargs)
        if events["fail_before_runtime"]:
            events["fail_before_runtime"] = False
            raise RuntimeError("simulated learner initialization failure")
        restored = kwargs["restore_runtime_state"]
        step = restored["step"] if restored else 0
        def save(step):
            kwargs["runtime_checkpoint_fn"](step, dict(schema="cat-ppo-runtime-v1", step=step,
                contract={"metadata": kwargs["runtime_metadata"]}))
        if restored is None:
            save(0)
        unit = spec["config"]["fine_tuning"]["effective_batch_geometry"]["transitions_per_update"]
        interval = spec["config"]["fine_tuning"]["retention_validation"]["interval_updates"] * unit
        for current in (step + interval - unit, step + interval):
            metrics = {"training/rollout_reward_mean": .3}
            kwargs["progress_fn"](current, metrics)
            kwargs["scored_checkpoint_fn"](current, None, "learned", {"upper_std_max": .1}, metrics, "training_proxy")
            save(current)
        return None, None, {"training/completed_steps": step + interval}

    monkeypatch.setattr(launcher, "prepare", prepare)
    validator_module = importlib.import_module("cat_ppo.furniture.retention_validation")
    monkeypatch.setattr(validator_module, "RetentionValidator", Validator)
    native = importlib.import_module("cat_ppo.learning.policy.ppo.train")
    monkeypatch.setattr(native, "train", learner)
    checkpoint = importlib.import_module("cat_ppo.furniture.checkpoint")
    monkeypatch.setattr(checkpoint, "native_writer", lambda params, network_config, step:
                        writer(f"{params}:{step}".encode(), network_config))
    return args, spec, events


def test_launcher_validates_interval_and_restores_original_baseline_on_resume(stabilized_run):
    args, spec, events = stabilized_run
    first = launcher.run(args, spec)
    baseline = (args.run_dir / "validation_baseline.json").read_bytes()
    identity = json.loads((args.run_dir / "wandb.json").read_text())["id"]
    assert [entry["params"] for entry in events["evaluations"]] == ["initial", "learned"]
    selected = BestCheckpointStore.open_existing(args.run_dir).selected()
    assert selected["step"] == first["completed_steps"]
    assert selected["provenance"]["action_distribution"]["upper_std_max"] == .1
    assert json.loads((Path(selected["path"]) / "ppo_network_config.json").read_text())["upper_std_max"] == .1
    args.resume = True
    launcher.run(args, spec)
    assert events["preparations"] == [True, False]
    assert [entry["params"] for entry in events["evaluations"]] == ["initial", "learned", "learned"]
    assert (args.run_dir / "validation_baseline.json").read_bytes() == baseline
    assert json.loads((args.run_dir / "wandb.json").read_text())["id"] == identity
    assert events["evaluations"][-1]["baseline"] == events["evaluations"][1]["baseline"]
    # The second validated learner is an exact score tie: keep first checkpoint.
    assert BestCheckpointStore.open_existing(args.run_dir).selected()["step"] == first["completed_steps"]


def test_baseline_survives_zero_progress_learner_failure_without_blocking_retry(stabilized_run):
    args, spec, events = stabilized_run
    events["fail_before_runtime"] = True
    with pytest.raises(RuntimeError, match="initialization failure"):
        launcher.run(args, spec)
    assert not (args.run_dir / "resume.msgpack").exists()
    saved = (args.run_dir / "validation_baseline.json").read_bytes()
    args.resume = True
    result = launcher.run(args, spec)
    assert result["startup_retry"]
    assert events["preparations"] == [True, True]
    assert (args.run_dir / "validation_baseline.json").read_bytes() == saved
    assert sum(entry["step"] == 0 for entry in events["evaluations"]) == 1
