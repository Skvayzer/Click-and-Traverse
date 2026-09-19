"""Runner boundary checks with a tiny stateful task, without CUDA or W&B."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cat_mjlab import runner
from cat_mjlab.learning import Learner, LearnerConfig
from train_cat_mjlab import parser


class FakeTask:
    def __init__(self, num_envs=6):
        self.num_envs = num_envs
        self.obs = dict(state=torch.zeros(num_envs, 3), privileged_state=torch.zeros(num_envs, 4))
        self.navigation_counts = torch.zeros(4, 2, dtype=torch.long)
        self.contrast_counts = torch.zeros(3, 2, dtype=torch.long)
        self.ticks = 0
        self.contract = {"actor_features": ["a", "b", "c"]}

    def step(self, action):
        self.ticks += 1
        self.obs["state"].add_(1)
        self.obs["privileged_state"].add_(1)
        done = torch.full((self.num_envs,), self.ticks % 2 == 0)
        self.navigation_counts[3] += torch.tensor([int(done.sum()), int(done.sum()) // 2])
        terminal = {key: value.clone() for key, value in self.obs.items()}
        if bool(done.any()):
            for value in self.obs.values():
                value[done] = -10  # Explicitly distinguish autoreset vs terminal observation.
        return dict(obs=self.obs, terminal_obs=terminal, reward=torch.ones(self.num_envs),
            done=done, truncated=torch.zeros_like(done), terminated=done,
            metrics=dict(episode_return=torch.ones(self.num_envs) * 2,
                         episode_length=torch.ones(self.num_envs) * 2))

    def state_dict(self):
        return dict(obs=self.obs, navigation_counts=self.navigation_counts,
                    contrast_counts=self.contrast_counts, ticks=self.ticks)

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value)


class FakeSimulation:
    def state_dict(self):
        return dict(qpos=torch.ones(6, 2))

    def load_state_dict(self, state):
        torch.testing.assert_close(state["qpos"], torch.ones(6, 2))


def tiny_config():
    return LearnerConfig(actor_obs=3, critic_obs=4, action_size=2, actor_hidden=(4,), critic_hidden=(5,),
        num_policies=3, embedding_dim=2, num_minibatches=2, num_updates_per_batch=1)


def test_collector_preserves_pre_step_observation_and_native_autoreset_next_obs():
    learner = Learner(tiny_config(), device="cpu")
    task = FakeTask()
    ids = torch.arange(3).repeat_interleave(2)
    rollout, collected = runner.collect_rollout(task, learner, unroll_length=3, trajectories=12, policy_ids=ids)
    assert rollout["state"].shape == (12, 3, 3)
    torch.testing.assert_close(rollout["state"][:6, 0], torch.zeros(6, 3))
    torch.testing.assert_close(rollout["state"][:6, 1], torch.ones(6, 3))
    torch.testing.assert_close(rollout["next_state"][:6, 1], torch.full((6, 3), -10.))
    assert collected["navigation_counts"][3].tolist() == [18, 9]
    # All chunks retain equal policy-group identities; no virtual samples added.
    assert rollout["policy_id"][:, 0].tolist() == ids.tolist() * 2


def test_success_section_contains_only_rates_and_omits_absent_scene_groups():
    window = runner.SuccessWindow()
    window.append(torch.tensor([[0, 0], [2, 1], [0, 0], [2, 1]]), torch.zeros(3, 2, dtype=torch.long))
    metrics = window.metrics()
    assert metrics["success/goal_success_rate"] == .5
    assert metrics["success/ordinary_clutter_goal_success_rate"] == .5
    assert "success/cat_goal_success_rate" not in metrics
    assert all(key.endswith("_success_rate") for key in metrics if key.startswith("success/"))


def test_bounded_training_checkpoint_and_resume_preserve_one_identity(tmp_path, monkeypatch):
    config = tiny_config()
    metadata = dict(contract=dict(asdict(config), sapg=dict(num_policies=3, embedding_dim=2), batch_size=3, unroll_length=2))
    monkeypatch.setattr(runner, "read_array_archive", lambda path: (metadata, []))
    monkeypatch.setattr(runner, "load_array_archive", lambda *args, **kwargs: {"exact_runtime_resume": False})
    tasks = []
    def factory(args, *, environment_config):
        task = FakeTask(args.num_envs)
        tasks.append(task)
        return task, FakeSimulation(), {"preserved": True}
    monkeypatch.setattr(runner, "create_task", factory)
    asset = tmp_path / "assets.json"
    asset.write_text("{}")
    directory = tmp_path / "run"
    arguments = ["verify", "--checkpoint-npz", str(asset), "--bank-manifest", str(asset),
        "--body-collision-bank", str(asset), "--body-collision-resets", str(asset),
        "--run-dir", str(directory), "--num-envs", "6", "--device", "cpu"]
    result = runner.run(parser().parse_args(arguments))
    assert result["env_steps"] == 12 and result["updates"] == 1
    assert result["status"] == "bounded_verification_complete"
    assert sorted(path.name for path in directory.glob("*.pt")) == ["best.pt", "resume.pt"]
    identity = json.loads((directory / "wandb.json").read_text())
    snapshot = torch.load(directory / "resume.pt", weights_only=True)
    assert snapshot["task"]["ticks"] == 2
    assert snapshot["learner"]["env_steps"] == 12
    result = runner.run(parser().parse_args(arguments + ["--resume"]))
    assert result["env_steps"] == 24
    assert tasks[-1].ticks == 4
    resumed_identity = json.loads((directory / "wandb.json").read_text())
    assert resumed_identity["id"] == identity["id"]
    assert resumed_identity["last_global_step"] == 24
    assert len((directory / "metrics.jsonl").read_text().splitlines()) == 2
    (directory / "STOP").touch()
    with pytest.raises(ValueError, match="STOP exists"):
        runner.run(parser().parse_args(arguments + ["--resume"]))


def test_verify_rejects_unbounded_or_online_jobs_before_loading_anything(tmp_path):
    args = SimpleNamespace(run_dir=tmp_path, max_updates=0, checkpoint_interval_updates=10,
                           command="verify", wandb_mode="disabled")
    with pytest.raises(ValueError, match="bounded"):
        runner.run(args)


def test_logger_suppresses_old_steps_after_resume(tmp_path):
    logger = runner.Logger(tmp_path, mode="disabled", project="test", entity="test", record={})
    logger.log(100, {"success/goal_success_rate": .5})
    resumed = runner.Logger(tmp_path, mode="disabled", project="test", entity="test", record={}, resume=True)
    resumed.log(90, {"success/goal_success_rate": .4})
    resumed.log(110, {"success/goal_success_rate": .6})
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["global_step"] for row in rows] == [100, 110]
