"""Recorder format and terminal capture tests; no actual policy rollout."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cat_mjlab.learning import Learner, LearnerConfig
from scripts.record_mjlab_rollout import load_policy, record_episode, verify_contract, sha256


class SyntheticTask:
    def __init__(self):
        self.num_envs = 1
        self.sim = SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(1, 36),
            qvel=torch.zeros(1, 35), time=torch.zeros(1)), capacity_report=lambda: {})
        self.obs = {}
        self.ticks = 0

    def reset(self, ids):
        if len(ids):
            self.sim.data.qpos.fill_(-999)
            self.sim.data.time.zero_()

    def step(self, action):
        self.ticks += 1
        self.sim.data.qpos.add_(1)
        self.sim.data.time.add_(.02)
        done = torch.tensor([self.ticks == 2])
        metrics = dict(resolved=done, successful=torch.tensor([False]),
                       episode_return=torch.tensor([.1 * self.ticks]), **{"episode/body_collision": done})
        self.reset(torch.nonzero(done).flatten())
        return dict(done=done, terminated=done, truncated=torch.tensor([False]), metrics=metrics)


def test_recorder_copies_terminal_pose_before_autoreset_and_restores_reset():
    task = SyntheticTask()
    original = task.reset
    def act(*args, **kwargs):
        assert kwargs == dict(policy_ids=0, deterministic=True)
        return dict(action=torch.zeros(1, 29))
    trace, outcome = record_episode(task, SimpleNamespace(act=act), policy_id=0, frames=5)
    np.testing.assert_array_equal(trace["qpos"][:, 0], [0, 1, 2])
    np.testing.assert_allclose(trace["time"], [0, .02, .04])
    assert task.sim.data.qpos[0, 0] == -999
    assert task.reset == original
    assert outcome["body_collision"] and not outcome["success"] and outcome["length"] == 2


@pytest.mark.parametrize("kind", ["best", "resume"])
def test_loader_accepts_native_schemas_and_preserves_selected_policy(tmp_path, kind):
    config = LearnerConfig(actor_obs=3, critic_obs=4, action_size=2, actor_hidden=(5,), critic_hidden=(6,),
                           num_policies=2, embedding_dim=3)
    source = Learner(config, device="cpu")
    source.env_steps = 123
    contract = dict(learner_config=json.loads(json.dumps(asdict(config))))
    if kind == "best":
        payload = dict(schema="cat-mjlab-best-v1", config=asdict(config), model=source.model.state_dict(),
                       step=123, contract=contract)
    else:
        payload = dict(schema="cat-mjlab-runtime-v1", learner=source.state_dict(), contract=contract)
    path = tmp_path / (kind + ".pt")
    torch.save(payload, path)
    target, metadata = load_policy(path, device="cpu", policy_id=1)
    obs = dict(state=torch.randn(3, 3), privileged_state=torch.randn(3, 4))
    torch.testing.assert_close(target.act(obs, policy_ids=1, deterministic=True)["action"],
                               source.act(obs, policy_ids=1, deterministic=True)["action"])
    assert metadata["step"] == 123 and metadata["checkpoint_sha256"] == sha256(path)
    with pytest.raises(ValueError, match="policy ID"):
        load_policy(path, device="cpu", policy_id=2)


def test_recorder_rejects_mismatched_scene_bank_before_simulation(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = SimpleNamespace(bank_manifest=manifest, body_collision_bank=manifest, body_collision_resets=manifest)
    with pytest.raises(ValueError, match="bank_sha256"):
        verify_contract(dict(bank_sha256="wrong"), args)
