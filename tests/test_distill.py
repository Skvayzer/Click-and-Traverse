"""CPU checks for DAgger label bookkeeping and the regression step."""
import types
import torch

from cat_mjlab.learning import Learner, LearnerConfig


def _controller(n=4, T=3, beta=1.):
    from cat_mjlab.distill import DistillController
    cfg = LearnerConfig(algorithm="ppo", actor_obs=6, critic_obs=8, action_size=3, actor_hidden=(16,), critic_hidden=(8,), num_minibatches=2)
    learner = Learner(cfg, device="cpu")
    c = DistillController.__new__(DistillController)
    c.learner, c.config, c.device = learner, cfg, torch.device("cpu")
    expert = Learner(cfg, device="cpu").model.eval()
    c.names, c.experts = ["a", "b"], [expert, expert]
    c.task = types.SimpleNamespace(num_envs=n, scene_ids=torch.arange(n) % 2, info=dict(step=torch.zeros(n, dtype=torch.long)))
    c.owner_of_scene = torch.tensor([0, 1])
    c.beta, c.beta_start, c.beta_end, c.beta_decay = beta, 1., 0., 2
    c.expert_drives = torch.ones(n, dtype=torch.bool)
    c.epochs, c.replay_rollouts, c.noise = 30, 2, 0.
    c.optimizer = torch.optim.Adam(learner.model.actor.parameters(), lr=1e-2)
    c.labels, c.replay, c.updates = [], [], 0
    return c, expert


def test_expert_drives_at_beta_one_and_labels_align():
    c, expert = _controller()
    obs = dict(state=torch.randn(4, 6), privileged_state=torch.randn(4, 8))
    out = c.act(obs)
    from cat_mjlab.learning import gaussian_parameters
    mean, _ = gaussian_parameters(expert.logits(obs["state"], torch.zeros(4, dtype=torch.long)))
    assert torch.allclose(out["raw_action"], mean)          # beta = 1: the expert's action is executed
    assert torch.equal(c.labels[0][1], torch.tensor([0, 1, 0, 1]))


def test_regression_moves_the_student_toward_the_labels():
    c, expert = _controller()
    T = 3; states = torch.randn(4, T, 6)
    for t in range(T):
        c.act(dict(state=states[:, t], privileged_state=torch.randn(4, 8)))
    rollout = dict(state=states, reward=torch.zeros(4, T))
    first = c.update(rollout)
    for _ in range(5):
        for t in range(T):
            c.act(dict(state=states[:, t], privileged_state=torch.randn(4, 8)))
        last = c.update(rollout)
    assert last["distill/a/action_mse"] < first["distill/a/action_mse"] * 0.5
    assert c.beta == 0.                                       # decayed over 2 updates
