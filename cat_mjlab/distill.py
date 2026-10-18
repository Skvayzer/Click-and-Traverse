"""DAgger distillation of per-domain experts into one student (experts + distillation structure).

Experts are frozen policies, each responsible for a set of scene types (reporting buckets from
runner.SCENE_BUCKETS). The student is the ordinary native actor-critic being trained.

Per control step (inside the unchanged rollout loop):
  * every expert computes its deterministic action mean for all envs (cheap MLPs);
  * each env is labelled by the expert that owns its current scene type;
  * the ACTION EXECUTED is the expert's with probability beta (per env, re-drawn at episode start),
    otherwise the student's (mean + small exploration noise). beta decays from beta_start to
    beta_end over beta_decay updates: early on the experts drive (good states), later the student
    drives and the experts label the states the student actually reaches (the DAgger fix for
    compounding error).
Per update: supervised regression of the student's action mean onto the expert labels over the
rollout (plus a replay of recent rollouts), several epochs of minibatches. No reward is used for
learning; task rewards and success metrics are still computed and logged for evaluation.

Experts:
  * .npz array archives (e.g. the released CAT generalist, outputs/mjlab_migration_20260919/
    original-cat-expanded.npz) -> actor on the first 222 observation features (base contract);
  * native checkpoints (resume.pt / best.pt) -> actor on as many features as it was trained with.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .learning import ActorCritic, Learner, LearnerConfig, gaussian_parameters, log_probability


def load_expert(path, device):
    path = Path(path)
    if path.suffix == ".npz":
        from .conversion import load_array_archive
        learner = Learner(LearnerConfig(algorithm="ppo"), device="cpu")
        load_array_archive(learner, path, restore_optimizer=False)
        model = learner.model
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = payload.get("learner", payload)
        config = dict(state["config"]); config.setdefault("style_critic", False)
        model = ActorCritic(LearnerConfig(**config))
        model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class DistillController:
    """Stands in for Learner in collect_rollout (act/config/device) and in the training loop (update)."""

    regularize = None       # plain DAgger unless __init__ sets the RL fine-tune mode

    def __init__(self, learner, task, experts, routing, *, beta_start=1., beta_end=0., beta_decay=60,
                 epochs=2, lr=3e-4, replay_rollouts=3, student_noise=.05, scene_routing=None, dropped=None,
                 regularize=None):
        from .runner import SCENE_BUCKETS, _scene_buckets
        self.learner, self.task = learner, task
        self.config, self.device = learner.config, learner.device
        self.names = list(experts)
        self.experts = [load_expert(experts[n], self.device) for n in self.names]
        bucket_of_scene = torch.as_tensor(_scene_buckets(task.bank.manifest), device=self.device)
        owner_of_bucket = torch.full((len(SCENE_BUCKETS),), -1, dtype=torch.long, device=self.device)
        for name, buckets in routing.items():
            if name == "_default":
                continue
            for b in buckets:
                owner_of_bucket[SCENE_BUCKETS.index(b)] = self.names.index(name)
        unowned = [SCENE_BUCKETS[i] for i in range(len(SCENE_BUCKETS)) if owner_of_bucket[i] < 0]
        self.default_owner = self.names.index(routing["_default"][0]) if "_default" in routing else len(self.names) - 1
        owner_of_bucket[owner_of_bucket < 0] = self.default_owner
        self.owner_of_scene = owner_of_bucket[bucket_of_scene]
        self.unowned = unowned
        self.scene_routed = 0
        if scene_routing:
            # Per-scene owners (scripts/build_distill_routing.py: the expert with the higher measured clean success
            # in THIS scene). Bucket routing cannot express the traversal/narrow split, which runs through the
            # middle of procedural_cat and published_cat by opening width. Unlisted scenes keep the bucket owner.
            index = {s["scene_id"]: i for i, s in enumerate(task.bank.manifest["scenes"])}
            unknown = sorted(set(scene_routing.values()) - set(self.names))
            if unknown:
                raise ValueError(f"scene routing names experts that were not given: {unknown}")
            missing = [sid for sid in scene_routing if sid not in index]
            if missing:
                raise ValueError(f"scene routing lists {len(missing)} scene ids not in the bank, e.g. {missing[:3]}")
            rows = torch.tensor([index[sid] for sid in scene_routing], dtype=torch.long, device=self.device)
            owners = torch.tensor([self.names.index(n) for n in scene_routing.values()], dtype=torch.long, device=self.device)
            self.owner_of_scene = self.owner_of_scene.clone()
            self.owner_of_scene[rows] = owners
            self.scene_routed = len(rows)
        if dropped:
            # Scenes without a competent teacher: no label (mask 0). Only meaningful in regularize mode; plain
            # DAgger never samples them (zero weight).
            index = {s["scene_id"]: i for i, s in enumerate(task.bank.manifest["scenes"])}
            rows = torch.tensor([index[sid] for sid in dropped if sid in index], dtype=torch.long, device=self.device)
            self.owner_of_scene = self.owner_of_scene.clone(); self.owner_of_scene[rows] = -1
        # RL fine-tune mode: PPO drives and learns (the student acts stochastically, values and log-probs from the
        # learner); expert labels ride along as a fading regulariser in the PPO loss; the first `critic_warmup`
        # updates train only the critic (actor frozen). dict(lambda_start, lambda_end, decay, critic_warmup).
        self.regularize = regularize
        self.beta_start, self.beta_end, self.beta_decay = beta_start, beta_end, beta_decay
        self.beta = beta_start
        self.expert_drives = torch.ones(task.num_envs, dtype=torch.bool, device=self.device)
        self.epochs, self.replay_rollouts, self.noise = epochs, replay_rollouts, student_noise
        self.optimizer = torch.optim.Adam(learner.model.actor.parameters(), lr=lr)
        self.labels, self.replay = [], []
        self.updates = 0

    # ---------------------------------------------------------------- rollout side
    def _expert_means(self, state):
        owner = self.owner_of_scene[self.task.scene_ids]
        label = torch.zeros((len(state), self.config.action_size), device=self.device)
        for k, model in enumerate(self.experts):
            rows = owner == k   # owner -1 (no competent teacher): label stays 0, masked out
            if bool(rows.any()):
                width = model.config.actor_obs
                mean, _ = gaussian_parameters(model.logits(state[rows, :width], 0))
                label[rows] = mean
        return label, owner

    @torch.no_grad()
    def act(self, obs, policy_ids=None, *, deterministic=False):
        state, privileged = obs["state"], obs["privileged_state"]
        if self.regularize is not None:
            label, owner = self._expert_means(state)
            self.labels.append((label, owner))
            return self.learner.act(state, privileged, policy_ids, deterministic=deterministic)
        n = len(state)
        # Re-draw who drives for envs that just started an episode (step counter 0).
        fresh = self.task.info["step"] == 0
        if bool(fresh.any()):
            draw = torch.rand(n, device=self.device) < self.beta
            self.expert_drives = torch.where(fresh, draw, self.expert_drives)
        label, owner = self._expert_means(state)
        ids = torch.zeros(n, dtype=torch.long, device=self.device)
        logits = self.learner.model.logits(state, ids)
        mean, scale = gaussian_parameters(logits)
        student = mean + self.noise * torch.randn_like(mean)
        raw = torch.where(self.expert_drives[:, None], label, student)
        self.labels.append((label, owner))
        return dict(action=raw.tanh(), raw_action=raw, log_prob=log_probability(logits, raw), mean=mean,
                    value=torch.zeros(n, device=self.device), action_std=scale, policy_id=ids)

    # ---------------------------------------------------------------- training side
    def update(self, rollout):
        trajectories, T = rollout["reward"].shape
        n = self.task.num_envs
        # Labels were appended in collect_rollout's loop order: for each env block, for each t.
        label = torch.empty((trajectories, T, self.config.action_size), device=self.device)
        owner = torch.empty((trajectories, T), dtype=torch.long, device=self.device)
        k = 0
        for begin in range(0, trajectories, n):
            for t in range(T):
                label[begin:begin + n, t], owner[begin:begin + n, t] = self.labels[k]
                k += 1
        self.labels.clear()
        if self.regularize is not None:
            return self._regularized_update(rollout, label, owner)
        batch = dict(state=rollout["state"].reshape(-1, rollout["state"].shape[-1]), label=label.reshape(-1, label.shape[-1]),
                     owner=owner.reshape(-1))
        self.replay.append(batch); self.replay = self.replay[-self.replay_rollouts:]
        data = {key: torch.cat([b[key] for b in self.replay]) for key in batch}
        rows = len(data["label"]); per = max(1, rows // self.config.num_minibatches)
        model = self.learner.model; ids0 = torch.zeros(per, dtype=torch.long, device=self.device)
        losses, per_owner = [], {}
        for _ in range(self.epochs):
            perm = torch.randperm(rows, device=self.device)
            for start in range(0, rows - per + 1, per):
                idx = perm[start:start + per]
                mean, _ = gaussian_parameters(model.logits(data["state"][idx], ids0[:len(idx)]))
                err = (mean - data["label"][idx]).square().mean(-1)
                loss = err.mean()
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
                self.optimizer.step(); losses.append(float(loss))
        with torch.no_grad():   # per-expert fit on the newest rollout
            mean, _ = gaussian_parameters(model.logits(batch["state"], torch.zeros(len(batch["state"]), dtype=torch.long, device=self.device)))
            err = (mean - batch["label"]).square().mean(-1)
            for j, name in enumerate(self.names):
                m = batch["owner"] == j
                if bool(m.any()):
                    per_owner[f"distill/{name}/action_mse"] = float(err[m].mean())
                    per_owner[f"distill/{name}/sample_share"] = float(m.float().mean())
        self.updates += 1
        self.beta = max(self.beta_end, self.beta_start - (self.beta_start - self.beta_end) * self.updates / max(1, self.beta_decay))
        self.learner.updates += 1; self.learner.env_steps += trajectories * T
        return dict(total_loss=sum(losses) / max(1, len(losses)), policy_loss=sum(losses) / max(1, len(losses)),
                    v_loss=0., entropy_loss=0., rollout_reward_mean=float(rollout["reward"].mean()),
                    physical_transitions=trajectories * T, optimizer_transitions=rows,
                    env_steps=self.learner.env_steps, updates=self.learner.updates,
                    **per_owner, **{"distill/beta_next": self.beta, "distill/expert_driving_share": float(self.expert_drives.float().mean())})

    def _regularized_update(self, rollout, label, owner):
        r = self.regularize
        progress = min(1., self.updates / max(1, int(r.get("decay", 200))))
        lam = float(r.get("lambda_start", 1.)) + (float(r.get("lambda_end", 0.)) - float(r.get("lambda_start", 1.))) * progress
        frozen = self.updates < int(r.get("critic_warmup", 0))
        self.learner.distill_lambda = 0. if frozen else lam
        self.learner.actor_frozen = frozen
        rollout = dict(rollout, distill_label=label, distill_mask=(owner >= 0).float())
        metrics = self.learner.update(rollout)
        per_owner = {}
        with torch.no_grad():
            state = rollout["state"].reshape(-1, rollout["state"].shape[-1])
            mean, _ = gaussian_parameters(self.learner.model.logits(state, torch.zeros(len(state), dtype=torch.long, device=self.device)))
            err = (mean - label.reshape(-1, label.shape[-1])).square().mean(-1); flat = owner.reshape(-1)
            for j, name in enumerate(self.names):
                m = flat == j
                if bool(m.any()):
                    per_owner[f"distill/{name}/action_mse"] = float(err[m].mean())
                    per_owner[f"distill/{name}/sample_share"] = float(m.float().mean())
            per_owner["distill/unlabelled_share"] = float((flat < 0).float().mean())
        self.updates += 1
        return dict(metrics, **per_owner, **{"distill/lambda": self.learner.distill_lambda, "distill/actor_frozen": float(frozen)})

