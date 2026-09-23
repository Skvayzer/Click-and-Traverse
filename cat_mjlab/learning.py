"""Native PyTorch CAT PPO/SAPG, independent of simulator and RSL-RL defaults.

``Learner.act(state, privileged_state, policy_ids)`` returns a mapping containing
``action`` (tanh bounded), ``raw_action``, ``log_prob``, ``mean``, ``value``,
``action_std`` and ``policy_id``. Observations may instead be a mapping with
``state`` and ``privileged_state`` keys. SAPG IDs are fixed per environment.

``Learner.update(rollout)`` expects every tensor to start with [trajectory,time]
and keys: state, privileged_state, next_state, next_privileged_state, raw_action,
log_prob, reward, discount, truncation, policy_id. discount is 1-done; truncation
is the separate native horizon flag. next observations must follow the original
CAT wrapper semantics. Shuffle trajectories, never individual time positions.

PPO recomputes Brax GAE and minibatch normalization on each optimizer pass.
SAPG freezes targets, augments one complete follower block for the leader, and
normalizes the complete augmented rollout once. Both share CAT's squashed
state-dependent Gaussian, Monte Carlo transformed entropy, and value factor .25.
No evaluation, retention gate, rollback, learning-rate schedule or reward changes.
"""
from __future__ import annotations

from .observation_contract import ACTOR_SIZE, CRITIC_SIZE

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LearnerConfig:
    algorithm: str = "sapg"
    actor_obs: int = ACTOR_SIZE
    critic_obs: int = CRITIC_SIZE
    action_size: int = 29
    actor_hidden: tuple[int, ...] = (512, 256, 128, 64)
    critic_hidden: tuple[int, ...] = (1024, 512, 256, 128)
    num_policies: int = 6
    embedding_dim: int = 16
    learning_rate: float = 3e-4
    entropy_cost: float = .003
    discounting: float = .98
    gae_lambda: float = .95
    clipping_epsilon: float = .2
    reward_scaling: float = 1.
    max_grad_norm: float | None = 1.
    normalize_advantage: bool = True
    num_minibatches: int = 64
    num_updates_per_batch: int = 4
    prepare_chunk_size: int = 64
    max_action_std: float | None = None

    def __post_init__(self):
        if self.algorithm not in ("ppo", "sapg"):
            raise ValueError("algorithm must be ppo or sapg")
        for name in ("actor_obs", "critic_obs", "action_size", "num_policies",
                     "embedding_dim", "num_minibatches", "num_updates_per_batch", "prepare_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.algorithm == "sapg" and self.num_policies < 2:
            raise ValueError("SAPG needs a leader and a follower")
        if not 0 <= self.clipping_epsilon < 1 or not 0 <= self.gae_lambda <= 1 or not 0 <= self.discounting <= 1:
            raise ValueError("Invalid clipping or discount/GAE coefficient")
        if any(not math.isfinite(v) or v < 0 for v in (self.learning_rate, self.entropy_cost, self.reward_scaling)):
            raise ValueError("Learning and reward coefficients must be finite and nonnegative")
        if self.max_action_std is not None and (not math.isfinite(self.max_action_std) or self.max_action_std <= .001):
            raise ValueError("max_action_std must exceed the distribution floor .001 or be None")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive or None")


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.silu(layer(x))
        return self.layers[-1](x)


class ActorCritic(nn.Module):
    """Single embedding table shared by both trunks, with joint gradients."""
    def __init__(self, config: LearnerConfig):
        super().__init__()
        self.config = config
        width = config.embedding_dim if config.algorithm == "sapg" else 0
        self.actor = MLP((config.actor_obs + width, *config.actor_hidden, 2 * config.action_size))
        self.critic = MLP((config.critic_obs + width, *config.critic_hidden, 1))
        if width:
            self.policy_embeddings = nn.Parameter(torch.randn(config.num_policies, width) * .01)
            with torch.no_grad():
                self.actor.layers[0].weight[:, -width:] = 0
                self.critic.layers[0].weight[:, -width:] = 0
        else:
            self.register_parameter("policy_embeddings", None)

    def condition(self, observations, policy_ids):
        if self.policy_embeddings is None:
            return observations
        ids = torch.as_tensor(policy_ids, device=observations.device)
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("policy_ids must be integers")
        ids = torch.broadcast_to(ids, observations.shape[:-1])
        # F.embedding checks both negative and upper out-of-range indices.
        return torch.cat((observations, F.embedding(ids, self.policy_embeddings)), dim=-1)

    def logits(self, state, policy_ids=0):
        logits = self.actor(self.condition(state, policy_ids))
        if self.config.max_action_std is None:
            return logits
        # CAT parameterizes sigma with softplus, not an independent log-sigma.
        # Capping this monotone head is equivalent to capping log(sigma).
        mean, raw_scale = logits.chunk(2, dim=-1)
        maximum = math.log(math.expm1(self.config.max_action_std - .001))
        return torch.cat((mean, raw_scale.clamp_max(maximum)), dim=-1)

    def value(self, privileged_state, policy_ids=0):
        return self.critic(self.condition(privileged_state, policy_ids)).squeeze(-1)


def gaussian_parameters(logits):
    mean, raw_scale = logits.chunk(2, dim=-1)
    return mean, F.softplus(raw_scale) + .001


def tanh_log_det(raw_action):
    return 2 * (math.log(2.) - raw_action - F.softplus(-2 * raw_action))


def log_probability(logits, raw_action):
    mean, scale = gaussian_parameters(logits)
    # Preserve Brax's arithmetic ordering, including x/std - mean/std.
    density = -.5 * (raw_action / scale - mean / scale).square()
    density -= .5 * math.log(2 * math.pi) + scale.log()
    return (density - tanh_log_det(raw_action)).sum(dim=-1)


def transformed_entropy(logits, noise=None):
    mean, scale = gaussian_parameters(logits)
    noise = torch.randn_like(mean) if noise is None else noise
    gaussian_entropy = .5 + .5 * math.log(2 * math.pi) + scale.log()
    return (gaussian_entropy + tanh_log_det(mean + scale * noise)).sum(dim=-1)


@torch.no_grad()
def compute_gae(truncation, termination, rewards, values, bootstrap_value, *, lambda_=.95, discount=.98):
    """Exact Brax target/advantage equations; inputs [B,T], bootstrap [B]."""
    mask = 1 - truncation
    following = torch.cat((values[:, 1:], bootstrap_value[:, None]), dim=1)
    delta = (rewards + discount * (1 - termination) * following - values) * mask
    accumulator = torch.zeros_like(bootstrap_value)
    residual = torch.empty_like(values)
    for t in range(values.shape[1] - 1, -1, -1):
        accumulator = delta[:, t] + discount * (1 - termination[:, t]) * mask[:, t] * lambda_ * accumulator
        residual[:, t] = accumulator
    target = residual + values
    next_target = torch.cat((target[:, 1:], bootstrap_value[:, None]), dim=1)
    advantage = (rewards + discount * (1 - termination) * next_target - values) * mask
    return target, advantage


def importance_clipped_surrogate(log_prob, old_log_prob, behavior_log_prob, log_importance,
                                 advantages, clipping_epsilon, *, log_normalizer=0.):
    """Overflow-resistant SAPG surrogate, without capping importance weights.

    This preserves the crash fix in cat_ppo/learning/policy/sapg/losses.py:
    combine current/behavior likelihoods directly, include |A| and 1/N before
    exponentiating, and mask zero advantages before exp. Do not replace with
    exp(mu)*min(exp(r)*A,clip(exp(r))*A).
    """
    if not 0 <= clipping_epsilon < 1:
        raise ValueError("clipping_epsilon must lie in [0,1)")
    lower = log_prob.new_tensor(1 - clipping_epsilon).log()
    upper = log_prob.new_tensor(1 + clipping_epsilon).log()
    weighted_unclipped = log_prob - behavior_log_prob
    weighted_clipped = log_importance + (log_prob - old_log_prob).clamp(lower, upper)
    log_weight = torch.where(advantages >= 0,
                             torch.minimum(weighted_unclipped, weighted_clipped),
                             torch.maximum(weighted_unclipped, weighted_clipped))
    nonzero = advantages != 0
    magnitude = torch.where(nonzero, advantages.abs(), torch.ones_like(advantages)).log()
    exponent = torch.where(nonzero, log_weight + magnitude - log_normalizer, torch.zeros_like(log_weight))
    return advantages.sign() * exponent.exp()


def _normalize(advantages):
    return (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)


def _validate_rollout(data, config):
    required = ("state", "privileged_state", "next_state", "next_privileged_state",
                "raw_action", "log_prob", "reward", "discount", "truncation", "policy_id")
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"Missing rollout tensors: {sorted(missing)}")
    shape = data["reward"].shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("rollout must have [trajectory,time] rewards")
    for key in required:
        if data[key].shape[:2] != shape:
            raise ValueError(f"{key} leading shape differs from rewards")
    for key, width in (("state", config.actor_obs), ("next_state", config.actor_obs),
                       ("privileged_state", config.critic_obs), ("next_privileged_state", config.critic_obs),
                       ("raw_action", config.action_size)):
        if data[key].shape != (*shape, width):
            raise ValueError(f"{key} trailing feature shape differs")
    for key in ("log_prob", "discount", "truncation", "policy_id"):
        if data[key].shape != shape:
            raise ValueError(f"{key} must have scalar [trajectory,time] shape")
    ids = data["policy_id"]
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("policy_id must be an integer tensor")
    if config.algorithm == "sapg":
        count = config.num_policies
        if shape[0] % count or not bool(torch.all(ids == ids[:, :1])):
            raise ValueError("SAPG requires equally sized groups and a fixed ID per trajectory")
        counts = (ids[:, :1] == torch.arange(count, device=ids.device)).sum(dim=0)
        if not bool(torch.all(counts == shape[0] // count)):
            raise ValueError("SAPG requires equal policy group counts with valid IDs")


@torch.no_grad()
def prepare_sapg_rollout(model, data, config, *, follower_id=None):
    _validate_rollout(data, config)
    size = data["reward"].shape[0]
    ids = data["policy_id"]
    baseline = torch.empty_like(data["reward"])
    bootstrap = torch.empty_like(data["reward"][:, 0])
    std = torch.empty_like(data["reward"])
    for begin in range(0, size, config.prepare_chunk_size):
        s = slice(begin, begin + config.prepare_chunk_size)
        baseline[s] = model.value(data["privileged_state"][s], ids[s])
        bootstrap[s] = model.value(data["next_privileged_state"][s, -1], ids[s, -1])
        std[s] = gaussian_parameters(model.logits(data["state"][s], ids[s]))[1].mean(dim=-1)
    truncation = data["truncation"]
    termination = (1 - data["discount"]) * (1 - truncation)
    target, advantage = compute_gae(truncation, termination, data["reward"] * config.reward_scaling,
                                    baseline, bootstrap, lambda_=config.gae_lambda, discount=config.discounting)
    if follower_id is None:
        follower_id = int(torch.randint(1, config.num_policies, ()).item())
    if not 1 <= follower_id < config.num_policies:
        raise ValueError("follower_id must select a follower")
    selected = torch.nonzero(ids[:, 0] == follower_id, as_tuple=True)[0]
    follower = {key: value[selected] for key, value in data.items()}
    leader_ids = torch.zeros_like(follower["policy_id"])
    leader_baseline = torch.empty_like(follower["reward"])
    leader_next = torch.empty_like(leader_baseline)
    leader_logprob = torch.empty_like(leader_baseline)
    for begin in range(0, len(selected), config.prepare_chunk_size):
        s = slice(begin, begin + config.prepare_chunk_size)
        leader_baseline[s] = model.value(follower["privileged_state"][s], leader_ids[s])
        leader_next[s] = model.value(follower["next_privileged_state"][s], leader_ids[s])
        leader_logprob[s] = log_probability(model.logits(follower["state"][s], leader_ids[s]), follower["raw_action"][s])
    leader_target = follower["reward"] * config.reward_scaling + config.discounting * (1 - termination[selected]) * leader_next
    leader_target = torch.where(follower["truncation"] != 0, leader_baseline, leader_target)
    leader_advantage = (leader_target - leader_baseline) * (1 - follower["truncation"])
    result = {key: torch.cat((value, follower[key]), dim=0) for key, value in data.items()}
    advantage = torch.cat((advantage, leader_advantage), dim=0)
    if 'task_bucket' in result:
        result['raw_advantage']=advantage.clone()
        result['initial_value_error']=torch.cat((target-baseline,leader_target-leader_baseline))
    if config.normalize_advantage:
        advantage = _normalize(advantage)
    advantage *= 1 - result["truncation"]
    result.update(
        target_value=torch.cat((target, leader_target)), advantage=advantage,
        old_target_log_prob=torch.cat((data["log_prob"], leader_logprob)),
        log_importance=torch.cat((torch.zeros_like(data["reward"]), leader_logprob - follower["log_prob"])),
        target_policy_id=torch.cat((ids, leader_ids)),
        action_std=torch.cat((std, std[selected])),
    )
    return result


@torch.no_grad()
def ratio_diagnostics(log_prob, data, config):
    """Detached minibatch diagnostics; ESS uses shifted weights to avoid overflow.

    Quantiles/ESS are averaged over nonempty minibatches by update(); maxima
    retain the maximum across the update. Counts include repeated PPO epochs.
    """
    behavior = data["policy_id"]
    target = data.get("target_policy_id", behavior)
    old = data.get("old_target_log_prob", data["log_prob"])
    importance = data.get("log_importance", torch.zeros_like(log_prob))
    ratio = log_prob.detach() - old
    lower, upper = math.log1p(-config.clipping_epsilon), math.log1p(config.clipping_epsilon)
    result = {}
    for name, mask in (("on_policy_leader", (target == 0) & (behavior == 0)),
                       ("followers", target != 0),
                       ("relabeled_leader", (target == 0) & (behavior != 0))):
        weights = importance[mask].double()
        if not weights.numel():
            continue
        prefix = f"diagnostics/{name}/"
        result[prefix + "sample_count"] = weights.new_tensor(weights.numel())
        for label, q in (("p01", .01), ("p50", .5), ("p95", .95), ("p99", .99)):
            result[prefix + "importance_log_weight_" + label] = torch.quantile(weights, q)
        result[prefix + "importance_log_weight_max"] = weights.max()
        shifted = (weights - weights.max()).exp()
        ess = shifted.sum().square() / shifted.square().sum()
        result[prefix + "importance_ess"] = ess
        result[prefix + "importance_ess_fraction"] = ess / weights.numel()
        result[prefix + "ppo_ratio_clip_fraction"] = ((ratio[mask] < lower) | (ratio[mask] > upper)).double().mean()
    if 'task_bucket' in data:
        bucket=data['task_bucket']
        for name,mask in [('retention',bucket<4),('narrow',bucket==4),('flat',bucket==5)]:
            selected=ratio[mask]
            if selected.numel():
                prefix=f'diagnostics/task_updates/{name}/'
                result[prefix+'sample_count']=selected.new_tensor(selected.numel())
                result[prefix+'ppo_ratio_clip_fraction']=((selected<lower)|(selected>upper)).float().mean()
    return result


def compute_loss(model, data, config, *, entropy_noise=None):
    ids = data["target_policy_id"] if config.algorithm == "sapg" else data["policy_id"]
    logits = model.logits(data["state"], ids)
    baseline = model.value(data["privileged_state"], ids)
    log_prob = log_probability(logits, data["raw_action"])
    if config.algorithm == "sapg":
        advantage = data["advantage"].detach()
        targets = data["target_value"].detach()
        policy_loss = -importance_clipped_surrogate(
            log_prob, data["old_target_log_prob"].detach(), data["log_prob"].detach(),
            data["log_importance"].detach(), advantage, config.clipping_epsilon,
            log_normalizer=log_prob.new_tensor(advantage.numel()).log()).sum()
    else:
        with torch.no_grad():
            bootstrap = model.value(data["next_privileged_state"][:, -1], ids[:, -1])
            termination = (1 - data["discount"]) * (1 - data["truncation"])
            targets, advantage = compute_gae(data["truncation"], termination,
                data["reward"] * config.reward_scaling, baseline.detach(), bootstrap,
                lambda_=config.gae_lambda, discount=config.discounting)
            if config.normalize_advantage:
                advantage = _normalize(advantage)
        ratio = (log_prob - data["log_prob"].detach()).exp()
        policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - config.clipping_epsilon,
                                                                 1 + config.clipping_epsilon) * advantage).mean()
    value_loss = .25 * (targets - baseline).square().mean()
    entropy_loss = -config.entropy_cost * transformed_entropy(logits, entropy_noise).mean()
    total = policy_loss + value_loss + entropy_loss
    return total, dict(total_loss=total, policy_loss=policy_loss, v_loss=value_loss, entropy_loss=entropy_loss) | ratio_diagnostics(log_prob, data, config)


@torch.no_grad()
def task_share_diagnostics(data):
    """Augmented SAPG batch, before optimization; descriptive shares, not gradients."""
    if 'raw_advantage' not in data:return {}
    raw=data['raw_advantage'];adv=data['advantage'];reward=data['reward'];err=data['initial_value_error']
    bucket=data['task_bucket'];prefix='diagnostics/task_share/'
    out={prefix+'raw_advantage_mean':float(raw.mean()),prefix+'raw_advantage_std':float(raw.std(correction=0))}
    denom_adv=adv.abs().sum().clamp_min(1e-12);denom_reward=reward.abs().sum().clamp_min(1e-12)
    denom_value=err.square().sum().clamp_min(1e-12)
    for name,mask in [('retention',bucket<4),('narrow',bucket==4),('flat',bucket==5)]:
        p=prefix+name+'/'
        count=int(mask.sum());out[p+'sample_count']=count;out[p+'sample_fraction']=count/raw.numel()
        if not count:continue
        out.update({p+'reward_mean':float(reward[mask].mean()),
            p+'absolute_reward_share':float(reward[mask].abs().sum()/denom_reward),
            p+'raw_advantage_mean':float(raw[mask].mean()),p+'raw_advantage_std':float(raw[mask].std(correction=0)),
            p+'normalized_advantage_abs_mean':float(adv[mask].abs().mean()),
            p+'normalized_advantage_mean':float(adv[mask].mean()),
            p+'normalized_advantage_positive_fraction':float((adv[mask]>0).float().mean()),
            p+'normalized_advantage_abs_share':float(adv[mask].abs().sum()/denom_adv),
            p+'initial_value_error_mse':float(err[mask].square().mean()),
            p+'initial_value_error_squared_share':float(err[mask].square().sum()/denom_value)})
    return out


class Learner:
    def __init__(self, config: LearnerConfig | None = None, *, device="cuda"):
        self.config = config or LearnerConfig()
        self.device = torch.device(device)
        self.model = ActorCritic(self.config).to(self.device)
        # Match Optax Adam defaults, with epsilon outside sqrt and no weight decay.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                           betas=(.9, .999), eps=1e-8, foreach=False)
        self.updates = 0
        self.env_steps = 0

    @torch.no_grad()
    def reset_action_std(self, sigma):
        if not math.isfinite(sigma) or sigma <= .001 or (self.config.max_action_std is not None and sigma > self.config.max_action_std):
            raise ValueError("Initial sigma must exceed .001 and not exceed its ceiling")
        if self.optimizer.state:
            raise ValueError("Resetting sigma requires a fresh optimizer")
        layer = self.model.actor.layers[-1]
        layer.weight[self.config.action_size:].zero_()
        layer.bias[self.config.action_size:].fill_(math.log(math.expm1(sigma - .001)))

    @torch.no_grad()
    def act(self, state, privileged_state=None, policy_ids=None, *, deterministic=False):
        if isinstance(state, dict):
            state, privileged_state = state["state"], state["privileged_state"]
        if privileged_state is None:
            raise ValueError("privileged_state is required for the asymmetric critic")
        ids = torch.zeros(state.shape[:-1], device=state.device, dtype=torch.long) if policy_ids is None else policy_ids
        ids = torch.broadcast_to(torch.as_tensor(ids, device=state.device), state.shape[:-1])
        logits = self.model.logits(state, ids)
        mean, scale = gaussian_parameters(logits)
        raw = mean if deterministic else mean + scale * torch.randn_like(mean)
        return dict(action=raw.tanh(), raw_action=raw, log_prob=log_probability(logits, raw), mean=mean,
                    value=self.model.value(privileged_state, ids), action_std=scale, policy_id=ids)

    def update(self, rollout):
        data = {key: value.detach() for key, value in rollout.items()}
        _validate_rollout(data, self.config)
        physical = data["reward"].numel()
        leader = data["policy_id"] == 0
        reward_proxy = data["reward"][leader].mean()
        if self.config.algorithm == "sapg":
            data = prepare_sapg_rollout(self.model, data, self.config)
        shares=task_share_diagnostics(data)
        for key in ('raw_advantage','initial_value_error'):data.pop(key,None)
        rows = data["reward"].shape[0]
        if rows % self.config.num_minibatches:
            raise ValueError("Augmented trajectory count must divide evenly into num_minibatches")
        totals = {}
        metric_steps = {}
        steps = 0
        for _ in range(self.config.num_updates_per_batch):
            permutation = torch.randperm(rows, device=self.device)
            for indices in permutation.reshape(self.config.num_minibatches, -1):
                mini = {key: value[indices] for key, value in data.items()}
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = compute_loss(self.model, mini, self.config)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite learner loss; refusing optimizer step")
                loss.backward()
                grads = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
                norm = torch.stack([gradient.square().sum() for gradient in grads]).sum().sqrt()
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite learner gradient; refusing optimizer step")
                if self.config.max_grad_norm is not None:
                    # torch clip_grad_norm_ adds 1e-6; Optax does not.
                    factor = torch.where(norm < self.config.max_grad_norm, torch.ones_like(norm),
                                         self.config.max_grad_norm / norm)
                    for gradient in grads:
                        gradient.mul_(factor)
                self.optimizer.step()
                for key, value in metrics.items():
                    value = value.detach()
                    if key.endswith("_max"):
                        totals[key] = torch.maximum(totals.get(key, value), value)
                    else:
                        totals[key] = totals.get(key, 0.) + value
                    metric_steps[key] = metric_steps.get(key, 0) + 1
                steps += 1
        self.updates += 1
        self.env_steps += physical
        return {key: float(value if key.endswith(("_max", "/sample_count")) else value / metric_steps[key]) for key, value in totals.items()} | {
            "rollout_reward_mean": float(reward_proxy), "optimizer_steps": steps,
            "physical_transitions": physical, "optimizer_transitions": data["reward"].numel(),
            "env_steps": self.env_steps, "updates": self.updates,
        } | shares

    def state_dict(self):
        return dict(schema="cat-mjlab-learner-v1", config=asdict(self.config), model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(), updates=self.updates, env_steps=self.env_steps)

    def load_state_dict(self, state):
        if state.get("schema") != "cat-mjlab-learner-v1" or dict(state.get("config", {}), max_action_std=state.get("config", {}).get("max_action_std")) != asdict(self.config):
            raise ValueError("Learner schema or configuration differs")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates, self.env_steps = int(state["updates"]), int(state["env_steps"])
