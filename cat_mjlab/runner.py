"""Continuous training-only mjlab runner with one run and bounded checkpoints.

No evaluators or retention/rollback mechanisms are constructed. Contrastive best
models use role-balanced leader training success. Other banks retain the leader
rollout reward criterion. One overwritten
resume.pt stores learner/Adam, task, simulator and PyTorch RNG state. Changing
physics backend starts a new W&B lineage even when weights/Adam are converted.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, fields
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time
import uuid

import torch

from .checkpoint_arrays import read_array_archive, sampling_state_from_archive
from .conversion import load_array_archive
from .learning import Learner, LearnerConfig


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_device_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_device_tree(item, device) for item in value)
    return value


def atomic_torch_save(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Checkpoint destination cannot be a symlink")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(_cpu_tree(value), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class SuccessWindow:
    """First-outcome counts, never virtual relabeled SAPG transitions."""
    def __init__(self, maxlen=100):
        self.navigation = deque(maxlen=maxlen)
        self.contrast = deque(maxlen=maxlen)
        self.roles = deque(maxlen=maxlen)

    def append(self, navigation, contrast, roles=None):
        self.navigation.append(navigation.detach().cpu().tolist())
        self.contrast.append(contrast.detach().cpu().tolist())
        self.roles.append((torch.zeros(4, 2, dtype=torch.long) if roles is None else roles).detach().cpu().tolist())

    def role_balanced_score(self):
        """Equal role weights, only after each role has resolved leader episodes."""
        if not self.roles:
            return None
        counts = torch.tensor(list(self.roles), dtype=torch.int64).sum(dim=0)
        if bool((counts[:, 0] == 0).any()):
            return None
        return float((counts[:, 1].double() / counts[:, 0]).mean())

    def metrics(self, *, contrastive=False):
        result = {}
        for rows, names in ((self.navigation, ("cat_goal", "ordinary_clutter_goal", "hand_protection_goal", "goal")),
                            (self.contrast, ("forward_protected", "narrow_passage", "posture_transition"))):
            if not rows:
                continue
            counts = torch.tensor(list(rows), dtype=torch.int64).sum(dim=0)
            for name, (resolved, success) in zip(names, counts.tolist()):
                if contrastive and name in ("cat_goal", "ordinary_clutter_goal", "hand_protection_goal"):
                    continue
                if resolved:
                    result[f"success/{name}_success_rate"] = success / resolved
                result[f"training/{name}_resolved_count"] = resolved
        return result

    def state_dict(self):
        return dict(maxlen=self.navigation.maxlen, navigation=list(self.navigation),
                    contrast=list(self.contrast), roles=list(self.roles))

    def load_state_dict(self, state):
        if state["maxlen"] != self.navigation.maxlen:
            raise ValueError("Success window configuration differs")
        self.navigation.clear(); self.navigation.extend(state["navigation"])
        self.contrast.clear(); self.contrast.extend(state["contrast"])
        self.roles.clear(); self.roles.extend(state["roles"])


class Logger:
    def __init__(self, directory, *, mode, project, entity, record, resume=False):
        self.directory = Path(directory)
        self.run = None
        self.path = self.directory / "wandb.json"
        if resume:
            self.identity = json.loads(self.path.read_text())
            if any(self.identity[key] != value for key, value in (("mode", mode), ("project", project), ("entity", entity))):
                raise ValueError("W&B destination differs from the saved run")
        else:
            self.identity = dict(id=uuid.uuid4().hex[:8], mode=mode, project=project, entity=entity,
                                 initialized=False, last_global_step=-1)
            atomic_json(self.path, self.identity)
        if mode != "disabled":
            import wandb
            self.run = wandb.init(project=project, entity=entity or None, id=self.identity["id"], mode=mode,
                resume=("must" if resume and self.identity["initialized"] else "allow" if resume else "never"),
                name=self.directory.name, dir=str(self.directory), config=record)
            self.run.define_metric("global_step")
            self.run.define_metric("*", step_metric="global_step")
            self.run.define_metric("success/*", step_metric="global_step", summary="max")
            self.identity.update(initialized=True, url=self.run.url)
            atomic_json(self.path, self.identity)

    def log(self, step, values):
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise FloatingPointError("Refusing nonfinite training metrics")
        # A crash can leave logs ahead of the last durable update. Resume the
        # same identity, suppress stale steps until learner progress catches up.
        if int(step) < self.identity.get("last_global_step", -1):
            return
        event = dict(global_step=int(step), **{key: float(value) for key, value in values.items()})
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if self.run is not None:
            self.run.log(event)
        self.identity["last_global_step"] = int(step)
        atomic_json(self.path, self.identity)
        print(json.dumps(event, sort_keys=True), flush=True)

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


@contextmanager
def stop_requests(directory):
    requested = [False]
    def handler(signum, frame):
        del signum, frame
        requested[0] = True
    previous = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield lambda: requested[0] or (Path(directory) / "STOP").exists()
    finally:
        for sig, value in previous.items():
            signal.signal(sig, value)


def collect_rollout(task, learner, *, unroll_length, trajectories, policy_ids):
    """Collect native [trajectory,time] chunks with bounded preallocated storage.

    The old wrapper exposes autoreset observations to the learner; preserve that
    choice rather than substituting the separately available terminal_obs.
    """
    n = task.num_envs
    if trajectories % n:
        raise ValueError("Trajectory count must be divisible by parallel environments")
    config = learner.config
    shape = (trajectories, unroll_length)
    rollout = {key: torch.empty((*shape, width), device=learner.device) for key, width in (
        ("state", config.actor_obs), ("privileged_state", config.critic_obs),
        ("next_state", config.actor_obs), ("next_privileged_state", config.critic_obs),
        ("raw_action", config.action_size))}
    rollout.update({key: torch.empty(shape, device=learner.device) for key in ("log_prob", "reward", "discount", "truncation")})
    rollout["policy_id"] = torch.empty(shape, dtype=torch.long, device=learner.device)
    before_nav = task.navigation_counts.clone()
    before_contrast = task.contrast_counts.clone()
    before_roles = task.role_counts.clone()
    completed = torch.zeros((), device=learner.device)
    sum_return = completed.clone(); sum_length = completed.clone(); collisions = completed.clone()
    std_sum = completed.clone()
    for begin in range(0, trajectories, n):
        section = slice(begin, begin + n)
        for t in range(unroll_length):
            # task.step mutates task.obs; persist current observations first.
            rollout["state"][section, t].copy_(task.obs["state"])
            rollout["privileged_state"][section, t].copy_(task.obs["privileged_state"])
            acted = learner.act(task.obs, policy_ids=policy_ids)
            transition = task.step(acted["action"])
            rollout["next_state"][section, t].copy_(transition["obs"]["state"])
            rollout["next_privileged_state"][section, t].copy_(transition["obs"]["privileged_state"])
            for key in ("raw_action", "log_prob", "policy_id"):
                rollout[key][section, t].copy_(acted[key])
            rollout["reward"][section, t].copy_(transition["reward"])
            rollout["discount"][section, t].copy_((~transition["done"]).float())
            rollout["truncation"][section, t].copy_(transition["truncated"].float())
            done = transition["done"]
            metrics = transition["metrics"]
            completed += done.sum()
            sum_return += torch.where(done, metrics["episode_return"], 0.).sum()
            sum_length += torch.where(done, metrics["episode_length"], 0.).sum()
            collisions += (done & metrics.get("episode/body_collision", torch.zeros_like(done))).sum()
            std_sum += acted["action_std"].sum()
    info = dict(navigation_counts=task.navigation_counts - before_nav,
                contrast_counts=task.contrast_counts - before_contrast,
                role_counts=task.role_counts - before_roles,
                action_std=std_sum / (trajectories * unroll_length * config.action_size))
    info["metrics"] = {"training/completed_episode_count": float(completed), "training/action_std": float(info["action_std"])}
    if bool(completed > 0):
        info["metrics"].update({"episode/return": float(sum_return / completed),
                               "episode/length": float(sum_length / completed),
                               "training/body_collision_rate": float(collisions / completed)})
    return rollout, info


def learner_config_from_archive(metadata, *, algorithm=None):
    source = metadata.get("contract", {})
    algorithm = algorithm or ("sapg" if source.get("sapg") else "ppo")
    accepted = {field.name for field in fields(LearnerConfig)}
    kwargs = {key: value for key, value in source.items() if key in accepted}
    kwargs.update(source.get("sapg", {}))
    kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    kwargs["algorithm"] = algorithm
    return LearnerConfig(**kwargs)


def environment_config_from_archive(metadata):
    source = metadata.get("contract", {})
    recorded = source.get("metadata", {}).get("config", {})
    config = recorded.get("env_config")
    if config is None:
        config = json.loads((ROOT / "configs/cat_generalist_released.json").read_text())["env_config"]
    randomize = source.get("randomize_initial_episode_steps",
                           recorded.get("policy_config", {}).get("randomize_initial_episode_steps", True))
    return dict(config, randomize_initial_episode_steps=randomize)


def create_task(args, *, environment_config=None):
    from .sim import CATSimulation
    from .scene_bank import SceneBank
    from .collision import CollisionChecker
    from .config import wholebody_config
    from .task import CATTask
    sim = CATSimulation(args.num_envs, device=args.device, nconmax=args.nconmax, njmax=args.njmax)
    bank = SceneBank(args.bank_manifest, device=args.device, reset_manifest=args.body_collision_resets,
                     collision_manifest=args.body_collision_bank)
    collision = CollisionChecker(sim.model, args.body_collision_bank, field_manifest=args.bank_manifest, device=args.device)
    config = wholebody_config(environment_config, stabilization=True, hand_protection=True, hand_contrast=bank.has_contrast)
    config["wholebody"]["body_collision"].update(enabled=True, bank_manifest=str(args.body_collision_bank),
                                                 reset_manifest=str(args.body_collision_resets))
    task = CATTask(sim, bank, config, collision=collision, seed=args.seed)
    if args.compile_task:
        task.enable_compilation()
    return task, sim, config


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_identity():
    paths = sorted((ROOT / "cat_mjlab").glob("*.py")) + [ROOT / "train_cat_mjlab.py"]
    hashes = {str(path.relative_to(ROOT)): _file_hash(path) for path in paths if path.is_file()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _versions():
    result = {}
    for name in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run(args):
    directory = Path(args.run_dir).resolve()
    if args.max_updates < 0 or args.checkpoint_interval_updates < 1:
        raise ValueError("Invalid update/checkpoint interval")
    if args.command == "verify" and (args.max_updates < 1 or args.wandb_mode != "disabled"):
        raise ValueError("Verification must be bounded and W&B disabled")
    if args.resume and args.fresh_optimizer:
        raise ValueError("--fresh-optimizer applies only to initial backend migration")
    if (directory / "STOP").exists():
        raise ValueError("STOP exists; retire it deliberately before resuming")
    if directory.is_symlink():
        raise ValueError("Run directory cannot be a symlink")
    if not args.resume:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Run directory is not empty; use --resume for a complete native resume")
        directory.mkdir(parents=True, exist_ok=True)
    elif not directory.is_dir():
        raise ValueError("Resume directory does not exist")
    metadata, archive_arrays = read_array_archive(args.checkpoint_npz)
    config = learner_config_from_archive(metadata, algorithm=args.algorithm)
    source = metadata.get("contract", {})
    batch_size = args.batch_size if args.batch_size is not None else source.get("batch_size", args.num_envs // config.num_minibatches)
    unroll = args.unroll_length if args.unroll_length is not None else source.get("unroll_length", 32)
    trajectories = batch_size * config.num_minibatches
    if min(batch_size, unroll, args.num_envs) <= 0 or trajectories % args.num_envs:
        raise ValueError("batch_size*num_minibatches must be divisible by num_envs")
    if config.algorithm == "sapg" and (args.num_envs % config.num_policies or batch_size % config.num_policies):
        raise ValueError("SAPG num_envs and batch_size must be divisible by the policy count")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    learner = Learner(config, device=args.device)
    migration = load_array_archive(learner, args.checkpoint_npz, restore_optimizer=not args.fresh_optimizer)
    config_source = environment_config_from_archive(metadata)
    task, sim, environment_config = create_task(args, environment_config=config_source)
    policies = config.num_policies if config.algorithm == "sapg" else 1
    policy_ids = torch.arange(policies, device=learner.device).repeat_interleave(args.num_envs // policies)
    task.set_policy_ids(policy_ids)
    if not args.resume:
        sampling, report = sampling_state_from_archive(metadata, archive_arrays, _file_hash(args.bank_manifest))
        migration["sampling"] = report
        if sampling is not None:
            task.restore_sampling_state(sampling, resample=True)
    del archive_arrays
    contract = dict(learner_config=asdict(config), num_envs=args.num_envs, batch_size=batch_size,
        unroll_length=unroll, seed=args.seed, nconmax=args.nconmax, njmax=args.njmax,
        compile_task=args.compile_task,
        bank_sha256=_file_hash(args.bank_manifest), collision_sha256=_file_hash(args.body_collision_bank),
        resets_sha256=_file_hash(args.body_collision_resets), source_sha256=_source_identity(), versions=_versions(),
        environment_config=environment_config, checkpoint_archive_sha256=_file_hash(args.checkpoint_npz))
    contract = json.loads(json.dumps(contract))  # tuples -> JSON lists consistently
    selection = "role-balanced leader training success" if task.hand_contrast else "leader mean physical rollout reward"
    record = dict(schema="cat-mjlab-run-v1", contract=contract, migration=migration,
                  backend="mjlab/MuJoCo Warp", evaluation=False, retention=False,
                  best_selection=selection, success="leader-only first outcomes; 100-update rolling window",
                  best_score_semantics="100-update training history; published weights are post-update; no checkpoint evaluation",
                  adaptive_sampling="all physical policies; posture-qualified successes for protected/transition roles",
                  physics_timestep=.002, control_timestep=.02,
                  state_resume="full state restoration; GPU Warp does not guarantee bitwise deterministic trajectories")
    window = SuccessWindow()
    best_score = None
    previous_walltime = 0.
    if args.resume:
        old_record = json.loads((directory / "run.json").read_text())
        if old_record["contract"] != contract:
            raise ValueError("Native mjlab resume contract differs from saved run")
        snapshot = torch.load(directory / "resume.pt", map_location="cpu", weights_only=True)
        if snapshot.get("schema") != "cat-mjlab-runtime-v1" or snapshot["contract"] != contract:
            raise ValueError("Native runtime contract differs")
        learner.load_state_dict(snapshot["learner"])
        sim.load_state_dict(_device_tree(snapshot["simulation"], learner.device))
        task.load_state_dict(_device_tree(snapshot["task"], learner.device))
        torch.set_rng_state(snapshot["rng_cpu"])
        if snapshot["rng_cuda"]:
            torch.cuda.set_rng_state_all(snapshot["rng_cuda"])
        window.load_state_dict(snapshot["success_window"])
        best_score, previous_walltime = snapshot["best_score"], snapshot["walltime"]
        # Best-model publication can be newer than the less frequent full
        # runtime snapshot. Preserve its selection score across crash recovery.
        if (directory / "best.pt").exists():
            incumbent = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            if (incumbent.get("schema") != "cat-mjlab-best-v1" or incumbent.get("contract") != contract
                    or not math.isfinite(incumbent["score"])):
                raise ValueError("Existing best checkpoint differs from the native run contract")
            best_score = incumbent["score"] if best_score is None else max(best_score, incumbent["score"])
        record = old_record
    else:
        atomic_json(directory / "run.json", record)
    logger = Logger(directory, mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity,
                    record=record, resume=args.resume)
    started = time.monotonic()
    local_updates = 0
    status = "running"

    def save_runtime():
        snapshot = dict(schema="cat-mjlab-runtime-v1", contract=contract, learner=learner.state_dict(),
            simulation=sim.state_dict(), task=task.state_dict(), rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            success_window=window.state_dict(), best_score=best_score,
            walltime=previous_walltime + time.monotonic() - started)
        atomic_torch_save(directory / "resume.pt", snapshot)
        atomic_json(directory / "status.json", dict(phase=status, env_steps=learner.env_steps,
            updates=learner.updates, best_score=best_score, resume="resume.pt", walltime=snapshot["walltime"]))

    try:
        if not args.resume:
            save_runtime()
        with stop_requests(directory) as should_stop:
            while not should_stop() and (args.max_updates == 0 or local_updates < args.max_updates):
                began = time.monotonic()
                rollout, collected = collect_rollout(task, learner, unroll_length=unroll,
                                                     trajectories=trajectories, policy_ids=policy_ids)
                sim.capacity_report()  # Latched overflow must stop before any gradient update.
                metrics = learner.update(rollout)
                del rollout
                local_updates += 1
                window.append(collected["navigation_counts"], collected["contrast_counts"], collected["role_counts"])
                elapsed = time.monotonic() - began
                score = window.role_balanced_score() if task.hand_contrast else metrics["rollout_reward_mean"]
                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    atomic_torch_save(directory / "best.pt", dict(schema="cat-mjlab-best-v1",
                        model=learner.model.state_dict(), config=asdict(config), contract=contract, step=learner.env_steps,
                        score=score, selected_policy_id=0, selection=selection,
                        score_semantics=record["best_score_semantics"],
                        observation_contract=task.contract))
                values = {"learner/" + key: value for key, value in metrics.items()
                          if key in ("total_loss", "policy_loss", "v_loss", "entropy_loss")}
                values.update(window.metrics(contrastive=task.hand_contrast)); values.update(collected["metrics"])
                values.update({"training/rollout_reward_mean": metrics["rollout_reward_mean"],
                    "performance/control_steps_per_second": metrics["physical_transitions"] / elapsed,
                    "performance/update_seconds": elapsed})
                if task.hand_contrast and score is not None:
                    values["training/checkpoint_selection_score"] = score
                if learner.device.type == "cuda":
                    free, total = torch.cuda.mem_get_info(learner.device)
                    values["performance/device_vram_used_gib"] = (total - free) / 2**30
                logger.log(learner.env_steps, values)
                if local_updates % args.checkpoint_interval_updates == 0:
                    save_runtime()
            status = "stopped_by_request" if should_stop() else "bounded_verification_complete" if args.command == "verify" else "completed_requested_updates"
        save_runtime()
    except BaseException as error:
        # Preserve the last committed snapshot: a partially failed update is not
        # a valid replacement for its finite pre-update learner/environment.
        status = "failed"
        atomic_json(directory / "failure.json", dict(error=type(error).__name__, message=str(error),
            observed_steps=learner.env_steps, local_updates=local_updates, durable_checkpoint="resume.pt"))
        logger.finish(exit_code=1)
        raise
    else:
        logger.finish()
    return dict(status=status, run_dir=str(directory), env_steps=learner.env_steps, updates=local_updates, best_score=best_score)
