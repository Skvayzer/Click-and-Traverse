"""Fixed-seed traversal validation sharing the learner's existing field bank.

Validation is synchronous and owns no second environment or scene-field copy.
The caller runs it between updates, never concurrently with learner tracing.
It uses the same network factory/distribution as PPO, with deterministic and
stochastic actions on paired reset/noise streams.  These are fixed training-bank
regression scenes, not an unseen-scene generalization benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time


# Fixed before the diagnostic rollouts in analysis/cat-noise-ablation-20260916/
# selection.json. Resolve by identity, since expanded-bank indices can differ.
FIXED_SCENE_IDS = (
    "D8G0L1O0S3", "D8G2L1O2S16", "published-forward", "published-hurdle1",
    "published-hurdle3", "published-crouch1", "published-side2",
    "published-side-hurdle-crouch1", "procedural-D4G2L0O0S2001",
    "procedural-D4G0L2O0S2001", "procedural-D7G1L2O1S2001",
    "procedural-D10G2L3O2S2001",
    "random-furniture-dense-train-004001-5ef9779babcf",
    "random-furniture-dense-train-004002-4f7ed1085e60",
    "random-generic_clutter-dense-train-005001-1eaef5f4115f",
    "random-generic_clutter-dense-train-005002-d4f59651508b",
)
MODES = ("deterministic", "stochastic")
CLUTTER_FAMILIES = frozenset(("furniture", "generic_clutter"))


def select_scenes(manifest, scene_ids=FIXED_SCENE_IDS):
    """Reject missing or ambiguous identities instead of changing the benchmark."""
    requested = tuple(scene_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Validation scene identities must be nonempty and unique")
    scenes = manifest["scenes"]
    lookup = {scene["scene_id"]: index for index, scene in enumerate(scenes)}
    if len(lookup) != len(scenes):
        raise ValueError("Scene bank contains duplicate identities")
    missing = set(requested) - lookup.keys()
    if missing:
        raise ValueError(f"Fixed validation scenes missing from bank: {sorted(missing)}")
    selected = [lookup[scene_id] for scene_id in requested]
    if not any(scenes[index]["family"] in CLUTTER_FAMILIES for index in selected):
        raise ValueError("Validation requires clutter scenes")
    if not any(scenes[index]["family"] not in CLUTTER_FAMILIES for index in selected):
        raise ValueError("Validation requires CAT retention scenes")
    return selected


def summarize_episodes(episodes):
    """Equal-weight episode rates, with separate CAT and clutter denominators."""
    if not episodes:
        raise ValueError("Cannot summarize an empty validation")

    def summary(rows):
        count = len(rows)
        if not count:
            raise ValueError("Empty validation subgroup")
        rate = lambda key: sum(bool(row[key]) for row in rows) / count
        successful = [row for row in rows if row["goal_reached"]]
        # No NaN/inf values are emitted to checkpoint metadata or W&B. The
        # completion-time denominator is logged explicitly when there is none.
        return {
            "episode_count": count,
            "goal_success_rate": rate("goal_reached"),
            "fall_rate": rate("fall"),
            "obstacle_violation_rate": rate("obstacle"),
            "hand_violation_rate": rate("hand_violation"),
            "elbow_violation_rate": rate("elbow_violation"),
            "self_contact_rate": rate("self_contact"),
            "numerical_failure_rate": rate("numerical"),
            "outside_bounds_rate": rate("outside_bounds"),
            "timeout_rate": rate("timeout"),
            "mean_episode_seconds": sum(row["seconds"] for row in rows) / count,
            "completion_time_seconds": (sum(row["seconds"] for row in successful) / len(successful)
                                        if successful else 0.0),
            "completion_time_count": len(successful),
            "mean_min_hand_clearance_m": sum(row["min_hand_clearance"] for row in rows) / count,
            "minimum_hand_clearance_m": min(row["min_hand_clearance"] for row in rows),
            "mean_return": sum(row["return"] for row in rows) / count,
        }

    result = summary(episodes)
    for label, rows in (
        ("cat", [row for row in episodes if row["family"] not in CLUTTER_FAMILIES]),
        ("clutter", [row for row in episodes if row["family"] in CLUTTER_FAMILIES]),
    ):
        result.update({f"{label}_{key}": value for key, value in summary(rows).items()})
    result["scenes"] = {
        scene_id: dict(summary(rows), family=rows[0]["family"])
        for scene_id in dict.fromkeys(row["scene_id"] for row in episodes)
        for rows in [[row for row in episodes if row["scene_id"] == scene_id]]
    }
    return result


def retention_selection(modes, baseline, *, max_cat_drop=0.05, max_scene_drop=0.125):
    """Gate CAT retention in both action modes, then rank clutter performance.

    Rates are absolute differences, not relative percentages. Per-scene gating
    prevents improved easy scenes from masking a lost hurdle/passage behavior.
    The default scene tolerance is two of the sixteen fixed seeds. The caller
    must not publish ineligible candidates, even when no incumbent exists.
    """
    if not (0 <= max_cat_drop <= 1 and 0 <= max_scene_drop <= 1):
        raise ValueError("Retention tolerances must lie in [0, 1]")
    reasons = []
    for mode in MODES:
        current, reference = modes[mode], baseline[mode]
        if set(current["scenes"]) != set(reference["scenes"]):
            raise ValueError("Baseline and candidate validation scene identities differ")
        if current["episode_count"] != reference["episode_count"]:
            raise ValueError("Baseline and candidate episode counts differ")
        for record in (current, reference):
            numeric = [value for key, value in record.items() if key != "scenes"]
            if not all(math.isfinite(float(value)) for value in numeric):
                raise ValueError("Selection requires finite validation metrics")
        if current["cat_goal_success_rate"] + max_cat_drop + 1e-12 < reference["cat_goal_success_rate"]:
            reasons.append(f"{mode}: aggregate CAT success below baseline tolerance")
        if current["numerical_failure_rate"] > 0:
            reasons.append(f"{mode}: numerical failure")
        for scene_id, scene in current["scenes"].items():
            old = reference["scenes"][scene_id]
            if scene["family"] != old["family"] or scene["episode_count"] != old["episode_count"]:
                raise ValueError("Baseline and candidate scene populations differ")
            if (scene["family"] not in CLUTTER_FAMILIES
                    and scene["goal_success_rate"] + max_scene_drop + 1e-12 < old["goal_success_rate"]):
                reasons.append(f"{mode}: {scene_id} success below baseline tolerance")
    # Evaluate both deployment and actual exploration behavior. A candidate
    # cannot improve the primary score by hiding poor stochastic behavior.
    score = (
        min(modes[mode]["clutter_goal_success_rate"] for mode in MODES),
        -max(modes[mode]["clutter_hand_violation_rate"] for mode in MODES),
        min(modes[mode]["cat_goal_success_rate"] for mode in MODES),
        min(modes[mode]["clutter_mean_min_hand_clearance_m"] for mode in MODES),
    )
    return {
        "eligible": not reasons, "score": list(score), "reasons": reasons,
        "max_cat_drop": max_cat_drop, "max_scene_drop": max_scene_drop,
        "ordering": ["minimum_mode_clutter_goal_success", "negative_maximum_mode_clutter_hand_violation",
                     "minimum_mode_cat_goal_success", "minimum_mode_clutter_mean_min_hand_clearance"],
    }


@dataclass(frozen=True)
class ValidationResult:
    step: int
    metrics: dict
    modes: dict
    episodes: dict
    selection: dict | None
    metadata: dict

    def as_dict(self):
        return dict(step=self.step, metrics=self.metrics, modes=self.modes,
                    episodes=self.episodes, selection=self.selection, metadata=self.metadata)


class RetentionValidator:
    """Use an existing raw CAT environment, factory and device-resident fields."""

    def __init__(self, environment, network_factory, *, scene_ids=FIXED_SCENE_IDS,
                 seeds=range(16), chunk_steps=25):
        import jax
        import jax.numpy as jp
        from cat_ppo.learning.policy.ppo.field_arguments import FieldArguments

        self.env = environment
        self.seeds = tuple(seeds)
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or seed < 0 for seed in self.seeds)):
            raise ValueError("Validation seeds must be distinct nonnegative integers")
        if type(chunk_steps) is not int or chunk_steps <= 0:
            raise ValueError("Validation chunk_steps must be a positive integer")
        self.chunk_steps = chunk_steps
        self.selected = select_scenes(environment.field_bank_manifest, scene_ids)
        self.pairs = [(index, seed) for index in self.selected for seed in self.seeds]
        self.count = len(self.pairs)
        self.scene_indices = jp.asarray([index for index, _ in self.pairs], dtype=jp.int32)
        # Keys use selected-scene order, not mutable bank indices. Consequently
        # adding unrelated training scenes never changes the regression draws.
        ordinals = {index: ordinal for ordinal, index in enumerate(self.selected)}
        self.reset_keys = jp.stack([jax.random.fold_in(jax.random.PRNGKey(seed), ordinals[index])
                                   for index, seed in self.pairs])
        self.noise_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0xCA7AB1))(self.reset_keys)
        self.binding = FieldArguments(environment)
        if self.binding.names[:3] != ("sdf", "bf", "gf"):
            raise ValueError("Retention validation requires the existing scene-bank field buffers")
        contract = environment.observation_contract()
        self.network = network_factory(
            {"state": (len(contract["actor_features"]),),
             "privileged_state": (len(contract["critic_features"]),)}, environment.action_size)
        self.limits = environment._pf_scene_episode_lengths[self.scene_indices]
        self.max_steps = int(jp.max(self.limits))
        self._reset = jax.jit(self._reset_impl)
        self._advance = jax.jit(self._advance_impl)

    def _reset_impl(self, fields):
        import jax
        with self.binding.bind(fields):
            return jax.vmap(self.env.reset_with_pf_id)(self.reset_keys, self.scene_indices)

    def _advance_impl(self, state, accumulator, active, index, params, stochastic, fields):
        import jax
        import jax.numpy as jp

        def advance(carry, _):
            state, acc, active, index = carry
            logits = self.network.policy_network.apply(params[0], params[1], state.obs)
            distribution = self.network.parametric_action_distribution
            keys = jax.vmap(lambda key: jax.random.fold_in(key, index))(self.noise_keys)
            sampled = jax.vmap(distribution.sample)(logits, keys)
            actions = jp.where(stochastic, sampled, distribution.mode(logits))
            stepped = jax.vmap(self.env.step)(state, actions)
            # Native raw task has no autoreset; freeze all leaves at first
            # completion. Inactive rows cannot contribute rewards or outcomes.
            state = jax.tree.map(lambda old, new: jp.where(
                active.reshape((self.count,) + (1,) * (new.ndim - 1)), new, old), state, stepped)
            episode = state.info["wholebody_episode"]
            faults = state.info["wholebody_faults"]
            failure = active & faults["any"]
            success = active & episode["goal_reached"] & ~failure
            length = acc["length"] + active.astype(jp.int32)
            timeout = active & (length >= self.limits) & ~failure & ~success
            # A positive native done that is not an identified failure must be
            # a timeout, never an invented successful termination.
            unexplained_done = active & (state.done > 0) & ~failure & ~success & ~timeout
            next_acc = {
                "length": length,
                "return": acc["return"] + jp.where(active, state.reward, 0.),
                "timeout": acc["timeout"] | timeout,
                "unexplained_done": acc["unexplained_done"] | unexplained_done,
                "goal_reached": acc["goal_reached"] | success,
                "min_hand_clearance": jp.minimum(acc["min_hand_clearance"], jp.where(
                    active, jp.min(state.info["handsdf"].reshape(self.count, -1), axis=-1), jp.inf)),
            }
            for key in ("fall", "obstacle", "self_contact", "numerical", "outside_bounds",
                        "hand_violation", "elbow_violation"):
                next_acc[key] = acc[key] | (active & episode[key])
            return (state, next_acc, active & ~failure & ~success & ~timeout & ~unexplained_done, index + 1), None

        with self.binding.bind(fields):
            return jax.lax.scan(advance, (state, accumulator, active, index), None,
                                length=self.chunk_steps)[0]

    def evaluate(self, params, *, step, baseline=None):
        """Return host summaries; params are native (normalizer, actor, critic).

        Completed means first valid goal, native failure, or native horizon.
        Goal completion requires root and both feet and rejects lateral bypass;
        the shared environment records these predicates before native counters
        reset. Failures on the goal-reaching transition take precedence.
        """
        import jax
        import jax.numpy as jp
        import numpy as np

        if type(step) is not int or step < 0:
            raise ValueError("Validation step must be a nonnegative integer")
        started = time.monotonic()
        fields = self.binding.values
        initial = self._reset(fields)
        jax.block_until_ready(initial)
        if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(params)):
            raise ValueError("Cannot evaluate nonfinite parameters")
        modes, episode_rows = {}, {}
        for mode in MODES:
            zeros = jp.zeros(self.count, dtype=jp.float32)
            false = jp.zeros(self.count, dtype=bool)
            acc = dict(length=jp.zeros(self.count, dtype=jp.int32), **{"return": zeros},
                       timeout=false, unexplained_done=false, goal_reached=false,
                       min_hand_clearance=jp.min(initial.info["handsdf"].reshape(self.count, -1), axis=-1))
            acc.update({key: false for key in ("fall", "obstacle", "self_contact", "numerical",
                                              "hand_violation", "elbow_violation")})
            acc["outside_bounds"] = initial.info["wholebody_episode"]["outside_bounds"]
            state, active, index = initial, jp.ones(self.count, dtype=bool), jp.int32(0)
            for _ in range(math.ceil(self.max_steps / self.chunk_steps)):
                state, acc, active, index = self._advance(
                    state, acc, active, index, params[:2], jp.asarray(mode == "stochastic"), fields)
                if not bool(jp.any(active)):
                    break
            arrays = {key: np.asarray(value) for key, value in acc.items()}
            if np.any(arrays["unexplained_done"]):
                raise RuntimeError("Native CAT terminated without a recorded failure or horizon")
            if bool(jp.any(active)):
                raise RuntimeError("Validation finished with incomplete episodes")
            rows = []
            for row_index, (scene_index, seed) in enumerate(self.pairs):
                scene = self.env.field_bank_manifest["scenes"][scene_index]
                row = {key: values[row_index].item() for key, values in arrays.items()
                       if key != "unexplained_done"}
                row.update(scene_id=scene["scene_id"], family=scene["family"], seed=seed,
                           seconds=row["length"] * self.env.dt,
                           horizon_steps=int(self.limits[row_index]))
                if not math.isfinite(row["min_hand_clearance"]):
                    # Numerical failures remain failed episodes. Give their
                    # clearance a conservative finite sentinel for aggregation.
                    row["min_hand_clearance"] = -1.0
                if not math.isfinite(row["return"]):
                    row["return"] = 0.0
                rows.append(row)
            episode_rows[mode] = rows
            modes[mode] = summarize_episodes(rows)
        selection = retention_selection(modes, baseline) if baseline is not None else None
        metrics = {}
        for mode, summary in modes.items():
            prefix = "validation/" if mode == "deterministic" else "validation/stochastic/"
            metrics.update({prefix + key: value for key, value in summary.items() if key != "scenes"})
        metrics["validation/evaluation_seconds"] = time.monotonic() - started
        if selection is not None:
            metrics["validation/retention_eligible"] = int(selection["eligible"])
        metadata = dict(schema="cat-fixed-retention-validation-v1", seeds=list(self.seeds),
                        scene_ids=[self.env.field_bank_manifest["scenes"][index]["scene_id"]
                                   for index in self.selected],
                        scenes=[dict(scene_id=self.env.field_bank_manifest["scenes"][index]["scene_id"],
                                     family=self.env.field_bank_manifest["scenes"][index]["family"],
                                     horizon_steps=int(self.env._pf_scene_episode_lengths[index]))
                                for index in self.selected],
                        episodes_per_mode=self.count, modes=list(MODES),
                        horizons="native per-scene 1000/4000 control steps",
                        stopping="first clean whole-body goal, native failure, or native timeout",
                        ablation_difference="earlier standalone noise diagnostic continued after reaching the goal",
                        baseline="released CAT expanded into the same whole-body architecture and distribution",
                        generalization="fixed training-bank regression scenes; not held-out layouts",
                        fields="same device buffers passed as dynamic operands; no copied scene bank",
                        top_level_metrics="deterministic deployment; stochastic mode has its own prefix")
        return ValidationResult(step, metrics, modes, episode_rows, selection, metadata)
