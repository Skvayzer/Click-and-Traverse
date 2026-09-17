"""Persistent regression decisions for the unchanged 22-scene hand benchmark.

This module does not save/load networks, alter learning rates, or log W&B
metrics. The launcher owns those operations and acknowledges a recovery only
after restoring the anchor. Fixed-seed rate thresholds are operational guards,
not statistical significance tests. Existing per-original-scene gates remain
additional requirements for publishing a best checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import numbers

from cat_ppo.furniture.retention_validation import FIXED_SCENE_IDS, MODES


SCHEMA = "cat-regression-recovery-v1"
GROUPS = (("cat", 192, .05), ("ordinary_clutter", 64, .10),
          ("hand_protection", 96, .10))


def _required(value, key, label):
    if not isinstance(value, dict) or key not in value:
        raise ValueError(f"Missing recovery evidence: {label}.{key}")
    return value[key]


def _number(value, label, *, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"Recovery requires numeric {label}")
    value = float(value)
    if (not math.isfinite(value) or value < 0
            or (maximum is not None and value > maximum)
            or (integer and value != int(value))):
        raise ValueError(f"Recovery requires finite in-range {label}")
    return int(value) if integer else value


def _count(value, key, expected, label):
    if _number(_required(value, key, label), f"{label}.{key}", integer=True) != expected:
        raise ValueError(f"Recovery benchmark population differs: {label}.{key}")


def _rate(value, key, count, label):
    rate = _number(_required(value, key, label), f"{label}.{key}", maximum=1.)
    if not math.isclose(rate * count, round(rate * count), rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Recovery success disagrees with integer episodes: {label}.{key}")
    return rate


def _nonfinite(value):
    if isinstance(value, dict):
        return any(_nonfinite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_nonfinite(child) for child in value)
    return isinstance(value, numbers.Real) and not math.isfinite(float(value))


def _benchmark(metadata):
    """Select identity fields, excluding descriptive provenance/run clock."""
    fields = ("schema", "scene_ids", "scenes", "seeds", "modes",
              "episodes_per_mode", "stopping")
    result = {key: deepcopy(_required(metadata, key, "validation metadata")) for key in fields}
    ids, rows = result["scene_ids"], result["scenes"]
    if (result["schema"] != "cat-fixed-retention-validation-v1"
            or not isinstance(ids, list) or len(ids) != 22
            or len(set(ids)) != 22 or tuple(ids[:16]) != FIXED_SCENE_IDS
            or not isinstance(rows, list) or len(rows) != 22):
        raise ValueError("Recovery requires the unchanged fixed 16-scene prefix plus six hand scenes")
    if result["seeds"] != list(range(16)) or tuple(result["modes"]) != MODES:
        raise ValueError("Recovery requires unchanged seeds 0..15 and both action modes")
    _count(result, "episodes_per_mode", 352, "metadata")
    families = ["original_cat"] * 2 + ["published_cat"] * 6 + ["procedural_cat"] * 4
    families += ["furniture"] * 2 + ["generic_clutter"] * 2
    for index, (identity, row) in enumerate(zip(ids, rows)):
        if row.get("scene_id") != identity:
            raise ValueError("Recovery scene rows and identities differ in order")
        family = _required(row, "family", identity)
        if ((index < 16 and family != families[index])
                or (index >= 16 and family not in ("furniture", "generic_clutter"))):
            raise ValueError("Recovery benchmark families differ")
        _count(row, "horizon_steps", 1000 if index < 12 else 4000, identity)
    return result


def _summaries(modes, benchmark, *, per_scene):
    compact = {}
    for mode in MODES:
        current = _required(modes, mode, "modes")
        _count(current, "episode_count", 352, mode)
        _count(current, "clutter_episode_count", 160, mode)
        summary = {}
        for group, count, _ in GROUPS:
            _count(current, group + "_episode_count", count, mode)
            key = group + "_goal_success_rate"
            summary[key] = _rate(current, key, count, mode)
        summary["numerical_failure_rate"] = _rate(current, "numerical_failure_rate", 352, mode)
        if per_scene:
            scenes = _required(current, "scenes", mode)
            if set(scenes) != set(benchmark["scene_ids"]):
                raise ValueError("Recovery candidate scene populations differ")
            rates = []
            for row in benchmark["scenes"]:
                identity = row["scene_id"]
                scene = scenes[identity]
                _count(scene, "episode_count", 16, identity)
                if scene.get("family") != row["family"]:
                    raise ValueError("Recovery candidate scene family differs")
                rates.append(_rate(scene, "goal_success_rate", 16, identity))
            for group, values in (("cat", rates[:12]), ("ordinary_clutter", rates[12:16]),
                                  ("hand_protection", rates[16:])):
                if not math.isclose(summary[group + "_goal_success_rate"], sum(values) / len(values),
                                    rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Recovery aggregate success differs from per-scene episodes")
        compact[mode] = summary
    return compact


def _state_copy(state):
    if state.get("schema") != SCHEMA:
        raise ValueError("Unknown recovery state schema")
    # State can be persisted as ordinary JSON; reject NaN rather than writing it.
    return json.loads(json.dumps(state, allow_nan=False))


def initialize_recovery_state(source_selection, checkpoint, *, confirmations=2,
                              max_cat_drop=.05, max_ordinary_clutter_drop=.10,
                              max_hand_protection_drop=.10):
    """Initialize from an eligible *22-scene* selected checkpoint archive.

    Archived source steps and new-run steps have different clocks. No new-run
    evaluation has occurred yet, so last_step starts at -1, not source_step.
    The protected initial baseline never changes when a new best is promoted.
    """
    metrics = _required(source_selection, "metrics", "source selection")
    selected = _required(metrics, "selection", "source metrics")
    if source_selection.get("selection_source") != "retention_validation" or selected.get("eligible") is not True:
        raise ValueError("Recovery requires an eligible retention-selected source checkpoint")
    benchmark = _benchmark(_required(_required(source_selection, "provenance", "source"),
                                      "validation", "source provenance"))
    archived = _required(metrics, "validation", "source metrics")
    if _nonfinite(archived):
        raise ValueError("Protected source must have finite validation metrics")
    modes = {}
    for mode in MODES:
        prefix = "validation/" if mode == "deterministic" else "validation/stochastic/"
        modes[mode] = {key[len(prefix):]: value for key, value in archived.items()
                       if key.startswith(prefix) and "/" not in key[len(prefix):]}
    summaries = _summaries(modes, benchmark, per_scene=False)
    if any(summary["numerical_failure_rate"] for summary in summaries.values()):
        raise ValueError("Protected source has numerical failures")
    source_step = _number(_required(source_selection, "step", "source selection"), "source step", integer=True)
    confirmations = _number(confirmations, "confirmations", integer=True)
    if confirmations < 2:
        raise ValueError("Recovery requires at least two failing evaluations")
    thresholds = {group: _number(value, group + " tolerance", maximum=1.) for group, value in zip(
        (group for group, _, _ in GROUPS),
        (max_cat_drop, max_ordinary_clutter_drop, max_hand_protection_drop))}
    anchor = dict(checkpoint=str(checkpoint), step=source_step, modes=summaries)
    return dict(schema=SCHEMA, benchmark=benchmark, baseline=deepcopy(anchor), anchor=anchor,
                thresholds=thresholds, confirmations=confirmations, consecutive_failures=0,
                recovery_count=0, last_step=-1, last_report=None,
                source_selection_sha256=hashlib.sha256(json.dumps(source_selection, sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode()).hexdigest())


def _compare(summaries, state):
    reasons, comparisons = [], {}
    for mode in MODES:
        comparisons[mode] = {}
        if summaries[mode]["numerical_failure_rate"]:
            reasons.append(f"{mode}: numerical failure")
        for group, _, _ in GROUPS:
            key = group + "_goal_success_rate"
            current = summaries[mode][key]
            # The immutable floor prevents a succession of slightly weaker
            # selected checkpoints from ratcheting away original performance.
            reference = max(state["baseline"]["modes"][mode][key], state["anchor"]["modes"][mode][key])
            comparisons[mode][key] = dict(current=current, reference=reference,
                absolute_drop=reference - current, allowed_drop=state["thresholds"][group])
            if current + state["thresholds"][group] + 1e-12 < reference:
                reasons.append(f"{mode}: {key} below protected tolerance")
    return reasons, comparisons


def assess_recovery(result, state, *, extra_reasons=(), nonfinite=False, step=None):
    """Return (new JSON state, report); never restore a model inside this helper.

    Valid benchmark evidence with poor rates needs two successive failures.
    Numerical failures/nonfinite policy values request immediate recovery.
    Malformed or changed benchmark evidence raises ValueError: it must not be
    mistaken for a comparable evaluation. ``extra_reasons`` can carry the
    launcher's complementary per-original-scene gate failures.

    For nonfinite parameters caught before validation, pass result=None,
    nonfinite=True, step=current_step. Repeated step IDs cannot count twice.
    """
    updated = _state_copy(state)
    if hasattr(result, "as_dict"):
        result = result.as_dict()
    summaries, comparisons, reasons = {}, {}, list(extra_reasons)
    if result is None:
        if not nonfinite or step is None:
            raise ValueError("Missing validation is only allowed for explicit nonfinite failure with step")
    else:
        benchmark = _benchmark(_required(result, "metadata", "candidate"))
        if benchmark != state["benchmark"]:
            raise ValueError("Recovery benchmark differs from protected source")
        candidate_step = _required(result, "step", "candidate")
        if step is not None and step != candidate_step:
            raise ValueError("Recovery candidate step differs from supplied step")
        step = candidate_step
        modes = _required(result, "modes", "candidate")
        nonfinite = bool(nonfinite or _nonfinite(modes))
        if not nonfinite:
            summaries = _summaries(modes, benchmark, per_scene=True)
            found, comparisons = _compare(summaries, state)
            reasons.extend(found)
            nonfinite = any(summary["numerical_failure_rate"] for summary in summaries.values())
    step = _number(step, "candidate step", integer=True)
    if step <= updated["last_step"]:
        raise ValueError("Recovery evaluations require strictly increasing new-run steps")
    if nonfinite:
        reasons.append("nonfinite parameters/validation or numerical episode failure")
    reasons = list(dict.fromkeys(reasons))
    updated["consecutive_failures"] = updated["consecutive_failures"] + 1 if reasons else 0
    rollback = bool(nonfinite or updated["consecutive_failures"] >= updated["confirmations"])
    report = dict(eligible=not reasons, rollback=rollback,
                  status="rollback_required" if rollback else ("confirmation_pending" if reasons else "healthy"),
                  reasons=reasons, step=step, consecutive_failures=updated["consecutive_failures"],
                  checkpoint=updated["anchor"]["checkpoint"], comparisons=comparisons,
                  modes=summaries, numerical_failure=bool(nonfinite))
    updated.update(last_step=step, last_report=report)
    return updated, deepcopy(report)


def promote_recovery_anchor(state, result, checkpoint, *, selected_best, gates_eligible):
    """Promote only after the launcher has successfully published a better best.

    The caller's existing lexicographic selection decides improvement and its
    per-original-scene gates remain mandatory. This additionally rechecks all
    three protected populations against the source and current anchor.
    """
    if not selected_best or not gates_eligible:
        raise ValueError("Recovery anchor promotion requires a selected best passing all gates")
    if hasattr(result, "as_dict"):
        result = result.as_dict()
    updated = _state_copy(state)
    if _benchmark(result["metadata"]) != state["benchmark"] or _nonfinite(result["modes"]):
        raise ValueError("Recovery anchor has incompatible or nonfinite evidence")
    summaries = _summaries(result["modes"], state["benchmark"], per_scene=True)
    reasons, _ = _compare(summaries, state)
    if reasons:
        raise ValueError("Recovery anchor fails protected success gates: " + "; ".join(reasons))
    step = _number(result["step"], "anchor step", integer=True)
    if step != updated["last_step"] or not (updated["last_report"] or {}).get("eligible"):
        raise ValueError("Recovery anchor must be the latest passing assessed evaluation")
    updated["anchor"] = dict(checkpoint=str(checkpoint), step=step, modes=summaries)
    return updated


def acknowledge_recovery(state):
    """Clear breach streak only after the learner restored the requested anchor."""
    updated = _state_copy(state)
    if not (updated["last_report"] or {}).get("rollback"):
        raise ValueError("No regression recovery is pending")
    updated["consecutive_failures"] = 0
    updated["recovery_count"] += 1
    updated["last_report"]["status"] = "recovered"
    updated["last_report"]["rollback"] = False
    return updated
