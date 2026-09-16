"""Absolute source-best retention gates for a changed hand-training profile.

The new profile's step-zero validation must not silently redefine already
learned CAT or ordinary-clutter performance. This pure helper checks the same
archived populations in both action modes, at startup and at later candidate
checkpoints. The bank/reset append builders separately prove that the old
scene geometry and reset poses are unchanged; this checks benchmark identity,
ordering, seeds, horizons, denominators and finite, consistent success rates.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers

from cat_ppo.furniture.retention_validation import CLUTTER_FAMILIES, FIXED_SCENE_IDS, MODES


def _required(mapping, key, label):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"Missing source-best guard metadata: {label}.{key}")
    return mapping[key]


def _number(value, label, *, minimum=0., maximum=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"Source-best guard requires a numeric {label}")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"Source-best guard requires finite in-range {label}")
    return result


def _count(mapping, key, expected, label):
    value = _number(_required(mapping, key, label), f"{label}.{key}")
    if value != expected:
        raise ValueError(f"Source-best guard population differs: {label}.{key}={value}, expected {expected}")


def _rate(mapping, key, count, label):
    value = _number(_required(mapping, key, label), f"{label}.{key}", maximum=1.)
    if not math.isclose(value * count, round(value * count), rel_tol=0., abs_tol=1e-8):
        raise ValueError(f"Success rate disagrees with integer episode population: {label}.{key}")
    return value


def _finite_numbers(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_numbers(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_numbers(child, f"{label}[{index}]")
    elif isinstance(value, numbers.Real) and not math.isfinite(float(value)):
        raise ValueError(f"Source-best guard requires finite {label}")


def _metadata(value, *, source):
    label = "source validation" if source else "candidate validation"
    if _required(value, "schema", label) != "cat-fixed-retention-validation-v1":
        raise ValueError("Source-best guard requires the fixed retention validation schema")
    ids = _required(value, "scene_ids", label)
    rows = _required(value, "scenes", label)
    if (not isinstance(ids, (list, tuple)) or len(set(ids)) != len(ids)
            or tuple(ids[:16]) != FIXED_SCENE_IDS or len(ids) < 16
            or (source and len(ids) != 16)
            or not isinstance(rows, (list, tuple)) or len(rows) != len(ids)):
        raise ValueError("Source-best guard requires the unchanged fixed 16-scene prefix")
    if (_required(value, "seeds", label) != list(range(16))
            or tuple(_required(value, "modes", label)) != MODES):
        raise ValueError("Source-best guard requires unchanged seeds 0..15 and both action modes")
    _count(value, "episodes_per_mode", len(ids) * 16, label)
    expected_families = (["original_cat"] * 2 + ["published_cat"] * 6
                         + ["procedural_cat"] * 4 + ["furniture"] * 2 + ["generic_clutter"] * 2)
    for index, (identity, row) in enumerate(zip(ids, rows)):
        family = _required(row, "family", label)
        if row.get("scene_id") != identity:
            raise ValueError("Source-best guard scene rows and IDs differ in order")
        if ((index < 16 and family != expected_families[index])
                or (index >= 16 and family not in CLUTTER_FAMILIES)):
            raise ValueError("Source-best guard requires unchanged CAT/clutter families")
        expected_horizon = 1000 if index < 12 else 4000
        _count(row, "horizon_steps", expected_horizon, f"{label}.{identity}")
    return list(ids), list(rows)


def source_best_retention_guard(result, source_selection, *, max_cat_drop=.05,
                                max_ordinary_clutter_drop=.125, source_archive=None):
    """Return eligibility against the protected v1 source best, without I/O.

    ``result`` is a ValidationResult or its ``as_dict()`` representation;
    ``source_selection`` is the archived checkpoint/selection.json dictionary.
    Malformed/mismatched evidence raises ValueError. Valid evidence with a
    performance regression returns eligible=False and explicit reasons.

    CAT compares the same 12x16 episodes (192). Ordinary clutter compares only
    the original 4x16 (64), so improvements on new hand tasks cannot mask a loss.
    The existing new-run per-scene retention gates remain complementary.
    """
    max_cat_drop = _number(max_cat_drop, "max_cat_drop", maximum=1.)
    max_ordinary_clutter_drop = _number(max_ordinary_clutter_drop, "max_ordinary_clutter_drop", maximum=1.)
    if hasattr(result, "as_dict"):
        result = result.as_dict()
    source_metrics = _required(source_selection, "metrics", "source selection")
    selected = _required(source_metrics, "selection", "source metrics")
    if (source_selection.get("selection_source") != "retention_validation"
            or selected.get("eligible") is not True):
        raise ValueError("Source-best guard requires an eligible retention-selected source checkpoint")
    source_step = _number(_required(source_selection, "step", "source selection"), "source step")
    candidate_step = _number(_required(result, "step", "candidate result"), "candidate step")
    if source_step <= 0 or source_step != int(source_step) or candidate_step != int(candidate_step):
        raise ValueError("Source-best guard requires integer source/candidate steps")
    source_meta = _required(_required(source_selection, "provenance", "source selection"),
                            "validation", "source provenance")
    candidate_meta = _required(result, "metadata", "candidate result")
    source_ids, source_rows = _metadata(source_meta, source=True)
    candidate_ids, candidate_rows = _metadata(candidate_meta, source=False)
    if candidate_rows[:16] != source_rows:
        raise ValueError("Source-best guard source benchmark metadata differs from candidate prefix")
    if (_required(source_meta, "stopping", "source validation")
            != _required(candidate_meta, "stopping", "candidate validation")):
        raise ValueError("Source-best guard validation stopping semantics differ")
    if len(candidate_ids) <= 16:
        raise ValueError("Hand-profile guard requires appended hand validation layouts")
    archived = _required(source_metrics, "validation", "source metrics")
    modes = _required(result, "modes", "candidate result")
    _finite_numbers(archived, "source metrics")
    _finite_numbers(modes, "candidate modes")
    reasons, comparisons = [], {}
    for mode in MODES:
        prefix = "validation/" if mode == "deterministic" else "validation/stochastic/"
        _count(archived, prefix + "episode_count", 256, "source metrics")
        _count(archived, prefix + "cat_episode_count", 192, "source metrics")
        _count(archived, prefix + "clutter_episode_count", 64, "source metrics")
        source_cat = _rate(archived, prefix + "cat_goal_success_rate", 192, "source metrics")
        source_clutter = _rate(archived, prefix + "clutter_goal_success_rate", 64, "source metrics")
        source_numerical = _number(_required(archived, prefix + "numerical_failure_rate", "source metrics"),
                                   "source numerical failure rate", maximum=1.)
        if source_numerical != 0.:
            raise ValueError("Protected source has numerical failures and cannot define retention")
        current = _required(modes, mode, "candidate modes")
        _count(current, "episode_count", len(candidate_ids) * 16, mode)
        _count(current, "cat_episode_count", 192, mode)
        _count(current, "clutter_episode_count", (len(candidate_ids) - 12) * 16, mode)
        _count(current, "ordinary_clutter_episode_count", 64, mode)
        cat = _rate(current, "cat_goal_success_rate", 192, mode)
        clutter = _rate(current, "ordinary_clutter_goal_success_rate", 64, mode)
        scene_results = _required(current, "scenes", mode)
        if set(scene_results) != set(candidate_ids):
            raise ValueError("Source-best guard candidate scene result population differs")
        rates = []
        for identity, metadata in zip(candidate_ids, candidate_rows):
            scene = scene_results[identity]
            _count(scene, "episode_count", 16, identity)
            if _required(scene, "family", identity) != metadata["family"]:
                raise ValueError("Source-best guard candidate result family differs from its metadata")
            rates.append(_rate(scene, "goal_success_rate", 16, identity))
        if (not math.isclose(cat, sum(rates[:12]) / 12, rel_tol=0., abs_tol=1e-12)
                or not math.isclose(clutter, sum(rates[12:16]) / 4, rel_tol=0., abs_tol=1e-12)):
            raise ValueError("Source-best guard aggregate success differs from unchanged per-scene populations")
        numerical = _number(_required(current, "numerical_failure_rate", mode),
                            "candidate numerical failure rate", maximum=1.)
        if numerical:
            reasons.append(f"{mode}: numerical failure")
        if cat + max_cat_drop + 1e-12 < source_cat:
            reasons.append(f"{mode}: CAT success below protected source-best tolerance")
        if clutter + max_ordinary_clutter_drop + 1e-12 < source_clutter:
            reasons.append(f"{mode}: ordinary-clutter success below protected source-best tolerance")
        comparisons[mode] = dict(
            source_cat_goal_success_rate=source_cat, cat_goal_success_rate=cat,
            cat_absolute_drop=source_cat - cat, cat_episode_count=192,
            source_ordinary_clutter_goal_success_rate=source_clutter,
            ordinary_clutter_goal_success_rate=clutter,
            ordinary_clutter_absolute_drop=source_clutter - clutter,
            ordinary_clutter_episode_count=64)
    # Fingerprint the complete selection metadata, binding rates to their exact
    # selected checkpoint file inventory as well as its step and provenance.
    digest = hashlib.sha256(json.dumps(source_selection, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return dict(schema="cat-source-best-retention-guard-v1", eligible=not reasons, reasons=reasons,
                source_step=int(source_step), candidate_step=int(candidate_step),
                max_cat_drop=max_cat_drop, max_ordinary_clutter_drop=max_ordinary_clutter_drop,
                comparisons=comparisons,
                provenance=dict(source_archive=None if source_archive is None else str(source_archive),
                    source_selection_sha256=digest, scene_ids=source_ids, seeds=list(range(16)),
                    archived_source_has_per_scene_rates=False,
                    source_per_scene_gate="unavailable; retain new-run per-scene CAT gates",
                    required_external_proof="append builders verify unchanged source fields, geometry and reset rows"))
