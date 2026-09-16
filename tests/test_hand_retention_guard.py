"""A new arm profile cannot redefine source-best regression performance."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cat_ppo.furniture.hand_retention_guard import source_best_retention_guard
from cat_ppo.furniture.retention_validation import FIXED_SCENE_IDS, MODES


def _rates(total, count):
    base, remainder = divmod(total, count)
    return [(base + int(index < remainder)) / 16 for index in range(count)]


def _set_outcomes(result, mode, *, cat=72, ordinary=50, hand=0):
    rows = result["metadata"]["scenes"]
    rates = _rates(cat, 12) + _rates(ordinary, 4) + _rates(hand, 6)
    result["modes"][mode] = dict(episode_count=352, cat_episode_count=192,
        clutter_episode_count=160, ordinary_clutter_episode_count=64,
        cat_goal_success_rate=cat / 192, ordinary_clutter_goal_success_rate=ordinary / 64,
        clutter_goal_success_rate=(ordinary + hand) / 160, numerical_failure_rate=0.,
        scenes={row["scene_id"]: dict(episode_count=16, family=row["family"], goal_success_rate=rate)
                for row, rate in zip(rows, rates)})


@pytest.fixture
def evidence():
    families = (["original_cat"] * 2 + ["published_cat"] * 6 + ["procedural_cat"] * 4
                + ["furniture"] * 2 + ["generic_clutter"] * 2)
    scenes = [dict(scene_id=identity, family=family, horizon_steps=1000 if index < 12 else 4000)
              for index, (identity, family) in enumerate(zip(FIXED_SCENE_IDS, families))]
    source_metadata = dict(schema="cat-fixed-retention-validation-v1", scene_ids=list(FIXED_SCENE_IDS),
                           scenes=scenes, seeds=list(range(16)), modes=list(MODES), episodes_per_mode=256,
                           stopping="first clean whole-body goal, native failure, or native timeout")
    archived = {}
    for mode, cat, ordinary in (("deterministic", 72, 50), ("stochastic", 63, 45)):
        prefix = "validation/" if mode == "deterministic" else "validation/stochastic/"
        archived.update({prefix + key: value for key, value in dict(episode_count=256,
            cat_episode_count=192, clutter_episode_count=64, cat_goal_success_rate=cat / 192,
            clutter_goal_success_rate=ordinary / 64, numerical_failure_rate=0.).items()})
    selection = dict(selection_source="retention_validation", step=104857600,
                     metrics=dict(selection=dict(eligible=True), validation=archived),
                     provenance=dict(validation=source_metadata), files={"weights": {"sha256": "a" * 64, "size": 1}})
    candidate_metadata = deepcopy(source_metadata)
    for family in ("furniture", "generic_clutter"):
        for level in range(3):
            identity = f"hand-{family}-{level}"
            candidate_metadata["scene_ids"].append(identity)
            candidate_metadata["scenes"].append(dict(scene_id=identity, family=family, horizon_steps=4000))
    candidate_metadata["episodes_per_mode"] = 352
    result = dict(step=0, metadata=candidate_metadata, modes={})
    _set_outcomes(result, "deterministic", cat=72, ordinary=50)
    _set_outcomes(result, "stochastic", cat=63, ordinary=45)
    return result, selection


def test_unchanged_source_performance_passes_with_explicit_provenance(evidence):
    result, source = evidence
    report = source_best_retention_guard(result, source, source_archive="/archive/best")
    assert report["eligible"] and report["reasons"] == []
    assert report["source_step"] == 104857600 and report["candidate_step"] == 0
    assert report["comparisons"]["deterministic"]["source_cat_goal_success_rate"] == .375
    assert report["comparisons"]["stochastic"]["source_ordinary_clutter_goal_success_rate"] == .703125
    assert report["provenance"]["source_archive"] == "/archive/best"
    assert len(report["provenance"]["source_selection_sha256"]) == 64
    assert not report["provenance"]["archived_source_has_per_scene_rates"]
    wrapped = SimpleNamespace(as_dict=lambda: result)
    assert source_best_retention_guard(wrapped, source)["eligible"]


@pytest.mark.parametrize("mode,baseline_cat,baseline_clutter", [("deterministic", 72, 50), ("stochastic", 63, 45)])
def test_cat_drop_is_absolute_in_each_mode_and_cannot_be_hidden_by_hand_gains(evidence, mode, baseline_cat, baseline_clutter):
    result, source = evidence
    _set_outcomes(result, mode, cat=baseline_cat - 9, ordinary=baseline_clutter, hand=96)
    assert source_best_retention_guard(result, source)["eligible"]  # 9/192 < .05
    _set_outcomes(result, mode, cat=baseline_cat - 10, ordinary=baseline_clutter, hand=96)
    report = source_best_retention_guard(result, source)
    assert not report["eligible"]
    assert report["reasons"] == [f"{mode}: CAT success below protected source-best tolerance"]


def test_original_clutter_cannot_be_masked_by_six_new_hand_layouts(evidence):
    result, source = evidence
    _set_outcomes(result, "deterministic", ordinary=42, hand=96)
    assert source_best_retention_guard(result, source)["eligible"]  # Exact .125 drop allowed.
    _set_outcomes(result, "deterministic", ordinary=41, hand=96)
    report = source_best_retention_guard(result, source)
    assert not report["eligible"] and "ordinary-clutter" in report["reasons"][0]
    # Even a later checkpoint is always measured against the source best,
    # independently of whether its new-run baseline happens to be weaker.
    result["step"] = 50000000
    assert not source_best_retention_guard(result, source)["eligible"]


@pytest.mark.parametrize("change,message", [
    ("missing_rate", "Missing"), ("nan", "finite"), ("other_nan", "finite"),
    ("source_seeds", "seeds"), ("new_seeds", "seeds"), ("reorder", "prefix"),
    ("family", "families"), ("horizon", "population"), ("count", "population"),
    ("source_count", "population"), ("aggregate", "aggregate"), ("scene_count", "population"),
    ("stopping", "stopping"), ("ineligible", "eligible"), ("missing_mode", "Missing"),
    ("added_cat", "families"), ("source_numeric", "numerical"),
])
def test_invalid_or_changed_benchmark_evidence_fails_closed(evidence, change, message):
    result, source = evidence
    archived = source["metrics"]["validation"]
    current = result["modes"]["deterministic"]
    if change == "missing_rate":
        del archived["validation/cat_goal_success_rate"]
    elif change == "nan":
        current["cat_goal_success_rate"] = float("nan")
    elif change == "other_nan":
        current["hand_mean_return"] = float("inf")
    elif change == "source_seeds":
        source["provenance"]["validation"]["seeds"][0] = 16
    elif change == "new_seeds":
        result["metadata"]["seeds"] = list(reversed(range(16)))
    elif change == "reorder":
        result["metadata"]["scene_ids"][:2] = reversed(result["metadata"]["scene_ids"][:2])
    elif change == "family":
        result["metadata"]["scenes"][0]["family"] = "published_cat"
    elif change == "horizon":
        result["metadata"]["scenes"][0]["horizon_steps"] = 4000
    elif change == "count":
        current["cat_episode_count"] = 191
    elif change == "source_count":
        archived["validation/clutter_episode_count"] = 63
    elif change == "aggregate":
        current["ordinary_clutter_goal_success_rate"] = 1.
    elif change == "scene_count":
        current["scenes"][FIXED_SCENE_IDS[0]]["episode_count"] = 15
    elif change == "stopping":
        result["metadata"]["stopping"] = "continue after goals"
    elif change == "ineligible":
        source["metrics"]["selection"]["eligible"] = False
    elif change == "missing_mode":
        del result["modes"]["stochastic"]
    elif change == "added_cat":
        result["metadata"]["scenes"][-1]["family"] = "procedural_cat"
    elif change == "source_numeric":
        archived["validation/numerical_failure_rate"] = 1 / 256
    with pytest.raises(ValueError, match=message):
        source_best_retention_guard(result, source)


def test_numerical_failure_rejects_candidate_without_changing_metric_evidence(evidence):
    result, source = evidence
    result["modes"]["stochastic"]["numerical_failure_rate"] = 1 / 352
    report = source_best_retention_guard(result, source)
    assert not report["eligible"] and report["reasons"] == ["stochastic: numerical failure"]


@pytest.mark.parametrize("settings", [{"max_cat_drop": float("nan")}, {"max_cat_drop": -.01},
                                      {"max_ordinary_clutter_drop": 1.01}])
def test_invalid_tolerances_are_rejected(evidence, settings):
    result, source = evidence
    with pytest.raises(ValueError, match="finite"):
        source_best_retention_guard(result, source, **settings)
