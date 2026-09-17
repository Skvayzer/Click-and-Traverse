"""Recovery decisions must protect every population in both action modes."""
from copy import deepcopy
import json

import pytest

from cat_ppo.furniture.recovery_guard import (
    acknowledge_recovery, assess_recovery, initialize_recovery_state,
    promote_recovery_anchor,
)
from cat_ppo.furniture.retention_validation import FIXED_SCENE_IDS, MODES


def _set_outcomes(result, mode, *, cat=76, ordinary=51, hand=59):
    rates = []
    for successes, count in ((cat, 12), (ordinary, 4), (hand, 6)):
        base, remaining = divmod(successes, count)
        rates.extend((base + int(index < remaining)) / 16 for index in range(count))
    result["modes"][mode] = dict(episode_count=352, cat_episode_count=192,
        clutter_episode_count=160, ordinary_clutter_episode_count=64,
        hand_protection_episode_count=96, cat_goal_success_rate=cat / 192,
        ordinary_clutter_goal_success_rate=ordinary / 64,
        hand_protection_goal_success_rate=hand / 96,
        clutter_goal_success_rate=(ordinary + hand) / 160, numerical_failure_rate=0.,
        scenes={row["scene_id"]: dict(episode_count=16, family=row["family"], goal_success_rate=rate)
                for row, rate in zip(result["metadata"]["scenes"], rates)})


@pytest.fixture
def evidence():
    families = (["original_cat"] * 2 + ["published_cat"] * 6 + ["procedural_cat"] * 4
                + ["furniture"] * 2 + ["generic_clutter"] * 2
                + ["furniture"] * 3 + ["generic_clutter"] * 3)
    ids = [*FIXED_SCENE_IDS, *[f"hand-{index}" for index in range(6)]]
    metadata = dict(schema="cat-fixed-retention-validation-v1", scene_ids=ids,
        scenes=[dict(scene_id=identity, family=family, horizon_steps=1000 if index < 12 else 4000)
                for index, (identity, family) in enumerate(zip(ids, families))],
        seeds=list(range(16)), modes=list(MODES), episodes_per_mode=352,
        stopping="first clean whole-body goal, native failure, or native timeout")
    result = dict(step=0, metadata=metadata, modes={})
    archived = {}
    for mode in MODES:
        _set_outcomes(result, mode)
        prefix = "validation/" if mode == "deterministic" else "validation/stochastic/"
        archived.update({prefix + key: value for key, value in result["modes"][mode].items() if key != "scenes"})
    source = dict(step=26214400, selection_source="retention_validation",
                  metrics=dict(selection=dict(eligible=True), validation=archived),
                  provenance=dict(validation=deepcopy(metadata)))
    return result, source


def test_unchanged_source_is_healthy_and_state_roundtrips_json(evidence):
    result, source = evidence
    original = initialize_recovery_state(source, "/archive/protected-26m")
    assert original["last_step"] == -1 and original["anchor"]["step"] == 26214400
    state, report = assess_recovery(result, original)
    assert report["eligible"] and report["status"] == "healthy"
    assert not report["rollback"] and state["consecutive_failures"] == 0
    assert original["last_step"] == -1  # Functional API does not mutate persisted caller state.
    assert json.loads(json.dumps(state, allow_nan=False)) == state


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("group,inside,outside", [("cat", 67, 66), ("ordinary", 45, 44), ("hand", 50, 49)])
def test_each_group_tolerance_requires_two_consecutive_failures(evidence, mode, group, inside, outside):
    result, source = evidence
    state = initialize_recovery_state(source, "protected")
    _set_outcomes(result, mode, **{group: inside})
    state, report = assess_recovery(result, state)
    assert report["eligible"]
    result["step"] = 10
    _set_outcomes(result, mode, **{group: outside})
    state, report = assess_recovery(result, state)
    assert not report["eligible"] and not report["rollback"]
    assert report["status"] == "confirmation_pending"
    result["step"] = 20
    state, report = assess_recovery(result, state)
    assert report["rollback"] and report["checkpoint"] == "protected"
    assert report["consecutive_failures"] == 2
    recovered = acknowledge_recovery(state)
    assert recovered["consecutive_failures"] == 0 and recovered["recovery_count"] == 1
    assert recovered["last_report"]["status"] == "recovered"
    result["step"] = 30
    _, report = assess_recovery(result, recovered)
    assert not report["rollback"] and report["consecutive_failures"] == 1


def test_passing_intervening_evaluation_clears_streak_and_duplicate_steps_cannot_count(evidence):
    result, source = evidence
    state = initialize_recovery_state(source, "protected")
    _set_outcomes(result, "stochastic", hand=0)
    state, _ = assess_recovery(result, state)
    with pytest.raises(ValueError, match="increasing"):
        assess_recovery(result, state)
    result["step"] = 10
    _set_outcomes(result, "stochastic")
    state, report = assess_recovery(result, state)
    assert report["consecutive_failures"] == 0
    result["step"] = 20
    _set_outcomes(result, "stochastic", hand=0)
    _, report = assess_recovery(result, state)
    assert not report["rollback"]


@pytest.mark.parametrize("failure", ["rate_nan", "diagnostic_inf", "numerical_episode", "parameters"])
def test_nonfinite_or_numerical_failure_recovers_immediately(evidence, failure):
    result, source = evidence
    state = initialize_recovery_state(source, "protected")
    kwargs = {}
    if failure == "rate_nan":
        result["modes"]["deterministic"]["hand_protection_goal_success_rate"] = float("nan")
    elif failure == "diagnostic_inf":
        result["modes"]["stochastic"]["mean_return"] = float("inf")
    elif failure == "numerical_episode":
        result["modes"]["stochastic"]["numerical_failure_rate"] = 1 / 352
    else:
        result = None
        kwargs = dict(nonfinite=True, step=1)
    state, report = assess_recovery(result, state, **kwargs)
    assert report["rollback"] and report["numerical_failure"]
    assert report["consecutive_failures"] == 1
    json.dumps(state, allow_nan=False)


def test_per_original_scene_gate_can_trigger_recovery_without_new_metric_keys(evidence):
    result, source = evidence
    state = initialize_recovery_state(source, "protected")
    reasons = ["deterministic: published-hurdle1 success below baseline tolerance"]
    state, report = assess_recovery(result, state, extra_reasons=reasons)
    assert report["reasons"] == reasons and not report["rollback"]
    result["step"] = 10
    _, report = assess_recovery(result, state, extra_reasons=reasons)
    assert report["rollback"]


@pytest.mark.parametrize("change", ["seeds", "horizon", "identity", "count", "scene_count", "aggregate", "missing_mode", "stopping"])
def test_changed_or_malformed_benchmark_fails_closed(evidence, change):
    result, source = evidence
    state = initialize_recovery_state(source, "protected")
    if change == "seeds":
        result["metadata"]["seeds"][0] = 16
    elif change == "horizon":
        result["metadata"]["scenes"][0]["horizon_steps"] = 4000
    elif change == "identity":
        result["metadata"]["scene_ids"][-1] = "different-hand-scene"
        result["metadata"]["scenes"][-1]["scene_id"] = "different-hand-scene"
    elif change == "count":
        result["modes"]["stochastic"]["hand_protection_episode_count"] = 95
    elif change == "scene_count":
        result["modes"]["stochastic"]["scenes"][FIXED_SCENE_IDS[0]]["episode_count"] = 15
    elif change == "aggregate":
        result["modes"]["stochastic"]["ordinary_clutter_goal_success_rate"] = 1.
    elif change == "missing_mode":
        del result["modes"]["stochastic"]
    elif change == "stopping":
        result["metadata"]["stopping"] = "continue after first goal"
    with pytest.raises(ValueError):
        assess_recovery(result, state)


@pytest.mark.parametrize("change", ["old_16_scene", "ineligible", "nan", "numerical", "missing_hand_count"])
def test_invalid_protected_source_cannot_redefine_baseline(evidence, change):
    _, source = evidence
    if change == "old_16_scene":
        source["provenance"]["validation"]["scene_ids"] = list(FIXED_SCENE_IDS)
        source["provenance"]["validation"]["scenes"] = source["provenance"]["validation"]["scenes"][:16]
    elif change == "ineligible":
        source["metrics"]["selection"]["eligible"] = False
    elif change == "nan":
        source["metrics"]["validation"]["validation/hand_protection_goal_success_rate"] = float("nan")
    elif change == "numerical":
        source["metrics"]["validation"]["validation/numerical_failure_rate"] = 1 / 352
    elif change == "missing_hand_count":
        del source["metrics"]["validation"]["validation/hand_protection_episode_count"]
    with pytest.raises(ValueError):
        initialize_recovery_state(source, "protected")


def test_only_published_passing_best_promotes_anchor_and_initial_floor_persists(evidence):
    result, source = evidence
    state = initialize_recovery_state(source, "protected")
    # A rank-improving hand result may still be slightly weaker on CAT.
    for mode in MODES:
        _set_outcomes(result, mode, cat=68, hand=70)
    state, _ = assess_recovery(result, state)
    for selected, eligible in ((False, True), (True, False)):
        with pytest.raises(ValueError, match="selected best"):
            promote_recovery_anchor(state, result, "new-best", selected_best=selected, gates_eligible=eligible)
    promoted = promote_recovery_anchor(state, result, "new-best", selected_best=True, gates_eligible=True)
    assert promoted["anchor"]["checkpoint"] == "new-best"
    assert promoted["baseline"]["checkpoint"] == "protected"
    result["step"] = 10
    for mode in MODES:
        _set_outcomes(result, mode, cat=60, hand=70)
    _, report = assess_recovery(result, promoted)
    assert not report["eligible"]  # Relative to initial76, not merely current68.
    assert report["comparisons"]["deterministic"]["cat_goal_success_rate"]["reference"] == 76 / 192
    with pytest.raises(ValueError, match="protected success gates"):
        promote_recovery_anchor(promoted, result, "bad-best", selected_best=True, gates_eligible=True)


def test_anchor_cannot_skip_assessment_or_use_stale_result(evidence):
    result, source = evidence
    state = initialize_recovery_state(source, "protected")
    with pytest.raises(ValueError, match="latest passing"):
        promote_recovery_anchor(state, result, "new-best", selected_best=True, gates_eligible=True)
    with pytest.raises(ValueError, match="No regression"):
        acknowledge_recovery(state)
    with pytest.raises(ValueError, match="Missing validation"):
        assess_recovery(None, state)
