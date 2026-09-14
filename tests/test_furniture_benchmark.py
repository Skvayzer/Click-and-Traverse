import json

import pytest

from cat_ppo.furniture.benchmark import EpisodeTracker, paired_comparison, route_projection, summarize


def scene():
    return {"scene_id": "test-room", "start": [0, 0, 0], "goal": [0, 2],
            "route": [[0, 0], [2, 0], [2, 2], [0, 2]], "time_budget": 20,
            "bottlenecks": [{"center": [2, 0]}, {"center": [2, 2]}]}


def test_collision_at_goal_is_arrival_but_never_strict_success():
    tracker = EpisodeTracker(scene())
    tracker.update(position=[0, 2], elapsed=10, furniture_contact=True,
                   hand_contact=True, contact_part="left_finger", minimum_clearance=-.01)
    result = tracker.finish()
    assert result.reached_goal and not result.strict_success
    assert result.termination_reason == "furniture_contact"
    assert result.first_contact_part == "left_finger"
    assert result.hand_contact
    assert result.progress_before_first_contact == 0
    json.dumps(result.to_dict(), allow_nan=False)


def test_contacts_are_latched_and_high_x_is_not_goal():
    tracker = EpisodeTracker(scene())
    tracker.update(position=[2, 0], elapsed=3, furniture_contact=True)
    assert not tracker.finish().reached_goal
    tracker.update(position=[0, 2], elapsed=10)
    assert not tracker.finish().strict_success
    assert tracker.finish().first_contact_time == 3


def test_winding_route_uses_arclength_and_sequential_gates():
    tracker = EpisodeTracker(scene())
    tracker.update(position=[2, 2], elapsed=5)
    assert tracker.finish().bottlenecks_cleared == 0  # First gate was skipped.
    assert route_projection([2, 1], scene()["route"])[0] == pytest.approx(3)
    tracker.update(position=[2, 0], elapsed=6)
    tracker.update(position=[2, 2], elapsed=9)
    tracker.update(position=[0, 2], elapsed=12)
    result = tracker.finish()
    assert result.strict_success and result.bottlenecks_cleared == 2


def test_failure_time_is_capped_and_empty_success_median_is_null():
    tracker = EpisodeTracker(scene())
    tracker.update(position=[1, 0], elapsed=2, fall=True)
    result = summarize([tracker.finish()])
    assert result["mean_capped_completion_time"] == 20
    assert result["median_strict_success_time"] is None


def test_budget_expiration_cannot_be_late_success():
    tracker = EpisodeTracker(scene())
    tracker.update(position=[0, 2], elapsed=21)
    assert tracker.finish().timeout and not tracker.finish().reached_goal


def test_hand_self_contact_is_not_mislabeled_furniture():
    tracker = EpisodeTracker(scene())
    tracker.update(position=[0, 2], elapsed=10, hand_contact=True, self_collision=True)
    result = tracker.finish()
    assert not result.strict_success and not result.furniture_contact


def test_explicit_environment_goal_condition_is_respected():
    tracker = EpisodeTracker(scene())
    tracker.update(position=[0, 2], elapsed=10, goal_reached=False)
    assert not tracker.finish().reached_goal


def test_environment_goal_flag_cannot_override_geometric_goal_check():
    tracker = EpisodeTracker(scene())
    tracker.update(position=scene()["start"][:2], elapsed=1, goal_reached=True)
    assert not tracker.finish().reached_goal
    assert not tracker.finish().strict_success
    tracker.update(position=scene()["goal"], elapsed=2, goal_reached=True)
    assert tracker.finish().strict_success


def test_numerical_failure_is_separate_from_physical_fall_and_latched():
    tracker = EpisodeTracker(scene())
    tracker.update(position=scene()["goal"], elapsed=1, numerical_failure=True, goal_reached=True)
    tracker.update(position=scene()["goal"], elapsed=2)
    record = tracker.finish()
    assert record.numerical_failure and not record.fall and not record.strict_success
    assert record.termination_reason == "numerical_failure"
    summary = summarize([record])
    assert summary["numerical_failure_rate"] == 1
    assert summary["fall_rate"] == 0
    tracker.update(position=scene()["goal"], elapsed=3, fall=True)
    assert tracker.finish().fall and tracker.finish().numerical_failure


def test_precise_first_contact_timestamp_is_preserved_and_validated():
    tracker = EpisodeTracker(scene())
    tracker.update(position=[1, 0], elapsed=.02)
    tracker.update(position=[1.1, 0], elapsed=.04, hand_contact=True,
                   first_contact_time=.025, contact_part="right_finger")
    tracker.update(position=[1.2, 0], elapsed=.06, hand_contact=True, first_contact_time=.025)
    record = tracker.finish()
    assert record.first_contact_time == .025
    assert record.first_contact_part == "right_finger"
    assert record.progress_before_first_contact == pytest.approx(1 / 6)
    for bad_time in (-1, float("nan"), .08):
        with pytest.raises(ValueError, match="first_contact_time"):
            tracker.update(position=[1.2, 0], elapsed=.06, first_contact_time=bad_time)


def test_historical_episode_summary_defaults_absent_numerical_flag_to_false():
    old = EpisodeTracker(scene()).finish().to_dict()
    old.pop("numerical_failure")
    assert summarize([old])["numerical_failure_rate"] == 0


def test_paired_comparison_rejects_mismatches_and_uses_shared_cases():
    base = EpisodeTracker(scene()).finish().to_dict()
    improved = dict(base, strict_success=True, elapsed=9)
    result = paired_comparison([base], [improved])
    assert result["strict_success_rate_difference"] == 1
    assert result["scene_cluster_bootstrap_95_percent_intervals"] is None
    with pytest.raises(ValueError, match="identical"):
        paired_comparison([base], [dict(improved, episode_seed=1)])
    with pytest.raises(ValueError, match="duplicate"):
        paired_comparison([base, base], [improved, improved])
