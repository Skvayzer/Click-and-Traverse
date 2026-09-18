import json

import pytest

from cat_ppo.furniture.generalist_logging import GeneralistLogger


class MustNotConvert:
    @property
    def ndim(self):
        raise AssertionError("Discarded scene diagnostics must not be inspected or converted")


def test_compact_logging_keeps_training_outcomes_and_discards_scene_forest(tmp_path):
    logger = GeneralistLogger(tmp_path, mode="disabled", compact_logging=True)
    metrics = {
        f"pf_sampling/scene_{i}/probability": MustNotConvert() for i in range(7100)
    }
    metrics.update({
        "training/goal_success_rate": .6, "training/resolved_count": 10,
        "training/goal_success_count": 6,
        "training/cat_goal_success_rate": .75, "training/cat_resolved_count": 4,
        "training/cat_goal_success_count": 3,
        "training/ordinary_clutter_goal_success_rate": .5,
        "training/ordinary_clutter_resolved_count": 4,
        "training/ordinary_clutter_goal_success_count": 2,
        "training/hand_protection_goal_success_rate": .5,
        "training/hand_protection_resolved_count": 2,
        "training/hand_protection_goal_success_count": 1,
        "training/policy_loss": -.03, "training/entropy_loss": -.2,
        "training/leg_std_mean": .13, "training/upper_std_mean": .2,
        "training/sps": 21000, "hand_curriculum/stage": 1,
        "episode/sum_reward": 20., "episode/reward/handsdf": -.2,
        "training/body_collision_hands_rate": .1,
        "training/fall_rate": .01, "selection/best_updated": 1,
        "rollout/pfid/4001/episodes": MustNotConvert(),
        "validation/cat_goal_success_rate": MustNotConvert(),
        "baseline/cat_goal_success_rate": MustNotConvert(),
    })
    logger.log(524288, metrics)
    event = json.loads((tmp_path / "metrics.jsonl").read_text())
    expected = {k: v for k, v in metrics.items() if not isinstance(v, MustNotConvert)}
    assert event == {**expected, "global_step": 524288, "health/nonfinite": 0}


def test_compact_empty_groups_stay_absent_instead_of_reporting_false_zero_success(tmp_path):
    logger = GeneralistLogger(tmp_path, mode="disabled", compact_logging=True)
    logger.log(1, {"training/resolved_count": 0, "training/goal_success_count": 0})
    event = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert event["training/resolved_count"] == 0
    assert "training/goal_success_rate" not in event


def test_compact_nonfinite_losses_are_reported_even_for_a_suppressed_old_step(tmp_path):
    logger = GeneralistLogger(tmp_path, mode="disabled", compact_logging=True)
    logger.log(100, {"training/total_loss": 1.})
    with pytest.raises(FloatingPointError, match="training/total_loss"):
        logger.log(80, {"training/total_loss": float("nan")})
    event = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[-1])
    assert event["health/nonfinite"] == 1
    assert event["global_step"] == 100 and event["source_step"] == 80
    assert event["nonfinite_keys"] == ["training/total_loss"]


def test_default_preserves_legacy_metrics_and_resume_monotonicity(tmp_path):
    logger = GeneralistLogger(tmp_path, mode="disabled")
    logger.log(10, {"pf_sampling/scene_4/probability": .2, "validation/example": .3})
    resumed = GeneralistLogger(tmp_path, mode="disabled", resume=True)
    resumed.log(9, {"validation/example": .4})
    resumed.log(11, {"validation/example": .5})
    events = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(events) == 2
    assert events[0]["pf_sampling/scene_4/probability"] == .2
    assert events[0]["validation/example"] == .3
    assert events[1]["global_step"] == 11


def test_compact_logging_requires_explicit_boolean(tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        GeneralistLogger(tmp_path, mode="disabled", compact_logging="true")
