"""Config checks for ending room episodes after a short stand at the goal, and for route recovery."""
import pytest


def test_goal_hold_and_route_recovery_flags():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config(goal_hold_seconds=1.0)
    assert c['goal_hold_seconds'] == 1.0 and 'route_recovery' not in c
    assert 'goal_hold_seconds' not in wholebody_config(c, goal_hold_seconds=0)
    c = wholebody_config(route_recovery_speed=.4)
    assert c['route_recovery'] == dict(speed=.4, lost_seconds=5.)
    assert 'route_recovery' not in wholebody_config(c, route_recovery_speed=0)
    for bad in (dict(goal_hold_seconds=-1.), dict(route_recovery_speed=-.1), dict(route_recovery_speed=.4, route_lost_seconds=0),
                dict(route_lost_seconds=3.)):
        with pytest.raises(ValueError):
            wholebody_config(**bad)


def test_defaults_leave_existing_runs_unchanged():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config()
    assert 'goal_hold_seconds' not in c and 'route_recovery' not in c
