"""Keyboard v10: blocking footprint, heading deadband, config wiring."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cat_mjlab.blocking import BlockingMap, blocking_sdf2d
from cat_mjlab import task_math as tm

DX = .04


def scene_sdf(boxes, shape=(100, 50, 40), origin=(0., 0., 0.)):
    """Exact SDF of axis-aligned boxes (min/max corners) on a grid with cell i at origin + i*dx."""
    axes = [origin[k] + DX * np.arange(shape[k]) for k in range(3)]
    p = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    sdf = np.full(shape, 10., np.float32)
    for lo, hi in boxes:
        lo, hi = np.asarray(lo), np.asarray(hi)
        c, h = (lo + hi) / 2, (hi - lo) / 2
        q = np.abs(p - c) - h
        d = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
        sdf = np.minimum(sdf, d.astype(np.float32))
    return sdf


def bank_of(sdfs, origin=(0., 0., 0.)):
    shapes = [s.shape for s in sdfs]
    offsets = np.cumsum([0] + [int(np.prod(s)) for s in shapes[:-1]])
    return SimpleNamespace(fields={"sdf": torch.as_tensor(np.concatenate([s.reshape(-1, 1) for s in sdfs]))},
                           shapes=torch.tensor(shapes), offsets=torch.as_tensor(offsets),
                           origins=torch.tensor([origin] * len(sdfs), dtype=torch.float32),
                           dxs=torch.full((len(sdfs),), DX))


BEAM = ((1.9, 0., 1.02), (2.1, 2., 1.6))        # crouch under it
HURDLE = ((1.9, 0., 0.), (2.0, 2., .20))       # step over it
WALL = ((1.9, 0., 0.), (2.0, 2., 1.6))         # blocks
TABLE = ((1.6, 0., .70), (2.4, .8, .75))       # top at hand height: blocks


@pytest.mark.parametrize("box,blocks", [(BEAM, False), (HURDLE, False), (WALL, True), (TABLE, True)])
def test_band_decides_what_blocks(box, blocks):
    dist, _ = blocking_sdf2d(scene_sdf([box]), 0., DX)
    assert (dist.min() <= 0) == blocks


def test_sample_distance_and_direction_away_from_wall():
    bm = BlockingMap(bank_of([scene_sdf([WALL])]))
    xy = torch.tensor([[[1.5, 1.0], [2.4, 1.0]]])
    dist, away = bm.sample(xy, torch.tensor([0]))
    assert dist[0, 0] == pytest.approx(.40, abs=.05) and dist[0, 1] == pytest.approx(.40, abs=.05)
    assert away[0, 0, 0] < -.9 and away[0, 1, 0] > .9            # in front: pushed back; behind: pushed on


def test_safe_target_stops_at_wall_not_under_beam():
    command = torch.tensor([[.5, 0.]])
    xy = torch.tensor([[[1.65, 1.0]]])                          # 0.25 m before the obstacle
    for box, expect in ((WALL, 0.), (BEAM, .5), (HURDLE, .5)):
        bm = BlockingMap(bank_of([scene_sdf([box])]))
        dist, away = bm.sample(xy, torch.tensor([0]))
        target = tm.safe_velocity_target(command, away, dist)
        assert float(target[0, 0]) == pytest.approx(expect, abs=.03), box


def test_heading_hysteresis():
    prev = torch.tensor([False, False, True, True])
    err = torch.tensor([.10, .25, .10, .05])
    assert tm.hysteresis(err, prev, on=.2, off=.08).tolist() == [False, True, True, False]


def test_config_keyboard_v10_wiring():
    from cat_mjlab.config import wholebody_config
    with pytest.raises(ValueError):
        wholebody_config(keyboard_v10=True)
    c = wholebody_config(teleop_fraction=.5, heading_align_weight=.4, teleop_heading_commands=True, keyboard_v10=True,
                         hand_clearance_tight=(.10, .40, .3))
    assert c['blocking_map'] and c['safe_tracking_teleop_only'] and c['heading_deadband'] == (.20, .08)
    assert c['teleop']['cat_scenes'] and c['teleop']['p_stop'] == .30 and c['teleop']['p_through_cat'] == .70
    assert c['hand_clearance_tight'] == dict(low=.10, high=.40, floor=.3, walls_only=False)


def test_config_v11_progress_stall_and_cat_fraction():
    from cat_mjlab.config import wholebody_config
    base = dict(teleop_fraction=.5, heading_align_weight=.4, teleop_heading_commands=True, keyboard_v10=True)
    c = wholebody_config(**base, teleop_progress_weight=6., teleop_stall_weight=-3., teleop_cat_fraction=.3)
    assert c['reward_config']['scales']['teleop_progress'] == 6. and c['reward_config']['scales']['teleop_stall'] == -3.
    assert c['teleop']['cat_fraction'] == .3 and c['teleop']['fraction'] == .5
    with pytest.raises(ValueError):
        wholebody_config(**base, teleop_stall_weight=3.)          # a stall must cost
    with pytest.raises(ValueError):
        wholebody_config(teleop_fraction=.5, heading_align_weight=.4, teleop_heading_commands=True, teleop_cat_fraction=.3)


def test_side_gap_bucket_and_handsdf_scaling():
    from cat_mjlab.runner import _scene_buckets, SCENE_BUCKETS
    manifest = dict(scenes=[dict(scene_id='sidegap-0000-w0.40', family='procedural_cat', source=dict(kind='side-gap')),
                            dict(scene_id='published-side1', family='published_cat', source={})])
    assert [SCENE_BUCKETS[i] for i in _scene_buckets(manifest)] == ['side_gap', 'published_cat']
    hands = torch.full((1, 2, 1), .062)                        # sideways in a 0.34 m gap
    assert float(tm.sdf_reward(hands, knee=torch.tensor([.05]))) < -8.          # unscaled: ~ -8.7
    assert float(tm.sdf_reward(hands, knee=torch.tensor([.015]))) > -2.         # room scale 0.3: ~ -1.8


def test_config_side_gap_curriculum_wiring():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config(teleop_fraction=.5, heading_align_weight=.4, teleop_heading_commands=True,
                         heading_lookaheads=(.3, .6, 1.), heading_blocking_probes=True, sideways_bonus_weight=.4,
                         side_gap_curriculum=True, teleop_side_prob=.25)
    assert c['heading_align']['lookaheads'] == [.3, .6, 1.] and c['heading_align']['blocking'] and c['blocking_map']
    assert c['reward_config']['scales']['sideways_bonus'] == .4 and c['teleop']['p_side'] == .25
    assert c['side_gap_curriculum']['thresholds'] == [.50, .42, .36, 0.]
    with pytest.raises(ValueError):
        wholebody_config(sideways_bonus_weight=-1., heading_align_weight=.4)


@pytest.mark.parametrize("box,blocks", [(BEAM, False), (HURDLE, False), (WALL, True), (TABLE, False)])
def test_full_height_walls_only(box, blocks):
    dist, _ = blocking_sdf2d(scene_sdf([box]), 0., DX, also=(1.0, 1.4))
    assert (dist.min() <= 0) == blocks


def test_config_walls_only_and_tumbling():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config(teleop_fraction=.5, heading_align_weight=.4, teleop_heading_commands=True, side_gap_curriculum=True,
                         side_gap_tumbling=True, hand_clearance_tight=(.10, .40, .3), hand_tight_walls_only=True)
    assert c['side_gap_curriculum']['tumbling'] and c['hand_clearance_tight']['walls_only']


def test_config_sideways_bonus_progress():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config(heading_align_weight=.4, sideways_bonus_weight=1., sideways_bonus_progress=.15)
    assert c['sideways_bonus_progress'] == .15
    with pytest.raises(ValueError):
        wholebody_config(heading_align_weight=.4, sideways_bonus_progress=.15)      # needs the bonus itself


def test_scene_skill_tags():
    from cat_mjlab.runner import scene_skill_tags
    wall = scene_sdf([((1.0, 0., 0.), (1.1, .8, 1.6)), ((1.0, 1.2, 0.), (1.1, 2., 1.6))])   # 0.40 m opening
    flat = np.full_like(wall, 5.)
    bank = bank_of([wall, flat, flat, flat])
    bank.manifest = dict(scenes=[dict(scene_id='procedural-D4G0', family='procedural_cat'),
                                 dict(scene_id='published-side-hurdle3', family='published_cat'),
                                 dict(scene_id='sidegap2-0001-angled-w0.38', family='procedural_cat',
                                      source=dict(kind='side-gap', side_gap=dict(variant='angled', gap_m=.38))),
                                 dict(scene_id='D8G0L1O0S3', family='original_cat')])
    names, matrix = scene_skill_tags(bank)
    tags = [sorted(n for n, on in zip(names, row) if on) for row in matrix]
    assert tags == [['proc_narrow'], ['pub_side_hurdle'], ['gap_angled', 'gapw_lt040'], ['original']]


def test_config_contact_penalty():
    from cat_mjlab.config import wholebody_config
    assert wholebody_config(contact_penalty=3.)['contact_penalty'] == 3.
    with pytest.raises(ValueError):
        wholebody_config(contact_penalty=-1.)
