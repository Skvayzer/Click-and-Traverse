"""Tracker v1 references (cat_mjlab/tracking.py): paths, the waiting clock, frames, rewards."""
import math
import unittest

import torch

from cat_mjlab import tracking as tr


def east(xy, ids):
    return torch.tensor([1., 0.]).expand(len(ids), 2)


class TrackingTest(unittest.TestCase):
    def make(self, n=4, mix=(('field', 1.),)):
        return tr.ReferenceTracker(n, 'cpu', mix=mix, generator=torch.Generator().manual_seed(0))

    def test_feature_names(self):
        self.assertEqual(len(tr.feature_names()), tr.FEATURES)

    def test_field_path_follows_the_field(self):
        t = self.make(); ids = torch.arange(4)
        t.reset(ids, torch.zeros(4, 2), torch.zeros(4), sample_dir=east, op_speed=torch.full((4,), .2))
        self.assertTrue(bool((t.count == 31).all()))
        self.assertTrue(torch.allclose(t.path[0, 30], torch.tensor([30 * tr.H, 0.]), atol=1e-6))

    def test_clock_advances_and_waits(self):
        t = self.make(); ids = torch.arange(4); v = torch.full((4,), .2)
        t.reset(ids, torch.zeros(4, 2), torch.zeros(4), sample_dir=east, op_speed=v)
        robot = torch.zeros(4, 2)
        for _ in range(50):                      # 1 s with the robot standing still
            t.step(robot, .02, sample_dir=east, op_speed=v)
        # the clock ran until the robot was MAX_LAG behind, then waited
        self.assertTrue(bool((t.s <= tr.MAX_LAG + .2 * .02 + 1e-6).all()), t.s)
        robot = torch.tensor([[.5, 0.]]).expand(4, 2)   # robot jumps ahead of the reference
        info = t.step(robot, .02, sample_dir=east, op_speed=v)
        r, tel = t.rewards(robot, torch.zeros(4), torch.full((4,), .78), torch.zeros(4, 17), info)
        self.assertTrue(bool((r['track_lead'] > 0).all()))

    def test_features_in_heading_frame(self):
        t = self.make(1); ids = torch.arange(1)
        t.reset(ids, torch.zeros(1, 2), torch.zeros(1), sample_dir=east, op_speed=torch.full((1,), .5))
        yaw = math.pi / 2                         # robot faces +y: an east path is on its right (-y local)
        c, s = math.cos(yaw), math.sin(yaw)
        navi = torch.tensor([[[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]])
        f = t.features(ids, torch.zeros(1, 2), torch.tensor([yaw]), navi)
        dx, dy = f[0, 1 + 5 * 3], f[0, 2 + 5 * 3]     # tau = 1.0 s: 0.5 m east
        self.assertAlmostEqual(float(dx), 0., places=5)
        self.assertAlmostEqual(float(dy), -.5, places=5)
        self.assertEqual(float(f[0, 0]), 1.)

    def test_none_source_is_blank(self):
        t = self.make(2, mix=(('none', 1.),)); ids = torch.arange(2)
        t.reset(ids, torch.zeros(2, 2), torch.zeros(2), sample_dir=east, op_speed=torch.full((2,), .3))
        f = t.features(ids, torch.zeros(2, 2), torch.zeros(2), torch.eye(3).expand(2, 3, 3))
        self.assertEqual(float(f.abs().sum()), 0.)

    def test_random_source_and_long_paths(self):
        t = self.make(3, mix=(('random', 1.),)); ids = torch.arange(3)
        t.reset(ids, torch.zeros(3, 2), torch.zeros(3), sample_dir=east, op_speed=torch.zeros(3))
        t.speed[:] = .6
        robot = torch.zeros(3, 2)
        for k in range(1500):                    # 30 s at 0.6 m/s = 18 m, much longer than the P-point buffer
            here = t._point(ids, t.s)
            robot = here.clone()                 # a perfect tracker
            t.speed[:] = .6; t.timer[:] = 10.
            t.step(robot, .02, sample_dir=east, op_speed=torch.zeros(3))
        self.assertTrue(bool((t.count <= tr.P).all()))
        self.assertFalse(bool(t.lost.any()))
        step = torch.linalg.vector_norm(t.path[:, 1:20] - t.path[:, :19], dim=-1)
        self.assertTrue(torch.allclose(step, torch.full_like(step, tr.H), atol=1e-5))


    def test_hold_worlds_get_no_reference(self):
        t = self.make(mix=(('field', .5), ('random', .5))); ids = torch.arange(4)
        hold = torch.tensor([True, False, True, False])
        t.reset(ids, torch.zeros(4, 2), torch.zeros(4), sample_dir=east, op_speed=torch.full((4,), .2),
                room_scene=torch.ones(4, dtype=torch.bool), hold=hold)
        self.assertFalse(bool(t.valid[hold].any()))
        self.assertTrue(bool(t.valid[~hold].all()))

    def test_upper_targets_look_ahead(self):
        t = self.make(n=1); ids = torch.arange(1)
        t.reset(ids, torch.zeros(1, 2), torch.zeros(1), sample_dir=east, op_speed=torch.full((1,), .2))
        t.mask_upper[:] = 1.
        t.upper_goal[:] = t.nominal_upper + 1.         # far goal: the look-ahead is rate-limited
        f = t.features(ids, torch.zeros(1, 2), torch.zeros(1), torch.eye(3)[None])
        upper = f[0, -18:-1]
        expected = torch.full((17,), t.upper_rate * tr.UPPER_HORIZON)
        self.assertTrue(torch.allclose(upper, expected, atol=1e-6), upper)


if __name__ == '__main__':
    unittest.main()
