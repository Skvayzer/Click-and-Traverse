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

    # ------------------------------------------------------------------ plan source (planner with a field reflex)
    def straight_plan(self, t, ids, speed=.5, yaw=0.):
        k = torch.arange(tr.PLAN_FRAMES, dtype=torch.float32)
        xy = torch.stack((k * tr.PLAN_DT * speed, torch.zeros_like(k)), -1)[None].expand(len(ids), -1, -1).clone()
        yaws = torch.full((len(ids), tr.PLAN_FRAMES), yaw)
        z = torch.full((len(ids), tr.PLAN_FRAMES), .70)
        upper = (t.nominal_upper + .3)[None, None].expand(len(ids), tr.PLAN_FRAMES, 17).clone()
        t.set_plan(ids, xy, yaws, z, upper, torch.full((len(ids),), .5))

    def test_continuity_state_matches_the_executing_plan(self):
        from cat_mjlab.planner.data import euler_zyx
        t = tr.ReferenceTracker(2, 'cpu', mix=(('plan', 1.),), continuity=dict(reanchor=.5),
                                generator=torch.Generator().manual_seed(0))
        ids = torch.arange(2)
        t.reset(ids, torch.zeros(2, 2), torch.zeros(2), sample_dir=east, op_speed=torch.full((2,), .3))
        k = torch.arange(tr.PLAN_FRAMES, dtype=torch.float32)
        xy = torch.stack((k * .02, torch.zeros_like(k)), -1)[None].repeat(2, 1, 1)      # 0.2 m/s along +x
        yaw = torch.full((2, tr.PLAN_FRAMES), .3); z = torch.full((2, tr.PLAN_FRAMES), .74)
        upper = (k[:, None] * .01).expand(-1, 17)[None].repeat(2, 1, 1)
        roll, pitch = torch.full((2, tr.PLAN_FRAMES), .05), torch.full((2, tr.PLAN_FRAMES), -.03)
        legs = (k[:, None] * .02).expand(-1, 12)[None].repeat(2, 1, 1)
        t.set_plan(ids, xy, yaw, z, upper, torch.full((2,), .5), roll=roll, pitch=pitch, legs=legs)
        t.set_plan(ids[1:], xy[1:], yaw[1:], z[1:], upper[1:], torch.full((1,), .5))   # world 1: no full state
        t.plan_t[:] = .25
        qpos, qvel = t.plan_state(ids[:1])
        self.assertTrue(torch.allclose(qpos[0, :3], torch.tensor([.05, 0., .74]), atol=1e-5))
        r, p_, y = euler_zyx(qpos[:, 3:7])
        self.assertAlmostEqual(float(r), .05, places=4); self.assertAlmostEqual(float(p_), -.03, places=4)
        self.assertAlmostEqual(float(y), .3, places=4)
        self.assertTrue(torch.allclose(qpos[0, 7:19], torch.full((12,), .05), atol=1e-5))   # legs at frame 2.5
        self.assertTrue(torch.allclose(qpos[0, 19:36], torch.full((17,), .025), atol=1e-5))
        self.assertTrue(torch.allclose(qvel[0, :3], torch.tensor([.2, 0., 0.]), atol=1e-4))
        self.assertTrue(torch.allclose(qvel[0, 6:18], torch.full((12,), .2), atol=1e-4))     # 0.02 rad per 0.1 s
        near = t.continuing(ids, torch.tensor([[.1, 0.], [.05, 0.]]))
        self.assertEqual(near.tolist(), [True, False])                                       # world 1 lacks the legs
        far = t.continuing(ids, torch.tensor([[.9, 0.], [.05, 0.]]))
        self.assertEqual(far.tolist(), [False, False])                                       # 0.85 m away: re-anchor
        goal_robot, _ = t.route_goal(ids, 2.)
        goal_plan, _ = t.route_goal(ids, 2., from_xy=torch.tensor([[.5, 0.], [.5, 0.]]))
        self.assertGreater(float(goal_plan[0, 0]), float(goal_robot[0, 0]) + .4)             # goal ahead of the plan

    def test_plan_worlds_wait_for_a_chunk(self):
        t = self.make(2, mix=(('plan', 1.),)); ids = torch.arange(2)
        t.reset(ids, torch.zeros(2, 2), torch.zeros(2), sample_dir=east, op_speed=torch.full((2,), .3))
        self.assertTrue(bool(t.is_plan.all()))
        self.assertFalse(bool(t.valid.any()))                     # no chunk yet: the policy sees no reference
        self.assertEqual(t.due().tolist(), [0, 1])
        self.assertTrue(bool((t.count == 51).all()))              # the guidance path is built: plan goals come from it
        goal, heading = t.route_goal(ids, 2.)
        self.assertTrue(torch.allclose(goal, torch.tensor([[.6, 0.], [.6, 0.]]), atol=1e-5))
        self.straight_plan(t, ids)
        self.assertTrue(bool(t.valid.all()))
        self.assertEqual(len(t.due()), 0)

    def test_plan_clock_targets_and_features(self):
        t = self.make(1, mix=(('plan', 1.),)); ids = torch.arange(1)
        t.reset(ids, torch.zeros(1, 2), torch.zeros(1), sample_dir=east, op_speed=torch.full((1,), .5))
        self.straight_plan(t, ids)
        robot = torch.zeros(1, 2)
        for _ in range(10):                                       # 0.2 s, the robot follows exactly
            robot = t.plan_at(ids, t.plan_t)[0]
            t.step(robot, .02, sample_dir=east, op_speed=torch.full((1,), .5))
        self.assertAlmostEqual(float(t.plan_t[0]), .2, places=5)
        self.assertAlmostEqual(float(t.replan_in[0]), .3, places=5)
        xy, yaw = t.targets(ids, 1.)
        self.assertAlmostEqual(float(xy[0, 0]), .6, places=5)     # (0.2 + 1.0) s at 0.5 m/s
        robot = t.plan_at(ids, t.plan_t)[0]                       # on the plan at 0.2 s: x = 0.1 m
        f = t.features(ids, robot, torch.zeros(1), torch.eye(3)[None])
        self.assertAlmostEqual(float(f[0, 1 + 5 * 3]), .5, places=4)     # dx at tau = 1.0 s, relative to the robot
        self.assertAlmostEqual(float(f[0, 5 + 5 * 3]), .70 - tr.NOMINAL_Z, places=5)   # plan height
        self.assertTrue(torch.allclose(f[0, -18:-1], torch.full((17,), .3), atol=1e-5))  # plan arm targets
        self.assertEqual(float(f[0, -1]), 1.)

    def test_plan_clock_waits_for_a_lagging_robot(self):
        t = self.make(1, mix=(('plan', 1.),)); ids = torch.arange(1)
        t.reset(ids, torch.zeros(1, 2), torch.zeros(1), sample_dir=east, op_speed=torch.full((1,), .5))
        self.straight_plan(t, ids)
        for _ in range(50):                                       # the robot stands still for 1 s
            t.step(torch.zeros(1, 2), .02, sample_dir=east, op_speed=torch.full((1,), .5))
        self.assertLess(float(t.plan_t[0]), .5)                   # the clock stopped near MAX_LAG ahead
        self.assertEqual(t.due().tolist(), [0])                   # and the replan is due

    def test_reflex_rule_releases_arms_without_paying_for_it(self):
        t = self.make(2, mix=(('plan', 1.),)); ids = torch.arange(2)
        t.reset(ids, torch.zeros(2, 2), torch.zeros(2), sample_dir=east, op_speed=torch.full((2,), .5))
        self.straight_plan(t, ids)
        info = t.step(torch.zeros(2, 2), .02, sample_dir=east, op_speed=torch.full((2,), .5))
        q = t.nominal_upper[None].expand(2, 17).clone(); q[:, 3:10] += 1.   # left arm far off its target
        released = torch.tensor([[False, False], [True, False]])
        r, tel = t.rewards(torch.zeros(2, 2), torch.zeros(2), torch.full((2,), .70), q, info, released=released)
        # world 1's left arm is released: its share is dropped, not paid; the rest still pays
        self.assertLess(float(r['track_upper'][1]), 10 / 17 + 1e-6)
        self.assertGreaterEqual(float(r["track_upper"][1]), float(r["track_upper"][0]) - 1e-5)
        far = torch.tensor([[.3, 0.], [.3, 0.]])
        r1, _ = t.rewards(far, torch.zeros(2), torch.full((2,), .70), q, info)
        r2, _ = t.rewards(far, torch.zeros(2), torch.full((2,), .70), q, info, deadband=torch.tensor([0., .2]))
        self.assertAlmostEqual(float(r1['track_pos'][0]), float(r2['track_pos'][0]), places=6)
        self.assertGreater(float(r2['track_pos'][1]), float(r1['track_pos'][1]))

    def test_commit_keeps_the_clock_running_and_charges_lag(self):
        """COMBO2: with commit the plan clock runs on while the robot stands, and the deadband forgives sideways error
        only. Without commit a standing robot keeps the reference beside it (the stall COMBO1 learned)."""
        for commit, expect_far in ((False, False), (True, True)):
            t = tr.ReferenceTracker(1, 'cpu', mix=(('plan', 1.),), commit=commit, generator=torch.Generator().manual_seed(0))
            ids = torch.arange(1)
            t.reset(ids, torch.zeros(1, 2), torch.zeros(1), sample_dir=east, op_speed=torch.full((1,), .5))
            self.straight_plan(t, ids)
            for _ in range(40):                                   # 0.8 s standing still at the start
                info = t.step(torch.zeros(1, 2), .02, sample_dir=east, op_speed=torch.full((1,), .5))
            gap = float(torch.linalg.vector_norm(t.plan_at(ids, t.plan_t)[0][0]))
            self.assertEqual(gap > .3, expect_far, (commit, gap))
        # deadband: 0.2 m sideways is forgiven, 0.2 m of lag is not
        t = tr.ReferenceTracker(2, 'cpu', mix=(('plan', 1.),), commit=True, generator=torch.Generator().manual_seed(0))
        ids = torch.arange(2)
        t.reset(ids, torch.zeros(2, 2), torch.zeros(2), sample_dir=east, op_speed=torch.full((2,), .5))
        self.straight_plan(t, ids)
        info = t.step(torch.zeros(2, 2), .02, sample_dir=east, op_speed=torch.full((2,), .5))
        here = t.plan_at(ids, t.plan_t)[0]
        robot = here + torch.tensor([[0., .2], [-.2, 0.]])      # world 0 beside the plan, world 1 behind it
        r, _ = t.rewards(robot, torch.zeros(2), torch.full((2,), .70), t.plan_at(ids, t.plan_t)[3], info,
                         deadband=torch.full((2,), .2))
        self.assertGreater(float(r['track_pos'][0]), .99)
        self.assertLess(float(r['track_pos'][1]), .2)

    def test_fallback_drops_the_plan_after_a_stall_and_replans_later(self):
        t = tr.ReferenceTracker(2, 'cpu', mix=(('plan', 1.),), fallback=dict(stall_s=1., hold_s=2., min_speed=.03),
                                generator=torch.Generator().manual_seed(0))
        ids = torch.arange(2); v = torch.full((2,), .3)
        t.reset(ids, torch.zeros(2, 2), torch.zeros(2), sample_dir=east, op_speed=v)
        self.straight_plan(t, ids)
        robot = torch.zeros(2, 2)
        for k in range(60):                                   # 1.2 s: world 0 stands, world 1 walks at 0.3 m/s
            robot[1, 0] += .3 * .02
            t.step(robot, .02, sample_dir=east, op_speed=v)
            if not bool(t.valid[0]): break
        self.assertFalse(bool(t.valid[0]))                    # world 0 lost its plan: reactive mode
        self.assertTrue(bool(t.valid[1]))                     # the walking world keeps its plan
        self.assertNotIn(0, t.due().tolist())                 # no replanning during the fallback
        for _ in range(101):                                  # 2 s of fallback
            t.step(robot, .02, sample_dir=east, op_speed=v)
        self.assertIn(0, t.due().tolist())                    # then a fresh plan is due
        self.assertEqual(float(t.fallback_t[0]), 0.)

    def test_hold_worlds_can_get_a_stand_still_plan(self):
        t = tr.ReferenceTracker(4, 'cpu', mix=(('field', 1.),), hold_plan=1., generator=torch.Generator().manual_seed(0))
        ids = torch.arange(4); hold = torch.tensor([True, False, True, False])
        t.reset(ids, torch.zeros(4, 2), torch.zeros(4), sample_dir=east, op_speed=torch.full((4,), .3), hold=hold)
        self.assertTrue(bool(t.is_plan[hold].all()))
        self.assertTrue(bool(t.still[hold].all()))
        self.assertTrue(bool((t.route_speed[hold] == 0).all()))
        self.assertTrue(bool((t.count[hold] == 1).all()))          # no guidance path for stand-still worlds


class PlannerStateTest(unittest.TestCase):
    def test_state_matches_the_dataset_features(self):
        from cat_mjlab.planner.data import features, quat_from_euler
        from cat_mjlab.planner.online import planner_state
        g = torch.Generator().manual_seed(0)
        T, N = 120, 3
        qpos = torch.randn(T + 1, N, 36, generator=g); qvel = torch.randn(T + 1, N, 35, generator=g)
        qpos[..., 3:7] = quat_from_euler(.1 * torch.randn(T + 1, N, generator=g), .1 * torch.randn(T + 1, N, generator=g),
                                         torch.randn(T + 1, N, generator=g))
        starts = torch.tensor([[5, 0], [7, 2]]).numpy()
        state, *_ = features(qpos, qvel, starts)
        ours = planner_state(qpos[[5, 7], [0, 2]], qvel[[5, 7], [0, 2]])
        self.assertTrue(torch.allclose(state, ours, atol=1e-5))


class RouteDirectionTest(unittest.TestCase):
    def test_follows_an_l_shaped_route_to_its_end(self):
        from cat_mjlab.navigation import route_direction
        route = torch.zeros(1, 5, 2); route[0, :3] = torch.tensor([[0., 0.], [2., 0.], [2., 2.]])
        count = torch.tensor([3])
        xy = torch.tensor([[0., .5]])                      # off the route, beside its first leg
        path = [xy.clone()]
        for _ in range(200):
            xy = xy + .05 * route_direction(xy, route, count); path.append(xy.clone())
        end = path[-1][0]
        self.assertLess(float(torch.linalg.vector_norm(end - torch.tensor([2., 2.]))), .06)    # reached the end, stopped
        mid = torch.stack(path)[:, 0]
        self.assertTrue(bool((mid[:, 0] <= 2.06).all()))                                        # turned at the corner
        self.assertTrue(torch.equal(route_direction(torch.tensor([[2., 2.]]), route, count), torch.zeros(1, 2)))


class ComboConfigTest(unittest.TestCase):
    def test_flags_reach_the_config(self):
        """--tracker-reflex passes an empty dict (the default bands): it must not be dropped as falsy (it was, once)."""
        from types import SimpleNamespace
        from cat_mjlab.runner import tracker_spec
        from cat_mjlab.config import wholebody_config
        args = SimpleNamespace(tracker=True, tracker_mix='plan=0.5,field=0.15,random=0.1,none=0.25', tracker_scales=None,
                               tracker_sigma=None, tracker_planner='/x/planner.pt', tracker_plan_samples=None,
                               tracker_plan_steps=None, tracker_plan_interval=None, tracker_hold_plan=0.5, tracker_reflex=True)
        tc = wholebody_config(tracker=tracker_spec(args), operator_speed=(0.1, 0.6))['tracker']
        self.assertEqual(tc['reflex'], dict(hand_band=.15, body_band=.35, deadband=.20))
        self.assertEqual(tc['hold_plan'], .5)
        self.assertEqual(tc['planner']['path'], '/x/planner.pt')
        self.assertEqual(tc['planner']['steps'], 6)
        args.tracker_reflex = False
        self.assertNotIn('reflex', wholebody_config(tracker=tracker_spec(args), operator_speed=(0.1, 0.6))['tracker'])
        self.assertNotIn('commit', tc)
        args.tracker_commit = True
        self.assertTrue(wholebody_config(tracker=tracker_spec(args), operator_speed=(0.1, 0.6))['tracker']['commit'])
        self.assertNotIn('continuity', tc); self.assertNotIn('route_go', tc)
        args.tracker_continuity, args.tracker_route_go = 0.5, True
        tc4 = wholebody_config(tracker=tracker_spec(args), operator_speed=(0.1, 0.6))['tracker']
        self.assertEqual(tc4['continuity'], dict(reanchor=.5)); self.assertTrue(tc4['route_go'])
        args.tracker_crouch_gate = .70
        self.assertEqual(wholebody_config(tracker=tracker_spec(args), operator_speed=(0.1, 0.6))['plan_crouch_gate'], .70)


class PrecisionConfigTest(unittest.TestCase):
    def test_far_threat_only_needs_the_far_target(self):
        from cat_mjlab.config import wholebody_config
        cfg = wholebody_config(reactive_hand_clearance=(.20, .40), reactive_far_threat_only=True)
        self.assertTrue(cfg['reactive_far_threat_only'])
        self.assertNotIn('reactive_far_threat_only', wholebody_config(reactive_hand_clearance=(.20, .40)))
        with self.assertRaises(ValueError):
            wholebody_config(reactive_far_threat_only=True)


if __name__ == '__main__':
    unittest.main()


class HeightCurriculumTest(unittest.TestCase):
    def test_config_and_scene_levels(self):
        import sys
        from pathlib import Path
        import numpy as np
        from cat_mjlab.config import wholebody_config
        self.assertEqual(wholebody_config(height_curriculum=(0.5, 300))['height_curriculum'], dict(advance=.5, window=300))
        self.assertNotIn('height_curriculum', wholebody_config())
        with self.assertRaises(ValueError):
            wholebody_config(height_curriculum=(1.5, 300))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        import generate_beam_bars as g
        rng = np.random.default_rng(0)
        boxes, info = g.scene_boxes(rng, 0, 'beam')
        self.assertEqual(info['beam_bottom_m'], 1.40)
        beam = boxes[0]
        self.assertAlmostEqual(beam['center'][2] - beam['half_size'][2], 1.40, places=3)     # underside
        self.assertGreaterEqual(2 * beam['half_size'][1], g.WIDTH)                           # spans the full width
        boxes, info = g.scene_boxes(rng, 7, 'beam_bar')
        self.assertEqual((info['bar_m'], info['beam_bottom_m']), (0.19, 0.90))
        bar, beam = boxes
        self.assertLess(bar['center'][0], beam['center'][0])                                 # the bar comes first


class PlannerDataFilterTest(unittest.TestCase):
    def test_drop_stopped_keeps_standing_scenes(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from cat_mjlab.planner.data import FILES, drop_stopped, load_split
        info = dict(scene_types=["procedural_cat", "original_cat", "clutter_dense", "reactive_standing"])
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "train"; d.mkdir()
            for k, (bucket, speed) in enumerate(((1, .0), (1, .2), (3, .0), (2, .01))):
                arrays = dict(grid=np.zeros((1, 2, 2, 2), np.uint8), state=np.zeros((1, 67), np.float32),
                              chunk=np.zeros((1, 20, 35), np.float32), goal=np.array([[0, 0, 0, 1, speed]], np.float32),
                              meta=np.array([[k, bucket, 0, 0, 0, 0]]), pose=np.zeros((1, 3), np.float32))
                for f in FILES:
                    np.save(d / f"{k:03d}_{f}.npy", arrays[f])
            out = load_split(tmp, "train", keep=drop_stopped(info))
            self.assertEqual(out["meta"][:, 0].tolist(), [1, 2, 3])     # only the stopped CAT window is dropped
            self.assertEqual(len(out["grid"]), 3)


class GaitConfigTest(unittest.TestCase):
    def test_cadence_contract_and_lipschitz(self):
        from dataclasses import replace
        import numpy as np
        from cat_mjlab.config import wholebody_config
        from cat_mjlab.observation_contract import actor_size, critic_size, observation_contract
        cfg = wholebody_config(cadence=(.55, .55, .6, 1.4, .2, 4800), hip_deviation_weight=-1.)
        self.assertEqual(cfg['cadence'], dict(a=.55, b=.55, lo=.6, hi=1.4, noise=.2, ramp_steps=4800., tau=.5))
        self.assertEqual(cfg['reward_config']['scales']['hip_deviation'], -1.)
        self.assertEqual(actor_size(True, True, True), 222 + 4 + 41 + 1)
        self.assertEqual(critic_size(True, True), 310 + 41 + 1)
        contract = observation_contract(True, True, True)
        self.assertEqual(contract['actor_features'][-1], 'gait.stride_frequency_hz')
        self.assertEqual(len(contract['actor_features']), 268)
        from cat_mjlab.learning import Learner, LearnerConfig, compute_loss, gaussian_parameters, log_probability
        config = LearnerConfig(algorithm='ppo', actor_obs=3, critic_obs=4, action_size=2, actor_hidden=(7, 5),
                               critic_hidden=(9, 6), num_policies=1, embedding_dim=2, num_minibatches=2,
                               num_updates_per_batch=2, prepare_chunk_size=2, entropy_cost=0.)
        torch.manual_seed(0); learner = Learner(config, device='cpu')
        rng = np.random.default_rng(7)
        data = {k: torch.tensor(rng.normal(0, .3, (6, 4, w)), dtype=torch.float32)
                for k, w in (('state', 3), ('next_state', 3), ('privileged_state', 4), ('next_privileged_state', 4))}
        data['policy_id'] = torch.zeros(6, 4, dtype=torch.long)
        with torch.no_grad():
            logits = learner.model.logits(data['state'], data['policy_id']); mean, std = gaussian_parameters(logits)
            data['raw_action'] = mean + std * torch.tensor(rng.normal(size=(6, 4, 2)), dtype=torch.float32)
            data['log_prob'] = log_probability(logits, data['raw_action'])
        data.update(reward=torch.tensor(rng.normal(size=(6, 4)), dtype=torch.float32), discount=torch.ones(6, 4),
                    truncation=torch.zeros(6, 4))
        base, _ = compute_loss(learner.model, data, config)
        smooth, metrics = compute_loss(learner.model, data, replace(config, lipschitz=.01))
        self.assertGreater(float(metrics['diagnostics/lipschitz_penalty']), 0.)
        self.assertAlmostEqual(float(smooth - base), .01 * float(metrics['diagnostics/lipschitz_penalty']), places=5)
        smooth.backward()                                      # the penalty is differentiable (double backward)
