"""CPU geometry/field contracts; no robot dynamics or policy training."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cat_ppo.furniture.scenes import (
    FAMILIES, GUIDANCE_METHOD, generate_scene, load_scene, route_guidance,
    signed_distance, validate_scene, write_scene_bundle,
)


class SceneGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dense = generate_scene(seed=7, family="mixed", split="test")
        cls.pilot = generate_scene(seed=2, family="mixed", difficulty="pilot")

    def test_dense_counts_and_part_identity(self):
        scene = self.dense
        self.assertEqual(scene["counts"]["tables"], 9)
        self.assertEqual(scene["counts"]["chairs"], 36)
        self.assertEqual(scene["counts"]["bottlenecks"], 6)
        names = [box["name"] for box in scene["boxes"]]
        self.assertEqual(len(names), len(set(names)))
        tables = {box["furniture_id"] for box in scene["boxes"] if box["category"] == "tabletop"}
        chairs = {box["furniture_id"] for box in scene["boxes"] if box["category"] == "chair_seat"}
        self.assertEqual(len(tables), 9)
        self.assertEqual(len(chairs), 36)
        self.assertEqual(sum(box["category"] == "table_leg" for box in scene["boxes"]), 36)
        self.assertEqual(sum(box["category"] == "chair_leg" for box in scene["boxes"]), 144)
        self.assertTrue(any(box["category"] == "overhead" for box in scene["boxes"]))

    def test_seed_reproducibility_and_physical_split_holdout(self):
        repeat = generate_scene(seed=7, family="mixed", split="test")
        self.assertEqual(json.dumps(self.dense, sort_keys=True), json.dumps(repeat, sort_keys=True))
        train = generate_scene(seed=7, family="mixed", split="train")
        train_tops = [box for box in train["boxes"] if box["category"] == "tabletop"]
        test_tops = [box for box in self.dense["boxes"] if box["category"] == "tabletop"]
        self.assertLess(max(box["half_size"][0] for box in train_tops),
                        min(box["half_size"][0] for box in test_tops))
        self.assertTrue({b["shape_identity"] for b in train_tops}.isdisjoint(
            {b["shape_identity"] for b in test_tops}))
        different_seed = generate_scene(seed=8, family="mixed", split="test")
        self.assertNotEqual(self.dense["geometry_hash"], different_seed["geometry_hash"])
        self.assertNotEqual(self.dense["route"], different_seed["route"])

    def test_all_families_preserve_root_route_and_have_different_geometry(self):
        hashes = set()
        for family in FAMILIES:
            scene = generate_scene(seed=3, family=family, split="validation")
            hashes.add(scene["geometry_hash"])
            self.assertEqual((scene["counts"]["tables"], scene["counts"]["chairs"]), (9, 36))
            self.assertTrue(scene["feasibility"]["root_route_validated"])
            self.assertGreater(scene["feasibility"]["root_margin_lower_bound_m"], 0)
            self.assertFalse(scene["feasibility"]["full_body_validated"])
            self.assertFalse(scene["feasibility"]["dynamic_feasibility_validated"])
        self.assertEqual(len(hashes), len(FAMILIES))

    def test_three_start_goals_preserve_ordered_route_progress(self):
        cases = self.dense["start_goals"]
        self.assertEqual(len(cases), 3)
        self.assertEqual(cases[0]["start"][:2], cases[2]["goal"])
        self.assertEqual(cases[0]["goal"], cases[2]["start"][:2])
        self.assertEqual([gate["id"] for gate in cases[0]["bottlenecks"]],
                         list(reversed([gate["id"] for gate in cases[2]["bottlenecks"]])))
        for case in cases:
            self.assertEqual(case["start"][:2], case["route"][0])
            self.assertEqual(case["goal"], case["route"][-1])
            distances = [gate["route_distance_m"] for gate in case["bottlenecks"]]
            self.assertEqual(distances, sorted(distances))
            self.assertEqual(len(distances), 6)
            self.assertTrue(all(0 < d < case["route_length_m"] for d in distances))
            self.assertTrue(all(0 < gate["route_progress_fraction"] < 1 for gate in case["bottlenecks"]))
            self.assertGreater(case["time_budget"], case["route_length_m"])

    def test_table_and_chair_holes_and_near_floor_remain_free(self):
        # Actual furniture centers, away from legs; a solid convex table/chair
        # mesh or historical lowest-six-voxel fill would fail these points.
        points = np.array([[1.75, .61, .20], [2.60, 2.13, .20], [3.20, 1.50, .01]])
        self.assertTrue(np.all(signed_distance(points, self.pilot["boxes"]) > .10))
        self.assertLess(float(signed_distance(np.array([1.75, .61, .75]), self.pilot["boxes"])), 0)
        self.assertLess(float(signed_distance(np.array([2.60, 2.13, .45]), self.pilot["boxes"])), 0)
        self.assertFalse(any(box["category"] == "floor" for box in self.pilot["boxes"]))
        self.assertFalse(self.pilot["generator"]["floor_in_obstacle_field"])

    def test_oriented_box_distance_uses_xy_yaw_and_preserves_height(self):
        box = dict(center=[1, 2, 3], half_size=[.2, .4, .1], yaw=math.pi / 2)
        actual = signed_distance(np.array([[1, 2.5, 3], [1.5, 2, 3], [1, 2, 3.5], [1, 2, 3]]), [box])
        np.testing.assert_allclose(actual, [.3, .1, .4, -.1], atol=1e-12)

    def test_open_floor_is_explicit_control(self):
        scene = generate_scene(difficulty="open-floor")
        self.assertEqual(scene["counts"]["tables"], 0)
        self.assertEqual(scene["counts"]["chairs"], 0)
        self.assertEqual(len(scene["boxes"]), 4)
        self.assertEqual(scene["bottlenecks"], [])

    def test_invalid_geometry_hash_or_route_is_rejected(self):
        changed = copy.deepcopy(self.pilot)
        changed["boxes"][0]["center"][0] += .1
        with self.assertRaisesRegex(ValueError, "geometry_hash"):
            validate_scene(changed)
        changed = copy.deepcopy(self.pilot)
        changed["goal"][0] += .1
        with self.assertRaisesRegex(ValueError, "endpoints"):
            validate_scene(changed)
        with self.assertRaises(ValueError):
            generate_scene(split="unknown")


class SceneFieldTest(unittest.TestCase):
    def test_bundle_grid_axis_centers_gradient_and_hashes(self):
        scene = generate_scene(difficulty="open_floor")
        with tempfile.TemporaryDirectory() as temporary:
            bundle = write_scene_bundle(scene, Path(temporary) / "room", voxel_size=.10, chunk_size=127)
            loaded = load_scene(bundle)
            self.assertEqual(loaded, load_scene(bundle / "scene.json"))
            grid = loaded["grid"]
            self.assertEqual(grid["axis_order"], "xyz")
            np.testing.assert_allclose(grid["sample_origin"], [-.05, -.05, .05])
            sdf = np.load(bundle / "sdf.npy", allow_pickle=False)
            bf = np.load(bundle / "bf.npy", allow_pickle=False)
            gf = np.load(bundle / "gf.npy", allow_pickle=False)
            self.assertEqual(sdf.shape, tuple(grid["shape"]))
            self.assertEqual(bf.shape, sdf.shape + (3,))
            self.assertEqual(gf.shape, bf.shape)
            self.assertEqual(sdf.dtype, np.float32)
            # Index xyz=(4,15,6) is world (.35,1.45,.65), .29m
            # from the west wall's x=.06 face. This catches xyz/zxy swaps.
            self.assertAlmostEqual(float(sdf[4, 15, 6]), .29, places=6)
            np.testing.assert_allclose(bf[4, 15, 6], [1, 0, 0], atol=1e-6)
            np.testing.assert_allclose(sdf[4, 15, :], .29, atol=1e-6)
            self.assertTrue(np.isfinite(sdf).all() and np.isfinite(bf).all() and np.isfinite(gf).all())
            self.assertLessEqual(float(np.linalg.norm(gf, axis=-1).max()), .600001)
            self.assertEqual(loaded["field_provenance"]["guidance"], GUIDANCE_METHOD)
            for field in ("sdf", "bf", "gf"):
                digest = hashlib.sha256((bundle / f"{field}.npy").read_bytes()).hexdigest()
                self.assertEqual(digest, loaded["fields"][field]["sha256"])
            with self.assertRaises(FileExistsError):
                write_scene_bundle(scene, bundle)

    def test_goal_variants_share_geometry_and_distance_but_reverse_guidance(self):
        scene = generate_scene(difficulty="open_floor")
        with tempfile.TemporaryDirectory() as temporary:
            first = write_scene_bundle(scene, Path(temporary) / "forward", voxel_size=.20, goal_index=0)
            last = write_scene_bundle(scene, Path(temporary) / "reverse", voxel_size=.20, goal_index=2)
            forward, reverse = load_scene(first), load_scene(last)
            self.assertEqual(forward["geometry_hash"], reverse["geometry_hash"])
            self.assertEqual(forward["fields"]["sdf"]["sha256"], reverse["fields"]["sdf"]["sha256"])
            self.assertNotEqual(forward["fields"]["gf"]["sha256"], reverse["fields"]["gf"]["sha256"])
            self.assertNotEqual(forward["field_provenance"]["guidance_hash"],
                                reverse["field_provenance"]["guidance_hash"])
            forward_gf, reverse_gf = np.load(first / "gf.npy"), np.load(last / "gf.npy")
            self.assertGreater(forward_gf[9, 7, 3, 0], .5)
            self.assertLess(reverse_gf[9, 7, 3, 0], -.5)
            self.assertEqual(reverse["goal_index"], 2)

    def test_guidance_follows_correct_aisle_and_stops_at_goal(self):
        route = [[0, 0], [4, 0], [4, 2], [0, 2], [0, 4], [4, 4]]
        points = np.array([[2, 0, .8], [2, 2, .8], [2, 4, .8], [4, 4, .8], [4.1, 4, .8]])
        guidance = route_guidance(points, route)
        np.testing.assert_allclose(guidance[:3, 0], [.6, -.6, .6], atol=1e-12)
        np.testing.assert_allclose(guidance[:, 2], 0)
        np.testing.assert_allclose(guidance[3], 0, atol=1e-12)
        self.assertLess(guidance[4, 0], 0)  # An overshoot returns to the goal, not past it.
        self.assertLess(abs(guidance[4, 0]), .6)

    def test_bad_numeric_generation_parameters_publish_nothing(self):
        scene = generate_scene(difficulty="open_floor")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "never-published"
            for kwargs in ({"voxel_size": 0}, {"voxel_size": float("nan")},
                           {"goal_index": 3}, {"chunk_size": 0}):
                with self.assertRaises(ValueError):
                    write_scene_bundle(scene, output, **kwargs)
                self.assertFalse(output.exists())

    def test_loader_rejects_tampered_field_and_stale_selected_guidance(self):
        scene = generate_scene(difficulty="open_floor")
        with tempfile.TemporaryDirectory() as temporary:
            output = write_scene_bundle(scene, Path(temporary) / "room", voxel_size=.20)
            field = output / "sdf.npy"
            original = field.read_bytes()
            tampered = bytearray(original)
            tampered[-1] ^= 1
            field.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "sha256"):
                load_scene(output)
            field.write_bytes(original)
            metadata = json.loads((output / "scene.json").read_text())
            for key in ("start", "goal", "route", "time_budget", "bottlenecks", "route_length_m"):
                metadata[key] = metadata["start_goals"][2][key]
            metadata["goal_index"] = 2
            (output / "scene.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "guidance provenance"):
                load_scene(output)

    def test_loader_rejects_shifted_grid_with_unchanged_field_bytes(self):
        scene = generate_scene(difficulty="open_floor")
        with tempfile.TemporaryDirectory() as temporary:
            output = write_scene_bundle(scene, Path(temporary) / "room", voxel_size=.20)
            metadata = json.loads((output / "scene.json").read_text())
            metadata["grid"]["edge_origin"][0] += .10
            metadata["grid"]["sample_origin"][0] += .10
            (output / "scene.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "grid_hash"):
                load_scene(output)


if __name__ == "__main__":
    unittest.main()
