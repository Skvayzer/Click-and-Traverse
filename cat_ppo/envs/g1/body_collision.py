"""Training-side full-body primitive/obstacle checks, without actor features.

The native floor/contact solver remains unchanged. Forbidden obstacle overlap
is latched at physics substeps and ends the policy transition with an explicit
event penalty. These discrete checks are not continuous collision detection.
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np

from cat_ppo.furniture.body_collision_geometry import (
    box_box_separation, capsule_box_separation, sphere_box_separation,
    compile_proposal, transform_primitives,
)


ROOT = Path(__file__).resolve().parents[3]
PROPOSAL = ROOT / "docs/assets/collision-proxy-proposal-20260916/proposal.json"
REGIONS = ("feet", "legs", "trunk", "head", "arms", "hands")


def body_collisions(compiled, bank, scene_id, xpos, xmat, *, max_candidates):
    """Exact primitive/box overlaps after conservative indexed broadphase.

    This returns only 35 flags. Candidate pairs are temporary computation and
    are never included in observations or stored in PPO trajectories.
    """
    from cat_ppo.furniture.body_collision_bank import lookup_collision_candidates
    world = transform_primitives(xpos, xmat, compiled["body_ids"],
                                 compiled["local_centers"], compiled["local_rotations"],
                                 compiled["local_endpoints"])
    result = jp.zeros(len(compiled["body_ids"]), dtype=bool)
    for kind in ("box", "capsule", "sphere"):
        indices = np.asarray(compiled["indices"][kind], dtype=np.int32)
        if not len(indices):
            continue
        centers = world["centers"][indices]
        candidate_ids, valid = lookup_collision_candidates(
            bank, scene_id, centers, max_candidates=max_candidates)
        # Bound narrow-phase temporaries at 16k environments: test eight
        # candidates at a time, reducing flags immediately. Every candidate
        # remains included; this is memory tiling, not a capped neighbor list.
        width = min(8, max_candidates)
        padding = (-max_candidates) % width
        candidate_ids = jp.pad(candidate_ids, ((0, 0), (0, padding)))
        valid = jp.pad(valid, ((0, 0), (0, padding)))
        chunks = lambda values: jp.swapaxes(values.reshape(len(indices), -1, width), 0, 1)

        def test_chunk(collided, inputs):
            ids, mask = inputs
            obstacle_center = bank["centers"][ids]
            obstacle_rotation = bank["rotations"][ids]
            obstacle_half = bank["half_sizes"][ids]
            if kind == "box":
                separation = box_box_separation(
                    centers[:, None], world["rotations"][indices, None],
                    jp.asarray(compiled["half_sizes"])[indices, None],
                    obstacle_center, obstacle_rotation, obstacle_half)
            elif kind == "capsule":
                ends = world["endpoints"][indices]
                separation = capsule_box_separation(
                    ends[:, None, 0], ends[:, None, 1],
                    jp.asarray(compiled["radii"])[indices, None],
                    obstacle_center, obstacle_rotation, obstacle_half)
            else:
                separation = sphere_box_separation(
                    centers[:, None], jp.asarray(compiled["radii"])[indices, None],
                    obstacle_center, obstacle_rotation, obstacle_half)
            return collided | jp.any(mask & (separation <= 0.), axis=-1), None

        collided, _ = jax.lax.scan(test_chunk, jp.zeros(len(indices), dtype=bool),
                                   (chunks(candidate_ids), chunks(valid)))
        result = result.at[indices].set(collided)
    return result


class BodyCollisionMixin:
    """Optional training extension; disabled configurations keep native behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        settings = self._config.wholebody.body_collision
        self.body_collision_enabled = bool(settings.enabled)
        if not self.body_collision_enabled:
            return
        if self.compatibility_mode:
            raise ValueError("Full-body collision training cannot be native compatibility mode")
        if not np.isfinite(settings.event_penalty) or settings.event_penalty <= 0:
            raise ValueError("Body collision event penalty must be finite and positive")
        from cat_ppo.furniture.body_collision_bank import load_body_collision_bank
        proposal_path = Path(settings.proposal).resolve()
        proposal = json.loads(proposal_path.read_text())
        self._body_primitives = compile_proposal(proposal, self.mj_model)
        self._body_region_mask = jp.asarray([
            [group == name for group in self._body_primitives["groups"]]
            for name in REGIONS], dtype=bool)
        bank_path = Path(settings.bank_manifest).resolve()
        arrays, metadata = load_body_collision_bank(
            bank_path, expected_field_manifest=self._config.pf_config.bank_manifest,
            expected_proxy_sha256=hashlib.sha256(proposal_path.read_bytes()).hexdigest())
        self._body_collision_bank = {key: jp.asarray(value) for key, value in arrays.items()}
        self.body_collision_metadata = metadata
        reset_path = Path(settings.reset_manifest).resolve()
        reset_meta = json.loads(reset_path.read_text())
        if (reset_meta["collision_bank_sha256"] != hashlib.sha256(bank_path.read_bytes()).hexdigest()
                or reset_meta["proxy_sha256"] != hashlib.sha256(proposal_path.read_bytes()).hexdigest()):
            raise ValueError("Validated reset poses belong to different collision geometry")
        pool_path = reset_path.parent / reset_meta["file"]
        if hashlib.sha256(pool_path.read_bytes()).hexdigest() != reset_meta["sha256"]:
            raise ValueError("Validated reset pose cache checksum differs")
        pool = np.load(pool_path, allow_pickle=False)
        if (pool.ndim != 3 or pool.shape[0] != self.num_pf_scenes or pool.shape[-1] != self.mj_model.nq
                or not np.isfinite(pool).all() or pool.shape[1] < 1):
            raise ValueError("Invalid full-body-clear reset pool")
        self._body_reset_pool = jp.asarray(pool)
        self.body_collision_contract = dict(
            schema="cat-body-collision-v1", shape_counts=proposal["shape_counts"],
            proxy_sha256=hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
            bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
            reset_sha256=hashlib.sha256(reset_path.read_bytes()).hexdigest(),
            cadence_seconds=float(self._config.sim_dt),
            detection="primitive-volume overlap at each physics pose and final integrated pose",
            grace_steps=0, event_penalty=float(settings.event_penalty),
            physical_obstacle_impulses=False, continuous_collision_detection=False,
            observations_added=0,
            reset="native random pose when clear; otherwise sampled certified clear pose in same scene")

    def observation_contract(self):
        contract = super().observation_contract()
        if getattr(self, "body_collision_enabled", False):
            contract["body_collision"] = self.body_collision_contract
            contract["obstacle_physics"] = "fields plus primitive-volume terminal collision checks"
        return contract

    def _body_collision_flags(self, data):
        shapes = body_collisions(self._body_primitives, self._body_collision_bank,
                                 self._field_pf_id, data.xpos, data.xmat,
                                 max_candidates=self.body_collision_metadata["static_candidate_count"])
        return jp.any(self._body_region_mask & shapes[None], axis=-1)

    def _validate_reset_pose(self, rng, qpos):
        if not getattr(self, "body_collision_enabled", False):
            return super()._validate_reset_pose(rng, qpos)
        # Kinematics only: no contact solver or obstacle impulses are introduced.
        data = mjx.kinematics(self.mjx_model, mjx.make_data(self.mjx_model).replace(qpos=qpos))
        invalid = jp.any(self._body_collision_flags(data))
        rng, key = jax.random.split(rng)
        index = jax.random.randint(key, (), 0, self._body_reset_pool.shape[1])
        fallback = self._body_reset_pool[self._field_pf_id, index]
        return rng, jp.where(invalid, fallback, qpos), {"wholebody_reset_replaced": invalid}

    def reset(self, rng):
        return self._initialize_body_collision_info(super().reset(rng))

    def reset_with_pf_id(self, rng, pf_id):
        return self._initialize_body_collision_info(super().reset_with_pf_id(rng, pf_id))

    def _initialize_body_collision_info(self, state):
        if not self.body_collision_enabled:
            return state
        state.info["wholebody_body_collision"] = jp.array(False)
        state.info["wholebody_collision_regions"] = jp.zeros(len(REGIONS), dtype=bool)
        state.metrics["reward/body_collision_event"] = jp.array(0.)
        return state

    def _physics_step(self, rng, data, motor_targets, info):
        if not self.body_collision_enabled:
            return super()._physics_step(rng, data, motor_targets, info)
        from cat_ppo.envs.g1.env_cat import torque_step

        def substep(carry, _):
            key, current, collided = carry
            key, advanced = torque_step(
                key, self.mjx_model, current, motor_targets,
                kps=self._kps, kds=self._kds, kp_scale=info["kp_scale"],
                kd_scale=info["kd_scale"], rfi_lim_scale=info["rfi_lim_scale"],
                torque_limit=self.torque_limit, n_substeps=1)
            # mjx.step exposes FK computed at the start of its integration.
            # These are consecutive true poses 0,2,...,18 ms, not stale info.
            collided |= self._body_collision_flags(advanced)
            return (key, advanced, collided), None

        (rng, data, regions), _ = jax.lax.scan(
            substep, (rng, data, jp.zeros(len(REGIONS), dtype=bool)), None,
            length=self.n_substeps)
        # Include the final 20 ms pose while preserving native CAT's returned
        # derived-data timing for observations, rewards and floor contacts.
        regions |= self._body_collision_flags(mjx.kinematics(self.mjx_model, data))
        info["wholebody_collision_regions"] = regions
        info["wholebody_body_collision"] = jp.any(regions)
        return rng, data

    def step(self, state, action):
        result = super().step(state, action)
        if not self.body_collision_enabled:
            return result
        penalty = -float(self._config.wholebody.body_collision.event_penalty) * result.info["wholebody_body_collision"]
        # Apply after native nonnegative reward clipping and WITHOUT dt scaling.
        # SamplePFWrapper preserves this terminal reward for collision training.
        result.metrics["reward/body_collision_event"] = penalty
        return result.replace(reward=result.reward + penalty)
