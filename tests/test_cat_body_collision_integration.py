"""Actual G1 physics and PPO transitions with the approved body proxies."""
from copy import deepcopy
import hashlib
import json

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np
import pytest

from cat_ppo.envs.g1.body_collision import PROPOSAL, REGIONS
from cat_ppo.envs.g1.env_cat import g1_loco_task_config
from cat_ppo.envs.g1.env_cat_wholebody import G1CatWholeBodyEnv, wholebody_config
from cat_ppo.furniture import body_collision_bank, generalist_fields
from cat_ppo.furniture.generalist_training import wrap_for_cat_wholebody_training
from cat_ppo.furniture.wholebody_stability import EPISODE_KEYS
from cat_ppo.learning.policy.ppo.train import _generate_unroll_with_scene_ids


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_obstacles():
    """One real indexed obstacle in each of two CAT scene slots; no floor."""
    center = np.array([[2., 0., .85]])
    half = np.array([[.30, .50, .60]])
    rotation = np.eye(3)[None]
    proposal = json.loads(PROPOSAL.read_text())
    index = body_collision_bank.build_scene_index(
        center, half, rotation,
        query_radius=body_collision_bank.proxy_bounding_radius(proposal))
    cells = len(index["counts"])
    arrays = dict(
        centers=np.concatenate([np.zeros((1, 3)), center]).astype(np.float32),
        half_sizes=np.concatenate([np.zeros((1, 3)), half]).astype(np.float32),
        rotations=np.concatenate([rotation, rotation]).astype(np.float32),
        scene_box_offsets=np.ones(2, np.int32),
        scene_box_counts=np.ones(2, np.int32),
        scene_grid_origins=np.tile(index["origin"], (2, 1)).astype(np.float32),
        scene_grid_shapes=np.tile(index["shape"], (2, 1)),
        scene_grid_offsets=np.array([0, cells], np.int32),
        cell_starts=np.tile(index["starts"] + 1, 2).astype(np.int32),
        cell_counts=np.tile(index["counts"], 2),
        candidate_ids=np.concatenate([np.zeros(1, np.int32), index["ids"] + 1]),
        cell_size=np.array(.25, np.float32),
    )
    return arrays, dict(static_candidate_count=index["max_candidates"])


@pytest.fixture(scope="module")
def collision_env(tmp_path_factory):
    directory = tmp_path_factory.mktemp("g1-body-collision")
    fields = directory / "fields"
    fields.mkdir()
    shape = (20, 21, 16)
    gf = np.broadcast_to(np.array([.6, 0., 0.], np.float32), (*shape, 3)).copy()
    for name, values in (("gf", gf), ("bf", np.zeros_like(gf)),
                         ("sdf", np.full(shape, 1., np.float32))):
        np.save(fields / f"{name}.npy", values)
    common = dict(path="fields", shape=list(shape), origin=[-2., -2., -.5], dx=.25,
                  start=[0., 0., .8], goal=[2., 0., .75], reset_yaw=0.,
                  sampling_weight=1., family="original_cat", reset_xy_scale=[1., 1.])
    manifest = dict(schema=generalist_fields.SCHEMA, scenes=[dict(common), dict(common)])
    field_manifest = directory / "fields-manifest.json"
    field_manifest.write_text(json.dumps(manifest))
    collision_manifest = directory / "collision-manifest.json"
    collision_manifest.write_text(json.dumps({"fixture": "two-scene-one-obstacle"}))
    arrays, metadata = _tiny_obstacles()

    base = g1_loco_task_config().env_config.copy_and_resolve_references()
    base.term_collision_threshold = 0.
    with pytest.MonkeyPatch.context() as patch:
        # Only disk-manifest validation is stubbed. G1 MJX physics, all approved
        # proxies, spatial lookup, geometric overlap and wrappers remain real.
        patch.setattr(generalist_fields, "load_generalist_manifest", lambda path: manifest)
        config = wholebody_config(base, bank_manifest=field_manifest)
        reference = G1CatWholeBodyEnv(config=config)
        reset_pool = np.tile(np.asarray(reference._init_q), (2, 1, 1))
        reset_pool[:, :, 0] = -1.
        pool_path = directory / "clear-poses.npy"
        np.save(pool_path, reset_pool)
        reset_manifest = directory / "reset-manifest.json"
        reset_manifest.write_text(json.dumps(dict(
            collision_bank_sha256=_hash(collision_manifest), proxy_sha256=_hash(PROPOSAL),
            file=pool_path.name, sha256=_hash(pool_path))))

        def load_bank(path, *, expected_field_manifest, expected_proxy_sha256):
            assert path == collision_manifest
            assert expected_field_manifest == str(field_manifest)
            assert expected_proxy_sha256 == _hash(PROPOSAL)
            return arrays, metadata

        patch.setattr(body_collision_bank, "load_body_collision_bank", load_bank)
        config.wholebody.body_collision.update(
            enabled=True, bank_manifest=str(collision_manifest),
            reset_manifest=str(reset_manifest), event_penalty=2.)
        return G1CatWholeBodyEnv(config=config)


@pytest.fixture(scope="module")
def clear_state(collision_env):
    state = jax.jit(collision_env.reset_with_pf_id)(jax.random.PRNGKey(42), jp.int32(0))
    jax.block_until_ready(state.reward)
    return state


def _at_obstacle(env, state):
    state = deepcopy(state)
    qpos = state.data.qpos.at[:2].set(jp.array([2., 0.]))
    data = mjx.kinematics(env.mjx_model, state.data.replace(qpos=qpos))
    # Match the physics pose's previous-step query history before moving; the
    # terminal event must come from primitive overlap, not a stale velocity.
    state.info["head_pos"] = data.site_xpos[env._head_site_id]
    state.info["feet_pos"] = data.site_xpos[env._feet_site_id]
    state.info["hands_pos"] = data.site_xpos[env._hands_site_id]
    state.info["odom_delay"] = data.qpos[:7]
    return state.replace(data=data)


def test_real_reset_is_clear_and_body_checks_add_no_observation_features(collision_env, clear_state):
    env, state = collision_env, clear_state
    assert env.action_size == 29
    assert state.obs["state"].shape == (222,)
    assert state.obs["privileged_state"].shape == (310,)
    assert set(state.info["wholebody_episode"]) == set(EPISODE_KEYS)
    assert not state.info["wholebody_body_collision"]
    np.testing.assert_array_equal(state.info["wholebody_collision_regions"], False)
    assert not jp.any(jax.jit(env._body_collision_flags)(state.data))
    assert env.observation_contract()["body_collision"]["observations_added"] == 0
    assert env.observation_contract()["body_collision"]["grace_steps"] == 0


def test_first_tick_body_overlap_terminates_and_penalty_survives_native_clip(collision_env, clear_state):
    env = collision_env
    state = _at_obstacle(env, clear_state)
    assert int(state.info["step"]) == 0
    assert bool(env._crossed_goal(state.data.qpos[:3]))
    assert bool(jp.all(env._crossed_goal(state.info["feet_pos"])))
    # Even previously latched goal credit must not survive this collision.
    state.info["wholebody_episode"]["goal_reached"] = jp.array(True)
    result = jax.jit(env.step)(state, jp.zeros(29))
    jax.block_until_ready(result.reward)
    assert result.done == 1.
    assert result.info["wholebody_body_collision"]
    assert result.info["wholebody_faults"]["body_collision"]
    assert result.info["wholebody_faults"]["obstacle"]
    assert not result.info["wholebody_episode"]["goal_reached"]
    for region, event in zip(REGIONS, result.info["wholebody_collision_regions"]):
        assert result.info["wholebody_episode"]["body_collision_" + region] == event
    assert jp.any(result.info["wholebody_collision_regions"])
    event = result.metrics["reward/body_collision_event"]
    assert event == -2.
    native_reward = jp.clip(sum(value for name, value in result.metrics.items()
                               if name.startswith("reward/") and name != "reward/body_collision_event")
                            * env.dt, 0., 10000.)
    np.testing.assert_allclose(result.reward, native_reward - 2., atol=1e-6)
    assert result.reward < 0.
    assert result.obs["state"].shape == (222,)
    assert result.obs["privileged_state"].shape == (310,)


def test_intermediate_physics_collision_is_latched_when_final_pose_is_clear(
        collision_env, clear_state, monkeypatch):
    env = collision_env
    # Control the predicate, not the stepping/latching code: real MJX physics
    # advances through one collision-positive 4 ms sample, then clear samples
    # through 20 ms. Exact shape intersections are exercised separately above.
    trunk = jp.arange(len(REGIONS)) == REGIONS.index("trunk")

    def intermediate_collision(data):
        return trunk & (data.time > .003) & (data.time < .005)

    monkeypatch.setattr(env, "_body_collision_flags", intermediate_collision)
    state = deepcopy(clear_state)
    assert not jp.any(intermediate_collision(state.data))
    # A distinct function forces tracing of the controlled predicate even if
    # another test already compiled the same bound env.step method.
    result = jax.jit(lambda initial: env.step(initial, jp.zeros(29)))(state)
    jax.block_until_ready(result.reward)
    assert float(result.data.time) == pytest.approx(env.dt)
    assert not jp.any(intermediate_collision(result.data))
    assert result.done == 1.
    assert result.info["wholebody_body_collision"]
    np.testing.assert_array_equal(result.info["wholebody_collision_regions"], trunk)
    assert result.info["wholebody_episode"]["body_collision_trunk"]
    assert result.metrics["reward/body_collision_event"] == -2.
    assert result.reward < 0.


def test_real_collision_at_timeout_reaches_ppo_then_resets_without_repeat(collision_env, clear_state):
    env = collision_env
    wrapped = wrap_for_cat_wholebody_training(env)
    key = jax.random.PRNGKey(71)
    batched = jax.jit(wrapped._reset_with_pf_id)(key[None], jp.array([0], jp.int32))
    moved = _at_obstacle(env, clear_state)
    batched = batched.replace(data=jax.tree.map(lambda x: x[None], moved.data))
    batched.info["steps"] = jp.array([999.])
    batched.info["wholebody_episode"]["goal_reached"] = jp.array([True])

    def policy(obs, rng):
        del rng
        return jp.zeros((obs["state"].shape[0], 29)), {}

    def collect(state):
        return _generate_unroll_with_scene_ids(
            wrapped, state, policy, key, 2,
            extra_fields=("truncation", "episode_metrics", "episode_done"))

    final, transitions = jax.jit(collect)(batched)
    jax.block_until_ready(final.reward)
    extra = transitions.extras["state_extras"]
    np.testing.assert_array_equal(extra["episode_done"][:, 0], [1., 0.])
    np.testing.assert_array_equal(extra["truncation"][:, 0], [0., 0.])
    np.testing.assert_array_equal(extra["episode_metrics"]["wb_body_collision"][:, 0], [1., 0.])
    np.testing.assert_array_equal(extra["episode_metrics"]["wb_goal_reached"][:, 0], [0., 0.])
    assert transitions.reward[0, 0] < 0.
    assert transitions.reward[1, 0] >= 0.
    np.testing.assert_array_equal(final.info["wholebody_body_collision"], False)
    np.testing.assert_array_equal(final.info["wholebody_collision_regions"], False)
    assert final.obs["state"].shape == (1, 222)
    assert final.obs["privileged_state"].shape == (1, 310)
