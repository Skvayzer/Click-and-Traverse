"""Paired native-CAT evaluations isolating upper-body action exploration.

This standalone diagnostic does not modify the learner, source task, reward,
reset distribution, observation contract, or checkpoint. Finished episodes are
retained without autoreset. Noise modes share reset and action-noise streams.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def validate_snapshot_contract(snapshot):
    """This legacy diagnostic must never replace repaired policy semantics."""
    contract = snapshot.get("contract", {})
    if contract.get("distribution") is not None:
        raise ValueError(
            "This diagnostic targets the original unbounded run. Bounded-policy snapshots "
            "require RetentionValidator with the saved network factory and environment setup."
        )
    if snapshot.get("normalization", False) or contract.get("normalize_observations", False):
        raise ValueError("This diagnostic requires an unnormalized CAT policy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-ids", nargs="+", required=True,
                        help="Integer bank indices or exact scene_id strings")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Explicit seeds; otherwise range(--seed-count)")
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--controllers", nargs="+", default=["current", "initial_wholebody"],
                        choices=["current", "initial_wholebody"])
    parser.add_argument("--modes", nargs="+", default=["full", "upper_deterministic", "deterministic", "upper_cap_0.1", "upper_cap_0.05"])
    parser.add_argument("--baseline-modes", nargs="+", default=["deterministic"])
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--chunk-steps", type=int, default=25)
    args = parser.parse_args()
    if args.max_steps < 1 or args.chunk_steps < 1:
        parser.error("Step counts must be positive")
    args.source_root = args.source_root.resolve()
    sys.path.insert(0, str(args.source_root))

    import jax
    import jax.numpy as jp
    import numpy as np
    from flax import serialization
    from brax.training.agents.ppo import networks
    from brax.training.acme import running_statistics, specs
    from ml_collections import ConfigDict
    from mujoco_playground._src import collision
    from cat_ppo.envs.g1.env_cat_wholebody import G1CatWholeBodyEnv, wholebody_config
    from cat_ppo.furniture.generalist_config import released_config
    from cat_ppo.furniture.learning import load_native, adapt_native_params
    from cat_ppo.furniture.control import legacy_observation_contract

    cause_names = ["fall_inverted", "fall_head_low", "self_contact", "head_field",
                   "pelvis_field", "torso_field", "feet_field", "hands_field",
                   "knees_field", "shoulders_field", "elbows_field", "numerical_nan"]

    class DiagnosticEnv(G1CatWholeBodyEnv):
        def _outside_traversal_bounds(self, data, info):
            scene = self._field_pf_id
            origin = self._pf_origins[scene]
            extent = origin + (self._pf_shapes[scene] - 1) * self._pf_dxs[scene]
            root = data.qpos[:3]
            points = jp.concatenate([root[None, :], info["feet_pos"]], axis=0)
            outside_lateral = jp.any((points[:, 1] < origin[1]) | (points[:, 1] > extent[1]))
            outside_xy = jp.any((root[:2] < origin[:2]) | (root[:2] > extent[:2]))
            # Native CAT starts may lie before the field's minimum x; those
            # legitimate approaches must not be mistaken for lateral bypasses.
            return jp.where(self._pf_crossed_is_x_plane[scene], outside_lateral, outside_xy)

        def _get_termination(self, data, info):
            # Instrument precisely where native step computes termination, before
            # it resets info['step'] to zero on terminal/timeout transitions.
            native = super()._get_termination(data, info)
            grace = info["step"] >= 50
            threshold = -self._config.term_collision_threshold
            self_contact = (
                collision.geoms_colliding(data, self._right_foot_geom_id, self._left_foot_geom_id)
                | collision.geoms_colliding(data, self._left_foot_geom_id, self._right_shin_geom_id)
                | collision.geoms_colliding(data, self._right_foot_geom_id, self._left_shin_geom_id))
            fields = [jp.any(info[key] < threshold) & grace for key in
                      ("headdf", "pelvdf", "torsdf", "feetdf", "handsdf", "kneesdf", "shldsdf")]
            causes = jp.stack([
                self.get_gravity(data, "pelvis")[2] < 0., info["head_pos"][2] < .7,
                self_contact & grace, *fields,
                jp.any(info["wholebody_elbow_clearance"] < threshold) & grace
                & self._config.terminate_on_elbow_collision,
                jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any(),
            ])
            info["diagnostic_causes"] = causes
            info["diagnostic_cause_mismatch"] = native != jp.any(causes)
            info["diagnostic_step"] = info["step"]
            scene = self._field_pf_id
            origin = self._pf_origins[scene]
            extent = origin + (self._pf_shapes[scene] - 1) * self._pf_dxs[scene]
            root = data.qpos[:3]
            within_xy = jp.all((root[:2] >= origin[:2]) & (root[:2] <= extent[:2]))
            root_goal = self._crossed_goal(root)
            feet_goal = jp.all(self._crossed_goal(info["feet_pos"]))
            info["diagnostic_root_goal"] = root_goal & within_xy
            info["diagnostic_body_goal"] = root_goal & feet_goal & within_xy
            info["diagnostic_outside_xy"] = ~within_xy
            info["diagnostic_outside_traversal_bounds"] = self._outside_traversal_bounds(data, info)
            info["diagnostic_upper_targets"] = info["motor_targets"][12:]
            return native

        def reset_with_pf_id(self, rng, pf_id):
            state = super().reset_with_pf_id(rng, pf_id)
            state.info.update(
                diagnostic_causes=jp.zeros(len(cause_names), dtype=bool),
                diagnostic_cause_mismatch=jp.array(False),
                diagnostic_step=jp.array(0, dtype=jp.int32),
                diagnostic_root_goal=jp.array(False),
                diagnostic_body_goal=jp.array(False),
                diagnostic_outside_xy=jp.array(False),
                diagnostic_outside_traversal_bounds=self._outside_traversal_bounds(state.data, state.info),
                diagnostic_upper_targets=state.info["motor_targets"][12:])
            return state

    snapshot = json.loads((args.snapshot_dir / "snapshot.json").read_text())
    validate_snapshot_contract(snapshot)
    payload = (args.snapshot_dir / "inference.msgpack").read_bytes()
    if hashlib.sha256(payload).hexdigest() != snapshot["inference_sha256"]:
        raise ValueError("Checkpoint inference hash differs from snapshot metadata")
    trained = serialization.msgpack_restore(payload)
    del payload
    released = released_config()
    config = wholebody_config(ConfigDict(released["env_config"]), bank_manifest=args.bank_manifest)
    env = DiagnosticEnv(config=config)
    scenes = env.field_bank_manifest["scenes"]
    scene_lookup = {scene["scene_id"]: i for i, scene in enumerate(scenes)}
    selected = [scene_lookup[value] if value in scene_lookup else int(value) for value in args.scene_ids]
    if any(index < 0 or index >= len(scenes) for index in selected) or len(set(selected)) != len(selected):
        raise ValueError("Scene IDs must be distinct valid bank indices")
    seeds = args.seeds if args.seeds is not None else list(range(args.seed_count))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and distinct")
    pairs = [(scene, seed) for scene in selected for seed in seeds]
    scene_ids = jp.asarray([scene for scene, _ in pairs], dtype=jp.int32)
    reset_keys = jp.stack([jax.random.fold_in(jax.random.PRNGKey(seed), scene) for scene, seed in pairs])
    # A separate stream is independent of sensor, physics and push randomness.
    noise_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0xCA7AB1))(reset_keys)
    count = len(pairs)
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(event="building_reset", episodes=count, devices=[str(device) for device in jax.devices()])), flush=True)
    reset = jax.jit(jax.vmap(env.reset_with_pf_id))
    initial = reset(reset_keys, scene_ids)
    initial.data.qpos.block_until_ready()
    obs_shapes = {key: value.shape[1:] for key, value in initial.obs.items()}
    net_config = released["policy_config"]["network_factory"]
    net = networks.make_ppo_networks(
        obs_shapes, env.action_size,
        policy_hidden_layer_sizes=tuple(net_config["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(net_config["value_hidden_layer_sizes"]),
        policy_obs_key="state", value_obs_key="privileged_state")
    template = net.policy_network.init(jax.random.PRNGKey(0))
    actors = {"current": trained["policy"]}
    if "initial_wholebody" in args.controllers:
        source = load_native(args.source_root / "data/furniture/native_generalist_v1")
        normalizer = running_statistics.init_state({key: specs.Array(shape, jp.dtype("float32")) for key, shape in obs_shapes.items()})
        expanded, adaptation = adapt_native_params(
            source, (normalizer, template, net.value_network.init(jax.random.PRNGKey(1))),
            legacy_observation_contract(), env.observation_contract(), new_action_std=.05)
        actors["initial_wholebody"] = expanded[1]
    for name, actor in actors.items():
        if jax.tree.structure(actor) != jax.tree.structure(template):
            raise ValueError(f"{name}: network tree mismatch")
        for actual, expected in zip(jax.tree.leaves(actor), jax.tree.leaves(template)):
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError(f"{name}: network shape/nonfinite mismatch")
        actors[name] = jax.tree.map(jp.asarray, actor)

    # Check the intervention itself on identical initial observations/noise.
    initial_distribution = net.parametric_action_distribution.create_dist(
        net.policy_network.apply(None, actors["current"], initial.obs))
    first_epsilon = jax.vmap(lambda key: jax.random.normal(
        jax.random.fold_in(key, 0), (env.action_size,)))(noise_keys)
    initial_full_action = net.parametric_action_distribution.postprocess(
        initial_distribution.loc + initial_distribution.scale * first_epsilon)
    initial_upper_off_action = net.parametric_action_distribution.postprocess(
        initial_distribution.loc + initial_distribution.scale.at[:, 12:].set(0.) * first_epsilon)
    parity = dict(
        full_vs_upper_deterministic_leg_max_error=float(jp.max(jp.abs(initial_full_action[:, :12] - initial_upper_off_action[:, :12]))),
        upper_deterministic_vs_mean_max_error=float(jp.max(jp.abs(initial_upper_off_action[:, 12:] - jp.tanh(initial_distribution.loc[:, 12:])))),
    )
    if any(error > 1e-7 for error in parity.values()):
        raise RuntimeError(f"Noise intervention parity failed: {parity}")

    def mode_parameters(mode):
        if mode == "full":
            return 1., float("inf"), 1.
        if mode == "upper_deterministic":
            return 1., 0., 1.
        if mode == "deterministic":
            return 0., 0., 1.
        if mode == "upper_nominal":
            return 0., 0., 0.
        if mode.startswith("upper_cap_"):
            cap = float(mode.removeprefix("upper_cap_"))
            if not np.isfinite(cap) or cap < 0:
                raise ValueError("Noise cap must be finite and nonnegative")
            return 1., cap, 1.
        raise ValueError(f"Unknown mode: {mode}")

    limits = jp.minimum(env._pf_scene_episode_lengths[scene_ids], args.max_steps)
    initial_xy = initial.data.qpos[:, :2]
    goals_xy = env._pf_scene_goals[scene_ids, :2]
    initial_goal_distance = jp.linalg.norm(initial_xy - goals_xy, axis=-1)

    def initial_metrics():
        zero = jp.zeros(count)
        return dict(
            length=jp.zeros(count, dtype=jp.int32), returns=zero,
            terminal=jp.zeros(count, dtype=bool), timeout=jp.zeros(count, dtype=bool),
            horizon=jp.zeros(count, dtype=bool), causes=jp.zeros((count, len(cause_names)), dtype=bool),
            cause_mismatch=jp.zeros(count, dtype=bool),
            root_goal=jp.zeros(count, dtype=bool), body_goal=jp.zeros(count, dtype=bool), raw_body_goal=jp.zeros(count, dtype=bool),
            goal_first_step=jp.full(count, -1, dtype=jp.int32), outside_xy=jp.zeros(count, dtype=bool),
            outside_traversal_bounds=initial.info["diagnostic_outside_traversal_bounds"],
            min_hand_clearance=jp.min(initial.info["handsdf"].reshape(count, -1), axis=-1),
            min_elbow_clearance=jp.min(initial.info["wholebody_elbow_clearance"], axis=-1),
            min_head_height=initial.info["head_pos"][:, 2],
            max_root_x=initial_xy[:, 0], max_goal_distance_reduction=zero,
            leg_std_sum=zero, upper_std_sum=zero, applied_upper_std_sum=zero,
            applied_upper_action_abs_sum=zero,
            upper_target_offset_square_sum=zero,
            upper_target_step_square_sum=zero,
            upper_joint_velocity_square_sum=zero,
        )

    def rollout_chunk(state, metrics, active, time_index, actor, leg_noise, upper_cap, upper_mean):
        def advance(carry, _):
            state, metrics, active, index = carry
            logits = net.policy_network.apply(None, actor, state.obs)
            distribution = net.parametric_action_distribution.create_dist(logits)
            loc, scale = distribution.loc, distribution.scale
            loc = loc.at[:, 12:].multiply(upper_mean)
            keys = jax.vmap(lambda key: jax.random.fold_in(key, index))(noise_keys)
            epsilon = jax.vmap(lambda key: jax.random.normal(key, (env.action_size,)))(keys)
            applied_scale = scale.at[:, :12].multiply(leg_noise)
            applied_scale = applied_scale.at[:, 12:].set(jp.minimum(scale[:, 12:], upper_cap))
            action = net.parametric_action_distribution.postprocess(loc + applied_scale * epsilon)
            previous_upper_targets = state.info["motor_targets"][:, 12:]
            stepped = jax.vmap(env.step)(state, action)
            # Keep every leaf of finished episodes fixed; never reset them.
            state = jax.tree.map(lambda old, new: jp.where(
                active.reshape((count,) + (1,) * (new.ndim - 1)), new, old), state, stepped)
            done = active & (state.done > 0)
            length = metrics["length"] + active.astype(jp.int32)
            expired = active & (length >= limits) & ~done
            native_timeout = expired & (length >= env._pf_scene_episode_lengths[scene_ids])
            horizon = expired & ~native_timeout
            clean = active & ~done
            raw_body_goal = clean & state.info["diagnostic_body_goal"]
            outside_traversal = metrics["outside_traversal_bounds"] | (active & state.info["diagnostic_outside_traversal_bounds"])
            new_body_goal = raw_body_goal & ~outside_traversal
            first_goal = new_body_goal & ~metrics["body_goal"]
            root_xy = state.data.qpos[:, :2]
            distance_reduction = initial_goal_distance - jp.linalg.norm(root_xy - goals_xy, axis=-1)
            metrics = dict(
                length=length, returns=metrics["returns"] + jp.where(active, state.reward, 0.),
                terminal=metrics["terminal"] | done,
                timeout=metrics["timeout"] | native_timeout,
                horizon=metrics["horizon"] | horizon,
                causes=metrics["causes"] | (done[:, None] & state.info["diagnostic_causes"]),
                cause_mismatch=metrics["cause_mismatch"] | (active & state.info["diagnostic_cause_mismatch"]),
                root_goal=metrics["root_goal"] | (clean & state.info["diagnostic_root_goal"]),
                body_goal=metrics["body_goal"] | new_body_goal,
                raw_body_goal=metrics["raw_body_goal"] | raw_body_goal,
                goal_first_step=jp.where(first_goal, length, metrics["goal_first_step"]),
                outside_xy=metrics["outside_xy"] | (active & state.info["diagnostic_outside_xy"]),
                outside_traversal_bounds=outside_traversal,
                min_hand_clearance=jp.minimum(metrics["min_hand_clearance"], jp.where(active, jp.min(state.info["handsdf"].reshape(count, -1), axis=-1), jp.inf)),
                min_elbow_clearance=jp.minimum(metrics["min_elbow_clearance"], jp.where(active, jp.min(state.info["wholebody_elbow_clearance"], axis=-1), jp.inf)),
                min_head_height=jp.minimum(metrics["min_head_height"], jp.where(active, state.info["head_pos"][:, 2], jp.inf)),
                max_root_x=jp.maximum(metrics["max_root_x"], jp.where(active, root_xy[:, 0], -jp.inf)),
                max_goal_distance_reduction=jp.maximum(metrics["max_goal_distance_reduction"], jp.where(active, distance_reduction, -jp.inf)),
                leg_std_sum=metrics["leg_std_sum"] + jp.where(active, jp.mean(scale[:, :12], axis=-1), 0.),
                upper_std_sum=metrics["upper_std_sum"] + jp.where(active, jp.mean(scale[:, 12:], axis=-1), 0.),
                applied_upper_std_sum=metrics["applied_upper_std_sum"] + jp.where(active, jp.mean(applied_scale[:, 12:], axis=-1), 0.),
                applied_upper_action_abs_sum=metrics["applied_upper_action_abs_sum"] + jp.where(active, jp.mean(jp.abs(action[:, 12:]), axis=-1), 0.),
                upper_target_offset_square_sum=metrics["upper_target_offset_square_sum"] + jp.where(active, jp.mean((state.info["diagnostic_upper_targets"] - env._default_qpos[None, 12:]) ** 2, axis=-1), 0.),
                upper_target_step_square_sum=metrics["upper_target_step_square_sum"] + jp.where(active, jp.mean((state.info["diagnostic_upper_targets"] - previous_upper_targets) ** 2, axis=-1), 0.),
                upper_joint_velocity_square_sum=metrics["upper_joint_velocity_square_sum"] + jp.where(active, jp.mean(state.data.qvel[:, 18:35] ** 2, axis=-1), 0.),
            )
            return (state, metrics, active & ~done & ~expired, index + 1), None
        return jax.lax.scan(advance, (state, metrics, active, time_index), xs=None, length=args.chunk_steps)[0]

    advance_chunk = jax.jit(rollout_chunk)
    metadata = dict(
        schema="cat-noise-ablation-v1", snapshot=snapshot,
        source_root=str(args.source_root), bank_manifest=str(args.bank_manifest.resolve()),
        bank_manifest_sha256=env.field_bank_manifest["manifest_sha256"],
        scene_ids=selected, seeds=seeds, observation_shapes=obs_shapes,
        normalization=False, environment_config=config.to_dict(),
        intervention_parity=parity,
        noise_matching="Identical 29-value standard-normal draws per scene/seed/step, independent of environment randomization; modes change scale only before native tanh postprocess.",
        upper_nominal="Separate posture diagnostic: deterministic learned legs, upper actions forced to0 so native target returns toward default pose. Changes mean actions; not a pure exploration ablation.",
        reset="Native reset_with_pf_id; identical seeded initial states; fresh episode clock; no training wrapper phase staggering or autoreset.",
        goal_definition="Diagnostic clean crossing: root and both feet satisfy native _crossed_goal and root is inside field XY bounds; CAT x>1.5, rooms each position within0.5m of goal. Valid body_goal additionally requires no previous/current root-or-feet lateral field exit forCAT, no rootXY field exit forrooms. InitialCAT x<-0.5approaches are allowed. raw_body_goal omits priorpath restriction. Not a native terminal or timeout success.",
        termination="Exact native fall, field-threshold/self-contact (50-step grace), elbow and NaN rules; simultaneous causes retained. Evaluation horizon is distinct from native timeout.",
        reward="Raw native step reward, including first and terminal transitions; not training logger's completed-episode accounting.",
        max_steps=args.max_steps, chunk_steps=args.chunk_steps,
        initial_wholebody="Original CAT expanded to222 observations/29actions with zero added input weights and upper std0.05; compact hand spheres/elbows retained, so not original12actionrobot benchmark.",
        devices=[str(device) for device in jax.devices()],
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []

    def clean_json(value):
        if isinstance(value, dict):
            return {key: clean_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean_json(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def summarize(rows):
        terminal = [row for row in rows if row["terminal"]]
        return dict(episodes=len(rows), mean_length=float(np.mean([row["length"] for row in rows])),
                    mean_return=float(np.mean([row["returns"] for row in rows])),
                    native_terminations=len(terminal), native_timeouts=sum(row["timeout"] for row in rows),
                    horizon_censored=sum(row["horizon"] for row in rows),
                    body_goal_crossings=sum(row["body_goal"] for row in rows),
                    raw_body_goal_crossings=sum(row["raw_body_goal"] for row in rows),
                    falls=sum(row["fall"] for row in rows), obstacle_field=sum(row["obstacle_field"] for row in rows),
                    self_contacts=sum(row["self_contact"] for row in rows),
                    fall_only=sum(row["fall"] and not row["collision"] for row in rows),
                    collision_only=sum(row["collision"] and not row["fall"] for row in rows),
                    fall_and_collision=sum(row["fall"] and row["collision"] for row in rows),
                    mean_goal_distance_reduction=float(np.mean([row["goal_distance_reduction"] for row in rows])))

    for controller in args.controllers:
        for mode in args.modes if controller == "current" else args.baseline_modes:
            leg_noise, upper_cap, upper_mean = mode_parameters(mode)
            state, metrics, active = initial, initial_metrics(), jp.ones(count, dtype=bool)
            index = jp.array(0, dtype=jp.int32)
            mode_started = time.monotonic()
            print(json.dumps(dict(event="mode_start", controller=controller, mode=mode, episodes=count)), flush=True)
            while True:
                state, metrics, active, index = advance_chunk(
                    state, metrics, active, index, actors[controller], jp.asarray(leg_noise, dtype=jp.float32), jp.asarray(upper_cap, dtype=jp.float32), jp.asarray(upper_mean, dtype=jp.float32))
                active_host, index_host = jax.device_get((active, index))
                if int(index_host) % 100 < args.chunk_steps or not active_host.any():
                    print(json.dumps(dict(event="progress", controller=controller, mode=mode,
                                          steps=int(index_host), active=int(active_host.sum()), wall_seconds=time.monotonic()-mode_started)), flush=True)
                if not active_host.any():
                    break
            measured, final_qpos = jax.device_get((metrics, state.data.qpos))
            if measured["cause_mismatch"].any():
                raise RuntimeError("Instrumentation differs from native termination; discard diagnostic")
            final_distance = np.linalg.norm(final_qpos[:, :2] - np.asarray(goals_xy), axis=-1)
            mode_rows = []
            for row_index, (scene_index, seed) in enumerate(pairs):
                row = {key: value[row_index].tolist() for key, value in measured.items() if key != "causes"}
                causes = {name: bool(measured["causes"][row_index, i]) for i, name in enumerate(cause_names)}
                row.update(controller=controller, mode=mode, scene_index=scene_index,
                           scene_id=scenes[scene_index]["scene_id"], family=scenes[scene_index]["family"], seed=seed,
                           causes=causes, fall=causes["fall_inverted"] or causes["fall_head_low"],
                           obstacle_field=any(causes[name] for name in cause_names[3:11]),
                           self_contact=causes["self_contact"],
                           collision=any(causes[name] for name in cause_names[2:11]),
                           final_root=final_qpos[row_index, :3].tolist(),
                           initial_root=np.asarray(initial.data.qpos[row_index, :3]).tolist(),
                           goal_distance_reduction=float(np.asarray(initial_goal_distance)[row_index] - final_distance[row_index]),
                           root_x_progress=float(final_qpos[row_index, 0] - np.asarray(initial_xy)[row_index, 0]))
                for key in ("leg_std", "upper_std", "applied_upper_std", "applied_upper_action_abs"):
                    row["mean_" + key] = row.pop(key + "_sum") / max(row["length"], 1)
                for key in ("upper_target_offset", "upper_target_step", "upper_joint_velocity"):
                    row["rms_" + key] = float(np.sqrt(row.pop(key + "_square_sum") / max(row["length"], 1)))
                mode_rows.append(row)
            results.extend(mode_rows)
            summary = summarize(mode_rows)
            print(json.dumps(clean_json(dict(event="mode_complete", controller=controller, mode=mode,
                                             wall_seconds=time.monotonic()-mode_started, **summary))), flush=True)
            (args.output_dir / "episodes.json").write_text(json.dumps(clean_json(results), indent=2, allow_nan=False) + "\n")
            summaries = []
            for label in sorted({(row["controller"], row["mode"]) for row in results}):
                selected_rows = [row for row in results if (row["controller"], row["mode"]) == label]
                summaries.append(dict(controller=label[0], mode=label[1], overall=summarize(selected_rows),
                                      families={family: summarize([row for row in selected_rows if row["family"] == family])
                                                for family in sorted({row["family"] for row in selected_rows})}))
            (args.output_dir / "summary.json").write_text(json.dumps(clean_json(summaries), indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="complete", wall_seconds=time.monotonic()-started, output=str(args.output_dir))), flush=True)


if __name__ == "__main__":
    main()
