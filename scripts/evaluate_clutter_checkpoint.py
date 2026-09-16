"""Replay the stabilized fixed benchmark and save faithful clutter trajectories.

Uses the frozen training source and RetentionValidator's actual step/termination
implementation. The derived field bank contains byte-identical selected scenes.
No learner state or W&B run is modified.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, help="Optional complete bank for regression evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--test-corrected-navigation", action="store_true",
                        help="Explicitly evaluate old weights under corrected room navigation; not a historical-score replay")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import jax
    import jax.numpy as jp
    import numpy as np
    from ml_collections import ConfigDict
    from cat_ppo.envs.g1.env_cat_wholebody import G1CatWholeBodyEnv, wholebody_config
    from cat_ppo.furniture.learning import load_native
    from cat_ppo.furniture.retention_validation import (
        RetentionValidator, ValidationResult, summarize_episodes, retention_selection,
    )
    from cat_ppo.learning.policy.ppo.wholebody_distribution import (
        make_ppo_networks, checkpoint_distribution_config,
    )

    output = args.evaluation_dir.resolve()
    run = json.loads((output / "run.json").read_text())
    selected = json.loads((output / "checkpoint/selection.json").read_text())
    native = output / "checkpoint/native"
    for name, record in selected["files"].items():
        raw = (native / name).read_bytes()
        if len(raw) != record["size"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Checkpoint mismatch: {name}")
    bank_manifest = (args.bank_manifest or output / "bank/manifest.json").resolve()
    config = wholebody_config(ConfigDict(run["config"]["env_config"]),
                             bank_manifest=bank_manifest, stabilization=True)
    env = G1CatWholeBodyEnv(config=config)
    actual_contract, saved_contract = env.observation_contract(), dict(run["observation_contract"])
    if args.test_corrected_navigation:
        saved_contract["room_navigation"] = "ordered-certified-route-v1"
    if actual_contract != saved_contract:
        raise ValueError("Checkpoint observation contract mismatch")
    network_config = json.loads((native / "ppo_network_config.json").read_text())
    if network_config["normalize_observations"] or checkpoint_distribution_config(network_config) is None:
        raise ValueError("Expected the stabilized unnormalized CAT policy")
    factory = functools.partial(make_ppo_networks, **network_config["network_factory_kwargs"])
    params = jax.tree.map(jp.asarray, load_native(native))
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
        raise ValueError("Nonfinite checkpoint parameters")
    # Keep all 16 original benchmark ordinals and 16 seeds, including CAT
    # retention scenes. Field-bank indices may change; seed ordinals must not.
    validator = RetentionValidator(env, factory, chunk_steps=1, seeds=args.seeds)
    fields = validator.binding.values
    initial = validator._reset(fields)
    jax.block_until_ready(initial)
    n = validator.count
    clutter_rows = [i for i, (scene_index, _) in enumerate(validator.pairs)
                    if env.field_bank_manifest["scenes"][scene_index]["family"] in ("furniture", "generic_clutter")]
    trace_indices = jp.asarray(clutter_rows)

    def advance_chunk(state, acc, active, index, params, stochastic, fields):
        def one(carry, _):
            carry = validator._advance_impl(*carry, params, stochastic, fields)
            state = carry[0]
            trace = {"qpos": state.data.qpos[trace_indices],
                     "qvel": state.data.qvel[trace_indices],
                     "time": state.data.time[trace_indices],
                     "hand_clearance": state.info["handsdf"][trace_indices],
                     "elbow_clearance": state.info["wholebody_elbow_clearance"][trace_indices]}
            if "room_navigation" in state.info:
                trace.update({"route_" + key: value[trace_indices]
                              for key, value in state.info["room_navigation"].items()})
                trace["command"] = state.info["command"][trace_indices]
                trace["command_delay"] = state.info["command_delay"][trace_indices]
            return carry, trace
        return jax.lax.scan(one, (state, acc, active, index), None, length=25)

    advance_chunk = jax.jit(advance_chunk)
    modes, episode_rows, deterministic_traces = {}, {}, None
    started = time.monotonic()
    for mode in ("deterministic", "stochastic"):
        zero, false = jp.zeros(n), jp.zeros(n, dtype=bool)
        acc = dict(length=jp.zeros(n, dtype=jp.int32), **{"return": zero},
                   timeout=false, unexplained_done=false, goal_reached=false,
                   min_hand_clearance=jp.min(initial.info["handsdf"].reshape(n, -1), axis=-1))
        acc.update({key: false for key in ("fall", "obstacle", "self_contact", "numerical", "hand_violation", "elbow_violation")})
        acc["outside_bounds"] = initial.info["wholebody_episode"]["outside_bounds"]
        state, active, index = initial, jp.ones(n, dtype=bool), jp.int32(0)
        chunks = []
        print(json.dumps(dict(event="mode_started", mode=mode, episodes=n,
                              step=selected["step"], devices=[str(d) for d in jax.devices()])), flush=True)
        for chunk in range(math.ceil(validator.max_steps / 25)):
            (state, acc, active, index), trace = advance_chunk(
                state, acc, active, index, params[:2], jp.asarray(mode == "stochastic"), fields)
            if mode == "deterministic":
                chunks.append(jax.device_get(trace))
            remaining = int(jp.sum(active))
            if chunk % 20 == 0 or not remaining:
                print(json.dumps(dict(event="progress", mode=mode, steps=int(index), active=remaining,
                                      seconds=time.monotonic()-started)), flush=True)
            if not remaining:
                break
        arrays = jax.device_get(acc)
        if np.any(arrays["unexplained_done"]) or bool(jp.any(active)):
            raise RuntimeError("Evaluation ended with unexplained/incomplete episodes")
        rows = []
        for row_index, (scene_index, seed) in enumerate(validator.pairs):
            scene = env.field_bank_manifest["scenes"][scene_index]
            row = {key: values[row_index].item() for key, values in arrays.items() if key != "unexplained_done"}
            row.update(scene_id=scene["scene_id"], family=scene["family"], seed=seed,
                       seconds=row["length"]*env.dt, horizon_steps=int(validator.limits[row_index]))
            if not math.isfinite(row["min_hand_clearance"]):
                row["min_hand_clearance"] = -1.0
            if not math.isfinite(row["return"]):
                row["return"] = 0.0
            rows.append(row)
        episode_rows[mode] = rows
        modes[mode] = summarize_episodes(rows)
        if mode == "deterministic":
            deterministic_traces = {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}
        print(json.dumps(dict(event="mode_finished", mode=mode,
                              clutter=modes[mode]["clutter_goal_success_rate"], cat=modes[mode]["cat_goal_success_rate"])), flush=True)

    baseline = json.loads((output / "validation_baseline.json").read_text())["result"]
    if args.test_corrected_navigation or args.seeds != list(range(16)):
        selection = dict(eligible=False, reason="Changed navigation or seed set: diagnostic regression only; no historical selection comparison")
    else:
        selection = retention_selection(modes, baseline["modes"])
    metadata = dict(source_root=str(args.source_root), checkpoint_step=selected["step"],
                    original_bank_sha256=run["bank_sha256"],
                    evaluation_bank_sha256=hashlib.sha256(bank_manifest.read_bytes()).hexdigest(),
                    evaluation_bank=str(bank_manifest), seeds=args.seeds,
                    corrected_navigation_regression=args.test_corrected_navigation,
                    scenes="Fixed 16 layouts; explicitly listed reset/noise seeds",
                    stopping="Shared RetentionValidator implementation; first clean goal/fault/native horizon",
                    generalization="Fixed training-bank regression scenes, not unseen layouts",
                    recording="All 64 deterministic clutter episodes; videos selected after evaluation",
                    devices=[str(d) for d in jax.devices()], evaluation_seconds=time.monotonic()-started)
    result = ValidationResult(selected["step"], {}, modes, episode_rows, selection, metadata).as_dict()
    (output / "evaluation.json").write_text(json.dumps(result, indent=2)+"\n")

    # Reuse exactly the robot XML used by evaluation; furniture is display-only,
    # because CAT training queries fields instead of obstacle contact dynamics.
    from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml
    xml_template = assemble_training_xml()
    episode_root = output / "episodes"
    episode_root.mkdir(exist_ok=True)
    palette = {"tabletop": ".72 .49 .27 1", "table_leg": ".40 .30 .22 1",
               "chair_seat": ".23 .47 .53 1", "chair_back": ".20 .42 .48 1",
               "chair_leg": ".20 .26 .30 1", "chair_armrest": ".20 .36 .40 1",
               "wall": ".62 .66 .70 1", "crate": ".68 .47 .29 1",
               "partition": ".43 .54 .66 1", "shelf_edge": ".47 .58 .42 1"}
    for trace_row, full_row in enumerate(clutter_rows):
        scene_index, seed = validator.pairs[full_row]
        record = env.field_bank_manifest["scenes"][scene_index]
        scene_path = bank_manifest.parent / record["path"] / "scene.json"
        scene = json.loads(scene_path.read_text())
        row = episode_rows["deterministic"][full_row]
        directory = episode_root / f"scene-{scene['seed']}-seed-{seed:02d}"
        directory.mkdir(exist_ok=False)
        xml = ET.fromstring(xml_template)
        world = xml.find("worldbody")
        for i, part in enumerate(scene["boxes"]):
            yaw = part["yaw"]
            ET.SubElement(world, "geom", name=f"clutter_box_{i}", type="box",
                          pos=" ".join(map(str, part["center"])), size=" ".join(map(str, part["half_size"])),
                          quat=f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}",
                          rgba=palette.get(part["category"], ".60 .40 .35 1"),
                          contype="0", conaffinity="0", group="4" if part["category"] == "wall" else "0")
        (directory / "model.xml").write_text(ET.tostring(xml, encoding="unicode"))
        (directory / "scene.json").write_bytes(scene_path.read_bytes())
        length = row["length"]
        trace = {}
        for key in ("qpos", "qvel", "time"):
            initial_value = np.asarray(getattr(initial.data, key))[full_row]
            trace[key] = np.concatenate([initial_value[None], deterministic_traces[key][:length, trace_row]], axis=0)
        if not np.isfinite(trace["qpos"]).all() or not np.isfinite(trace["qvel"]).all():
            raise ValueError("Cannot render nonfinite trajectory")
        np.savez_compressed(directory / "trajectory.npz", **trace)
        np.savez_compressed(directory / "telemetry.npz", **{
            key: deterministic_traces[key][:length, trace_row]
            for key in deterministic_traces if key not in ("qpos", "qvel", "time")})
        meta = dict(scene_id=record["scene_id"], scene_seed=scene["seed"], family=record["family"], seed=seed,
                    checkpoint_steps=selected["step"], outcome=row, dt=env.dt, inference="deterministic",
                    source_run="de6ae369", original_bank_sha256=run["bank_sha256"],
                    corrected_navigation_regression=args.test_corrected_navigation,
                    evaluation_bank_sha256=metadata["evaluation_bank_sha256"],
                    scene_sha256=record["scene_sha256"],
                    trajectory_sha256=hashlib.sha256((directory/"trajectory.npz").read_bytes()).hexdigest(),
                    reset="Exact validation reset key using full benchmark ordinal; native randomization retained",
                    geometry="Actual primitive training scene; display-only obstacle geoms")
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(dict(event="finished", output=str(output), episodes_recorded=len(clutter_rows),
                          seconds=time.monotonic()-started)), flush=True)


if __name__ == "__main__":
    main()
