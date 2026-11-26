#!/usr/bin/env python3
"""C3: COS scenes as harness scenes (fields bank) + full-path plans (docs/cos_planner_reflex.md, C3).

  python scripts/cos_bank.py select --manifest /l/.../scenes_v2/manifest.json --splits /l/.../splits_v2 --split train \
      --per-family 34 --out /l/.../cos/c3/train_scenes.jsonl
  python scripts/cos_bank.py build --base-manifest data/furniture/tablegap_v1_packed/manifest.json \
      --scenes /l/.../cos/c3/train_scenes.jsonl --output /l/.../cos/c3/bank_train_packed --plans /l/.../cos/c3/plans_train

Each COS scene (chord frame: start pelvis at the origin, x toward the window end) becomes a CAT-style scene: its boxes
(levelled, all roles; heights capped at 2.0 m) plus a perimeter, start (0, 0, heading 0), goal = the reference's final
pelvis xy, fields by the standard make_clutter_fields pipeline, runtime frame = the chord frame (like the planning
benchmark's runtime shift). The plan is the SONIC-tracked rollout (physical posture, re-anchored path) when the scene is
feasible and clear, else the levelled M; plan npz = {qpos, modes, clips, meta(json: scene, ok)}.
The output is then run through build_body_collision_bank.py, build_body_collision_resets.py and
build_goal_seed_cache.py exactly as hpc/overnight_build_planning.sbatch does.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, shutil, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import cat_mjlab  # noqa: F401,E402  (installs CAT_PATH_MAP before the pinned bank is read)
FAMS = ("duck", "crawl", "high_step", "sideways", "arms_in", "arms_tucked", "arms_raised", "detour", "multi")
ZCAP = 2.0


def select(a):
    from cos_splits import take_of
    man = json.load(open(a.manifest)); takes = json.load(open(Path(a.splits) / "take_split.json"))
    rng = random.Random(a.seed); pool = collections.defaultdict(list)
    for folder in man["scene_folders"]:
        for l in open(Path(folder) / "scenes.jsonl"):
            s = json.loads(l)
            if not s.get("verified") or takes[take_of(s["clip"])] != a.split:
                continue
            if not (s.get("tracked") and s.get("tracked_clear") is not None and s["tracked_clear"] >= a.min_tracked_clear):
                continue
            pool[s["family"]].append(dict(scene=s["scene"], window=s["window"], take=take_of(s["clip"]), family=s["family"],
                                          source=s["source"], folder=folder))
    out = []
    for f in FAMS:
        rng.shuffle(pool[f]); seen = set(); got = []
        for c in pool[f]:
            if c["window"] in seen:
                continue
            got.append(c); seen.add(c["window"])
            if len(got) >= a.per_family:
                break
        out += got
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fo:
        for c in out:
            fo.write(json.dumps(c) + "\n")
    print(len(out), collections.Counter(c["family"] for c in out))


def scene_dict(s, ref):
    from cat_ppo.furniture.scenes import SCHEMA, _box, _digest
    boxes = []
    for i, b in enumerate(s["boxes"]):
        c = list(b["center"]); h = list(b["half"])
        top = min(c[2] + h[2], ZCAP); bot = c[2] - h[2]
        if top - bot < 0.02:
            continue
        c[2] = (top + bot) / 2; h[2] = (top - bot) / 2
        boxes.append(dict(name=f"{b['role']}_{i}", center=c, half_size=h, yaw=float(b.get("yaw", 0.)),
                          category="wall" if b["role"] == "wall" else "generic", cos_role=b["role"], cos_kind=b.get("kind", b["role"])))
    goal = [float(ref[-1, 0]), float(ref[-1, 1])]
    pts = np.array([[*b["center"][:2]] for b in boxes] + [[0, 0], goal])
    ext = np.array([max(b["half_size"][0], b["half_size"][1]) * 1.42 for b in boxes] + [0, 0])
    lo = np.minimum((pts - ext[:, None]).min(0), [-1.0, -1.0]) - 0.3
    hi = np.maximum((pts + ext[:, None]).max(0), [goal[0] + 1.0, 1.0]) + 0.3
    lo[0] = max(lo[0], -1.5); hi[0] = min(hi[0], goal[0] + 1.5)          # the room ends 1.5 m behind start / past goal
    shift = -lo                                                            # room frame = chord frame + shift
    dims = [float(hi[0] - lo[0]), float(hi[1] - lo[1]), ZCAP]
    for b in boxes:
        b["center"] = [b["center"][0] + shift[0], b["center"][1] + shift[1], b["center"][2]]
    x, y, z = dims
    boxes += [_box("wall_west", [0, y / 2, z / 2], [.06, y / 2, z / 2], "wall"),
              _box("wall_east", [x, y / 2, z / 2], [.06, y / 2, z / 2], "wall"),
              _box("wall_south", [x / 2, 0, z / 2], [x / 2, .06, z / 2], "wall"),
              _box("wall_north", [x / 2, y, z / 2], [x / 2, .06, z / 2], "wall")]
    start = [float(shift[0]), float(shift[1]), 0.]
    g = [goal[0] + shift[0], goal[1] + shift[1]]
    L = float(np.linalg.norm(np.diff(ref[:, :2], axis=0), axis=1).sum())
    digest = _digest(dict(boxes=boxes, room_dimensions=dims))
    return dict(schema=SCHEMA, scene_id=f"cos-{s['family']}-{digest[:12]}", family="generic_clutter", split="train",
                units="metres", coordinate_system="right-handed-z-up", boxes=boxes, room_dimensions=dims,
                geometry_hash=digest, start=start, goal=g, route=[start[:2], g], time_budget=max(20., L / 0.12 + 8.),
                bottlenecks=[], counts=dict(tables=0, chairs=0, generic_objects=len(boxes), primitive_boxes=len(boxes),
                                            bottlenecks=0),
                feasibility=dict(method="COS verified human motion (synth.verify) + SONIC tracked", root_route_validated=True),
                cos=dict(scene=s["scene"], family=s["family"], source=s["source"], window=s["window"], path_m=L)), shift


def write_case(job):
    from cat_mjlab.packing.field_packing import pack_direction
    from cat_ppo.furniture.generalist_fields import _field_records, make_clutter_fields, sha256
    c, output, plans = job
    s = next(json.loads(l) for l in open(Path(c["folder"]) / "scenes.jsonl") if json.loads(l)["scene"] == c["scene"]) \
        if "boxes" not in c else c
    mot = np.load(Path(c["folder"]) / "motions" / f"{s['window']}.npz")
    use_tracked = "tracked" in mot.files and s.get("tracked_clear") is not None and s["tracked_clear"] >= 0.0
    ref = mot["tracked"].astype(np.float64) if use_tracked else mot["M"].astype(np.float64)
    scene, shift = scene_dict(s, ref)
    relative = Path("scenes") / scene["scene_id"]; directory = output / relative
    record = make_clutter_fields(scene, directory)
    for name in ("bf", "gf"):
        codes, valid = pack_direction(np.load(directory / f"{name}.npy"))
        np.save(directory / f"{name}.npy", codes); np.save(directory / f"{name}_valid.npy", valid)
    sh = np.array([shift[0], shift[1], 0.])
    spec = json.loads((directory / "scene.json").read_text())
    for box in spec["boxes"]:
        box["center"] = (np.asarray(box["center"]) - sh).tolist()
    spec["start"] = [0., 0., 0.]; spec["goal"] = (np.asarray(spec["goal"]) - sh[:2]).tolist()
    spec["route"] = (np.asarray(spec["route"]) - sh[:2]).tolist(); spec["runtime_shift_m"] = (-sh).tolist()
    (directory / "scene.json").write_text(json.dumps(spec, indent=2) + "\n")
    steps = int(math.ceil(scene["time_budget"] / 0.02))
    record.update(family="procedural_cat", task_kind="cat", reset_mode="cat", crossed_mode="goal_radius",
                  sampling_group="procedural_cat", episode_length=max(1000, steps), reset_xy_scale=[1., 1.],
                  start=[0., 0., .8], goal=(np.asarray(record["goal"]) - sh).tolist(),
                  origin=(np.asarray(record["origin"]) - sh).tolist())
    source = dict(record["source"], kind="cos-scene", cos=scene["cos"], flat_balance="cos-scene")
    (directory / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    source["metadata_sha256"] = sha256(directory / "source.json")
    record.update(path=str(relative), source=source, fields=_field_records(directory),
                  scene_sha256=sha256(directory / "scene.json"))
    if plans:
        from cat_mjlab.fullpath_tracker import FullPathBank  # noqa: F401  (format check only)
        Path(plans).mkdir(parents=True, exist_ok=True)
        meta = dict(scene=scene["scene_id"], cos_scene=s["scene"], family=s["family"], ok=True,
                    reference="tracked" if use_tracked else "M", path_m=scene["cos"]["path_m"])
        np.savez_compressed(Path(plans) / f"{scene['scene_id']}.npz", qpos=ref.astype(np.float32),
                            modes=np.zeros((len(ref), 5), bool), clips=np.zeros(len(ref), np.int16), meta=json.dumps(meta))
    return record


def build(a):
    from cat_ppo.furniture.generalist_fields import _json_hash, load_generalist_manifest, scene_directory, sha256
    base_path = Path(a.base_manifest).resolve(); output = Path(a.output).resolve()
    base = load_generalist_manifest(base_path, verify_files=False)
    output.mkdir(parents=True, exist_ok=False)
    cands = [json.loads(l) for l in open(a.scenes)]
    jobs = [(c, output, a.plans) for c in cands]
    records, failed = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for c, fut in zip(cands, [ex.submit(write_case, j) for j in jobs]):
            try:
                records.append(fut.result()); print(records[-1]["scene_id"], flush=True)
            except Exception as e:                      # disconnected start/goal etc.: reported, not fatal
                failed.append(dict(scene=c["scene"], error=str(e)[:200])); print("FAIL", c["scene"], e, flush=True)
    for record in base["scenes"]:
        source = record["source"]
        if source.get("hand_contrast") or source.get("flat_balance"):
            parent = scene_directory(base, base_path, record); destination = output / record["path"]
            destination.mkdir(parents=True)
            for file in parent.iterdir():
                try:
                    os.link(file, destination / file.name)
                except OSError:
                    shutil.copy2(file, destination / file.name)
    fragment = output / "addition.json"
    fragment.write_text(json.dumps(dict(kind="cos-scenes", scenes=records, failed=failed), indent=1) + "\n")
    marker = dict(base["flat_balance"], schema="cat-extended-balance-v1",
                  parent=dict(manifest=str(base_path), sha256=sha256(base_path)),
                  additions=[dict(kind="cos-scenes", manifest=str(fragment), sha256=sha256(fragment), count=len(records))])
    manifest = dict(base, scenes=base["scenes"] + records, scene_count=base["scene_count"] + len(records),
                    flat_balance=marker, fields_bytes=base["fields_bytes"] +
                    sum(f["size_bytes"] for r in records for f in r["fields"].values()))
    manifest.pop("manifest_sha256"); manifest["manifest_sha256"] = _json_hash(manifest)
    pending = output / "manifest.pending.json"
    pending.write_text(json.dumps(manifest, indent=1) + "\n")
    load_generalist_manifest(pending, verify_files=False)
    pending.replace(output / "manifest.json")
    (output / "scene_ids.txt").write_text("\n".join(r["scene_id"] for r in records) + "\n")
    print(f"BUILT {len(records)} scenes, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("select", "build"))
    p.add_argument("--manifest"); p.add_argument("--splits"); p.add_argument("--split", default="train")
    p.add_argument("--per-family", type=int, default=34); p.add_argument("--min-tracked-clear", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out")
    p.add_argument("--base-manifest"); p.add_argument("--scenes"); p.add_argument("--output"); p.add_argument("--plans")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(); (select if a.mode == "select" else build)(a)
