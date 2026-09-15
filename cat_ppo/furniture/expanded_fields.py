"""Materialize CAT's available procedural range and irregular furniture banks.

All new procedural CAT fields call the released generator and field functions
directly. This expands public coverage; it is not a claim to recover unpublished
paper seeds or indoor crops. Completed scenes are resumable independently and a
training manifest is published only after the complete bank passes validation.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import urllib.request

import numpy as np

from cat_ppo.furniture.generalist_config import released_config
from cat_ppo.furniture.generalist_fields import (
    DATASET_REPO, DATASET_REVISION, DEFAULT_SAMPLING_GROUP_MASSES,
    EXPANDED_SCHEMA, FIELD_NAMES, RELEASED_CONFIG_SHA256, _download_file,
    _field_records, _json_hash, load_generalist_manifest, make_clutter_fields,
    sha256,
)
from cat_ppo.furniture.legacy_scenes import SOURCE_ROOT, TYPICAL_SCENES, UPSTREAM_COMMIT, _upstream
from cat_ppo.furniture.random_rooms import generate_random_room


BUILDER_VERSION = "cat-procedural-and-irregular-room-bank-v2"
DEFAULT_DIFFICULTIES = (.4, .5, .6, .7, .8, .9, 1.)
DEFAULT_PROCEDURAL_SEEDS = (2001, 2002)
DEFAULT_ROOM_SEEDS = tuple(range(4001, 4025))
DEFAULT_GENERIC_SEEDS = tuple(range(5001, 5025))
DEFAULT_OUTPUT = Path("data/furniture/cat_diversity_v2_20260916")
DEFAULT_BASE_BANK = Path("data/furniture/cat_generalist/manifest.json")


class SceneAdmissionError(ValueError):
    """Expected geometry rejection, recorded without disguising I/O/code errors."""


def _write_json(path, value):
    """Atomic small-file publication; incomplete field directories remain unlisted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _source_hashes():
    hashes = {name: sha256(SOURCE_ROOT / name) for name in
              ("random_obstacle.py", "grid_config.py", "pf_grid_config.yaml", "pf_modular.py")}
    here = Path(__file__).parent
    hashes.update({f"furniture/{name}": sha256(here / name) for name in
                   ("random_rooms.py", "scenes.py", "generalist_fields.py", "expanded_fields.py")})
    return hashes


def _integers(values, label, *, minimum=0, maximum=None):
    values = tuple(values)
    if not values or len(set(values)) != len(values) or any(
            type(v) is not int or v < minimum or (maximum is not None and v > maximum)
            for v in values):
        raise ValueError(f"{label} must contain distinct integers in the supported range")
    return values


def generation_plan(*, difficulties=DEFAULT_DIFFICULTIES, seeds=DEFAULT_PROCEDURAL_SEEDS,
                    ground_counts=range(4), lateral_counts=range(10), overhead_counts=range(4),
                    furniture_seeds=DEFAULT_ROOM_SEEDS, generic_seeds=DEFAULT_GENERIC_SEEDS,
                    typical_scenes=TYPICAL_SCENES):
    """A deterministic ordered list, with evaluation seeds explicitly held out."""
    difficulties = tuple(float(d) for d in difficulties)
    if (not difficulties or len(set(difficulties)) != len(difficulties)
            or any(not math.isfinite(d) or not 0 <= d <= 1 or abs(d * 10 - round(d * 10)) > 1e-8
                   for d in difficulties)):
        raise ValueError("Difficulties must be distinct multiples of .1 in [0,1]")
    seeds = _integers(seeds, "Procedural seeds")
    if set(seeds) & set(range(101, 111)):
        raise ValueError("Seeds 101–110 are reserved by CAT's released evaluation specification")
    ground_counts = _integers(ground_counts, "Ground counts", maximum=3)
    lateral_counts = _integers(lateral_counts, "Lateral counts", maximum=9)
    overhead_counts = _integers(overhead_counts, "Overhead counts", maximum=3)
    for values in (furniture_seeds, generic_seeds):
        if values:
            _integers(values, "Room seeds")
    typical_scenes = tuple(typical_scenes)
    if len(set(typical_scenes)) != len(typical_scenes) or set(typical_scenes) - set(TYPICAL_SCENES):
        raise ValueError("Typical scenes must be distinct members of the 27 released layouts")
    jobs = []
    for name in typical_scenes:
        jobs.append(dict(kind="published_cat", name=name, scene_id=f"published-{name}",
                         path=f"published/{name}"))
    for difficulty, ground, lateral, overhead, seed in itertools.product(
            difficulties, ground_counts, lateral_counts, overhead_counts, seeds):
        if ground == lateral == overhead == 0:
            continue
        name = f"D{round(difficulty * 10)}G{ground}L{lateral}O{overhead}S{seed}"
        jobs.append(dict(kind="procedural_cat", scene_id=f"procedural-{name}", path=f"procedural/{name}",
                         parameters=dict(difficulty=difficulty, seed=seed, n_rect_F=ground,
                                         n_rect_L=lateral, n_rect_R=lateral, n_rect_C=overhead)))
    for kind, room_seeds in (("furniture", furniture_seeds), ("generic_clutter", generic_seeds)):
        for seed in room_seeds:
            jobs.append(dict(kind=kind, seed=seed, scene_id=f"{kind}-S{seed}",
                             path=f"rooms/{kind}/S{seed}", split="train", difficulty="dense"))
    count = Counter(job["kind"] for job in jobs)
    return dict(builder_version=BUILDER_VERSION, jobs=jobs, original_anchor_count=37,
                requested_generated_count=len(jobs), requested_total_count=37 + len(jobs),
                family_counts=dict(count), difficulties=list(difficulties), seeds=list(seeds),
                counts=dict(ground=list(ground_counts), lateral=list(lateral_counts), overhead=list(overhead_counts)),
                room_seeds=dict(furniture=list(furniture_seeds), generic_clutter=list(generic_seeds)),
                excluded_evaluation_seeds=list(range(101, 111)),
                sampling_group_masses=DEFAULT_SAMPLING_GROUP_MASSES,
                exact_unpublished_paper_collection_reproduced=False,
                indoor_crops_included=False,
                deduplication="Preserve 37 anchors and published layouts; omit exact duplicate generated fields and retain all requested aliases")


def _array_hash(array):
    array = np.ascontiguousarray(array, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def check_free_space_connectivity(occupancy, sample_origin, dx, start, goal):
    """Check point-space 6-connected start/goal, without claiming robot clearance."""
    from scipy import ndimage

    occupancy = np.asarray(occupancy, dtype=bool)
    if occupancy.ndim != 3 or min(occupancy.shape) < 3 or occupancy.all():
        raise SceneAdmissionError("Occupancy must contain 3-D free space")
    indices = [np.rint((np.asarray(point) - sample_origin) / dx).astype(int) for point in (start, goal)]
    for index in indices:
        if np.any(index < 0) or np.any(index >= occupancy.shape):
            raise SceneAdmissionError("Start/goal lies outside the field grid")
    labels, _ = ndimage.label(~occupancy)
    first, last = (int(labels[tuple(index)]) for index in indices)
    if not first or first != last:
        raise SceneAdmissionError("Start/goal is occupied or disconnected in the 3-D voxel free space")
    return dict(method="six-connected-3d-free-voxel-components", start_goal_connected=True,
                start_index=indices[0].tolist(), goal_index=indices[1].tolist(),
                free_voxel_fraction=float(np.mean(~occupancy)),
                root_radius_validated=False, full_body_validated=False,
                dynamic_feasibility_validated=False)


def _task_semantics(record):
    task = "room" if record["family"] in ("furniture", "generic_clutter") else "cat"
    record.update(task_kind=task, reset_mode=task, crossed_mode="goal_radius" if task == "room" else "x_plane",
                  episode_length=4000 if task == "room" else 1000,
                  sampling_group=(record["family"] if record["family"] != "published_cat" else "original_cat"))
    return record


def _metadata_record(directory, source):
    _write_json(directory / "source.json", source)
    return dict(source, metadata_sha256=sha256(directory / "source.json"))


def _base_cat_record(job, directory, source):
    pf = released_config()["env_config"]["pf_config"]
    records = _field_records(directory)
    return _task_semantics(dict(scene_id=job["scene_id"], family=job["kind"], path=job["path"],
        shape=records["sdf"]["shape"], origin=list(pf["origin"]), dx=float(pf["dx"]),
        start=[0., 0., .8], goal=[2., 0., .75], reset_xy_scale=[1., 1.], reset_yaw=0.,
        sampling_weight=1., fields=records, source=_metadata_record(directory, source),
        sample_coordinates="Original CAT runtime origin retained, including released half-cell convention"))


def generate_procedural_fields(job, directory, source_hashes):
    """Exact upstream calls, native coordinates, no primitive-room adapter."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generator, module = _upstream("random_obstacle"), _upstream("pf_modular")
    cfg = generator.Cfg(**job["parameters"])
    occupancy, *axes = generator.generate_and_save(cfg, save=False)
    sample_origin = np.asarray([float(axis[0]) for axis in axes])
    connectivity = check_free_space_connectivity(occupancy, sample_origin, cfg.voxel,
                                                  [0., 0., .8], cfg.goal_w)
    sdf = module.make_sdf(occupancy, cfg.voxel)
    bf = module.grad3(sdf, cfg.voxel)
    _, gf = module.make_guidance_field_progressive(module.PFConfig(), np.meshgrid(*axes, indexing="ij"),
                                                  occupancy, cfg.goal_w, bf, sdf)
    for name, array in (("sdf", sdf), ("bf", bf), ("gf", gf)):
        if not np.isfinite(array).all():
            raise SceneAdmissionError(f"Nonfinite original CAT generated {name}: {job['scene_id']}")
        np.save(directory / f"{name}.npy", np.asarray(array, dtype=np.float32), allow_pickle=False)
    np.save(directory / "obs.npy", np.asarray(occupancy, dtype=np.uint8), allow_pickle=False)
    source = dict(kind="new-original-CAT-procedural-generation", upstream_commit=UPSTREAM_COMMIT,
        source_hashes=source_hashes, generation_parameters=job["parameters"],
        pipeline="original-generate_and_save-to-SDF-gradient-progressive-3D-FMM",
        generation_sample_origin=sample_origin.tolist(), generation_voxel_m=float(cfg.voxel),
        occupancy_sha256=sha256(directory / "obs.npy"), geometry_sha256=_array_hash(occupancy),
        connectivity=connectivity, split="train", unpublished_paper_seed_claim=False,
        physical_obstacles_in_training=False, runtime_nominal_pose_collision_check_required=True)
    return _base_cat_record(job, directory, source)


def _fetch_published_fields(job, directory, source_hashes, download):
    relative = f"assets_v1/TypiObs/{job['name']}"
    metadata_file = directory / "release_metadata.json"
    if metadata_file.exists():
        cached = json.loads(metadata_file.read_text())
        if cached.get("revision") != DATASET_REVISION or cached.get("scene_path") != relative:
            raise ValueError("Cached typical-scene source pin differs")
        metadata = cached["files"]
    elif download:
        url = f"https://huggingface.co/api/datasets/{DATASET_REPO}/tree/{DATASET_REVISION}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = {Path(entry["path"]).name: entry for entry in json.load(response)}
        _write_json(metadata_file, dict(repo=DATASET_REPO, revision=DATASET_REVISION,
                                       scene_path=relative, files=metadata))
    else:
        raise FileNotFoundError(f"Missing typical source metadata {metadata_file}; use --download")
    for name in (*FIELD_NAMES, "obs"):
        record = metadata[f"{name}.npy"]
        expected = record.get("lfs", {}).get("oid", "")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("Published typical asset lacks an immutable LFS SHA256")
        if record.get("path") != f"{relative}/{name}.npy":
            raise ValueError("Published typical metadata points outside its pinned scene")
        target = directory / f"{name}.npy"
        if download:
            _download_file(record["path"], target, expected)
        if not target.exists() or sha256(target) != expected:
            raise ValueError(f"Missing/corrupt pinned typical asset: {target}")
    grid = _upstream("grid_config").load_grid_config()
    occupancy = np.load(directory / "obs.npy", allow_pickle=False)
    sample_origin = np.asarray(grid["origin_w"]) + .5 * grid["voxel"]
    connectivity = check_free_space_connectivity(occupancy, sample_origin, grid["voxel"],
                                                  [0., 0., .8], grid["goal_w"])
    source = dict(kind="published-typical-CAT-fields", repo=DATASET_REPO,
        revision=DATASET_REVISION, scene_path=relative, files=metadata, arrays_unchanged=True,
        source_hashes=source_hashes, generation_sample_origin=sample_origin.tolist(),
        occupancy_sha256=sha256(directory / "obs.npy"), geometry_sha256=_array_hash(occupancy),
        connectivity=connectivity, physical_obstacles_in_training=False)
    return _base_cat_record(job, directory, source)


def _generate_room_fields(job, directory, source_hashes):
    try:
        scene = generate_random_room(job["seed"], split=job["split"], kind=job["kind"], difficulty=job["difficulty"])
        record = make_clutter_fields(scene, directory, dx=released_config()["env_config"]["pf_config"]["dx"])
    except (ValueError, RuntimeError) as error:
        if any(message in str(error) for message in ("No connected randomly packed room", "Clutter start and goal", "Nonfinite CAT FMM")):
            raise SceneAdmissionError(str(error)) from error
        raise
    record["path"] = job["path"]
    occupancy = np.load(directory / "obs.npy", allow_pickle=False)
    connectivity = check_free_space_connectivity(occupancy, np.asarray(record["origin"]), record["dx"],
                                                  record["start"], record["goal"])
    source = dict(record["source"], kind="new-irregular-room", source_hashes=source_hashes,
                  generation_parameters={key: job[key] for key in ("seed", "split", "kind", "difficulty")},
                  geometry_sha256=scene["geometry_hash"], occupancy_sha256=sha256(directory / "obs.npy"),
                  connectivity=connectivity)
    record["source"] = _metadata_record(directory, source)
    return _task_semantics(record)


def _validate_cached_record(record, directory):
    if record.get("rejected"):
        if sha256(directory / "source.json") != record["source"]["metadata_sha256"]:
            raise ValueError("Cached rejection provenance differs")
        return
    if record["fields"] != _field_records(directory):
        raise ValueError(f"Cached fields differ: {directory}")
    if sha256(directory / "source.json") != record["source"]["metadata_sha256"]:
        raise ValueError("Cached source metadata differs")
    if record.get("scene_sha256") and sha256(directory / "scene.json") != record["scene_sha256"]:
        raise ValueError("Cached room geometry differs")
    expected = record["source"].get("occupancy_sha256")
    if expected and sha256(directory / "obs.npy") != expected:
        raise ValueError("Cached occupancy differs")


def _build_job(payload):
    job, output, hashes, download = payload
    directory = Path(output) / job["path"]
    directory.mkdir(parents=True, exist_ok=True)
    key = _json_hash(dict(job=job, source_hashes=hashes, builder=BUILDER_VERSION,
                         dataset_revision=DATASET_REVISION))
    cache = directory / "record.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("cache_key") != key:
            raise ValueError(f"Scene cache describes different generation code/parameters: {directory}")
        _validate_cached_record(saved["record"], directory)
        return saved["record"]
    try:
        if job["kind"] == "published_cat":
            record = _fetch_published_fields(job, directory, hashes, download)
        elif job["kind"] == "procedural_cat":
            record = generate_procedural_fields(job, directory, hashes)
        else:
            record = _generate_room_fields(job, directory, hashes)
    except SceneAdmissionError as error:
        if job["kind"] == "published_cat":
            # Released fields are mandatory anchors, not substitute candidates.
            raise
        source = dict(kind="geometry-admission-rejection", generation=job,
                      source_hashes=hashes, reason=str(error))
        record = dict(rejected=True, scene_id=job["scene_id"], family=job["kind"],
                      path=job["path"], reason=str(error), source=_metadata_record(directory, source))
    _validate_cached_record(record, directory)
    _write_json(cache, dict(cache_key=key, record=record))
    return record


def _copy_anchors(base, base_directory, output):
    scenes = []
    for source_record in base["scenes"]:
        if source_record["family"] != "original_cat":
            continue
        record = _task_semantics(copy.deepcopy(source_record))
        source_dir = base_directory / source_record["path"]
        destination = output / "original" / source_record["scene_id"]
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("sdf.npy", "bf.npy", "gf.npy", "source.json", "obs.npy"):
            original, target = source_dir / name, destination / name
            if not original.exists():
                if name == "obs.npy":
                    continue
                raise FileNotFoundError(original)
            if target.exists():
                if sha256(target) != sha256(original):
                    raise ValueError(f"Existing anchor differs from the verified base bank: {target}")
                continue
            try:
                os.link(original, target)
            except OSError:
                shutil.copy2(original, target)
        record["path"] = str(destination.relative_to(output))
        _validate_cached_record(record, destination)
        scenes.append(record)
    return scenes


def deduplicate_generated(anchors, generated, jobs):
    """Omit identical generated tasks; preserve explicit parameter aliases for audit."""
    def fingerprint(scene):
        return _json_hash(dict(fields={name: scene["fields"][name]["sha256"] for name in FIELD_NAMES},
            origin=scene["origin"], shape=scene["shape"], dx=scene["dx"],
            task_kind=scene["task_kind"], start=scene["start"], goal=scene["goal"]))
    kept, candidates, known = list(anchors), [], {}
    for scene in anchors:
        known.setdefault(fingerprint(scene), scene["scene_id"])
    for job, scene in zip(jobs, generated, strict=True):
        if scene.get("rejected"):
            candidates.append(dict(scene_id=scene["scene_id"], path=scene["path"], family=scene["family"],
                                   generation=job, included_in_bank=False, duplicate_of=None,
                                   rejected=True, rejection_reason=scene["reason"]))
            continue
        signature = fingerprint(scene)
        canonical = known.get(signature)
        duplicate = canonical is not None and scene["family"] != "published_cat"
        candidates.append(dict(scene_id=scene["scene_id"], path=scene["path"], family=scene["family"],
                               generation=job, task_fields_sha256=signature,
                               geometry_sha256=scene["source"].get("geometry_sha256"),
                               included_in_bank=not duplicate, duplicate_of=canonical if duplicate else None))
        if not duplicate:
            known.setdefault(signature, scene["scene_id"])
            kept.append(scene)
    requested = {(job["parameters"]["difficulty"],
                  bool(job["parameters"]["n_rect_F"]), bool(job["parameters"]["n_rect_L"]),
                  bool(job["parameters"]["n_rect_C"])) for job in jobs if job["kind"] == "procedural_cat"}
    covered = set()
    # Duplicate aliases retain a represented task, even when a particular count
    # or difficulty genuinely generates the same geometry as another request.
    for candidate in candidates:
        job = candidate["generation"]
        if job["kind"] == "procedural_cat" and not candidate.get("rejected"):
            p = job["parameters"]
            covered.add((p["difficulty"], bool(p["n_rect_F"]), bool(p["n_rect_L"]), bool(p["n_rect_C"])))
    if requested != covered:
        raise ValueError("Deduplicated coverage lost a requested difficulty/obstacle family")
    requested_rooms = {job["kind"] for job in jobs if job["kind"] in ("furniture", "generic_clutter")}
    if not requested_rooms.issubset({scene["family"] for scene in kept}):
        raise ValueError("Every requested room family needs at least one admitted scene")
    return kept, candidates


def prepare_expanded_fields(base_manifest=DEFAULT_BASE_BANK, output=DEFAULT_OUTPUT, *,
                            workers=4, download=False, progress=None, **plan_options):
    """Resume completed scenes and atomically publish the fully validated v2 bank."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    base_manifest, output = Path(base_manifest).resolve(), Path(output).resolve()
    if output == base_manifest.parent:
        raise ValueError("Expanded output must be separate from the original bank")
    base = load_generalist_manifest(base_manifest)
    plan = generation_plan(**plan_options)
    hashes = _source_hashes()
    plan["source_hashes"] = hashes
    plan["base_manifest_sha256"] = base["manifest_sha256"]
    plan["dataset_revision"] = DATASET_REVISION
    plan["plan_sha256"] = _json_hash(plan)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "preparation_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise FileExistsError("Output contains a different preparation plan; use a new directory")
    else:
        _write_json(plan_path, plan)
    destination = output / "manifest.json"
    if destination.exists():
        existing = load_generalist_manifest(destination)
        if existing.get("preparation_plan_sha256") != plan["plan_sha256"]:
            raise ValueError("Existing completed manifest differs from preparation plan")
        return destination
    anchors = _copy_anchors(base, base_manifest.parent, output)
    generated = [None] * len(plan["jobs"])
    jobs = [(job, str(output), hashes, download) for job in plan["jobs"]]
    if workers == 1:
        for index, payload in enumerate(jobs):
            generated[index] = _build_job(payload)
            if progress:
                progress(index + 1, len(jobs), generated[index]["scene_id"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_build_job, payload): index for index, payload in enumerate(jobs)}
            completed = 0
            for future in as_completed(futures):
                index = futures[future]
                generated[index] = future.result()
                completed += 1
                if progress:
                    progress(completed, len(jobs), generated[index]["scene_id"])
    scenes, coverage = deduplicate_generated(anchors, generated, plan["jobs"])
    coverage_record = dict(plan_sha256=plan["plan_sha256"], requested_generated_count=len(generated),
                           unique_generated_count=len(scenes) - len(anchors), candidates=coverage,
                           rejected_count=sum(bool(c.get("rejected")) for c in coverage),
                           duplicate_count=sum(c.get("duplicate_of") is not None for c in coverage),
                           retained_family_counts=dict(Counter(s["family"] for s in scenes)))
    _write_json(output / "generation_coverage.json", coverage_record)
    manifest = dict(schema=EXPANDED_SCHEMA, released_config_sha256=RELEASED_CONFIG_SHA256,
        original_count=37, scene_count=len(scenes), scenes=scenes,
        byte_verified_original_count=base["byte_verified_original_count"],
        reconstructed_original_count=base["reconstructed_original_count"],
        upstream_commit=UPSTREAM_COMMIT, dataset_repo=DATASET_REPO, dataset_revision=DATASET_REVISION,
        sampling_group_masses=DEFAULT_SAMPLING_GROUP_MASSES,
        storage="ragged-concatenated-xyz; original CAT clipping and interpolation retained",
        fields_bytes=sum(f["size_bytes"] for scene in scenes for f in scene["fields"].values()),
        preparation_plan_sha256=plan["plan_sha256"], generation_coverage_sha256=sha256(output / "generation_coverage.json"),
        requested_generated_count=len(generated), duplicate_generated_count=coverage_record["duplicate_count"],
        rejected_generated_count=coverage_record["rejected_count"],
        requested_family_counts=plan["family_counts"], retained_family_counts=coverage_record["retained_family_counts"],
        unpublished_paper_training_collection_reproduced=False, indoor_crops_included=False,
        procedural_generation="Unmodified public CAT generator over explicitly recorded parameters; not unpublished paper seed identity")
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary = output / ".manifest.validating.json"
    _write_json(temporary, manifest)
    load_generalist_manifest(temporary)
    temporary.replace(destination)
    return destination
