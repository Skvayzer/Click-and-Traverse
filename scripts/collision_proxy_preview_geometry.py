"""Offline proposal geometry; nothing here changes training or observations.

The proposal encloses the compiled robot's visual mesh triangles in primitives
attached to their own articulated links.  Feet are two flat boxes, with mesh
triangles clipped at the sole/upper-foot division before fitting.  Existing
Dex3 protection spheres are retained exactly.  This is a review artifact, not
a runtime collision implementation or a claim of measured training overhead.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation


SCHEMA = "cat-offline-collision-proxy-proposal-v1"
FOOT_SPLIT_Z_M = -.008


def _rotation(quaternion):
    return Rotation.from_quat(np.asarray(quaternion), scalar_first=True).as_matrix()


def _world_vertices(model, data, geom_id):
    mesh = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
    # Compiled mesh vertices and geom poses already include mesh_pos/mesh_quat
    # and asset scaling. Applying those transformations again would be wrong.
    return (np.einsum("ij,nj->ni", data.geom_xmat[geom_id].reshape(3, 3), vertices)
            + data.geom_xpos[geom_id])


def _body_vertices(model, data, geom_id, body_id):
    return np.einsum("ji,nj->ni", data.xmat[body_id].reshape(3, 3),
                     _world_vertices(model, data, geom_id) - data.xpos[body_id])


def _faces(model, geom_id):
    mesh = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
    return np.asarray(model.mesh_face[start:start + count], dtype=np.int64)


def _descendant(model, child, ancestor):
    while child and child != ancestor:
        child = int(model.body_parentid[child])
    return child == ancestor


def _mesh_records(model):
    import mujoco

    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected one free-root robot")
    root_body = int(model.jnt_bodyid[free[0]])
    unique = {}
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if (model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH
                or not _descendant(model, body, root_body)):
            continue
        mesh = int(model.geom_dataid[geom])
        key = (body, mesh, tuple(model.geom_pos[geom]), tuple(model.geom_quat[geom]))
        if key in unique:
            unique[key]["duplicate_geom_ids"].append(geom)
            continue
        unique[key] = {
            "geom_id": geom, "body_id": body, "body_name": model.body(body).name,
            "mesh_id": mesh, "mesh_name": model.mesh(mesh).name,
            "geom_name": model.geom(geom).name or f"unnamed_geom_{geom}",
            "duplicate_geom_ids": [],
            "vertex_count": int(model.mesh_vertnum[mesh]),
            "triangle_count": int(model.mesh_facenum[mesh]),
        }
    return list(unique.values())


def _assignment(record, clip=None):
    result = {key: record[key] for key in (
        "geom_id", "geom_name", "mesh_name", "body_name", "duplicate_geom_ids")}
    if clip is not None:
        result["clip"] = clip
    return result


def _clipped_triangle_points(vertices, faces, clip):
    """Vertices of every triangle after clipping by one body-local halfspace.

    Keeping edge-plane intersection points is essential: merely dividing the
    original vertices between two boxes would not certify crossing triangles.
    Every resulting polygon is convex, so a convex primitive enclosing all its
    vertices encloses the whole polygon, including its interior.
    """
    if clip is None:
        return vertices
    if (clip.get("axis") not in (0, 1, 2)
            or clip.get("side") not in ("lower", "upper")
            or not math.isfinite(float(clip.get("position_m", math.nan)))):
        raise ValueError("Mesh clip requires a finite axis-aligned lower/upper halfspace")
    axis, split = int(clip["axis"]), float(clip["position_m"])
    lower = clip["side"] == "lower"
    distance = vertices[:, axis] - split
    inside = distance <= 0. if lower else distance >= 0.
    points = [vertices[inside]]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    first, second = vertices[edges[:, 0]], vertices[edges[:, 1]]
    first_d, second_d = distance[edges[:, 0]], distance[edges[:, 1]]
    crosses = ((first_d < 0.) & (second_d > 0.)) | ((first_d > 0.) & (second_d < 0.))
    if np.any(crosses):
        fraction = first_d[crosses] / (first_d[crosses] - second_d[crosses])
        cut = first[crosses] + fraction[:, None] * (second[crosses] - first[crosses])
        cut[:, axis] = split
        points.append(cut)
    result = np.concatenate(points)
    if not len(result):
        raise ValueError("Empty clipped mesh piece")
    return result


def _assigned_points(model, data, shape, assignment):
    geom = int(assignment["geom_id"])
    if model.mesh(int(model.geom_dataid[geom])).name != assignment["mesh_name"]:
        raise ValueError("Proposal geom indices do not match this model")
    vertices = _body_vertices(model, data, geom, int(shape["body_id"]))
    return _clipped_triangle_points(vertices, _faces(model, geom), assignment.get("clip"))


def _box(points, margin):
    lower, upper = points.min(axis=0), points.max(axis=0)
    return {"kind": "box", "center": ((lower + upper) / 2).tolist(),
            "quat": [1., 0., 0., 0.],
            "half_size": ((upper - lower) / 2 + margin).tolist()}


def _canonical_axis(axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    if axis[np.argmax(np.abs(axis))] < 0.:
        axis = -axis
    return axis


def _axis_frame(axis):
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(reference, axis)
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(axis, first), axis))


def _capsule(points, margin):
    """Conservative capsule, selecting among body/PCA axes by fitted volume.

    This is a deterministic compact fit, not a globally optimal enclosing
    capsule. For each axis the radius and segment endpoints account for actual
    hemispherical caps, instead of adding a full-radius bulge to mesh endpoints.
    """
    centered = points - points.mean(axis=0)
    covariance = np.einsum("ni,nj->ij", centered, centered) / len(points)
    _, pca = np.linalg.eigh(covariance)
    axes = [_canonical_axis(axis) for axis in list(np.eye(3)) + list(pca.T)]
    candidates = []
    for axis in axes:
        frame = _axis_frame(axis)
        coordinates = np.einsum("ni,ij->nj", points, frame)
        xy = (coordinates[:, :2].min(axis=0) + coordinates[:, :2].max(axis=0)) / 2
        radial_squared = np.sum((coordinates[:, :2] - xy) ** 2, axis=1)
        height = coordinates[:, 2]
        minimum_radius = float(np.sqrt(radial_squared.max()))
        maximum_radius = math.sqrt(minimum_radius ** 2 + (.5 * np.ptp(height)) ** 2)

        def segment(radius):
            cap = np.sqrt(np.maximum(radius ** 2 - radial_squared, 0.))
            start, end = float(np.min(height + cap)), float(np.max(height - cap))
            if start > end:
                start = end = .5 * (start + end)
            return start, end

        def volume(radius):
            start, end = segment(radius)
            return math.pi * radius ** 2 * (end - start) + 4 * math.pi * radius ** 3 / 3

        if maximum_radius - minimum_radius > 1e-10:
            result = minimize_scalar(volume, bounds=(minimum_radius, maximum_radius),
                                     method="bounded", options={"xatol": 1e-9})
            radius = min((minimum_radius, float(result.x), maximum_radius), key=volume)
        else:
            radius = minimum_radius
        start, end = segment(radius)
        endpoints = np.einsum("ij,nj->ni", frame,
                              np.array([[*xy, start], [*xy, end]]))
        fitted_radius = radius + margin
        score = (math.pi * fitted_radius ** 2 * (end - start)
                 + 4 * math.pi * fitted_radius ** 3 / 3)
        candidates.append((score, {
            "kind": "capsule", "center": endpoints.mean(axis=0).tolist(),
            "quat": Rotation.from_matrix(frame).as_quat(scalar_first=True).tolist(),
            "endpoints": endpoints.tolist(), "radius": fitted_radius,
        }))
    result = min(candidates, key=lambda item: item[0])[1]
    if np.linalg.norm(np.diff(result["endpoints"], axis=0)) < 1e-8:
        result["kind"] = "sphere"
        result.pop("endpoints")
    return result


def _signed_distance(shape, points):
    if shape["kind"] == "box":
        local = np.einsum("ni,ij->nj", points - np.asarray(shape["center"]),
                          _rotation(shape["quat"]))
        q = np.abs(local) - shape["half_size"]
        return np.linalg.norm(np.maximum(q, 0.), axis=1) + np.minimum(np.max(q, axis=1), 0.)
    if shape["kind"] == "sphere":
        return np.linalg.norm(points - shape["center"], axis=1) - shape["radius"]
    start, end = np.asarray(shape["endpoints"])
    line = end - start
    length_squared = float(np.dot(line, line))
    fraction = np.clip(np.einsum("ni,i->n", points - start, line)
                       / max(length_squared, 1e-30), 0., 1.)
    return np.linalg.norm(points - (start + fraction[:, None] * line), axis=1) - shape["radius"]


def validate_proposal(model, data, proposal, tolerance_m=2e-6):
    """Validate compiled triangles at current FK, including split-face pieces.

    Call mj_forward before this function. The test verifies containment by the
    proposed robot proxies; it is not a robot/obstacle collision evaluation.
    """
    if proposal.get("schema") != SCHEMA:
        raise ValueError("Unknown collision proposal schema")
    if not proposal.get("shapes"):
        raise ValueError("A collision proposal must contain at least one shape")
    report = []
    covered = set()
    assignment_coverage = defaultdict(list)
    for shape in proposal["shapes"]:
        kind = shape.get("kind")
        if kind not in ("box", "capsule", "sphere"):
            raise ValueError(f"Unsupported collision proposal primitive: {kind!r}")
        center = np.asarray(shape.get("center"), dtype=float)
        quaternion = np.asarray(shape.get("quat"), dtype=float)
        if (center.shape != (3,) or not np.isfinite(center).all()
                or quaternion.shape != (4,) or not np.isfinite(quaternion).all()
                or np.linalg.norm(quaternion) < 1e-12):
            raise ValueError("Shape requires a finite center and nonzero quaternion")
        if not shape.get("mesh_assignments"):
            raise ValueError("Each proposal shape must cover assigned mesh geometry")
        margin = float(shape.get("margin_m", math.nan))
        if not math.isfinite(margin) or margin < 0.:
            raise ValueError("Shape margin must be finite and nonnegative")
        if kind == "box":
            half_size = np.asarray(shape.get("half_size"), dtype=float)
            if (half_size.shape != (3,) or not np.isfinite(half_size).all()
                    or np.any(half_size <= 0.)):
                raise ValueError("Box half sizes must be three finite positive values")
        else:
            radius = float(shape.get("radius", math.nan))
            if not math.isfinite(radius) or radius <= 0.:
                raise ValueError("Sphere/capsule radius must be finite and positive")
            if kind == "capsule":
                endpoints = np.asarray(shape.get("endpoints"), dtype=float)
                if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
                    raise ValueError("Capsule endpoints must be two finite XYZ positions")
        body = int(shape["body_id"])
        if model.body(body).name != shape["body_name"]:
            raise ValueError("Proposal body indices do not match this model")
        minimum, checked = math.inf, 0
        for assignment in shape["mesh_assignments"]:
            geom = int(assignment["geom_id"])
            # Never silently attach one proxy across a moving joint. Fingers
            # may be descendants because their joints are welded in this task.
            source = int(model.geom_bodyid[geom])
            if not _descendant(model, source, body):
                raise ValueError("Proxy assigned outside its body subtree")
            while source != body:
                if model.body_jntnum[source] != 0:
                    raise ValueError("Proxy assignment crosses an articulated joint")
                source = int(model.body_parentid[source])
            points = _assigned_points(model, data, shape, assignment)
            distances = _signed_distance(shape, points)
            if not np.isfinite(distances).all():
                raise ValueError("Nonfinite proposal geometry")
            minimum = min(minimum, float(-distances.max()))
            checked += len(points)
            covered.add(geom)
            assignment_coverage[geom].append(assignment.get("clip"))
            # Duplicates are identical only in the compiled model used to fit.
            # Verify those transforms again instead of relying on a name match.
            for duplicate in assignment["duplicate_geom_ids"]:
                original_world = _world_vertices(model, data, geom)
                duplicate_world = _world_vertices(model, data, int(duplicate))
                if not np.allclose(original_world, duplicate_world, atol=tolerance_m, rtol=0.):
                    raise ValueError("An assigned duplicate geom has moved independently")
                covered.add(int(duplicate))
                assignment_coverage[int(duplicate)].append(assignment.get("clip"))
        required = float(shape["margin_m"])
        report.append({"shape_id": shape["id"], "minimum_margin_m": minimum,
                       "required_margin_m": required, "checked_points": checked,
                       "covered": bool(minimum >= required - tolerance_m)})
    records = _mesh_records(model)
    expected = {geom for record in records
                for geom in [record["geom_id"], *record["duplicate_geom_ids"]]}
    missing = sorted(expected - covered)
    incomplete = []
    for geom in sorted(expected & covered):
        clips = assignment_coverage[geom]
        if any(clip is None for clip in clips):
            continue
        halfspaces = defaultdict(set)
        for clip in clips:
            halfspaces[(int(clip["axis"]), float(clip["position_m"]))].add(clip["side"])
        if not any(sides == {"lower", "upper"} for sides in halfspaces.values()):
            incomplete.append(geom)
    return {"all_covered": not missing and not incomplete and all(item["covered"] for item in report),
            "missing_geom_ids": missing, "incomplete_mesh_geom_ids": incomplete,
            "shape_count": len(report),
            "covered_mesh_geom_count_including_duplicates": len(covered),
            "minimum_margin_m": min(item["minimum_margin_m"] for item in report),
            "triangle_coverage_method": "Convex enclosure of full mesh triangles; foot triangles clipped at split plane before enclosure",
            "shapes": report}


def build_proposal(model, data, margin_m=.003):
    """Return review-only JSON primitives fitted to the current compiled robot.

    Call mj_forward first. All positions/endpoints are relative to body_name;
    box/capsule quaternions use MuJoCo's wxyz convention. Capsules' endpoints
    are sphere centers (not their outer tips). No model or data is modified.
    """
    from cat_ppo.furniture.grippers import hand_sphere

    if not math.isfinite(margin_m) or not 0. <= margin_m <= .02:
        raise ValueError("Expected a modest finite offline fit margin")
    records = _mesh_records(model)
    groups = defaultdict(list)
    for record in records:
        name, body = record["mesh_name"], record["body_name"]
        if name.startswith(("left_hand_", "right_hand_")):
            groups[(name.split("_", 1)[0] + "_hand", "hand")].append(record)
        elif body == "torso_link":
            groups[("head" if name == "head_link" else "torso", "box")].append(record)
        elif body == "pelvis":
            groups[("pelvis", "box")].append(record)
        elif body.endswith("ankle_roll_link"):
            groups[(body, "foot")].append(record)
        elif body.endswith("ankle_pitch_link") or body.startswith("waist_"):
            groups[(body, "box")].append(record)
        else:
            groups[(body, "capsule")].append(record)
    shapes = []
    for (label, kind), entries in groups.items():
        if kind == "hand":
            side = label.split("_", 1)[0]
            body_id = int(model.body(f"{side}_wrist_yaw_link").id)
            sphere = hand_sphere(side)
            pieces = [(label, {"kind": "sphere", "center": sphere["center"],
                              "quat": [1., 0., 0., 0.], "radius": sphere["radius"]},
                       [_assignment(record) for record in entries], sphere["margin_m"])]
        else:
            body_id = entries[0]["body_id"]
            if kind == "foot":
                pieces = []
                for side, suffix in (("lower", "sole"), ("upper", "upper_foot")):
                    clip = {"axis": 2, "position_m": FOOT_SPLIT_Z_M, "side": side}
                    assignments = [_assignment(record, clip) for record in entries]
                    points = np.concatenate([_assigned_points(model, data, {"body_id": body_id}, item)
                                             for item in assignments])
                    pieces.append((label.replace("ankle_roll_link", suffix),
                                   _box(points, margin_m), assignments, margin_m))
            else:
                assignments = [_assignment(record) for record in entries]
                points = np.concatenate([_body_vertices(model, data, item["geom_id"], body_id)
                                         for item in assignments])
                geometry = _box(points, margin_m) if kind == "box" else _capsule(points, margin_m)
                pieces = [(label, geometry, assignments, margin_m)]
        for identifier, geometry, assignments, margin in pieces:
            group = ("hands" if kind == "hand" else "feet" if kind == "foot"
                     else "head" if label == "head"
                     else "trunk" if label in ("pelvis", "torso") or label.startswith("waist_")
                     else "legs" if any(value in label for value in ("hip_", "knee_", "ankle_"))
                     else "arms")
            shape = {"id": identifier, "body_id": body_id,
                     "body_name": model.body(body_id).name, "group": group,
                     **geometry, "margin_m": margin,
                     "mesh_assignments": assignments,
                     "assigned_geom_ids": [item["geom_id"] for item in assignments],
                     "assigned_geom_names": [item["geom_name"] for item in assignments],
                     "assigned_mesh_names": [item["mesh_name"] for item in assignments],
                     "preserves_existing_hand_sphere": kind == "hand"}
            points = np.concatenate([_assigned_points(model, data, shape, item) for item in assignments])
            shape["mesh_piece_bounds_body_local"] = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
            shapes.append(shape)
    proposal = {
        "schema": SCHEMA, "status": "offline_review_only_not_enabled_in_training",
        "shapes": shapes, "shape_counts": dict(Counter(shape["kind"] for shape in shapes)),
        "group_counts": dict(Counter(shape["group"] for shape in shapes)),
        "body_margin_m": margin_m, "foot_split_body_z_m": FOOT_SPLIT_Z_M,
        "unique_mesh_geom_count": len(records),
        "unique_mesh_vertex_count": sum(item["vertex_count"] for item in records),
        "unique_mesh_triangle_count": sum(item["triangle_count"] for item in records),
        "observation_changes": False, "physics_changes": False,
        "training_changes": False, "measured_runtime_cost": None,
        "limitations": [
            "Enclosing primitives include empty space around mesh surfaces; this is an approximation.",
            "Coverage validates robot mesh enclosure, not obstacle clearance or continuous collision detection.",
            "Meshes and poses are inspected offline; no collision checker is enabled by this proposal.",
        ],
    }
    proposal["validation"] = validate_proposal(model, data, proposal)
    if not proposal["validation"]["all_covered"]:
        raise ValueError("Fitted proposal failed compiled-triangle coverage validation")
    return proposal
