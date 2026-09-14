"""Unitree three-finger Dex3 geometry, fixed for the CAT traversal task.

The seven finger joints per hand are welded at the source model's open-hand
stand pose (thumb opposition clamped to its published joint limit). Body
control therefore remains 29-DoF. The source wrist/palm and finger inertials
are retained; visual geometry and protection boxes have zero added density.
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np


ASSET_ROOT = Path(__file__).resolve().parents[2] / "data/assets/unitree_g1/dex3"
MODEL_NAME = "unitree_dex3_three_finger_fixed_stand_v1"
ENVELOPE_MARGIN_M = 0.005


def _values(text, default):
    return np.fromstring(text, sep=" ") if text is not None else np.asarray(default, dtype=float)


def _rotation(quat):
    w, x, y, z = np.asarray(quat) / np.linalg.norm(quat)
    return np.asarray([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


def _numbers(values):
    return " ".join(format(float(value), ".12g") for value in values)


@lru_cache(maxsize=1)
def source_hands():
    """Verify vendored provenance before using source transforms or meshes."""
    manifest = json.loads((ASSET_ROOT / "provenance.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((ASSET_ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dex3 asset checksum mismatch: {name}")
    return ET.parse(ASSET_ROOT / "source-hands.xml").getroot()


def fixed_wrist(side):
    """Return source wrist and fixed finger bodies with their native inertias."""
    if side not in ("left", "right"):
        raise ValueError("hand side must be left or right")
    wrist = deepcopy(source_hands().find(f"body[@name='{side}_wrist_yaw_link']"))
    for body in wrist.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("class") != "visual":
                body.remove(geom)
        for joint in list(body.findall("joint")):
            # The published stand keyframe sets thumb_1 to +/-1.05 rad;
            # use the actual +/-1.0472 rad limit instead of exceeding it.
            if "hand_thumb_1_joint" in joint.get("name", ""):
                lo, hi = _values(joint.get("range"), (0., 0.))
                angle = float(np.clip(1.05 if side == "left" else -1.05, lo, hi))
                axis = _values(joint.get("axis"), (0., 0., 1.))
                if body.get("quat") is not None or joint.get("pos") is not None:
                    raise ValueError("Re-derive Dex3 fixed transform for changed source joint")
                body.set("quat", _numbers(np.r_[np.cos(angle/2), axis*np.sin(angle/2)]))
            body.remove(joint)
    return wrist


def _stl_vertices(path):
    raw = Path(path).read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"Expected canonical binary STL: {path}")
    triangles = np.frombuffer(raw, dtype=np.dtype([
        ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]),
        offset=84, count=count)
    return np.asarray(triangles["vertices"], dtype=float).reshape(-1, 3)


def hand_mesh_vertices(side):
    """All palm/finger mesh vertices expressed in the wrist-yaw body frame."""
    result = {}

    def visit(body, rotation, translation, is_wrist=False):
        if not is_wrist:
            translation = translation + rotation @ _values(body.get("pos"), (0., 0., 0.))
            rotation = rotation @ _rotation(_values(body.get("quat"), (1., 0., 0., 0.)))
        for geom in body.findall("geom"):
            mesh = geom.get("mesh", "")
            if "_hand_" not in mesh:
                continue
            local_rotation = rotation @ _rotation(_values(geom.get("quat"), (1., 0., 0., 0.)))
            local_position = translation + rotation @ _values(geom.get("pos"), (0., 0., 0.))
            vertices = _stl_vertices(ASSET_ROOT / "assets" / (mesh + ".STL"))
            result[mesh] = np.einsum("ij,nj->ni", local_rotation, vertices) + local_position
        for child in body.findall("body"):
            visit(child, rotation, translation)

    visit(fixed_wrist(side), np.eye(3), np.zeros(3), is_wrist=True)
    return result


@lru_cache(maxsize=2)
def hand_envelope(side):
    """Bounds cover every mesh triangle, including the thumb, plus 5 mm."""
    vertices = np.concatenate(list(hand_mesh_vertices(side).values()))
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite Dex3 mesh vertices")
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    return {"center": ((lower + upper) / 2).tolist(),
            "half_size": ((upper - lower) / 2 + ENVELOPE_MARGIN_M).tolist(),
            "mesh_lower": lower.tolist(), "mesh_upper": upper.tolist(),
            "margin_m": ENVELOPE_MARGIN_M}


def geometry_contract():
    provenance = json.loads((ASSET_ROOT / "provenance.json").read_text())
    return {"model": MODEL_NAME, "finger_control": "fixed_source_stand_pose",
            "actuated_finger_dofs": 0, "physical_finger_dofs_per_hand": 7,
            "fixed_thumb_1_angle_rad": {"left": 1.0472, "right": -1.0472},
            "other_fixed_finger_angles_rad": 0.0,
            "wrist_palm_and_finger_mass_kg_per_side": 0.7811226,
            "source_upstream_commit": "c20ca8f1fe5e519474c6c8d10b1ce5c719dd7a65",
            "source_mjcf_sha256": provenance["source_mjcf_sha256"],
            "asset_files_sha256": provenance["files_sha256"],
            "envelopes": {side: hand_envelope(side) for side in ("left", "right")}}


def validate_hand_envelopes(model, data):
    """Check native MuJoCo mesh vertices against the physical envelope boxes.

    Call after mj_forward for any body/wrist pose. This uses compiled mesh
    transforms rather than the STL traversal used to derive the bounds, so it
    also catches assembly and renderer transform errors.
    """
    import mujoco
    report = {}
    for side in ("left", "right"):
        envelope_id = model.geom(f"furniture_{side}_hand_envelope").id
        envelope_rotation = data.geom_xmat[envelope_id].reshape(3, 3)
        minimum_margin = float("inf")
        vertex_count, meshes = 0, []
        for geom_id in range(model.ngeom):
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            mesh_name = model.mesh(mesh_id).name
            if not mesh_name.startswith(f"{side}_hand_"):
                continue
            start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            vertices = model.mesh_vert[start:start + count]
            world = np.einsum("ij,nj->ni", data.geom_xmat[geom_id].reshape(3, 3), vertices) + data.geom_xpos[geom_id]
            local = np.einsum("ji,nj->ni", envelope_rotation, world - data.geom_xpos[envelope_id])
            margin = model.geom_size[envelope_id] - np.abs(local)
            if not np.isfinite(margin).all():
                raise ValueError(f"Nonfinite native Dex3 geometry: {mesh_name}")
            minimum_margin = min(minimum_margin, float(margin.min()))
            vertex_count += count
            meshes.append(mesh_name)
        if len(meshes) != 8 or minimum_margin < ENVELOPE_MARGIN_M - 2e-6:
            raise ValueError(f"{side} Dex3 hand not contained with 5 mm margin: {minimum_margin}")
        report[side] = {"checked_meshes": meshes, "checked_vertices": vertex_count,
                        "minimum_margin_m": minimum_margin, "contained": True}
    return report


def install_fixed_hands(root):
    """Replace the old rubber hands and assumed .8 kg wrist/hand inertia."""
    assets = root.find("asset")
    for mesh in list(assets.findall("mesh")):
        if "rubber_hand" in mesh.get("name", ""):
            assets.remove(mesh)
    for side in ("left", "right"):
        source = fixed_wrist(side)
        target = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        for element in list(target):
            if element.tag in ("inertial", "geom"):
                target.remove(element)
        target.insert(0, deepcopy(source.find("inertial")))
        for element in source:
            if element.tag in ("geom", "body"):
                target.append(deepcopy(element))
        for mesh_name in hand_mesh_vertices(side):
            ET.SubElement(assets, "mesh", name=mesh_name,
                          file=str(ASSET_ROOT / "assets" / (mesh_name + ".STL")))
        # Keep original CAT hand-thigh self-collision semantics using the
        # corrected sole collision envelope, without duplicate hand capsules.
        for pair in root.findall("./contact/pair"):
            for attr in ("geom1", "geom2"):
                if pair.get(attr) == f"{side}_hand_collision":
                    pair.set(attr, f"furniture_{side}_hand_envelope")
