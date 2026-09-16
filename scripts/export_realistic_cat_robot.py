#!/usr/bin/env python3
"""Export the native compact CAT G1 visual meshes in a static pose as glTF.

MuJoCo supplies forward kinematics from DEFAULT_QPOS. Fixed Dex3 fingers are
inherited from the assembled training robot. No policy or rollout is loaded.
The output is glTF Y-up, ready for Blender's standard glTF import conversion.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".robot-export-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def export_robot(output: Path, expected_mesh_count: int = 69) -> dict:
    import mujoco
    import numpy as np
    import trimesh
    from cat_ppo.envs.g1.constants import DEFAULT_QPOS
    from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml

    def valid_vertices(vertices, name):
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError(f"Invalid vertices for {name}: {vertices.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError(f"Non-finite vertices for {name}")
        if np.max(np.abs(vertices)) > 5:
            raise ValueError(f"Implausible robot extent for {name}")

    def bounds(vertices):
        return np.stack((vertices.min(axis=0), vertices.max(axis=0))).tolist()

    output.mkdir(parents=True, exist_ok=True)
    assembled_xml = assemble_training_xml()
    model = mujoco.MjModel.from_xml_string(assembled_xml)
    data = mujoco.MjData(model)
    data.qpos[:] = DEFAULT_QPOS
    mujoco.mj_forward(model, data)
    with tempfile.TemporaryDirectory(prefix="cat-compiled-xml-") as temporary_dir:
        xml_path = Path(temporary_dir) / "compiled.xml"
        mujoco.mj_saveLastXML(str(xml_path), model)
        compiled_xml_sha256 = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    scene = trimesh.Scene()
    records = []
    all_zup = []
    all_yup = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] <= 0 or model.geom_group[geom_id] >= 3:
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        local_vertices = np.asarray(
            model.mesh_vert[vertex_start:vertex_start + vertex_count], dtype=np.float64
        )
        faces = np.asarray(
            model.mesh_face[face_start:face_start + face_count], dtype=np.int64
        ).copy()
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        # Keep this explicit einsum: some local BLAS-backed matmul paths report
        # numerical warnings even for these small, bounded rigid transforms.
        vertices_zup = np.einsum("ij,kj->ik", local_vertices, rotation, optimize=False) + position
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        name = f"{body_name}_{geom_name}"
        valid_vertices(vertices_zup, name)
        if faces.min() < 0 or faces.max() >= len(vertices_zup):
            raise ValueError(f"Invalid triangle indices for {name}")
        # R_x(-pi/2): MuJoCo Z-up -> glTF Y-up, no matrix/BLAS operation.
        vertices_yup = np.column_stack((vertices_zup[:, 0], vertices_zup[:, 2], -vertices_zup[:, 1]))
        valid_vertices(vertices_yup, name)
        material_id = int(model.geom_matid[geom_id])
        rgba = np.asarray(model.mat_rgba[material_id] if material_id >= 0 else model.geom_rgba[geom_id])
        material = trimesh.visual.material.PBRMaterial(
            name=f"native_{name}",
            baseColorFactor=np.rint(np.clip(rgba, 0, 1) * 255).astype(np.uint8),
            metallicFactor=0.15,
            roughnessFactor=0.4,
        )
        mesh = trimesh.Trimesh(vertices=vertices_yup, faces=faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
        scene.add_geometry(mesh, geom_name=name, node_name=name)
        all_zup.append(vertices_zup)
        all_yup.append(vertices_yup)
        records.append({
            "name": name,
            "body": body_name,
            "geom_id": geom_id,
            "mesh_id": mesh_id,
            "vertex_count": vertex_count,
            "triangle_count": face_count,
            "native_rgba": rgba.tolist(),
            "bounds_mujoco_zup_m": bounds(vertices_zup),
            "vertices_gltf_yup_sha256": hashlib.sha256(vertices_yup.tobytes()).hexdigest(),
            "triangles_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
        })
    if len(records) != expected_mesh_count:
        raise ValueError(f"Expected {expected_mesh_count} native visual meshes; found {len(records)}")
    zup_vertices = np.concatenate(all_zup)
    yup_vertices = np.concatenate(all_yup)
    zup_extent = zup_vertices.max(axis=0) - zup_vertices.min(axis=0)
    if not (0.8 < zup_extent[2] < 2.2 and np.all(zup_extent[:2] < 2)):
        raise ValueError(f"Implausible G1 dimensions: {zup_extent.tolist()}")
    payload = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    reloaded = trimesh.load(io.BytesIO(payload), file_type="glb", force="scene", process=False)
    if len(reloaded.geometry) != len(records):
        raise ValueError("GLB reload changed the mesh count")
    reloaded_vertices = []
    for name, mesh in reloaded.geometry.items():
        valid_vertices(np.asarray(mesh.vertices), f"reloaded/{name}")
        reloaded_vertices.append(np.asarray(mesh.vertices))
    reloaded_vertices = np.concatenate(reloaded_vertices)
    if not np.allclose(bounds(reloaded_vertices), bounds(yup_vertices), atol=1e-6, rtol=0):
        raise ValueError("GLB reload changed the geometry bounds")
    glb_path = output / "g1_static.glb"
    atomic_write(glb_path, payload)
    record = {
        "source": "Native compact CAT assemble_training_xml; MuJoCo FK; fixed Dex3 hands",
        "policy_loaded": False,
        "simulation_steps": 0,
        "qpos": data.qpos.tolist(),
        "source_xml_sha256": hashlib.sha256(assembled_xml.encode()).hexdigest(),
        "compiled_xml_sha256": compiled_xml_sha256,
        "compiled_xml_hash_method": "mujoco.mj_saveLastXML after compilation",
        "gltf_axis_conversion": "R_x(-pi/2): native(x,y,z) -> glTF(x,z,-y)",
        "mesh_count": len(records),
        "bounds_mujoco_zup_m": bounds(zup_vertices),
        "bounds_gltf_yup_m": bounds(yup_vertices),
        "finite_vertices_checked_before_and_after_glb_reload": True,
        "glb_path": str(glb_path.resolve()),
        "glb_sha256": hashlib.sha256(payload).hexdigest(),
        "glb_size_bytes": len(payload),
        "meshes": records,
    }
    atomic_write(output / "g1_static.json", (json.dumps(record, indent=2) + "\n").encode())
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-mesh-count", type=int, default=69)
    args = parser.parse_args()
    record = export_robot(args.output.expanduser().resolve(), args.expected_mesh_count)
    print(json.dumps({key: value for key, value in record.items() if key not in ("meshes", "qpos")}, indent=2))


if __name__ == "__main__":
    main()
