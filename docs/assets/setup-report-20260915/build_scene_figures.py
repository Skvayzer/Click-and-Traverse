"""Rebuild report geometry figures from the actual CAT bank and Dex3 assets.

Run from the repository with:
    .venv/bin/python docs/assets/setup-report-20260915/build_scene_figures.py

No simulation, learned rollout, training process, or remote connection is used.
"""
from pathlib import Path
import hashlib
import itertools
import json
import logging
import math
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

from cat_ppo.furniture.generalist_fields import load_generalist_manifest
from cat_ppo.furniture.grippers import (
    ENVELOPE_MARGIN_M, geometry_contract, hand_envelope, hand_mesh_vertices,
)

OUTPUT = Path(__file__).resolve().parent
BANK = ROOT / "data/furniture/cat_generalist"
logging.getLogger("fontTools").setLevel(logging.WARNING)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 10, "text.color": "#23354b", "axes.labelcolor": "#42546c",
    "xtick.color": "#64748b", "ytick.color": "#64748b",
    "axes.edgecolor": "#cbd5e1", "pdf.fonttype": 42})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(figure, name, dpi=240):
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=dpi,
                       bbox_inches="tight", pad_inches=.055)
    plt.close(figure)


def corners(box):
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    hx, hy = box["half_size"][:2]
    xy = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return xy @ np.array([[c, s], [-s, c]]) + box["center"][:2]


def build_layouts(scenes):
    norm, cmap = Normalize(.35, 1.4), plt.get_cmap("YlGnBu")
    figure, axes = plt.subplots(1, 2, figsize=(12.9, 5.15))
    figure.subplots_adjust(left=.01, right=.885, bottom=.115, top=.88, wspace=.045)
    for index, (axis, scene) in enumerate(zip(axes, scenes)):
        axis.set_facecolor("#f7f9fc")
        axis.set_axisbelow(True)
        axis.grid(color="#e7edf4", lw=.6)
        for box in sorted(scene["boxes"], key=lambda b: b["center"][2] + b["half_size"][2]):
            height = box["center"][2] + box["half_size"][2]
            wall, overhead = box["category"] == "wall", box["category"] == "overhead"
            axis.add_patch(Polygon(corners(box), closed=True,
                facecolor="#8796a8" if wall else "#bca7eb" if overhead else cmap(norm(height)),
                edgecolor="#8065a5" if overhead else "#496173", lw=.32,
                alpha=.35 if overhead else 1, hatch="////" if overhead else None, zorder=3))
        route = np.array(scene["route"])
        line, = axis.plot(route[:, 0], route[:, 1], color="#d95479", lw=1.3,
                          ls=(0, (3, 2)), zorder=6)
        line.set_path_effects([patheffects.Stroke(linewidth=2.8, foreground="white", alpha=.8),
                              patheffects.Normal()])
        for first, second in zip(route[:-1], route[1:]):
            axis.annotate("", xy=first + .57 * (second - first),
                xytext=first + .48 * (second - first),
                arrowprops={"arrowstyle": "-|>", "lw": 1.15, "color": "#c9436a", "mutation_scale": 9}, zorder=7)
        if index == 0:
            tables = sorted((b for b in scene["boxes"] if b["category"] == "tabletop"),
                            key=lambda b: (b["center"][1], b["center"][0]))
            for number, box in enumerate(tables, 1):
                axis.text(*box["center"][:2], f"T{number}", ha="center", va="center",
                          fontsize=8.1, weight="bold", color="#173e50", zorder=7)
        for number, gate in enumerate(scene["bottlenecks"], 1):
            x, y = gate["center"]
            axis.scatter(x, y, s=71, facecolor="white", edgecolor="#526479", lw=.75, zorder=8)
            axis.text(x, y, str(number), ha="center", va="center", fontsize=7.1, weight="bold", zorder=9)
        axis.scatter(*scene["start"][:2], c="#16a078", s=65, edgecolor="white", lw=1.1, zorder=10)
        axis.scatter(*scene["goal"], c="#db9c25", marker="*", s=104, edgecolor="white", lw=1, zorder=10)
        for label, xy, color in (("START", scene["start"][:2], "#128561"), ("GOAL", scene["goal"], "#a7771a")):
            axis.text(xy[0] + .35, xy[1] + .25, label, color=color, fontsize=8.3,
                weight="bold", zorder=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1.4})
        axis.set_aspect("equal")
        axis.set_xlim(-.1, 9.1)
        axis.set_ylim(-.1, 9.80)
        axis.set_xticks([0, 3, 6, 9])
        axis.set_yticks([0, 3, 6, 9])
        axis.tick_params(labelsize=9)
        axis.set_xlabel("x (m)", labelpad=0)
        axis.set_ylabel("y (m)", labelpad=0)
        axis.set_title("9 tables + 36 chairs" if index == 0 else "45 generic clutter objects",
                       loc="left", fontsize=13.3, weight="bold", pad=19)
        axis.text(0, 1.018, "298 primitive boxes" if index == 0 else "91 primitive boxes",
                  transform=axis.transAxes, fontsize=9, color="#758296")
    bar_axis = figure.add_axes([.902, .365, .012, .38])
    bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar_axis)
    bar.set_label("Primitive top height (m)", fontsize=9, labelpad=8)
    bar.ax.tick_params(labelsize=8.2)
    bar.outline.set_visible(False)
    figure.text(.894, .224, "9.00 × 9.69 m\n4 cm voxels", fontsize=10.2,
                linespacing=1.45, weight="bold", color="#41546d")
    figure.text(.06, .009, "Dashed pink: geometric reference route  •  Hatched purple: overhead obstruction  •  Circled numbers: six narrow passages",
                fontsize=9.7, color="#627189")
    save(figure, "scene-layouts")


def draw_section(axis, entry, scene, *, compact):
    origin, shape, dx = np.array(entry["origin"]), np.array(entry["shape"]), entry["dx"]
    sdf = np.load(BANK / entry["path"] / "sdf.npy", mmap_mode="r")
    gf = np.load(BANK / entry["path"] / "gf.npy", mmap_mode="r")
    ix = int(round((4.50 - origin[0]) / dx))
    ys = origin[1] + np.arange(shape[1]) * dx
    zs = origin[2] + np.arange(shape[2]) * dx
    mesh = axis.pcolormesh(ys, zs, sdf[ix].T, cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-.2 if compact else -.25, vcenter=0, vmax=.6 if compact else .65),
        shading="nearest", rasterized=True)
    axis.contour(ys, zs, sdf[ix].T, levels=[0], colors="#20364d", linewidths=.65)
    yi, zi = np.arange(6, shape[1] - 5, 8 if compact else 6), np.arange(2, 41, 4 if compact else 3)
    y, z = np.meshgrid(ys[yi], zs[zi], indexing="ij")
    u, v = np.array(gf[ix][np.ix_(yi, zi)][..., 1]), np.array(gf[ix][np.ix_(yi, zi)][..., 2])
    free = np.array(sdf[ix][np.ix_(yi, zi)]) > .04
    axis.quiver(y, z, np.where(free, u, np.nan), np.where(free, v, np.nan), color="#20364d",
        angles="xy", scale_units="xy", scale=2.8 if compact else 3.4,
        width=.0042 if compact else .0025, headwidth=3.4, headlength=4)
    axis.set_xlim(0, scene["room_dimensions"][1])
    axis.set_ylim(0, 1.62)
    axis.set_xticks([0, 2, 4, 6, 8])
    axis.set_yticks([0, .4, .8, 1.2, 1.6])
    axis.tick_params(labelsize=8.7 if compact else 10, pad=2)
    axis.set_xlabel("y through the room (m)", fontsize=9.2 if compact else 10, labelpad=2)
    axis.set_ylabel("Height z (m)", fontsize=9.2 if compact else 10, labelpad=3)
    axis.annotate("Tabletop", xy=(2.23, .75), xytext=(.5, 1.40), fontsize=8.3 if compact else 9,
        color="#24374c", arrowprops={"arrowstyle": "->", "lw": .65, "color": "#24374c"},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95, "pad": 1.6})
    return mesh, ix, float(origin[0] + ix * dx)


def build_fields(entry, scene):
    figure, axis = plt.subplots(figsize=(6.25, 3.67))
    figure.subplots_adjust(left=.12, right=.98, bottom=.30, top=.81)
    mesh, ix, x = draw_section(axis, entry, scene, compact=True)
    figure.text(.12, .953, "Actual 3D CAT field through the furniture", fontsize=12.2, weight="bold")
    figure.text(.12, .885, f"Vertical slice at x = {x:.2f} m; arrows show projected guidance",
                fontsize=8.9, color="#64748b")
    bar_axis = figure.add_axes([.12, .112, .86, .035])
    bar = figure.colorbar(mesh, cax=bar_axis, orientation="horizontal", ticks=[-.2, 0, .2, .4, .6])
    bar.ax.tick_params(labelsize=8.2, pad=1)
    bar.set_label("Signed distance (m): obstacle interior < 0; free space > 0", fontsize=8.1, labelpad=3)
    bar.outline.set_visible(False)
    save(figure, "scene-fields", dpi=260)

    figure = plt.figure(figsize=(14, 6.3))
    grid = figure.add_gridspec(1, 2, width_ratios=[2.65, 1], left=.06, right=.955, bottom=.18, top=.75, wspace=.22)
    axis, notes = figure.add_subplot(grid[0]), figure.add_subplot(grid[1])
    notes.axis("off")
    mesh, _, _ = draw_section(axis, entry, scene, compact=False)
    figure.text(.06, .942, "Why the furniture field is three-dimensional", fontsize=22, weight="bold")
    figure.text(.06, .889, f"Actual stored SDF and guidance vectors through the center table column: x = {x:.2f} m",
                fontsize=11.7, color="#65748b")
    axis.set_title("Signed distance + guidance projected into this vertical slice", fontsize=11.4, loc="left", pad=23)
    bar = figure.colorbar(mesh, ax=axis, orientation="horizontal", pad=.27, fraction=.06, aspect=45)
    bar.set_label("Signed distance to obstacle surface (m)", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    bar.outline.set_visible(False)
    items = [("01  Occupancy", "Canonical furniture boxes\nvoxelized on a 4 cm grid."),
             ("02  CAT potential fields", "Original SDF → boundary gradient\n→ progressive 3D fast marching."),
             ("03  Whole-body queries", "Hands and body sample clearance\nand guidance in 3D."),
             ("39 scenes in one bank", "36 byte-verified original scenes\n+ 1 explicit reconstruction\n+ 2 dense FMM clutter scenes")]
    for index, (heading, body) in enumerate(items):
        ypos = .94 - index * .235
        notes.text(0, ypos, heading, fontsize=11.2, weight="bold", va="top")
        notes.text(0, ypos - .075, body, fontsize=10.1, va="top", linespacing=1.6, color="#617087")
    figure.text(.06, .045, "Blue: free space. Red: obstacle interior. Arrows are stored field values projected into the y–z plane; the full guidance also has an x component.",
                fontsize=9.1, color="#617086")
    figure.text(.06, .019, "This is a field visualization of the current training geometry, not a learned rollout or a collision-free traversal certificate.",
                fontsize=8.8, color="#8894a4")
    save(figure, "scene-field-section", dpi=220)
    return {"family": "furniture", "fixed_x_index": ix, "fixed_x_m": x,
            "displayed_components": ["gf_y", "gf_z"], "other_component_omitted": "gf_x",
            "sdf_file_sha256": entry["fields"]["sdf"]["sha256"],
            "gf_file_sha256": entry["fields"]["gf"]["sha256"]}


def build_hand():
    meshes, envelope = hand_mesh_vertices("left"), hand_envelope("left")
    center, half = np.array(envelope["center"]), np.array(envelope["half_size"])
    vertices = np.concatenate(list(meshes.values()))
    margins = half - np.abs(vertices - center)
    assert np.min(margins) >= ENVELOPE_MARGIN_M - 1e-10
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    box = center + signs * half
    figure = plt.figure(figsize=(6, 3.67))
    axis = figure.add_axes([.03, .07, .94, .83], projection="3d", computed_zorder=False)
    axis.set_proj_type("ortho")
    axis.view_init(elev=23, azim=-64, roll=0)
    axis.set_box_aspect(half * 2)
    light = np.array([-.25, -.65, .72])
    light /= np.linalg.norm(light)
    for name, mesh in meshes.items():
        triangles = mesh.reshape(-1, 3, 3)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        # Elementwise reduction avoids platform BLAS floating-status warnings
        # for tiny but finite STL triangles; no geometry is dropped.
        intensity = .40 + .60 * np.abs(np.sum(normals * light[None, :], axis=1))
        base = np.array([.91, .62, .23]) if "thumb" in name else np.array([.48, .60, .72])
        colors = np.clip(base[None, :] * intensity[:, None], 0, 1)
        assert np.isfinite(colors).all()
        axis.add_collection3d(Poly3DCollection(triangles * 1000, facecolors=colors,
            edgecolors="none", linewidths=0, rasterized=True, zorder=3))
    for first in range(8):
        for second in range(first + 1, 8):
            if np.count_nonzero(signs[first] != signs[second]) == 1:
                axis.plot(*((box[[first, second]] * 1000).T), color="#1192a5", lw=1.0, alpha=.85, zorder=5)
    axis.scatter(*(box * 1000).T, s=18, c="#11697a", edgecolors="white", linewidths=.6, depthshade=False, zorder=10)
    axis.set_xlim(*(np.array([center[0] - half[0], center[0] + half[0]]) * 1000))
    axis.set_ylim(*(np.array([center[1] - half[1], center[1] + half[1]]) * 1000))
    axis.set_zlim(*(np.array([center[2] - half[2], center[2] + half[2]]) * 1000))
    axis.set_axis_off()
    figure.text(.035, .95, "Fixed Dex3 hand and safety envelope", fontsize=12.2, weight="bold")
    figure.text(.035, .884, "Native STL geometry • fingers unactuated", fontsize=9.3, color="#64748b")
    figure.text(.035, .045, "5 mm padding on every side • all 8 corner probes shown", fontsize=9.3, color="#486279")
    figure.canvas.draw()
    thumb_point = np.concatenate([mesh for name, mesh in meshes.items() if "thumb" in name]).mean(axis=0) * 1000
    projected = proj3d.proj_transform(*thumb_point, axis.get_proj())[:2]
    axis.annotate("Thumb included", xy=projected, xycoords="data", xytext=(.72, .22), textcoords="axes fraction",
                  fontsize=9, color="#9c6a1d", arrowprops={"arrowstyle": "->", "lw": .9, "color": "#9c6a1d"},
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    save(figure, "hand-envelope", dpi=260)
    contract = geometry_contract()
    return {"side": "left", "model": contract["model"], "meshes": len(meshes),
        "stl_triangle_vertices_with_duplicates": len(vertices),
        "unique_vertices": len(np.unique(vertices, axis=0)),
        "minimum_mesh_vertex_margin_m": float(np.min(margins)),
        "all_mesh_vertices_contained": True, "envelope": envelope,
        "corners_wrist_frame_m": box.tolist(), "finger_control": contract["finger_control"],
        "actuated_finger_dofs": contract["actuated_finger_dofs"],
        "source_mjcf_sha256": contract["source_mjcf_sha256"],
        "source_upstream_commit": contract["source_upstream_commit"],
        "asset_files_sha256": contract["asset_files_sha256"]}


def scene_facts(manifest, entries, scenes, section, hand):
    result = {"manifest_path": str(BANK / "manifest.json"), "manifest_sha256": manifest["manifest_sha256"],
        "scene_count": manifest["scene_count"], "byte_verified_original_count": manifest["byte_verified_original_count"],
        "reconstructed_original_count": manifest["reconstructed_original_count"], "new_clutter_scenes": 2,
        "reconstructed_scene": "D8G2L3O2S13", "dataset_revision": manifest["dataset_revision"],
        "config_sha256": manifest["released_config_sha256"], "upstream_commit": manifest["upstream_commit"],
        "field_bytes_total": manifest["fields_bytes"], "plots": [], "scenes": [],
        "cross_section": section, "hand_geometry": hand,
        "builder": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
        "report_use": {"room_comparison": "scene-layouts.png/pdf, approximately 774×310 pt",
                       "field_section": "scene-fields.png/pdf, approximately 375×220 pt",
                       "hand": "hand-envelope.png/pdf, approximately 360×220 pt",
                       "route_note": "Geometric check route, not a policy trajectory; route is not used for FMM guidance."}}
    for entry, scene in zip(entries, scenes):
        counts, boxes = scene["counts"], scene["boxes"]
        assert counts["primitive_boxes"] == len(boxes)
        if entry["family"] == "furniture":
            assert len({b["furniture_id"] for b in boxes if b.get("furniture_type") == "table"}) == counts["tables"] == 9
            assert len({b["furniture_id"] for b in boxes if b.get("furniture_type") == "chair"}) == counts["chairs"] == 36
        else:
            assert len({b["object_id"] for b in boxes if b.get("object_id")}) == counts["generic_objects"] == 45
        shape, origin, dx = np.array(entry["shape"]), np.array(entry["origin"]), entry["dx"]
        result["scenes"].append({"scene_id": scene["scene_id"], "family": entry["family"],
            "geometry_sha256": scene["geometry_hash"], "scene_json_sha256": entry["scene_sha256"],
            "counts": counts, "room_dimensions_m": scene["room_dimensions"], "grid_shape": entry["shape"],
            "voxel_m": dx, "sample_origin_m": origin.tolist(), "last_sample_m": (origin + (shape - 1) * dx).tolist(),
            "grid_extent_m": (shape * dx).tolist(), "grid_edge_min_m": (origin - .5 * dx).tolist(),
            "grid_edge_max_m": (origin + (shape - .5) * dx).tolist(), "start_xyz_m": entry["start"],
            "goal_xyz_m": entry["goal"], "geometric_route_length_m": scene["route_length_m"],
            "bottleneck_width_min_m": min(g["width_m"] for g in scene["bottlenecks"]),
            "bottleneck_width_max_m": max(g["width_m"] for g in scene["bottlenecks"]),
            "root_geometric_admission": scene["feasibility"], "field_generation": entry["source"]})
    for file in sorted(OUTPUT.iterdir()):
        if file.suffix in (".png", ".pdf") and (file.name.startswith("scene-") or file.stem == "hand-envelope"):
            result["plots"].append({"path": str(file), "sha256": digest(file), "bytes": file.stat().st_size})
    return result


def main():
    manifest = load_generalist_manifest(BANK / "manifest.json")
    entries = [next(entry for entry in manifest["scenes"] if entry["family"] == family)
               for family in ("furniture", "generic_clutter")]
    scenes = [json.loads((BANK / entry["path"] / "scene.json").read_text()) for entry in entries]
    build_layouts(scenes)
    section = build_fields(entries[0], scenes[0])
    hand = build_hand()
    facts = scene_facts(manifest, entries, scenes, section, hand)
    payload = json.dumps(facts, indent=2) + "\n"
    (OUTPUT / "scene-facts.json").write_text(payload)
    print(json.dumps({"plots": [record["path"] for record in facts["plots"]],
                      "facts": str(OUTPUT / "scene-facts.json"), "hand_validation": hand}, indent=2))


if __name__ == "__main__":
    main()
