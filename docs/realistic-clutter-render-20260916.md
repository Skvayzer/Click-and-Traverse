# Realistic clutter appearance study — 16 September 2026

The rendered room uses the layout of `random-furniture-dense-train-004008-73ed3af356e1` from the active CAT training bank: **9 tables and 35 chairs**, in a **10.099 × 9.604 m** room. Furniture positions, bounding dimensions and yaw angles come from scene seed **4008**. Chair backs are oriented toward the generator's local negative Y; generated armrest boxes are retained, with visual support posts added.

This is a **static appearance study**, not a policy rollout. The imported furniture meshes are fitted to each object's bounding dimensions; their detailed surfaces, legs, seat heights and gaps differ from the box geometry used to generate the training SDF. The render does not change training geometry or establish collision-free traversal with these detailed meshes. Cutaway walls, baseboards, noticeboard, tile surface and lighting are visual staging. G1 stands at the scene's start position and heading using native `DEFAULT_QPOS`, with fixed Dex3 fingers. No simulation steps or policy inference are performed.

The furniture and floor assets are [Poly Haven CC0 assets](https://polyhaven.com/license). The downloads preserve the official API responses and verify every file against the API's size and MD5. `assets/manifest.json` additionally records SHA256 hashes, source URLs and author credits.

| Asset | Author | Download used |
| --- | --- | --- |
| [Wooden Table 02](https://polyhaven.com/a/wooden_table_02) | Serhii Khromov | 2K glTF, binary mesh and three PBR textures |
| [School Chair 01](https://polyhaven.com/a/SchoolChair_01) | Ethan Place | 2K glTF, binary mesh and three PBR textures |
| [Floor Tiles 02](https://polyhaven.com/a/floor_tiles_02) | Rob Tuytel | 2K color, OpenGL normal, roughness and displacement textures |

The downloaded asset files total **6,939,173 bytes**. The renderer uses floor color, normal and roughness maps; the downloaded displacement map is retained but does not displace the floor. Asset downloads are reused only after checksum verification.

## Reproduction

Run from the repository root. Set the artifact directory below. The included `docs/assets/realistic-clutter-20260916/active-scenes.json` is the existing scene snapshot bundle produced for the clutter previews. It contains `active_bank_sha256` and a `scenes` list whose entries contain the bank `record` and complete generated `scene`, including boxes, start, goal, room dimensions and geometry hash. It is an input artifact, not a layout reconstructed from the rendered image.

The source bank SHA256 is:

```text
dc97e6e19c7767c5d5b567f5abdde0a4fb8a85873893be22adff7863e1b8f6c0
```

The asset fetcher uses the Python standard library. Robot export uses the project's existing environment and dependencies, tested with Python **3.12.9**, MuJoCo **3.3.1**, trimesh **5.1.0** and NumPy **2.1.3**, together with the repository's CAT dependencies and robot assets. Rendering uses a separate environment, tested with Python **3.11.16**, `bpy` **4.5.3** and NumPy **1.26.4**. Keeping the render environment separate avoids changing training dependencies.

```bash
ARTIFACT_DIR=/path/to/CAT-Realistic-Clutter-20260916
SOURCE_BUNDLE=docs/assets/realistic-clutter-20260916/active-scenes.json

python3 scripts/fetch_realistic_furniture.py \
  --output "$ARTIFACT_DIR/assets" --resolution 2k

.venv/bin/python scripts/export_realistic_cat_robot.py \
  --output "$ARTIFACT_DIR"

python3.11 -m venv "$ARTIFACT_DIR/.render-venv"
"$ARTIFACT_DIR/.render-venv/bin/python" -m pip install \
  'bpy==4.5.3' 'numpy==1.26.4'

"$ARTIFACT_DIR/.render-venv/bin/python" scripts/render_realistic_clutter.py \
  --source "$SOURCE_BUNDLE" \
  --assets "$ARTIFACT_DIR/assets" \
  --robot "$ARTIFACT_DIR/g1_static.glb" \
  --output "$ARTIFACT_DIR" \
  --seed 4008 --samples 96
```

Adding `--preview` produces smaller 24-sample previews. Full rendering uses Blender Cycles, preferring a supported Metal GPU on macOS and otherwise falling back to CPU. It produces overview and interior PNGs at **2400 × 1688**, plus a `.blend` scene with packed textures. Render speed depends on the available device.

`g1_static.json` records native pose, source and compiled XML hashes, bounds, per-mesh checksums and the GLB checksum. The exporter verifies **69 visual meshes**, transforms MuJoCo Z-up coordinates to glTF Y-up, and checks finite vertices and unchanged bounds after reloading the exported GLB. Blender's importer converts them back to Z-up.

`render-provenance.json` records source scene/bank hashes, asset and robot hashes, all furniture placements, rendering settings and output image hashes. The `.blend` file permits inspecting the detailed meshes directly; the source scene bundle and training fields remain the authority for training geometry.
