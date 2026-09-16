# Generated CAT clutter examples

`clutter-examples.png` shows three exact scenes from the active training bank on ws008090, with a 3D cutaway view and matching floor plan. Individual views are `scene-4008.png`, `scene-4016.png`, and `scene-5001.png`; separate 3D and plan images are also included.

| Seed | Contents | Room | Checked root route |
| --- | --- | --- | --- |
| 4008 | 9 tables, 35 chairs | 10.1 × 9.6 m | 10.0 m |
| 4016 | 8 tables, 36 chairs | 9.8 × 10.9 m | 8.1 m |
| 5001 | 43 crates, shelves, blocks and partitions | 9.5 × 9.4 m | 5.7 m |

Objects retain their generated positions, orientations and dimensions. The G1 with fixed Dex3 hands is shown in its nominal starting pose. Blue A is the start, orange B is the goal, and green is the generator's geometrically checked route for a 23 cm root radius. These are static geometry previews, not learned rollouts or evidence of whole-body dynamic feasibility. Walls are cut away in 3D for visibility. Colors and lighting are presentation settings.

The active bank manifest SHA256 is `dc97e6e19c7767c5d5b567f5abdde0a4fb8a85873893be22adff7863e1b8f6c0`. The selected scene files were read from ws008090 and checked against that bank's hashes. `active-scenes.json` contains the exact scene objects and their bank records. `render-provenance.json` records the scene identities and rendered image hash. Rendering ran locally on the Mac without policy inference or training changes.

Reproduce from the project checkout:

```sh
.venv/bin/python scripts/render_clutter_examples.py \
  --source /Users/konstantinsmirnov/research/CAT-Clutter-Examples-20260916/active-scenes.json \
  --output /Users/konstantinsmirnov/research/CAT-Clutter-Examples-20260916
```

## Realistic asset recommendation

Start with **Poly Haven**, whose public library offers CC0 models without signup or a paywall: https://polyhaven.com/

- [Wooden Table 02](https://polyhaven.com/a/wooden_table_02): a rectangular wooden table with four straight legs; a close starting point for the generated table shape.
- [School Chair 01](https://polyhaven.com/a/SchoolChair_01): a textured plastic school chair with metal legs.
- [School Desk 01](https://polyhaven.com/a/SchoolDesk_01): another useful classroom/table variation.

The first two models provide Blender, glTF, USD and FBX downloads, with diffuse, normal, roughness and metal maps. For room surfaces and lighting, [ambientCG](https://ambientcg.com/) provides [PBR materials and HDRIs](https://docs.ambientcg.com/asset-types/) under [CC0](https://docs.ambientcg.com/license/). For a larger object catalogue later, [Objaverse's API](https://objaverse.allenai.org/docs/objaverse-1.0/) downloads GLB models and exposes per-object metadata; licenses and geometry need individual filtering.

Suggested implementation: preserve generated poses, fit a few reusable furniture models to the intended dimensions, use 1K–2K textures, and render recorded robot trajectories with those assets. This adds visual realism without adding policy observations or textured meshes to the training process.

For truthful hand-protection demos, matching an overall bounding box is insufficient. Table edges and undersides, chair backs, legs, and armrests must agree with the obstacle fields. Either fit the visual mesh closely to the existing primitives or derive simplified geometry from the asset, regenerate its SDF/navigation fields, and recheck the route. A visually protruding chair part must not be absent from the obstacle field used by the policy.

Asset sources were checked on 16 September 2026. Assets are recommendations and have not been installed into these scenes.
