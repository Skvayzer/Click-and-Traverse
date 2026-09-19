# Table-height hand-protection examples

These are actual MuJoCo renders of the canonical oriented-box training geometry
with prescribed static G1 / Dex3 poses. They are **not learned policy rollouts**.
Only perimeter walls are drawn low for visibility. Table slabs, aprons, legs and
exterior lane blockers keep their collision dimensions.

- `table-edge-scenes.png`: three generated examples, seeds 20260919–20260921,
  matching the new training bank.
- `table-edge-hand-comparison.png`: identical geometry, root position and camera
  with nominal versus raised arms, for seed 20260919.
- `scene-*.json`: complete scene geometry and static safety certificates.
- `render-audit.json`: exact pose and separation measurements used in labels.
- Individual close views and whole-room overviews are also provided.

The three table heights are 71.9, 70.5 and 71.5 cm; gaps are approximately
43.7, 44.4 and 43.9 cm. The raised hand spheres' lowest points are 83.6 cm above
the floor, leaving 11.8, 13.2 and 12.1 cm above tabletop height. In the first
comparison, nominal hand envelopes intersect the furniture by 10.7 cm;
the raised pose has 13.7 cm minimum hand-envelope separation and 4.1 cm
minimum separation over the approved body collision primitives.

Vertical height above the tabletop is a different quantity from minimum
Euclidean separation to all obstacle boxes; both are recorded separately.
Geometry certificates are sampled kinematic checks, not guarantees of dynamic
traversal or proofs that no alternative whole-body posture exists.

Regenerate from repository root:

```sh
JAX_PLATFORMS=cpu MUJOCO_GL=glfw .venv/bin/python scripts/render_table_edge_passages.py
```

The renderer always regenerates the scene from the current generator, avoiding
stale cached geometry. macOS requires a working CoreGraphics connection.
