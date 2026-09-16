# Original CAT procedural environment recordings

Three deterministic seed-0 rollouts use the selected best whole-body checkpoint at **157,286,400 steps** from run `de6ae369`. The scenes come from the unmodified CAT procedural generator and the current complete training bank. Training remains stopped.

| Video | Scene | Recorded duration | Outcome |
|---|---|---:|---|
| `01_hurdles.mp4` | `procedural-D4G2L0O0S2001` | 2.12 s | Goal reached |
| `02_narrow_passage.mp4` | `procedural-D4G0L2O0S2001` | 2.10 s | Goal reached |
| `03_mixed_obstacles.mp4` | `procedural-D7G1L2O1S2001` | 2.38 s | Goal reached |

The MP4s are on the Mac in `/Users/konstantinsmirnov/Downloads/CAT-Original-Environments-20260916`. Each has a second camera view, real-time playback and a labeled two-second final-pose hold. Obstacle translucency and hidden palm-sphere visualizations affect presentation only. No motion was synthesized during rendering.

Saved occupancy hashes match the bank. Mesh extraction matches CAT's original `better_mesh` and `marching_cubes_mesh` functions, with the original `[-0.5, -1, 0]` translation. An exact vertex/face comparison passed on the mixed scene. All three MP4s decoded successfully and their contact sheets were inspected. These are illustrative rollouts, not a success-rate benchmark.

**Collision interpretation:** goal outcomes, `PASS` overlays and metadata `strict_success` use the existing task/field checks, not full-body geometry certification. A subsequent read-only audit found no robot–obstacle mesh intersection in the narrow-passage example at 106 recorded poses plus intermediate poses (421 total, 5 ms spacing). That result is specific to this recording and sampled geometry. See the [remaining collision coverage limitation and audit evidence](CAT_COLLISION_COVERAGE_LIMITATION_20260916.md).

Recordings: `analysis/cat-procedural-videos-20260916` on ws008090; original trajectories/meshes copied to `/Users/konstantinsmirnov/research/CAT-Procedural-Videos-20260916/episodes` on the Mac. [Hashes and outcomes](assets/cat-procedural-videos-20260916/recordings.json).

The recorder now accepts `--record-scene-ids` and `--deterministic-only`. Original CAT mesh export additionally needs `scikit-image` and `trimesh`; ws008090 uses scikit-image 0.25.2 and lazy-loader 0.4, installed without upgrading existing training packages. The exact native meshing arithmetic avoids importing unrelated PyTorch visualization helpers.
