# Contrastive hand-protection examples

These images render the canonical MuJoCo G1/Dex3 scene geometry with **illustrative static poses**. They do not show new learned behavior or a dynamic rollout.

- `contrastive-scenes.png`: one matched group, with open, forward-protected, narrow, and protected/narrow/protected variants.
- `protected-postures.png`: the identical protected corridor with nominal, raised, and tucked arms. Hand spheres are red when intersecting and green when clear.
- The four `*-scene.json` files are the exact certified scenes shown. `render-audit.json` gives full-body and hand-sphere clearances for the pictured poses.

The sealed-lane v2 training bank (`data/furniture/contrastive_hand_v2`) contains 16 matched groups (seeds 20260919–20260934), 64 scenes, all in the training split. Each group retains the same route, cabinet dimensions, small scene rotation, and translation. It changes corridor width: approximately 1.51 m open, 0.735 m protected, and 0.405 m narrow. Each room has three modules with open spaces for arm transitions and turning; six full-height outer baffles seal the lanes between cabinet ends and perimeter walls, so the intended central gaps cannot be bypassed along the sides; the transition role alternates protected/narrow/protected.

The open role certifies hand-field clearance above 0.30 m throughout the checked poses, so the existing neutral-arm posture regularizer reactivates without an extra reward override. Cabinets are 1.24–1.28 m high. Their full height is important: the narrow variant intersects the checked nominal, raised, and tucked **forward** poses, while the checked sideways pose clears. Low tabletops would permit a raised forward alternative even in the narrow variant. Only perimeter walls are displayed low for visibility; canonical training geometry and certification keep their full height.

An analytic OBB cross-section interval check verifies that each module has exactly one floor-level opening, equal to its central gap. A regression removes one baffle and confirms that certification rejects the resulting exterior bypass. Certification also checks the approved 35 collision primitives along the route, sampled arm interpolation and turns, two floor-referenced crouches, and two kinematic stepping stances. It also checks the actual conservative 4 cm hand-distance field. Hand target boxes are checked over their straight swept route against analytic hand-sphere clearance and all adjacent SDF grid values. These are sampled kinematic certificates, not proofs of dynamic feasibility, exhaustive whole-body search, or safety of every arm configuration reaching a target box.

Regenerate using:

```sh
JAX_PLATFORMS=cpu MUJOCO_GL=glfw .venv/bin/python scripts/render_contrastive_passages.py
```
