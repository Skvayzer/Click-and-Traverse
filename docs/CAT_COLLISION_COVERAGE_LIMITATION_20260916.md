# Original limitation: incomplete body–obstacle collision coverage

**Implementation update:** the approved [full-body primitive training checks](CAT_BODY_COLLISION_TRAINING_20260916.md) now address this sparse-point coverage gap, with explicit training penalties and termination. The report below records the earlier limitation and evidence. Primitive approximation, voxel geometry and discrete-time limitations remain; this is not a continuous full-mesh collision guarantee.

Status on **2026-09-16**: confirmed limitation of the pinned released CAT training implementation and the inherited checks in this branch. **Not fixed by this documentation update.** Training remains stopped; observations, rewards, termination rules, physics and checkpoint selection are unchanged.

CAT can miss a robot surface penetrating an obstacle when its sampled body points remain outside. An episode can consequently continue through an actual geometric intersection. The reset mechanism responds to detected failures; the incomplete part is collision detection. A reported goal success or absence of an obstacle-field violation does **not** establish collision-free whole-body traversal.

## What released CAT checks

The reference is upstream commit `866ba392f1c1e84b92ad75fa66550f26e8af8e48` and the [pinned released generalist configuration](../configs/cat_generalist_released.json).

- The code samples the obstacle signed-distance field at **11 site centers**: head, pelvis, torso, both feet, both palms, both knees and both shoulders. These are **point samples**, not enclosing collision spheres. It does not subtract a foot or shin radius. [Upstream sampling](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/cat_ppo/envs/g1/env_cat.py#L637-L658).
- Obstacle termination tests whether any sampled distance is below `-term_collision_threshold`. The released generalist uses **0.0 m**. The generic source default, **0.04 m**, permits sampled-point penetration; it is not a protective clearance margin. Obstacle and selected self-contact termination have a **50-control-step / 1-second grace period**. Falls and numerical failures have separate checks. [Upstream termination](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/cat_ppo/envs/g1/env_cat.py#L821-L847).
- The distance-reward helper's **5 cm soft margin** encourages point clearance; it neither defines a collision sphere nor guarantees clearance of the surrounding body. [Upstream reward helper](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/cat_ppo/envs/g1/env_cat.py#L1200-L1210).
- Training simulates foot–floor and selected self-contacts. The flat training scene has no passage-wall or furniture geometry producing robot–obstacle contact forces. A visual obstacle mesh in a recording does not establish physical obstacle contacts. [Training scene](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain.xml#L29-L39), [explicit contact pairs](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/data/assets/unitree_g1/g1_mjx_feetonly_torque.xml#L466-L475).

The count above describes the released code we use. The [paper's representation description](https://arxiv.org/html/2601.16035v1#S3.SS1.SSS2) states 13 sampling locations; the two should not be conflated. Grid resolution and interpolation introduce additional approximation beyond the body-coverage limitation.

## How a foot or shin collision can be missed

The original foot **collision box** is 18 × 6 × 1.6 cm. Its sole-center query site leaves approximately **9 cm of toe/heel extent and 3 cm of lateral extent**, measured in the foot's local frame, beyond the sampled point. A toe can enter a wall while the point still has positive signed distance. These dimensions describe the configured collision box, not a measurement of the visual foot mesh. [Foot box and site](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/data/assets/unitree_g1/g1_mjx_feetonly_torque.xml#L198-L206).

Similarly, querying a knee and a foot does not check the shin surface between them or the thigh above the knee. The existing shin capsules participate in selected self-contacts; they do not supply whole-leg wall collision detection. The same coverage problem can affect torso and arm surfaces between samples. [Knee and shin geometry](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/data/assets/unitree_g1/g1_mjx_feetonly_torque.xml#L174-L190).

## Coverage in our current extension

The [compact setup](CAT_COMPACT_SPHERE_SETUP_20260916.md) replaces the two original palm point clearances with enclosing-hand sphere clearances and adds two elbow spheres. Nine original locations remain point samples. The [room correction](ROOM_NAVIGATION_FIX_20260916.md) additionally checks a swept trunk cylinder against canonical room geometry, without an initial grace period. That guard is room-only and does not cover every articulated body part.

Corrected furniture voxelization addresses missing obstacle geometry, and corrected guidance addresses route following. Neither supplies complete moving-body collision coverage. The current **222 actor inputs, 310 critic inputs and 29 actions** are unchanged.

## Evidence and interpretation of recorded results

The [earlier clutter evaluation](CLUTTER_CHECKPOINT_EVALUATION_20260916.md#visible-limitation) documented a visible tabletop–trunk intersection in room 4002, seed 8, before an elbow-field failure ended the episode. That recording predates the room trunk guard. It illustrates the earlier coverage gap, not a claim that the corrected guard failed on that trajectory.

The latest `02_narrow_passage.mp4` is **not a confirmed penetration example**. A separate read-only audit of scene `procedural-D4G0L2O0S2001`, seed 0, checked all **49 unique robot meshes** at **106 saved poses**, plus joint-manifold interpolation at 5 ms spacing (**421 poses total**). Each robot mesh's world-space bounding box was disjoint from every connected obstacle component's bounding box. Disjoint enclosing boxes establish mesh separation at the checked poses. The conservative separation lower bound was **28.27 cm for all body meshes** and **28.39 cm for legs/feet**; these are lower bounds, not exact nearest-surface distances. The apparent overlap is consistent with camera occlusion.

Actual compiled obstacle placement was checked against the saved OBJ and its `[-0.5, -1, 0]` translation; maximum vertex discrepancy was approximately `1.03e-7 m`. This audit covers the rendered geometry at sampled/interpolated poses, not continuous swept motion, physical collision proxies or other videos. [Audit evidence and source hashes](assets/collision-coverage-20260916/narrow-passage-audit.json), [recording details](CAT_PROCEDURAL_VIDEOS_20260916.md).

Existing video `PASS` labels, metadata `strict_success` fields and validation success rates retain their recorded task/field-based meaning. They must not be interpreted as full-body mesh collision certification. This report does not relabel historical results or change checkpoint eligibility.

## Implications for further work

An evaluation geometry audit can expose missed collisions and distinguish goal completion from geometrically clear traversal. **Evaluation alone cannot correct the training signal:** an intersection that no training check detects may produce no corresponding termination or collision penalty.

Resolving that learning limitation requires broader **training-side** body coverage, for example validated body collision proxies or additional internal surface checks. Those checks can affect reward or termination **without adding observations or feeding mesh data to the policy**. Such a change would be an explicit extension beyond released CAT and requires separate implementation, coverage validation and GPU-capacity measurement. No such additional change has been applied by this report, and no full-body collision-free guarantee is claimed.

The [mesh-fitted collision-shape proposal](COLLISION_PROXY_PROPOSAL_20260916.md)
was approved, including flat foot boxes and close-up foot views. Its subsequent
training implementation is documented in the update linked at the top.
