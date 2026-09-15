# Exact CAT paper reproduction: verified status

Date: 16 September 2026, Dubai. **Exact reproduction is blocked on missing paper artifacts. It has not been implemented or claimed complete.** The running experiment, its frozen source, and its field bank were not changed by this audit.

## Requirement and scope

The requested baseline is the paper's actual training collection and matching setup. Arbitrarily generating 216 procedural scenes and 139 indoor crops would reproduce the counts, not the collection. Our whole-body controls and hand-protection changes remain separately identified extensions; they are not evidence that the original paper baseline has been reproduced.

## What was checked

- Official repository: `GalaxyGeneralRobotics/Click-and-Traverse`, current `main` at `866ba392f1c1e84b92ad75fa66550f26e8af8e48`; its published branch/tag/release metadata and local tracked history.
- Official project website and paper, including the downloadable paper link.
- Public dataset: `Axian12138/Click-and-Traverse`, revision `db8c202fa724bc4d3dba2cb9fead1267f15fd811`; all 19 published dataset revisions, with complete recursive listings and no unprocessed pagination.
- Model release: revision `46ce4b57ba0639168d51741b661ff62f7ce6f045`, including the final generalist configuration.
- The two archives linked through the official Google Drive and Tsinghua Cloud mirrors. Archive directory metadata and small configuration members were inspected without downloading complete archives.
- Maintainer replies in issues #11, #9, #8, #1, #6 and #15.

The source inventory is saved in [public-inventory.json](assets/cat-exact-reproduction-20260916/public-inventory.json). Mirror results are recorded in [mirror-inventory.json](assets/cat-exact-reproduction-20260916/mirror-inventory.json).

## Collection coverage

| Collection | Verified finding |
| --- | --- |
| Paper | 139 accepted cropped 3D-FRONT scenes plus 216 procedural scenes, 355 total |
| Current public random assets | 36 distinct SDF files, all already represented by original bytes in our current bank |
| Public typical assets | 27 distinct SDF geometries across both release versions; the 22 older typical SDF files duplicate geometries in the 27 newer files |
| Historical public dataset | The same 63 unique SDF hashes across all 19 revisions; no historical-only geometry found |
| Official mirror archives | 36 random assets; 26 typical layouts in the older archive and 27 in the generalist archive; no 3D-FRONT crop collection |
| Pinned final generalist config | 37 random scene paths at difficulty 0.8, with a corresponding 37-teacher list |
| Missing configured scene | `D8G2L3O2S13` is named by checkpoint metadata but its exact asset is absent from the inspected releases; our existing bank contains an explicitly labeled reconstruction |

Identical SDF hashes establish byte identity of those files, not membership in the paper's undisclosed split. Nor do scene names alone prove identity with paper scenes. The README's statement of 42 random scenes is not consistent with the inspected 37-path checkpoint config and 36-asset inventory; no extra scene list should be invented from that discrepancy.

The authors confirm that the 139 indoor-scene specialist models were not open-sourced and that some procedural specialists covered multiple similar scenes. **355 scenes does not mean 355 distinct specialist models.** See [author clarification in issue #11](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/issues/11#issuecomment-4147292151). This statement concerns model availability; absence of the crop assets is a separate finding from the inspected file inventories.

## Required artifacts

1. **139 accepted indoor crops:** exact 3D-FRONT scene/room identifiers, object asset versions, crop centers/transforms, initial positions, and inclusion decisions after manual filtering; or the final geometry/field assets with authoritative hashes.
2. **216 procedural scenes:** complete parameters and seeds, exact generator revision and dependencies, field-grid settings, and hashes; or the exact resulting assets.
3. **Matching paper training implementation:** body measurement locations, observation/action contract, reward implementation/coefficients, reset and goal sampling, curriculum, all PPO/DAgger stage settings, and specialist-to-scene assignments.
4. **Model and evaluation provenance:** matching paper teacher/generalist checkpoints if reusing trained models, plus evaluation manifests and success criteria.

The generic 3D-FRONT dataset and the current procedural generator cannot identify the missing crop selections, manual filtering decisions, or unknown procedural seeds.

## Why the released setup is insufficient evidence of paper identity

The [paper's Section III-B](https://arxiv.org/html/2601.16035v2#S3.SS2) describes 5 m by 5 m indoor crops selected after planar walkable-space erosion by 0.1 m, goals sampled on a 2 m circle, and manual removal of unsuccessful scenes. It reports specialist PPO with 32,768 parallel environments and 5,000,000 episodes, followed by DAgger and a difficulty curriculum. These descriptions do not supply an exact accepted-scene manifest or all launch configurations.

Several public implementation details need explicit reconciliation with the paper-version code:

- The paper describes 13 measured body parts; the released native task uses 11 field sample sites and a 162-input actor.
- The released field metadata has a fixed goal. Root state randomization is not equivalent to generating a new random-goal field per episode.
- The released reward contains distance-windowed cosine alignment and a crossed-region baseline. The authors discuss implementation differences from the paper's mathematical reward in [issue #9](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/issues/9#issuecomment-4066969772).
- Our existing experiment performs direct PPO fine-tuning of the released generalist using 8,192 environments and 39 fixed scene slots. It does not execute the paper's full specialist-to-generalist training procedure.

These are documented differences and missing provenance, not instructions to guess replacement settings. Matching a public default or passing a unit test cannot establish exact paper reproduction.

## Next concrete step

Obtain the missing artifacts or a definitive mapping from the authors. A [reviewable request](CAT_PAPER_ARTIFACT_REQUEST.md) is prepared but has not been sent. Once assets arrive, verify their source, hashes, scene membership, coordinate conventions, and matching configurations before changing the experiment. No approximate replacement dataset or new training run was created during this audit.
