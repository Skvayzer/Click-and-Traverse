# Option C: existing hand scenes with certified posture objectives

Built on 2026-09-19. No training was launched or live process changed. All geometry,
field, certificate, reward and runtime checks ran on CPU. The source hand bank and
its collision/reset manifests remain unchanged. This is a fresh-run bank, not an
exact resume of the currently running width experiment.

## Published assets and composition

- Fields/metadata: `data/furniture/cat_hand_posture_v1_20260919/manifest.json`
- Collision: `data/furniture/cat_hand_posture_v1_20260919_collision/manifest.json`
- Resets: `data/furniture/cat_hand_posture_v1_20260919_resets/manifest.json`
- Parent: `data/furniture/cat_hand_protection_v1_20260917/manifest.json`

| Population | Scenes | Reset mass |
|---|---:|---:|
| Original CAT | 37 | Shares 20% original/published bucket |
| Published CAT | 27 | Shares 20% original/published bucket |
| Procedural CAT | 2,226 | 40% |
| Ordinary furniture | 24 | 12.5% |
| Ordinary generic clutter | 24 | 7.5% |
| Hand table aisles | **12** | **12.5%** |
| Hand shelf passages | **12** | **7.5%** |
| Total | **2,362** | **100%** |

The first 2,338 records are identical to the parent, in the same order, with all
field hashes unchanged. Each hand family has four easy, four medium and four hard
scenes. There are **no cabinet width rungs** in this bank. The existing single hand
curriculum remains: 64 first outcomes and 35% all-policy raw clean goals at the
current level; earlier levels remain eligible. Passing this gate does not prove
posture compliance. Family and hand-subgroup masses remain fixed under adaptation.

Retention fields are referenced through a hash-pinned parent, avoiding a 13+ GiB
copy across filesystems. The 24 hand scenes have byte-identical field arrays and
new scene/source metadata. Collision arrays and every reset pose are unchanged;
their new manifests bind the new field manifest and retain old-manifest provenance.
Geometry, reset laws, robot, observations, action dimensions, rewards outside the
new zones, and native SDF knees remain unchanged.

## Objectives and certification

Each hand scene has a route-length forward-protection zone, with 0.15 m boundary
fades. Its heading cost is the existing mean `1 - cos(yaw error)` for pelvis and
torso. Its paired hand-region cost is the existing `d²/(d²+0.15²)` construction.
Both retain scale -1.0 before dt=0.02. No heading-weight increase, SDF change,
reward-floor change, motion-penalty reduction or observation addition was made.
Targets use route-relative/root-relative XY and absolute world Z. Neutral-arm
regularization is released in the active hand zone, as in the contrastive bank.

All 12 tables allow **raised only** (`region_valid=[true,false]`). This is an
intentional raising objective, not a claim that every other posture collides.
Shelf choices:

- Easy `007001`–`007004`: **raised or tucked**. Complete zero-cost tucked target
  boxes clear the stored SDF by **7.77–22.17 mm**, and analytic hand-sphere geometry
  by **51.87–64.45 mm**. At the full 5 cm metric tolerance the conservative analytic
  margins are **1.87–14.45 mm** along the certified route.
- Medium `007005`–`007008` and hard `007009`–`007012`: **raised only**. The prescribed
  95% tucked pose and/or complete target box fails the conservative training-field
  check. An older full-limit point-pose certificate for medium `007006` does not
  certify this complete target region. This is not a proof that all possible
  tucked trajectories are impossible.

The shelf qualification tolerance does not guarantee positive *voxel SDF* at every
point within that tolerance. It does have a positive analytic sphere bound on the
certified route; runtime collision/field termination still governs actual safety.
Root deviations and arbitrary inverse-kinematic solutions are not covered by the
centreline certificate.

Across all scenes, raised target boxes have at least **100.98 mm analytic** and
**41.24 mm stored-field** clearance. The 5 cm tolerance envelope retains at least
**38.64 mm analytic clearance**. Certifications use actual FK, all 35 approved
body proxies, stored fields, sampled endpoint arm transitions, and swept box bounds.
When the coarse grid bound is inconclusive, a clipped-cell multiaffine bound
preserves CAT's historical interpolation corner ordering and both cell limits.
There is no claim of exhaustive IK or dynamic feasibility certification.

## Reward acceptance measurements

`reward-audit.json` contains every configured term and per-position totals, using
actual `CATTask._rewards`, current saved scales, real MuJoCo FK and stored fields.
The travel model prescribes 0.6 m/s translation and uniformly averages gait phases,
with static bias torque estimates. It is a kinematic reward comparison, not a
learned policy rollout or a dynamically balanced gait demonstration.

The table gives ranges across scenes of mean **reward per control step**, evaluated
in the fully active zone. Collision-event penalties are excluded; colliding paths
are counterfactual and would terminate with the additional -1 event.

| Family / pose | Pre-clamp mean | Post-clamp mean |
|---|---:|---:|
| Table, forward + raised | **0.23081–0.23546** | Same |
| Table, forward + nominal arms | -2.04606 to -1.33179 | 0.02011–0.02475 |
| Table, sideways + nominal arms | 0.17154–0.19391 | Same |
| Shelf, forward + raised | **0.21030–0.21282** | Same |
| Shelf, forward + nominal arms | -1.90327 to -1.37515 | 0.02537–0.03017 |
| Shelf, sideways + nominal arms | 0.17712–0.19430 | Same |

Raised-forward wins on **all 24 scene-average comparisons**. Its margin over the
sideways dodge is **+0.04150–0.06030/step for tables**, and
**+0.01601–0.03546/step for shelves**, including native balance and clearance terms.
This is not pointwise dominance at every position or a global optimality proof.

No held raised-forward or sideways traversal sample clamps in the quadrature.
Raised traversal minimum over all sampled phases is **+0.06930/step**. In the first
0.5 m approach, minimum rewards are **+0.13225 raised** and **+0.05404 sideways**;
no new approach plateau is introduced by weight 1. The raised endpoint, unlike
nominal forward arms, clears the approved collision proxies.

The four allowed tucked shelf targets are geometrically attainable but still
receive substantial native proximity penalties: mean post-clamp rewards are
0.05425–0.06838, with approximately 39–65% phase-weighted clamping. Allowing a safe
region is not a promise that native reward makes that region equally attractive.
Raising remains the better sampled option; no clearance coefficients were changed.

### Acquisition is a separate acceptance check

`acquisition-audit.json` includes actual finite-difference hand velocities, upper
motion histories, the gait clock, and double-support contact penalties while
paused. A naive linear 0.4 s raise while moving at 0.6 m/s often reaches the
obstacle too early and clamps; some hard cases collide. This failure is retained
in the report, not excluded from the evidence.

A **0.6 s paused, staged acquisition** (0.2 s inward shoulder-roll motion, then
0.4 s raising) reaches the same target in all 24 scenes without clamping. Maximum
upper-joint speed is 1.9 rad/s, below the unchanged 2 rad/s slew limit; offsets stay
within the action range. Minimum sampled reward is **+0.03912/step**, mean rewards
are **0.12459–0.14697**, and its advantage over matched stationary nominal arms is
**+0.03709–0.05947/step**. Minimum body clearance is **114.42 mm**, minimum hand-field
clearance **68.27 mm**. Thus a sampled acquisition path and rewarded traversal both
exist. The bank does not prescribe this motion to the policy; learning it remains
an empirical question. Finite heading costs do not physically forbid dodging.

## Runtime integration and diagnostics

The existing hand sampler remains authoritative; no dual curriculum was added.
Only the hand scenes carry contrastive metadata; retention keeps role -1. The
upgrade validator pins the original parent and rejects changed retention records,
field arrays, masses, table target choices, scene order or geometry.

Checkpoint selection is scoped to the declared protected role for this bank;
otherwise the old four-role requirement would prevent saving any `best.pt`.
Old banks retain their previous selection behaviour. Selection remains a rolling
training statistic, not an independent evaluation. Retention goal metrics remain
available separately; this selection score does not enforce retention quality.

Watch these leader-only rolling metrics separately:

- `training/leader_table_hand_contrast_hand_region_compliance_fraction`
- `training/leader_table_hand_contrast_heading_compliance_fraction`
- `training/leader_shelf_hand_contrast_hand_region_compliance_fraction`
- `training/leader_shelf_hand_contrast_heading_compliance_fraction`
- Their corresponding mean costs, sample counts, zone progress and collision rates.
- `hand_curriculum/stage` and per-level raw clean-goal counts/rates.
- `success/cat_goal_success_rate` and `success/ordinary_clutter_goal_success_rate`.

Table targets are raised-only, so table compliance is the direct raising signal.
Shelf compliance can include tucking on the four easy layouts. Pooled compliance
must not be mistaken for proof that tables have been solved.

## Validation and resource budget

45 targeted CPU tests pass. The complete 2,362-scene bank was loaded and eight
native CPU MuJoCo test worlds reset/stepped through production task/collision code,
covering retention and both hand families at every level. Rewards and observations
were finite, dimensions stayed 222/310/29, role sentinel values were correct, and
native task state round-tripped. `runtime-smoke.json` records the result. This does
not test GPU compilation or learned performance.

Logical resident fields: **13.542 GiB**, versus **15.700 GiB** in the live width bank.
A read-only logged measurement from the current 30,720-env run was **41.19 GiB**
device use. Subtracting the 2.159 GiB field difference gives roughly **39.03 GiB**
at comparable allocation state. Expect approximately **39–42 GiB**, with additional
peak/workspace uncertainty; budget a 48 GiB device. This new run was not profiled on
GPU. Same envs, 768 batch size, 40 minibatches and 32-step unroll keep rollout and
optimizer sizes comparable.

New bank allocation on this compressed filesystem is approximately **209 MiB**
(fields/metadata/history 167 MiB, collision 36 MiB, resets 6.3 MiB). No 13 GiB retention
copy was made. The current equivalent runtime snapshot is 567 MiB and `best.pt`
5.1 MiB. Atomic replacement can temporarily require two runtime snapshots
(~1.11 GiB). Reserve at least **2 GiB** for the future run including logs/compiler
caches; those caches are not hard-bounded. Checkpoints overwrite fixed filenames;
the proposed interval is 50 updates.

## Prepared launch — not executed

After the user cleanly stops the current GPU job, launch the separate fresh run:

```bash
bash configs/pilots/hand_posture_v1_30720.sh
```

The script warm-starts weights from the width run's durable `resume.pt`, discards
optimizer/sampler state, preserves learned per-dimension sigma with the 0.15
ceiling, inherits gamma=0.98, enables the 0.35 hand gate, and uses 30,720 envs,
768 batch size, 40 minibatches, unroll 32, four inherited epochs, and 300 updates.
It creates a separate online W&B run and output directory. No `--resume`, sigma
reset, or knee-only passage overlay is used.

Reproduction tools:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  .venv-mjlab/bin/python scripts/build_hand_posture_bank.py --output /path/to/new/bank
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  .venv-mjlab/bin/python scripts/verify_hand_posture_pilot.py --manifest /path/to/new/bank/manifest.json
```

The builder refuses to overwrite a published bank. Existing artifact paths in the
CPU smoke script are intentionally fixed to this published build.
