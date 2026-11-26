# COS planner + reflex — log and pre-registrations (opened 2026-11-26 16:30 +04)

Branch `feature/cos-planner-reflex` (from `feature/cos-obstacles` 4cb5dab). Data on Lustre `/l/users/konstantin.smirnov/cos/`;
small summaries in `~/.local/share/openresearch/files/humanoid-protection/cos-policy/`.
Starting point: scenes v1.1 (`scenes_v1/manifest.json`, 43,524 verified scenes, 132.5 h), see `docs/cos_obstacles.md`.

Trackers available at the start: COMBO_FIX (seeds 0–2, u150) and S3 tracker v2 (`feature/fullpath-plan`, jobs 325267/325268,
both finished 150 updates; evals on the 384 scenes with start replay: seed 0 64.5/29.7/5.7 clean/contact/refused at 0.15 m/s,
seed 1 68.5/28.5/2.6; E4 layouts 47–58 % clean).

## C0 — data cleaning (pre-registered 16:35, before any count)

**Hypothesis.** The user's bad examples (jumps/tricks, a stair descent sinking into the floor) are a few-percent minority
that kinematic tests plus clip-name keywords remove without emptying any family.

**Unit.** A *window* (the 3–12 s clip segment a scene is built on, `motions/<window>.npz` key `M`, 50 Hz G1 qpos in the
chord frame; z is world height, floor z = 0). A window that fails any test removes all its scenes. Features come from the
approved 35-primitive proxy (`cat_mjlab/cos/features.py`): `foot_z` = lowest point of each foot's sole/foot/ankle
primitives.

| reason | test (any frame of the window) |
|---|---|
| (a) `flight` | both feet > 0.05 m above the floor level for >= 0.10 s (5 frames) |
| (a) `pelvis_vz` | pelvis vertical speed (0.1 s central difference) > 1.0 m/s |
| (a) `acrobatic` | pelvis upside down (pelvis z-axis world z < 0), or lateral roll > 60° (pelvis y-axis world z > 0.87 in abs), or head below pelvis, or pelvis angular speed > 8 rad/s (0.1 s) |
| (a) `trick_name` | clip name matches flip, cartwheel, roll, handstand, parkour, vault, acrobat, somersault, tumbl, jump, hop, leap, skip, breakdanc, trick, gymnast, spin, twirl, pirouette, dive |
| (b) `uneven_floor` | the floor level (min over both feet, per 0.5 s block) varies by > 0.05 m across the window (stairs, ramps, platforms) |
| (b) `terrain_name` | name matches stair, step, climb, ramp, slope, platform, ladder, hill, incline, curb, ledge, ascend, descend; **except** step-over / side-step names (`step_?over`, `side_?step`, `stepping_?over`), which are the high-step and sideways families themselves and are judged by the kinematic tests only (count reported) |
| (c) `foot_penetration` | lowest foot point < −0.02 m |

Floor level for `flight` = 5th percentile of the per-frame min of both feet over the window. Reasons are counted
non-exclusively (a window can have several) and also by the first matching reason in the table order. `trick_event` (info):
removed windows whose trick frames lie inside an event's frames ±0.5 s. Knees/hands below −2 cm are reported, not removed.

**Re-verification.** Every kept scene is re-run through `synth.verify` (M clear of all boxes by the margin, D touches the
main obstacle); the stored flags must reproduce (pass: >= 99.5 % agreement).

**Metric.** Windows/scenes removed per reason and per family; kept verified, feasible (SONIC-tracked) and feasible+clear
scenes per family.

**Pass rule.** C0 passes when re-verification agrees >= 99.5 %, every family that had >= 100 feasible scenes keeps >= 50 %
of them, and a 24-scene random contact sheet of removed windows shows mostly true positives (eyeballed, reported).
Manifest v2: `scenes_v2/manifest.json` (+ per-source `scenes.jsonl` of kept scenes and `removed.jsonl`).

### C0 amendment 1 (2026-11-27 00:05, after the 200-window smoke run, before any full count)

**Smoke run (`scenes_v2_smoke`, 200 amass_gmr windows):** foot_penetration flagged 192/200 windows, flight 91 (incl. 72 of
the 90 crawl scenes); 38/1,357 scenes kept. The tests were miscalibrated, not the data.

**Diagnosis (`scripts/cos_floor_probe.py`, 300 windows per source, `cos/floor_probe.json`).** Window floor level = p5 of the
per-frame min foot z (proxy), median [p5, p95]:

| source | M floor | M min foot | M min − own floor | SONIC-tracked floor (physical) |
|---|---|---|---|---|
| amass_gmr | −4.0 [−5.8, −0.9] cm | −4.5 cm | −0.45 [−1.6, −0.05] cm | −1.0 cm |
| ember | −6.3 [−11.2, −2.3] cm | −7.3 cm | −0.88 [−3.5, −0.16] cm | −1.1 cm |
| phuma_full | −3.1 [−9.1, −1.7] cm | −3.8 cm | −0.48 [−2.2, −0.10] cm | −1.0 cm |
| lafan1 | −1.6 [−2.4, −0.9] cm | −2.4 cm | −0.74 [−1.4, −0.28] cm | −1.1 cm |

The physical rollouts put the proxy's lowest foot point at −1.0 cm (proxy sole slightly below MuJoCo's contact geometry plus
soft-contact sink); the retargeted clips sit a per-window constant 1–11 cm lower (retargeting height offset). Within a window
the feet dip only 0.5–0.9 cm below their own floor. So:

1. **Levelling.** Each window is shifted by dz = −0.010 − floor_M (to the physical proxy floor); M and all its boxes move by the
   same dz. A box that stood on the floor (bottom within 1 cm of z = 0) keeps its bottom at z = 0 and its top moves by dz
   (stretched/shortened), so ground obstacles stay grounded; overhead and other boxes move rigidly. D (default motion) is
   levelled to the same physical floor by its own dz_D (its floor distribution is reported). The SONIC-tracked rollout is
   physical and is not shifted; its clearance is recomputed against the levelled boxes. Every kept scene is re-verified
   with `synth.verify` (M margin, D touches the main box); scenes that no longer verify are dropped and counted
   (`reverify_fail`).
2. **foot_penetration** becomes: min foot z < floor_M − 0.02 m (2 cm below the window's own floor, i.e. below the floor
   after levelling).
3. **flight** uses the lowest point of the **whole body** (all 35 proxy primitives: knees, shins, hands count as support):
   whole-body lowest point > floor + 0.05 m for >= 0.10 s. **uneven_floor** likewise uses the whole-body lowest point per
   0.5 s block (crawling or kneeling on the floor is flat ground). Everything else is unchanged.
4. Output: levelled motions in `scenes_v2/<source>/motions/<window>.npz` (M, D, tracked, dz, dz_D); scenes carry levelled
   boxes. Pass rule unchanged, with re-verification agreement computed against the stored v1 flags.

### C0 amendment 2 (00:25, after a second 200-window smoke run, before any full count)
Smoke 2 (levelled): penetration 5/200 windows; but `uneven_floor` removed 78 of 90 crawl scenes because the whole-body
block minimum *range* is dominated by hands/knees sinking into the floor while crawling (retargeting), not by a rising
floor. Fix: `uneven_floor` = the whole-body support level of some 0.5 s block rises > 0.05 m above the window's foot floor
(stairs up or down and ramps still trip it: the window floor is the lowest level). `flight` is split for reporting: airborne
5–30 frames (0.1–0.6 s) = `flight` (a); held > 0.05 m above the floor for > 0.6 s = `floating` (b, a floor inconsistency of
the retargeted clip, e.g. long BMLmovi stretches 6–12 cm above the inferred floor). Removal rule otherwise unchanged.

## C1 — splits (pre-registered 00:45, before any split is drawn)

**Rule.** Splits are by source *clip* (all windows and scenes of a clip share a split; PHUMA chunks of one sequence and
`__rK` runs count as one clip, the v1 `take_of` rule): sha1(take) mod 100 < 10 → **test**, 10–14 → **val**, else **train**.
No take is in two splits (checked). **Test-200:** from test takes, scenes that are C0-kept, re-verified, SONIC-tracked and
clear with the tracked posture (tracked_clear >= 0) — "feasible, verified"; at most one scene per window and two per
take; 9 families (duck, crawl, high_step, sideways, arms_in, arms_tucked, arms_raised, detour, multi), quota 23 each
(200 − 9·22 = 2 extra to the largest pools); a family with fewer candidates gives all it has and the rest is spread over
the others (reported). Seeded (seed 0). Files: `cos/splits_v2/{take_split.json, test200.jsonl, summary.json}`.

## C2 — planner on COS (pre-registered 00:45, before any planner data is built)

**Data.** Planner v1 format (`cat_mjlab/planner/data.py`: state 67, 2 s chunk of 20 frames × 35, goal 5, robot-centred
40×32×36 SDF grid, 5 cm cells, uint8). COS examples from C0-kept, re-verified **train** scenes: the levelled M (and the
SONIC-tracked rollout when tracked_clear >= 0.03 m), velocities by finite differences, chunk starts every 0.2 s whose
2 s chunk overlaps an event ±1 s, at most 6 per (scene, trajectory); grid = analytic SDF of the scene's boxes (all
roles; floor excluded) in the chunk frame. COS val takes → the trainer's held-out split; test takes → a separate
`cos_test` set. Mixed with planner v1's policy-rollout data (`runs/wbp/planner_v1/all`, train subsampled to the size of
the COS train set, at most 300k; its test split kept).

**Training.** Warm start from planner v1 (`runs/wbp/planner_v1/train/planner.pt`, same format), its hyper-parameters,
20k steps (one GPU job), EMA weights evaluated.

**Evaluation (kinematic, no physics).** Situations: 2,000 random chunk starts from COS **test** takes (event-overlapping,
levelled M as the start state and goal source; stratified by family) and 1,000 from planner v1's held-out test half.
Per situation the planner draws K = 4 chunks (goal = the reference's chunk end, planner v1's protocol; the online
heading offsets ±90° are not used here) and the chosen chunk is the best by the 497-sphere grid score (largest minimum
clearance, as `planner_eval.py best_of`). **Primary metric:** exact-geometry collision rate of the chosen chunk on COS
test = share of chunks whose 35-primitive proxy (`synth.Body.sep`, 50 Hz frames at the chunk's 10 Hz keyframes plus
midpoints) penetrates any box (clearance < 0). Secondary: single-sample collision rate, hand clearance, goal error,
posture diversity (`metrics.diversity`, mean pairwise distance of the 4 samples), per family; on the benchmark test
half the grid-sphere collision-free rate (planner v1's own metric, tol 1 cm), since exact boxes do not exist for all of
those scenes. Same situations and seeds for planner v1 and the new planner.

**Pass rule.** COS-test exact collision rate of the chosen chunk <= 20 % **and** lower than planner v1's on the same
situations (paired bootstrap 95 % CI of the difference below 0). Guard: the benchmark grid-free rate must not drop by more
than 3 pp vs planner v1 (reported either way). Prediction: planner v1 collides on > 40 % of COS situations (it has never
seen tight boxes or postural avoidance), the COS planner on 10–20 %.

### C1 amendment (00:30, after drawing the take split, before any evaluation on test scenes)
The held-out test takes contain few windows of the rare families (feasible+clear: duck 10 windows in 7 takes, detour 12,
high step 2, sideways 22, crawl 0), so one scene per window gave duck 8, detour 10, high step 2. A family short of its quota
now also takes other obstacle fits of its windows (different geometry, same human motion): <= 3 scenes per window, 6 per
take. Crawl has no feasible test scene (SONIC tracks no crawl) and is absent from the test-200; its share goes to the others.

### C0 result (00:20) — PASS
Full run `cos/scenes_v2` (job 332431, code e5caec5): 5,876 windows with verified scenes; **28,302 of 43,524 verified scenes kept
(65 %)**; feasible (SONIC-tracked) 23,799 of 33,444, feasible+clear 22,395 of 31,198.
Re-verification after levelling agrees 99.87 % (36 scenes no longer verify: duck 26, arms_raised 8, other 2; dropped).
Levelling dz median: amass_gmr +2.8 cm, ember +4.6, phuma +1.8, lafan1 +0.6; D +3.0–3.4 cm.

Windows removed (non-exclusive): uneven_floor 1,011, floating 1,020, flight 756, foot_penetration 501, acrobatic 171,
pelvis_vz 163, terrain_name 87, trick_name 55 (11 step-over/side-step name exceptions judged kinematically; 554 windows
have hands/knees > 2 cm below the floor — reported, kept). Scenes by first reason: flight 5,137, uneven_floor 5,601,
foot_penetration 1,904, acrobatic 1,036, pelvis_vz 631, floating 300, terrain_name 318, trick_name 414.
By source (scenes): phuma_full 8,055 removed (uneven 3,133, flight 2,039, penetration 1,625, acrobatic 702), ember 4,489
(flight 2,108, uneven 1,634, terrain names 283 = KIT up/downstairs), amass_gmr 2,308, lafan1 334.

| family | verified before | kept | feasible before | kept feasible | kept feasible+clear |
|---|---|---|---|---|---|
| arms_raised | 16,555 | 10,037 | 13,137 | 9,101 (69 %) | 8,749 |
| arms_tucked | 9,365 | 6,518 | 8,557 | 6,067 (71 %) | 5,457 |
| arms_in | 6,277 | 4,570 | 5,145 | 3,993 (78 %) | 3,931 |
| multi | 4,306 | 2,885 | 3,368 | 2,424 (72 %) | 2,173 |
| duck | 3,763 | 2,436 | 989 | 679 (69 %) | 592 |
| detour | 1,634 | 1,009 | 1,383 | 888 (64 %) | 875 |
| sideways | 815 | 593 | 642 | 502 (78 %) | 496 |
| high_step | 619 | 193 | 223 | 145 (65 %) | 122 |
| crawl | 190 | 61 | 0 | 0 | 0 |

Pass rule: agreement 99.87 % >= 99.5 %; every family with >= 100 feasible scenes keeps >= 64 % (>= 50 %); contact sheet
(`cos-policy/c0_removed_sheet.png`, 24 removed windows, 3 per first reason): clear true positives — a handstand (pelvis
up-z −1.0), KIT up/downstairs clips, a CMU playground climb (lowest body point 87 cm above the floor), a crawl roll, foot
dips of 2–9 cm; borderline — 4 walking clips held 5–7 cm above their inferred floor for 0.2–0.7 s (retargeting floor
drift); false positives — 3 PHUMA descriptive names (“sleeves”, “Coaster”, “on one foot”) hit the trick keywords
(55 windows in total carry trick_name). **PASS.** Viewer refreshed from scenes_v2 (540 scenes, 21 MB, service restarted).

### C1 result (00:35)
Takes: train 2,413 / val 133 / test 277 (disjoint by construction; one split per take key). Kept verified scenes: train
24,445, val 1,250, test 2,607. Test-200 (`cos/splits_v2/test200.jsonl`): duck 22, high_step 5, sideways 22, arms_in 33,
arms_tucked 34, arms_raised 34, detour 22, multi 28, crawl 0 (no feasible test scene). C3 training scenes
(`cos/c3/train_scenes.jsonl`): 259 feasible+clear train scenes, one per window, 34 per family (high_step 21, crawl 0).

### State / next / jobs (00:40)
- Jobs: 332535 planner data (ws-ia, → `cos/planner_data_v1`), 332536 C3 train bank (ws-ia, → `cos/c3/bank_train_{packed,
  collision,resets,goal}` + `cos/c3/plans_train`), 332537 C4 test bank (gpu partition, CPU only, → `cos/c4/bank_test_*`,
  `cos/c4/plans_test`).
- Next: `hpc/cos_planner.sbatch` (COSDATA=planner_data_v1, OUT=cos/c2/planner_cos_v1) → C2 gate; C3 smoke
  (`hpc/cos_c3.sbatch SMOKE=1`) as soon as the train bank exists (planner v1 online until the C2 planner exists), then seed 0.
- Code clone on the cluster: `~/cat_worktrees/cos_6f455fb` (deploy: `hpc/cos_deploy.sh`, deploys the current branch).

### C1/C3 amendment (00:55, before any rollout): harness episode format
The harness's CAT scenes have a fixed 20 s episode (1,000 steps, validated by the bank loader), so a COS scene must be
traversable in 20 s at 0.15 m/s: test-200 and C3 training scenes are restricted to windows with chord <= 2.5 m (median
chord of the feasible scenes 1.5 m; 90th pct 3.0 m). Harness room = the motion's corridor (1 m behind the start and past
the goal, 1.3 m beside the pelvis path, 2.0 m high), perimeter-closed; boxes entirely outside are dropped (mostly the COS
room walls); start = the reference's first pose at the chord origin, goal = its final pelvis xy. Test-200 is redrawn with
the same rule and the same seed.
