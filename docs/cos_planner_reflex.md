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
