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
