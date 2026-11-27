# BlinkMimic tracker on COS motions — log and pre-registrations (opened 2026-11-27 15:32 +04)

Task (user): "train tracker with blinkmimic then. don't use planer-reflex policy. just track motions from the dataset
cleanly". A whole-body motion tracker for the G1 trained **from scratch** with BlinkMimic (arXiv 2611.16422,
github.com/fuyichen0725/blinkmimic, commit d264658) on the C0-cleaned COS motions. No warm start or teacher from any
planner–reflex / COMBO_FIX / S3 / V7r / C3 checkpoint; no planner and no obstacles in training.
Code: BlinkMimic clone `~/robotics/blinkmimic` branch `cos-tracking` (cluster: `~/blinkmimic`); conversion and evaluation
scripts on this branch (`feature/cos-blinkmimic` of cat-energy-cos). Data/checkpoints: Lustre `/l/users/konstantin.smirnov/cos/tracker/`.
Summaries: artifacts `cos-tracker/`. Environment: Python 3.12 venv `cos/tracker/venv` from BlinkMimic's uv.lock (exported
with `uv export --frozen`, installed with pip; uv is not installed on the cluster).

## B0 — install and reproduce (pre-registered 15:32, before any BlinkMimic run)
**Hypothesis.** The released recipe (`PRESET[main]`, seed 0, 3,500 iterations) runs on one RTX 5000 Ada 32 GB (ws-ia/gpu)
and reaches the README's strict target.
**Metric.** Their `tools/eval_batch.py` (all 40 LAFAN1 clips, 4 replicas, one checkpoint per 100 iterations) and
`tools/time_to_target.py`: training minutes to (a) first normalized return >= 0.82 and (b) return >= 0.82 with 0 failed clips
for 3 checkpoints in a row.
**Pass rule.** (b) is reached within 3 × their reported 3.4 min = 10.2 min of training wall-clock (their definition: training
time incl. compilation, excl. start-up and data loading). The RTX 5000 Ada is slower than a 4090, so a miss up to the 3,500
iterations with the target reached counts as "install works, slower hardware" and is reported as such; a run that never
reaches 0.82 / 0 failed fails B0. Purpose: proves the install, nothing about our data.

## B1 — data (pre-registered 15:47, before the export)
**Choice: unique C0-kept windows, not whole clips.** C0 cleaned and levelled *windows* (3–12 s around the avoidance events),
not whole clips: the parts of a clip outside its windows were never tested for flight, uneven floor or penetration, and the
levelling dz is per window. So the motion set = every window with >= 1 kept, re-verified scene (the levelled M of
`scenes_v2/<source>/motions/<window>.npz`, 50 Hz), one file per window, >= 2 s. These are also exactly the references of
the COS test scenes (B3). Splits by take (C1 `take_split.json`): train takes → training; val and test takes held out.
**Checks.** Joint order (ours vs BlinkMimic's `G1_JOINT_NAMES`), frame rate (50 Hz; their loader resamples to 50 Hz),
kinematics of the two G1 models (FK of the 14 tracked bodies on 2,000 random frames of our motions, max position difference),
foot height on the floor with their collision geometry (p5 of the lowest foot point per window should be within ±1 cm of 0).
**Report.** Clips and hours per split and family. No pass rule (a data gate); a kinematic mismatch > 1 cm or a floor offset
> 1 cm is fixed (re-levelled) before B2.

### B1 result (15:50)
`cos/tracker/motions_v1/{train,val,test}/<window>.npz` (`scripts/cos_bm_export.py`): **train 3,045 windows, 3.34 h, 2,413
takes; val 151, 0.15 h; test 340, 0.38 h** (windows >= 2 s; median ≈ 4 s). Families (windows; a window can hold several):
train arms_raised 1,635, arms_tucked 914, multi 815, arms_in 698, duck 275, sideways 205, detour 149, high_step 31, crawl 10;
test arms_raised 181, arms_tucked 97, arms_in 83, multi 83, duck 30, sideways 23, detour 15, high_step 2, crawl 2.
Checks: joint order identical (29 joints, Unitree SDK order); 50 Hz (no resampling); FK of the 14 tracked bodies of our
training G1 and BlinkMimic's g1_29dof_rev_1_0 MJCF identical on 2,000 frames (max difference 0.0 mm); with their collision
geometry (foot spheres) the levelled windows stand at −1.1 mm (p5 of the lowest foot point per window; 5–95 %: −4.4 /
+5.4 mm) → no re-levelling. Their model is 33.3 kg vs 34.4 kg for ours (same kinematics, different inertials; B3 uses our
harness). Loader: `BM_MOTION_GLOB=*.npz` (`load_qpos_npz`, branch cos-tracking 1d0e700).

## B2 — train from scratch on COS motions (pre-registered 15:50, before any COS training run)
**Runs** (PRESET[main] unchanged unless stated; seed 0; one RTX 5000 Ada): S40 = 40 random train windows, 3,500 iterations;
S500 = 500 random train windows, 3,500 iterations; SALL = all 3,045 train windows (3.34 h, 1.4 × LAFAN1's 2.45 h), 3,500
iterations first, extended to 7,000 if the return is still rising. Changes needed for scale (memory, probe sweep) are logged
as amendments.
**Metrics** (`tools/eval_metrics.py`, deterministic, every clip once from its first frame, no randomization; patched with
`--motion_dir`, a global body-position error and the failed-clip names): normalized return, failed clips (BeyondMimic
termination = fall/reset), global body-position RMS (14 bodies, not re-anchored), re-anchored body-position RMS, root
position error, joint error; on the run's training windows and on the 340 held-out **test** windows (val for checkpoint
choice only if needed). Wall-clock from their logs.
**Targets / pass rule (SALL).** Training windows: return >= 0.82 and <= 1 % failed. Held-out test windows: <= 3 % failed,
global body position <= 5 cm, re-anchored body position <= 4 cm, joint error <= 5°. Reference points: BlinkMimic on LAFAN1
3.44 cm re-anchored body error, 4.2° joints; SONIC tracked NVIDIA's walk within 2.5 cm in our harness (different metric,
stated for orientation only). Prediction: the short COS windows (mostly walking with arm postures) are easier than LAFAN1 per
second, so SALL reaches the training target within 3,500 iterations; held-out error 1–2 cm above training error.

### B0 result (16:22) — install works; time rule missed on slower hardware
Job 337353 (ws-ia, RTX 5000 Ada 32 GB), PRESET[main], seed 0: 3,500 iterations in 15.1 min of training wall-clock (their
4090: 7.8 min); 6.6 it/s in steady state (theirs 7.5) but ≈ 7.6 min to iteration 1,000 including compilation. Evaluation
(40 clips, 4 replicas, every 100 iterations): first return >= 0.82 at 9.19 min (it 1,400; theirs 2.9 min), strict target
(>= 0.82 and 0 failed for 3 checkpoints in a row) at **11.44 min** (it 2,200; theirs 3.4 min; rule 10.2 min). Final 0.861 / 0
failed (theirs 0.862 / 0). Verdict by the pre-registered rule: target reached, time 3.4 × theirs (rule 3 ×) → "install works,
slower hardware"; the iteration counts (1,400 / 2,200 vs their 1,300 / ~1,550) match.
