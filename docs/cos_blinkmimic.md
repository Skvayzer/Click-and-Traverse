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

### B0 result (16:22) — pre-registered time rule MISSED (11.4 min vs 10.2 min limit)
Job 337353 (ws-ia, RTX 5000 Ada 32 GB), PRESET[main], seed 0: 3,500 iterations in 15.1 min of training wall-clock (their
4090: 7.8 min); 6.6 it/s in steady state (theirs 7.5) but ≈ 7.6 min to iteration 1,000 including compilation. Evaluation
(40 clips, 4 replicas, every 100 iterations): first return >= 0.82 at 9.19 min (it 1,400; theirs 2.9 min), strict target
(>= 0.82 and 0 failed for 3 checkpoints in a row) at **11.44 min** (it 2,200; theirs 3.4 min; rule 10.2 min). Final 0.861 / 0
failed (theirs 0.862 / 0). **The pre-registered wall-clock rule is missed: 11.4 min against the 10.2 min limit.** Reasons: the same iteration counts
(1,400 / 2,200 vs their 1,300 / ~1,550) and the same final return as the authors, about 7 min of compilation before
iteration 1,000, and a slower card (6.6 vs 7.5 it/s). The recipe itself reproduces (target reached, final 0.861 / 0 failed).

## B1-full — whole-clip cleaning (pre-registered 16:25, before any count; requested after B1: 3.34 h of windows is a
small slice of the 132.5 h dataset)
Per whole clip of the four v1.1 libraries (dropped sources skipped), `scripts/cos_full_segments.py` applies the C0 tests
**per frame** with a local floor (p5 of the min foot height within ±2 s): flight (whole-body lowest point > floor + 5 cm
for 5–30 frames), floating (> 30 frames), pelvis vertical speed > 1 m/s, acrobatic (pelvis upside down, roll > 60°, head
below pelvis, angular speed > 8 rad/s), uneven floor (0.5 s support block > floor + 5 cm), foot penetration (min foot <
floor − 2 cm); clip names with the C0 trick/terrain keywords are dropped whole. Bad frames are dilated by 0.5 s, good runs
>= 2 s become segments, a segment whose local floor varies by > 5 cm is dropped, and each segment is levelled to the
physical proxy floor (dz = −0.010 − p5 of its min foot height, C0 amendment 1). Split by take with the C1 rule (C1's
take_split.json where the take had scenes, else the same sha1 rule) → no test take in training. Report hours and segments
per split and source, hours flagged per reason. No pass rule (data gate); a 40-clip contact sheet check is skipped (the
tests are C0's, already eyeballed).

## B2-full (pre-registered 16:25, before any B2-full run)
Train BlinkMimic from scratch on **all train segments** of B1-full (PRESET[main]; changes needed for memory — e.g. reference
storage, probe-sweep size — or a cap on hours if the library does not fit on a 32 GB card are logged as amendments and the
largest fitting subset is used), then evaluate with `tools/eval_metrics.py` on the **same 340 held-out test windows** as B2.
Comparison: B2-full vs B2 SALL (windows only) on held-out global body-position error, joint error and failed clips, and
both in B3. Hypothesis: 10–30 × more motion lowers held-out error and failures (more varied gaits and postures); risk: the
fixed iteration budget covers the larger library less densely → also report the training-set return. Pass (vs SALL):
held-out failed clips not higher and global body error lower by >= 0.5 cm.

## B3 — tracking through the obstacle scenes (pre-registered 16:27, before any B3 rollout)
**Setup** (`~/blinkmimic/tools/cos_obstacle_eval.py`, branch cos-tracking). BlinkMimic's own MuJoCo-Warp simulator (their
G1 collision geometry: convex hulls + foot spheres) with the boxes of each of the 200 COS test scenes (levelled, all roles)
added as static colliding geoms; scene k is placed 30 m along x from scene k−1 and world k tracks scene k's **held-out human
reference** (levelled M of the scene's window, 50 Hz, its own timing) from the reference's first frame, evaluation
conditions (no randomization, deterministic). One rollout per scene. A control run without the boxes is also made.
Deviations from C4, stated up front: different simulator and robot collision model (theirs vs our harness's body collision
bank); start = the reference's first frame (C4: the SONIC-tracked first pose re-anchored to it — same xy, near-identical
posture); the human's own timing instead of 0.15/0.30 m/s operator speeds; no goal-seeking policy (the reference ends at
the goal). Compared with C4 numbers as context, not as a paired test.
**Metrics.** Per scene: physical robot–box contact (any geom, penetration < 0), hand contact (wrist/hand bodies),
tracking failure (BeyondMimic termination = fall/reset), completion, goal reached (clip completed and final pelvis within
0.5 m of the reference's), **clean = completed without failure and without contact**; min clearance from the recorded
rollouts with our 35-primitive exact proxy (`synth.Body.sep`, same as C2/C0); per family; Wilson 95 % CI on clean.
**Pass rule (proposed by the user, fixed now):** clean >= 85 % on the 200 scenes with the B2 tracker chosen by the held-out
metrics (SALL, or B2-full if it is better on held-out error). Context: C4 plan-off 76.3 / 74.8 %, C3 oracle arm 73.8 %.
Prediction: 80–90 % clean; contacts concentrated in tight families (arms_in, sideways, duck), since M clears the boxes by
only >= 5 cm and any tracking error of > 5 cm on the arms touches.

### B2 amendment (16:43, after S40 / SALL metrics, before any variant run): global drift is unobservable
BlinkMimic's policy observation (`mdp.obs`) holds joints, joint velocities, angular velocity, gravity, previous action and the
reference-relative root **orientation** — no root **position** error; the reward's root-position term is weak (σ 0.3 m,
weight 0.5 of 5). Global drift is therefore invisible to the policy: SALL tracks held-out windows with 4.4 cm re-anchored
body error but 17 cm mean root drift (32 cm mean global body error). For B3 (world-frame obstacles cleared by >= 5 cm) this is
decisive, so two variants are added (seed 0, all 3,045 train windows, PRESET[main], 3,500 iterations), flag
`BM_ANCHOR_OBS=1`: the reference root position in the robot's yaw frame (3 values, 2 cm uniform noise) appended to the
observation (BeyondMimic's motion_anchor_pos_b; branch cos-tracking 731474c):
* **SALL-A**: anchor observation, reward unchanged;
* **SALL-AR**: anchor observation + root-position reward σ 0.3 → 0.1 m (`BM_ROOT_POS_STD=0.1`).
Same metrics; the B3 tracker is chosen among SALL / SALL-A / SALL-AR (and B2-full) by held-out median global body error
with failed clips <= 3 %, before any B3 rollout. Prediction: the anchor observation cuts held-out root drift from ~17 cm to
< 5 cm at equal re-anchored error.

### B2 results so far (16:44; `eval_metrics` patched to report per-clip values and medians — pooled means are
dominated by a tail of large drifts; seed 0; 3,500 iterations each; training wall-clock ≈ 15 min each)
| run | windows | train: return / failed / med. global / re-anch. body / joint | held-out test (340): return / failed / med. global / med. root / re-anch. body / joint |
|---|---|---|---|
| S40 | 40 | 0.928 / 0 / 5.6 cm / 3.4 cm / 3.0° | 0.744 / 11 (+2 non-finite) / 44.0 / 35.7 cm / 7.9 cm / 4.8° |
| S500 | 500 | 0.932 / 0 / 7.6 / 4.1 / 3.7° | 0.880 / 2 / 13.6 / 11.3 / 5.4 / 4.2° |
| SALL | 3,045 | 0.925 / 1 / 8.7 / 4.0 / 3.7° | **0.909 / 0 / 10.2 / 8.8 / 4.4 / 3.9°** |
S40 reached the LAFAN-style target on its own 40 clips at 6.2 min (it 500; 0.952 / 0 failed by 3,500). SALL meets the
held-out failure target (0 / 340) and the re-anchored target (4.4 vs <= 4 cm: just missed) and joints (3.9 <= 5°), but **not
the global target** (median 10.2 cm vs <= 5 cm; mean 32 cm) — drift, see the amendment above. One SALL training window
failed; a few worlds became non-finite in S40 (counted, excluded from pooled errors).

### B2 metrics, corrected (17:07): eval_metrics fix — finished worlds keep stepping idle and some blow up; `alive *
NaN` poisoned the sums (all "non-finite clips" were this artifact, short PHUMA windows; blinkmimic 1b8c56a). Re-evaluated:
| run | train: ret / failed / global med·mean·p90 / root med / re-anch. / joint | test (340): ret / failed / global med·mean·p90 / root med·mean / re-anch. / joint |
|---|---|---|---|
| SALL | 0.925 / 1 / 8.8·13.4·17.6 / 7.4 / 4.0 / 3.7° | 0.909 / 0 / 10.2·32.1·27.8 / 8.7·17.1 / 4.4 / 3.8° |
| SALL-A | 0.931 / 1 / 6.5·8.3·12.0 / 5.2 / 4.0 / 3.9° | 0.913 / 2 / 7.3·13.5·16.9 / 5.8·8.2 / 4.6 / 4.1° |
| SALL-AR | 0.914 / 1 / 5.2·7.1·9.4 / 3.3 / 4.2 / 4.4° | **0.897 / 0 / 5.7·12.5·13.1 / 3.8·6.1 / 4.6 / 4.6°** |
The anchor observation removes most of the drift (held-out root median 8.7 → 3.8 cm, p90 global 27.8 → 13.1 cm) at +0.2 cm
re-anchored / +0.7° joint error. Running: SALL-AR for 7,000 iterations (338020).

### B1-full result (17:09) — job 337990 (`cos/tracker/motions_full`, 199 s on 12 CPUs; three earlier attempts died on
EFAULT writes from the gpu-01 node and were rerun on ws-ia with buffered writes)
All 83,053 clips / 132.5 h → **91.4 h kept in 64,852 segments (69 %)**: train 77.6 h / 55,126 segments (70,624 clips);
val 4.3 h / 3,022; test 9.4 h / 6,704. By source (in → kept h): PHUMA 69.3 → 53.7, AMASS ember 49.1 → 25.9 (floating
floors), AMASS GMR 11.7 → 9.7, LAFAN1 2.5 → 2.1. Flagged hours (non-exclusive, before margins): floating 6.5, flight 4.1,
acrobatic 0.7, pelvis_vz 0.1, uneven floor 0.1, foot penetration 0.1; trick names 3.4 h and terrain names 3.2 h dropped
whole; short pieces (< 2 s after cutting) 5.8 h, short clips 2.7 h, segments with > 5 cm floor drift 1.3 h.

### B2-full amendment (17:09, before the run)
Scale changes (blinkmimic c5e1c8e): reference FK arrays filled in place (no second copy; 77.6 h ≈ 14 M frames ≈ 19 GB of
references on the GPU, training itself used 5.6 GB at SALL scale), per-clip contact floors by slices, and the probe sweep
capped at a fixed random 6,000 of its ~56 k tiles (`--eprobe_max_worlds 6000`). Because B3 needs the anchor observation
(B2 amendment), B2-full uses the SALL-AR settings (`BM_ANCHOR_OBS=1`, root σ 0.1 m) and is compared with SALL-AR (same
settings, windows only). Budget 15,000 iterations (the library is 23 × SALL's; ≈ 1 h), checkpoints every 500. Training-set
metrics on a fixed random 2,000-segment subset of the train segments; held-out on the same 340 test windows.
