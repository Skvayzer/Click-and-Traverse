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
