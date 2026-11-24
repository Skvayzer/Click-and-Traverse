# Evaluation overnight log (branch `feature/eval-overnight`)

Opened 2026-11-24 22:20 Dubai by the evaluation/figures/website agent. Evaluation only: no checkpoint is trained or changed.
Frozen policies: planner–reflex = COMBO_FIX seeds 0–2 (u150), reflex-off = COMBO_NOREFLEX seeds 0–2 (u150), reactive = V7r u600.
Code: this branch, forked from `feature/energy-price` 16ad2fe. The paper repo is read only.

## Pre-registration (written before any A1–A3 number was seen)

Common rules. Episodes use the original 384 narrow scenes (retention list `scenes.txt`), 2 episodes per scene unless stated.
Intervals are paired 95% percentile bootstrap CIs over **scenes** (10,000 resamples; the same resampled scene indices are used for
both policies, and the per-seed deltas are averaged within each resample). Seed-level intervals (t, 2 d.f.) are reported next to them.
With three seeds the t interval is the conservative one; a result is called *supported* only when the scene CI excludes zero **and** the
seed-mean has the same sign in all three seeds. Rates are percentages, deltas percentage points (pp).

### A1. Dynamic threats while walking (new benchmark)
* Protocol. During traversal, up to three objects per episode are thrown at the live hands, one at each speed tier 0.4, 0.8 and 1.2 m/s
  in a random order, first launch at 3–8 s and 6 s spacing (`cat_mjlab/threats.py`). Each object is aimed at the hand's current position
  (led by its measured velocity) from the front or the outer side, never from behind, along the clearest of 8 pre-drawn rays, so the hand is the
  first body part the object meets (the rule of the standing bank's `rearm`). Object: the plan-blind 5.4 × 5.4 × 17 cm bar, the same physical
  contact pairs and the same policy-side field merge as E3. The schedule is fixed per episode row and identical for every policy. Operator
  speeds 0.15 and 0.30 m/s, nominal physics. Any object contact ends the episode like any other contact.
* Policies: planner–reflex (3 seeds), reflex-off (3 seeds), reactive V7r (1 checkpoint).
* Metrics (per episode): hand contact with the thrown object (primary), any contact with it, clean traversal, refused, any hand contact.
* H1a. The reflex lowers hand contact with the thrown object. Prediction: planner–reflex minus reflex-off ≤ −2 pp pooled over speeds and tiers,
  with clean traversal no more than 2 pp lower. Pass: scene CI of the hand-contact delta excludes zero (upper bound < 0), and the clean delta's lower bound ≥ −2 pp.
  Falsified: CI includes zero or the delta is positive. Prior: E3's inserted-object contact at 0.30 m/s moving was −1.26 pp (slow box); faster objects should give the reflex more room, if it works.
* H1b. Planner–reflex is not worse than reactive on object hand contact (non-inferiority margin +2 pp, upper CI bound below +2 pp).
* Secondary: the same by speed tier, and the refusal cost of the reflex.

### A2. Robustness (reactive vs planner–reflex)
* Protocol. The checkpoint's own robustness settings (action latency up to 20 ms, pose noise, mount error, friction 0.5–1.2, mass ±10%), without
  `--nominal`: setting R1 = those, setting R2 = R1 plus the random pushes of training (every 5–10 s, 0.1–1.0 of their nominal magnitude). 384 scenes, 2 episodes, 0.15 and 0.30 m/s.
* Policies: planner–reflex (3 seeds), reflex-off (3 seeds, if time), reactive V7r.
* H2. The planner–reflex policy keeps its clean-traversal level: robust clean ≥ 0.9 × its nominal clean (R1) and it stays within 3 pp of reactive
  clean-or-better on R1 and R2, with any-contact no more than 3 pp above reactive. Falsified if clean falls below 0.9 × nominal or the contact gap to reactive exceeds 3 pp (CI lower bound).

### A3. Plan-input ablation and speed sweep (E4 held-out layouts)
* Protocol. The 72 held-out layouts (24 U-traps, 24 early-turn, 24 table gaps), two paired starts each, at 0.15, 0.30 and 0.50 m/s.
  The same planner–reflex network (same weights) is run with its plan input on (`--tracker-mix plan=1`, E4 results) and off (`--tracker-mix none=1`). Three seeds.
* Metrics: clean, any contact, hand contact, refused, time to the first clean goal (on clean arrivals), mean episode-minimum palm clearance.
* H3. Plan input on raises clean traversal at 0.50 m/s on the U and early-turn families combined by ≥ 5 pp without more hand contact, and lowers no family's clean rate
  by more than 5 pp at any speed. Pass: scene CI lower bound of the U+turn clean delta > 0 and point estimate ≥ 5 pp. Prior (seed 0 vs V7r, E4): +3.1 pp at 0.5 m/s, below the bar; this tests whether it is the plan or something else.
* Secondary: time to goal (the plan is expected to be slower), minimum palm clearance.

### A4. Collection of pending results
Third standing flip seed (319350_0, _1), α sensitivity (318296), E5 (320668–320670 and 319436) and E6: collected and tabulated without new interpretation beyond the pre-registered E-protocols.

## Run log
(Entries are appended below as the runs start and finish.)

### 22:20–23:00 Dubai, setup
* Code: `cat_mjlab/threats.py` (A1 thrown objects, `ThrownObstacle`), `scripts/make_throw_guide.py`, `--blind-variant throw` and `--pushes` in `scripts/eval_per_scene.py`,
  `scripts/eval_stats.py` (paired scene bootstrap), `scripts/overnight_data.py` (all numbers into `results.json`), `scripts/overnight_figures.py`, `hpc/eval_overnight.sbatch`.
  Commits on `feature/eval-overnight` (local worktree `~/robotics/cat-energy-eval`).
* Smoke (flat 1,500 steps of the full 768-episode throw benchmark, COMBO_FIX seed 1, 0.3 m/s): 1,256 throws launched, 0 skipped, 143 physical hand contacts, no capacity overflow, no exception.
* Throw guides `throw_0.15.npz`, `throw_0.3.npz` copy the 768 episode assignments and initial states of the E3 plan-blind guides (same bank hash) and add the schedule (seed 20261125). They live in `/tmp/cat-eval-overnight/guides` and on the cluster in `~/cat_worktrees/eval_overnight_inputs`.
* Cluster (gpu partition): A1 planner array 322712 (seeds 1–2; seed 0 runs locally), noreflex 322713 (1–2), A3 plan-off array 322692 (seed 1 ran, seeds 0 and 2 hit foreign GPU memory on gpu-02/03; seed 0 resubmitted as 322716). The gpu partition allows one GPU per user at a time here, so most queued jobs wait.
* Local GPU (dep-1, shared lock): A1 seed 0 planner–reflex 0.15 m/s started 22:35; the remaining local jobs run two at a time inside one lock hold (`/tmp/cat-eval-overnight/run-batch-local.sh`, log `batch-local.log`).
* Local outputs: `/tmp/cat-eval-overnight/runs/{A1,A2,A3}_*_local`.

### A4 collection (status at 22:50)
* Standing flip seed 2 (319350_0/_1) collected: arm×1 → ×30 moves arm-first 40.4% → 33.0%, failure 12.1% → 17.1%, body-first 0.95% → 0.77%, worlds with a foot moved 8.0% → 26.2% (`results.json`, standing).
* α sensitivity (seed 1): arm-first 43.7 / 38.9 / 33.1 % and failure 9.5 / 11.9 / 14.2 % at 250 / 500 / 1,000 W; body-first stays below 1%.
* E5 (reactive 320668, price reference 320669, plan-only 319436) and E6 BRACE 384 (320664) JSONs are mirrored under `/tmp/cat-eval-overnight/data/cluster`; BRACE 384 is in the figures. E5 no-price 320670 still running.
* SONIC plain (G2 384 and E4) from the SONIC agent's cluster outputs is included in the figures and the website as "SONIC (plain)".
