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

### A4: E5 ladder on the plan-blind obstacles (collected 22:58; no-price 320670 still running, 2 of 4 protocols done)
Full table with paired scene-bootstrap CIs: `artifacts/overnight/e5_table.md` (script `scripts/e5_table.py`). Headlines (planner–reflex = 3 seeds, others 1 checkpoint except plan-only 3 seeds):
* Reflex on minus off: no protocol shows a clean or any-contact change whose CI excludes zero. Inserted-object hand contact is lower in all four protocols except moving 0.15 (point estimates −0.9, −0.6, +0.6, −1.3 pp), CI excludes zero only at moving 0.30 m/s (−1.26 [−2.52, −0.00]), which is E3's single localized effect.
* Masking the actor's field inputs (plan only): 1–2% clean and 65–75% hand contact, i.e. the plan alone cannot carry the policy.
* Reactive V7r is cleaner than planner–reflex in all four plan-blind protocols (planner–reflex minus reactive clean −3.1, −4.6, −3.7, −6.6 pp, all CIs exclude zero) and has lower any inserted-object contact (+11.1, +5.6, +15.5, +4.8 pp for planner–reflex), while planner–reflex has lower hand contact with the inserted object at 0.15 m/s static (−4.5 [−7.2, −1.7]).
* The older priced COMBO1b u500 and the price-free continuation have 15–21 pp *less* any contact than the current planner–reflex at similar clean, an unresolved difference between checkpoints (not a price effect: the budgets differ).

### A3 prior evidence on the 384 scenes (existing runs, COMBO1b u500, one seed, 2 episodes per scene; `speed_sweep/COMBO1b_u500_{plan,none}`)
The same network with its plan input on (`--tracker-mix plan=1`) or off (`none=1`). Plan on minus plan off (percentage points, 95% scene-bootstrap CI): at 0.15 m/s clean −8.6 [−11.7, −5.5], any contact −12.1 [−15.2, −9.1], refused +20.8 [+17.2, +24.6], hand contact −1.2 [−2.6, +0.1]; at 0.30 m/s clean −5.7 [−8.3, −3.1], contact −10.2 [−13.5, −6.9], refused +15.9 [+12.6, +19.3], hand contact −1.6 [−2.9, −0.4]. Following the plan trades contact for refusals and loses clean traversal on these known-obstacle scenes. The pre-registered A3 test below uses the current COMBO_FIX networks on the held-out layouts.

### A1 interim (03:50): walking at 0.15 m/s, object thrown at the live hand
Runs: planner–reflex seeds 0 (local), 1, 2 (cluster 322712); reflex-off seed 1 (cluster 322713_1). 768 episodes per run, 95% of episodes had at least one throw launched (0 skipped); 1,500 throws per run, split evenly over the three speed tiers.
* Planner–reflex (3 seeds): clean 27.3 [24.8, 29.9]%, any object contact 46.8 [44.4, 49.1]%, object hit a hand in 19.8 [18.1, 21.7]% of episodes. Per throw the hand is hit in 2.8 / 10.0 / 17.0% at 0.4 / 0.8 / 1.2 m/s.
* Reflex off (seed 1): clean 27.5, object hand contact 19.8 [17.1, 22.5]%, per-tier 3.1 / 9.2 / 17.7%.
* Planner–reflex minus reflex-off (all three on-seeds against the one off-seed): object hand contact +0.04 pp [−2.91, +2.95], clean −0.17 [−3.17, +2.86]; seed-1-only matched pair: object hand contact −0.65 [−4.30, +2.99], clean +0.13 [−3.78, +4.04]. Pre-registered H1a (≤ −2 pp with CI upper < 0) is **not met** at 0.15 m/s. The per-tier differences are −0.4, +0.7 and −1.2 pp, none different from zero.
* The thrown-object benchmark is hard for every policy (clean falls from ~61% to ~27%), so the absence of a reflex effect is not a floor or ceiling artefact: hand hits scale from 3% to 17% with object speed, which a working reflex should have reduced.

### A1 interim (04:20): walking at 0.3 m/s
Planner–reflex seeds 1, 2 (cluster; seed 0 local pending) and reflex-off seed 1. Seed-1 matched pair (on minus off): object hand contact **+4.56 pp [+1.04, +8.20]**, any contact with the object +2.08 [−2.21, +6.25], any hand contact +4.04 [+0.39, +7.81], clean −1.43 [−5.21, +2.34], refused −0.52 [−2.08, +0.91]. Hand hits at the 1.2 m/s tier: 107/445 (24.0%) with the reflex vs 84/440 (19.1%) without. Pre-registered H1a fails at both speeds (at 0.3 m/s the sign is opposite to the prediction). Caveat: one reflex-off seed so far, and the two policies are separately fine-tuned networks (the contrast is not a switch on one network).
Note: the object travels 2.4 cm per control step at 1.2 m/s, so the 5 cm hand band is entered about two control steps before contact; a reflex that acts through a reward-gated arm cannot respond in that time. This explanation is a hypothesis; the per-flight reflex activation counter (`reflex_released_during_flight`, added after these runs started) will test it on the local runs.

### A3 interim (04:55): plan input on vs off, seed 0 (job 322716_0 on the cluster; seeds 1–2 resubmitted as 324268 after foreign-GPU OOM)
Same COMBO_FIX seed-0 network on the 72 held-out layouts, 144 episodes per speed. Plan on (E4 local run) minus plan off (`--tracker-mix none=1`), scene-bootstrap 95% CI:
| Speed | Clean | Any contact | Hand contact | Refused | U-trap + early-turn clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.15 | −5.56 [−12.50, +0.69] | +4.17 [−2.08, +10.42] | +4.17 [−1.39, +10.42] | +1.39 [−1.39, +4.86] | +3.12 [−4.17, +10.42] |
| 0.30 | −9.72 [−16.67, −2.78] | +6.25 [0.00, +12.50] | −5.56 [−11.11, 0.00] | +3.47 [+0.69, +7.64] | +3.12 [−1.04, +7.32] |
| 0.50 | −13.89 [−21.53, −6.25] | +9.03 [+2.08, +15.97] | −2.08 [−9.03, +4.86] | +4.86 [+1.39, +9.03] | −2.08 [−9.38, +5.21] |
By family (clean, plan on vs off): table gaps 58.3 vs 81.2 (0.15), 54.2 vs 89.6 (0.30), 43.8 vs 81.2 (0.50); U-traps 95.8 vs 87.5, 100 vs 91.7, 91.7 vs 93.8; early-turn 10.4 vs 12.5, 2.1 vs 4.2, 4.2 vs 6.2. Plan-off clean rates (60.4, 61.8, 60.4) are above reactive V7r's (58.3, 56.2, 52.1).
Pre-registered H3 (plan input raises clean on U+early-turn at 0.5 m/s by ≥ 5 pp, CI lower bound > 0): **fails** (−2.1 pp [−9.4, +5.2]); more strongly, the plan lowers clean traversal at every speed, mainly in table gaps, where plan-following refuses and contacts more. Using the pooled 3 on-seeds against the one off-seed (results.json) gives the same sign. Seeds 1–2 pending; the family pattern is from seed 0 only.

### A3 final (05:58): plan input on vs off, three matched training seeds (plan-off jobs 322716_0, 324268_1, 324268_2; plan-on = E4 planner seeds 0–2)
72 held-out layouts, 144 episodes per speed and seed. Plan on minus plan off; scene-bootstrap 95% CI over the pooled seeds (percentage points); seed-level t interval for clean in the last column.
| Speed | Clean | Any contact | Refused | Hand contact | U-trap + early-turn clean | Clean, seed-level CI (per-seed values) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.15 | −8.33 [−13.66, −3.47] | +7.64 [+3.01, +12.50] | +0.69 [−0.46, +2.55] | +6.94 [+2.78, +11.34] | +0.69 [−4.17, +4.86] | [−16.2, −0.4] (−5.6, −7.6, −11.8) |
| 0.30 | −8.80 [−13.19, −4.86] | +6.25 [+2.78, +10.19] | +2.55 [+0.69, +4.63] | −1.62 [−6.94, +3.24] | −2.08 [−6.60, +1.39] | [−14.3, −3.3] (−9.7, −6.3, −10.4) |
| 0.50 | −11.81 [−17.59, −6.02] | +8.80 [+4.17, +13.66] | +3.01 [+0.46, +5.56] | −3.24 [−7.87, +1.16] | +0.69 [−3.47, +5.56] | [−33.8, +10.2] (−13.9, −2.1, −19.4) |
Clean by family, plan on vs off (pooled seeds): table gaps 58.3 vs 84.7 (0.15), 66.0 vs 88.2 (0.30), 50.0 vs 86.8 (0.50); U-traps 97.9 vs 91.0, 93.8 vs 93.1, 91.7 vs 86.8; early-turn 5.6 vs 11.1, 2.8 vs 7.6, 3.5 vs 6.9. Time to clean goal (all layouts, s): plan on 29.1 / 19.2 / 14.6, plan off 19.8 / 16.8 / 14.6, reactive V7r 20.5 / 15.8 / 12.9; table gaps alone 47.1 vs 24.7 s at 0.15 m/s. Mean episode-minimum palm clearance: U-traps 7.3 (plan on) vs 5.7 cm (plan off) at 0.15, 7.4 vs 5.6 cm at 0.5; all layouts 3.3 / 3.7 / 3.6 vs 3.5 / 3.4 / 3.3 cm.
**H3 fails.** The plan does not raise U-trap + early-turn clean traversal at 0.5 m/s (+0.69 pp [−3.47, +5.56]). It lowers clean traversal in every speed with all three seeds negative at 0.15 and 0.3 m/s (at 0.5 m/s all three seeds are negative but the seed-level interval is wide), and the loss comes from table gaps. What the plan does buy: more clearance in U-traps (palm clearance +1.7 cm) and slightly higher U-trap clean at 0.15 and 0.5 m/s; it costs time in table gaps. The no-plan mode of the same network is the best held-out policy here (clean 62.3 / 63.0 / 60.2 vs reactive V7r 58.3 / 56.2 / 52.1).

### A1 final-so-far (06:10): all three policies, both speeds
Planner–reflex: seeds 1–2 (cluster) at 0.30 m/s, seeds 0–2 at 0.15 m/s; reflex-off: seed 1 only (cluster 322713_1); reactive V7r: 322950. Local seed-0 runs (reflex-off, planner at 0.3 m/s; the local lock was held by other jobs all night) have not started. Percent of episodes, scene-bootstrap 95% CI; per-throw hand hit rate at 0.4 / 0.8 / 1.2 m/s.
| Speed | Policy | Clean | Any contact | Object hit a hand | Object touched anything | Per-throw hand hit (0.4/0.8/1.2) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0.15 | Planner–reflex (3 seeds) | 27.3 | 70.4 | 19.8 [18.1, 21.7] | 46.8 | 2.8 / 10.0 / 17.0 |
| 0.15 | Reflex off (1 seed) | 27.5 | 69.9 | 19.8 [17.1, 22.5] | 46.6 | 3.1 / 9.2 / 17.7 |
| 0.15 | Reactive V7r | 27.3 | 70.4 | 19.7 [16.8, 22.5] | 40.5 | 3.6 / 12.6 / 19.4 |
| 0.30 | Planner–reflex (2 seeds) | 25.9 | 70.8 | 19.9 [17.6, 22.1] | 45.1 | 3.4 / 11.2 / 20.4 |
| 0.30 | Reflex off (1 seed) | 27.3 | 69.3 | 18.0 [15.1, 20.8] | 44.1 | 2.3 / 10.3 / 19.1 |
| 0.30 | Reactive V7r | 28.6 | 69.5 | 16.4 [13.8, 19.1] | 39.8 | 3.2 / 13.1 / 17.3 |
Paired differences (percentage points): planner–reflex minus reflex-off, object hand contact +0.04 [−2.91, +2.95] (0.15), +1.89 [−1.17, +5.01] (0.30); planner–reflex minus reactive, object hand contact +0.17 [−2.99, +3.34] (0.15), **+3.45 [+0.26, +6.51]** (0.30), any contact with the object +6.3 [+2.4, +10.2] and +5.3 [+1.4, +9.1]. Clean traversal differences are all within ±3 pp with intervals including zero (planner–reflex minus reactive at 0.30: −2.73 [−6.18, +0.72]).
**Verdicts.** H1a (the reflex lowers hand contact with thrown objects by ≥ 2 pp): **fails** at both speeds (point estimates +0.04 and +1.89, wrong sign at 0.30 m/s). H1b (non-inferior to reactive within +2 pp): **fails** at 0.30 m/s (upper bound +6.5) and is inconclusive at 0.15 m/s (point +0.17, CI half-width ±3.2 exceeds the margin). Descriptively, the reflex-on and reflex-off policies and the reactive policy are indistinguishable at 0.15 m/s, and the reactive policy touches the thrown objects less often (about −5 to −6 pp any object contact) at both speeds. The hand-hit rate rises with object speed for every policy (about 3% to 17–20%), so a speed-dependent reaction limit, not the policy family, sets the outcome. Caveats: one reflex-off seed and one reactive checkpoint; the on/off contrast compares separately trained networks; no oracle bound for avoidability was computed.

### Integration note (06:30): improve agent's I4 (plan input off, 384 scenes)
The improve agent's `--tracker-mix none=1` runs on the 384 scenes (`improve_none_seed0_0772f2b`, seed 0 so far) agree with A3: plan off 64.8 / 63.8% clean at 0.15 / 0.30 m/s against 61.1 / 61.8% for the plan-on policy (3-seed pool; plan on minus off −3.8 [−5.7, −1.9] and −2.0 [−4.2, +0.3] pp, any contact +4.3 [+1.8, +6.9] and −1.3 [−4.2, +1.5]). They are included in `results.json` as "Planner–reflex, plan off" (narrow benchmark) and appear in the figures and on the website.
