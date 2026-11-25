# Improving the planner–reflex policy (overnight 2026-11-24/25)

Summary: to be written at the end of the night.

Branch `feature/improve-overnight` from `feature/energy-price` 16ad2fe. Worktree `~/robotics/cat-energy-improve`.
History and baselines: `docs/overnight_20261123.md` (E0–E7). Baselines on the 384 narrow scenes, 2 episodes per scene per
speed: reactive V7r u600 (314948), planner–reflex COMBO_FIX seeds 0–2 (315810, 318295_0, 318295_2), reflex-off COMBO_NOREFLEX.

## Common protocol and decision rule (pre-registered 22:55 +04)

- Benchmarks: the 384-scene list `/tmp/cat-plan-blind/guides/scenes.txt` on `tablegap_v1` with the goal-column cache, nominal,
  deterministic, 2 episodes per scene at 0.15 and 0.30 m/s (1536 worlds, 21,000 steps); and the 72 held-out E4 layouts at
  0.15/0.30/0.50 m/s with the certified starts.
- Paired comparisons are against COMBO_FIX (same checkpoint, default rule: 4 candidates, clearest) and against reactive V7r.
  I report per-seed means ± s.d., seed-level paired t intervals (3 seeds) and a scene-cluster bootstrap over the pooled
  seed × scene pairs (10,000 resamples of scenes).
- A variant **counts as an improvement** at a speed only if the clean-rate change has a 95% CI lower bound above 0 and the
  contact-rate change has a point estimate ≤ 0 (its CI upper bound is reported). Pilot gate for spending seeds 1–2 on an
  eval-only variant: on seed 0, mean clean gain ≥ +1.5 pp over the two speeds and contact change ≤ +0.5 pp at each speed.
- Every eval-only variant also runs a control with the same code and GPU (the default rule) to separate code/GPU drift from
  the rule change.

## I1 — value-guided plan selection (eval-only)

Hypothesis: the planner's "clearest" rule picks plans the tracker cannot execute (E0: refusals start at a stall with plan
tracking credit intact). The tracker's own critic V(s, plan) — the asymmetric critic with the 41-dim reference block rebuilt
from each candidate — ranks candidates by expected return under the actual tracker, so choosing max V among clearance-safe
candidates should pick executable plans.
Implementation (`b2a5e5c`): `--plan-select value_safe` takes max V among candidates with ≥ 5 cm body clearance (or within 1 cm
of the clearest if none has 5 cm); `--plan-select value` ignores clearance. Stand-still worlds and the first plan after a
reset keep the clearance rule (no fresh observation yet). The switch and sideways penalties are not applied under value
selection.
Smoke (256 worlds, 300 steps, seed 0): 48% of replans pick a different candidate from the clearest one; the candidate value
spread is 0.28 (critic units).
Prediction: on seed 0, value_safe raises clean by ≥ 1.5 pp averaged over the two speeds, mostly by lower refusal, with
contact within +0.5 pp. Falsifier: clean gain < 1.5 pp or contact up > 0.5 pp at either speed.

## I2 — more candidates (eval-only)

Hypothesis: with 4 samples (2 forward, 2 sideways) the forward choice is often between two similar chunks; 8 samples give the
selection rule more executable options. Variants: clear rule with 8 samples; value_safe with 8 samples.
Prediction: K=8 clear changes clean by < 1 pp (clearance is already capped at 10 cm); K=8 value_safe beats K=4 value_safe
by ≥ 1 pp clean. Falsifier: the opposite ordering.

## I3 — value-based mode arbitration (eval-only)

Hypothesis: when even the best plan looks bad to the critic relative to acting with no reference, the same network should
act reactively (its 'none' training source, 25% of training worlds) for 0.6 s and then retry.
Smoke finding before any full run: V(no reference) exceeds V(best plan) by 0.91 on average, so an uncalibrated δ = 0 rejects
70% of replans (nearly the reactive mode). The threshold is therefore set from the gain distribution of the full I1 seed-0
run to reject the lowest ~10% of plans (δ = 10th percentile of V(plan) − V(none)). This uses the same scenes, which I flag as
a calibration leak; the held-out E4 layouts give a clean check.
Prediction: arbitration lowers refusal by ≥ 2 pp versus value_safe alone without raising contact. Falsifier: refusal
unchanged or contact up > 0.5 pp.

## I4 — reactive mode of the same network and per-scene oracle (eval-only)

`--tracker-mix none=1` on COMBO_FIX (no reference at all) measures the reactive mode of the planner–reflex network. The
per-scene oracle takes, per scene and speed, the better of reactive V7r and COMBO_FIX (clean first, then no contact), as an
upper bound on what arbitration between the two policies could reach. It is reported only as a bound.

## I5 — expert iteration of the planner (one planner fine-tune, frozen trackers)

Hypothesis: planner v1 learned from reactive policies' motions (V7, E2, T2); the tracker it now drives executes some of its
chunks poorly. Retraining it on the planner–reflex tracker's own contact-free windows (hindsight goals: what the tracker
actually did in the next 2 s), especially its successful close passes, moves the planner's distribution toward motions this
tracker executes, so plans stall less.
Recipe (`99fb69c`, `hpc/improve_expert_iter.sbatch`, job 322717 on ws-ia): collect from COMBO_FIX seed 0 with the plan
source only, 4096 worlds × 1500 steps × 3 rounds plus a held-out round, the 384 evaluation scenes excluded from collection
(asserted), side-gap/narrow/protected/transition/table-edge scene counts doubled, at most 150k new training windows; merge
with planner v1's V7/E2/T2 data; fine-tune planner v1 (warm start, normalizer kept) for 20k steps at lr 1e-4. Then
evaluate the frozen COMBO_FIX trackers (seeds 0–2) with the new planner on the 384 scenes and the 72 held-out layouts.
Planner v1's original training scenes already overlap the 384 evaluation scenes; the E4 layouts were never collected.
Prediction: refusal falls ≥ 2 pp and clean rises ≥ 1.5 pp on seed 0, with contact ≤ +0.5 pp; on E4 clean does not fall.
Falsifier: clean gain < 1.5 pp, or contact up > 0.5 pp, or E4 clean down > 3 pp.
A tracker fine-tune with the new planner (the second half of the brief's item 4) costs ~5 h per seed on the free ws-ia
slots and is only scheduled if the frozen-tracker result passes.

## Batch C decision rules (fixed 23:58, before any batch B result)

`scripts/improve_local/plan_batch_c.py` runs when batch C gets the GPU lock and applies the rules above to batch B's seed-0
files: (1) I3 arbitration with δ = 10th percentile of V(plan) − V(none) from the value_safe K=4 run, on top of value_safe
K=4 and of the default clearest rule; (2) seeds 1–2 for the passing variant with the largest mean clean gain (none if no
variant passes the pilot gate); (3) E4 held-out seed-0 runs for the same-code control and that variant; (4) if the expert-
iteration planner is ready, the seed-1 and seed-2 trackers with it (seed 0 runs on the cluster in 322842).

## I6 — stall-triggered planning: plan off by default (pre-registered 05:20, after the I4 seed-0 result and the eval agent's A3 report that plan-off beats plan-on on E4)

Hypothesis: the planner–reflex network is best without a reference in most scenes (I4, A3), but a plan is still needed where
reactive following stalls (U-trap dead ends, route commitment). Rule (`9f497ab`, `--tracker-stall-plan 2 4`): plan worlds act
without a reference; after 2 s without 5 cm of route progress a fresh plan is engaged for 4 s, then released (re-engaged on the
next stall). CPU smoke on published-side-hurdle-crouch1: the plan engages after the first stall (3.4 s) and toggles every
~4 s while the robot stays stuck.
Prediction: on the 384 scenes it keeps ≥ 80% of the no-reference clean gain over plan-on at 0.15 and lowers contact versus the
no-reference mode at 0.30 (plan-mode stops in CAT/side gaps); on E4 it keeps the U-trap rate of plan-on and the table-gap rate
of plan-off. Falsifier: clean below the no-reference mode by > 1 pp at both speeds, or E4 U-trap clean below plan-on by > 5 pp.
Runs: seed 0 (384 + E4) in batch D2 on dep-1; seeds 1–2 on the 384 scenes as cluster 324542 (after 323469); E4 seeds 1–2 in D2.

## Results

### I4 seed 0 — the planner–reflex network without any reference (02:45)

`--tracker-mix none=1` on COMBO_FIX seed 0 (`0772f2b`, dep-1), against the original seed-0 evaluation 315810 (cluster, older
code; the same-code control from batch B follows). Scene-bootstrap 95% intervals, pp.

| Speed | clean | contact | refused | hands |
| --- | --- | --- | --- | --- |
| 0.15 | 59.77 → 64.84, +5.08 [+2.73, +7.55] | 30.08 → 26.82, −3.26 [−6.51, 0.00] | 10.03 → 8.07, −1.95 [−4.43, +0.39] | 2.60 → 2.99 |
| 0.30 | 61.59 → 63.80, +2.21 [−0.39, +4.82] | 25.52 → 29.43, +3.91 [+0.78, +7.16] | 12.37 → 6.64, −5.73 [−8.46, −2.99] | 2.34 → 2.99 |

Against reactive V7r (314948) the no-reference mode gains +5.86 [+2.47, +9.38] clean and −9.24 [−12.89, −5.60] contact at
0.15, and +0.91 [−2.47, +4.30] clean, −3.12 [−6.51, +0.26] contact at 0.30. By group, it raises clean traversal in cabinets
(81 → 87/89%) and furniture rooms and lowers contact there; in CAT courses and side gaps it converts plan-mode refusals into
contacts (at 0.30: CAT contact 58 → 73%, side gaps 39 → 58%). The best-of-both oracle per scene reaches 67–68% clean.
Reading: training with planner references (and 25% reference-free worlds) produced a better reactive policy than V7r; at
inference the plan helps mainly by making the robot stop in front of passages it would otherwise hit. Seed 0 only.

Batch C rule change (02:45, before any other batch B result was read): the no-reference mode gets seeds 1–2 and an E4 run,
and value arbitration at δ = 0 ("follow the plan only when the critic prefers it to no reference", ~70% of replans rejected
in the smoke) is added; at most 5 full runs plus 2 E4 runs fit in GPU memory, so lower-priority runs are deferred.

### E2 seed 0 — refusal-pricing fine-tune (322679_0, 03:20)

150 updates from COMBO_FIX seed 0 with `--refusal-training`, then the standard 384-scene evaluation (cluster, same lineage as
315810). Scene-bootstrap 95% intervals.

| Speed | clean | contact | refused | hands |
| --- | --- | --- | --- | --- |
| 0.15 | 59.77 → 60.55, +0.78 [−1.82, +3.39] | 30.08 → 31.64, +1.56 [−1.56, +4.56] | 10.03 → 7.55, −2.47 [−4.30, −0.78] | 2.60 → 1.82 |
| 0.30 | 61.59 → 62.24, +0.65 [−1.82, +3.12] | 25.52 → 27.08, +1.56 [−1.17, +4.43] | 12.37 → 10.55, −1.82 [−4.04, +0.39] | 2.34 → 2.34 |

Pre-registered prediction (≥ 2 pp clean, ≥ 2 pp fewer refusals, no more contact at either speed): **not met** — refusals
fall, but as with the E1 recovery rules they turn mostly into CAT-course contacts (contact 64 → 73% at 0.15) rather than clean
passes; side gaps gain +10 pp clean at 0.15 [+1.2, +20.0]. By the seed-0 pilot rule, E2 seeds 1–2, E4 and plan-blind runs
are not spent. Checkpoint: `/l/users/konstantin.smirnov/runs/stationary_hands_20261113_overnight_E2_seed0_322679/run/resume.pt`.

### Same-code control, I1 and I2 on seed 0 (batch B, finished 03:53)

The same-code control (default rule, `0772f2b`, `--subset-fields`, dep-1) reproduces the cluster evaluation 315810 within
noise: clean +0.13 [−2.47, +2.73] and −0.26 [−2.86, +2.21] pp, contact +0.65 and +1.69 pp. All deltas below are against this
control; scene-bootstrap 95% intervals, pp.

| Variant (seed 0) | clean 0.15 | contact 0.15 | clean 0.30 | contact 0.30 | gate |
| --- | --- | --- | --- | --- | --- |
| I1 value_safe, K=4 | +3.12 [+0.65, +5.60] | −2.99 [−5.73, −0.26] | −1.17 [−3.78, +1.43] | +1.69 [−0.91, +4.43] | fail (mean +0.98) |
| I2 clearest, K=8 | −1.30 [−3.52, +0.91] | +0.26 [−2.47, +2.99] | −1.56 [−4.17, +1.04] | +0.26 [−2.60, +2.99] | fail |
| I2 value_safe, K=8 | +2.47 [+0.26, +4.69] | −2.60 [−5.21, 0.00] | −0.13 [−2.47, +2.21] | +1.04 [−1.30, +3.39] | fail (mean +1.17) |
| I4 no reference | +4.95 [+2.86, +7.16] | −3.91 [−7.16, −0.65] | +2.47 [0.00, +4.95] | +2.21 [−0.91, +5.47] | fail (contact at 0.30) |

Control absolute rates: 59.90 / 30.73 / 8.98 (clean / contact / refused) at 0.15 and 61.33 / 27.21 / 11.07 at 0.30.
Value selection switches away from the clearest candidate in 45% of replans; the candidate value spread has median 0.22
(critic units); V(chosen plan) − V(no reference) has median −0.79 and 10th percentile −3.72 (the I3 threshold). The
value_safe gain at 0.15 comes from cabinets (80.3 → 86.6% clean [+2.5, +10.0], contact 16.2 → 9.7%); nothing changes in CAT
courses. I2's prediction (K=8 clearest changes clean by < 1 pp) is borderline (−1.3/−1.6 pp, intervals include 0); its
second prediction (K=8 value_safe beats K=4 value_safe by ≥ 1 pp) is not met. Pattern across all eval-only variants: gains
appear at 0.15 m/s and vanish at 0.30, where plan-mode stops (refusals) are what keeps contact down.

### I4 three seeds and I5 seeds 1–2 (07:00)

No-reference mode, seeds 0–2 (dep-1, `0772f2b`) against plan-on (seed 0: same-code control; seeds 1–2: the round-4 cluster
evaluations 318295_0/_2, whose seed-0 counterpart the same-code control reproduced). Mean ± s.d. over seeds; t = seed-level
paired t interval, b = scene bootstrap over pooled seed pairs.

| Comparison | Speed | clean | contact | refused | hands |
| --- | --- | --- | --- | --- | --- |
| plan-on | 0.15 | 61.11 ± 1.39 | 31.38 ± 0.65 | 7.38 ± 1.51 | 2.52 ± 0.15 |
| no reference | 0.15 | 63.80 ± 0.98 | 29.56 ± 2.38 | 6.47 ± 1.40 | 2.82 ± 0.54 |
| Δ vs plan-on | 0.15 | +2.69 t[−2.33, +7.72] b[+1.22, +4.25] | −1.82 t[−6.32, +2.67] b[−4.21, +0.52] | −0.91 b[−2.91, +1.04] | +0.30 b[−0.65, +1.26] |
| Δ vs reactive V7r | 0.15 | +4.82 t[+2.38, +7.26] b[+1.65, +8.16] | −6.51 t[−12.41, −0.61] b[−9.94, −3.21] | +1.78 b[−0.04, +3.60] | −0.17 |
| plan-on | 0.30 | 61.76 ± 1.61 | 28.69 ± 2.44 | 9.38 ± 1.54 | 2.86 ± 0.52 |
| no reference | 0.30 | 64.15 ± 0.96 | 29.69 ± 0.26 | 6.03 ± 1.17 | 3.17 ± 0.15 |
| Δ vs plan-on | 0.30 | +2.39 t[+0.76, +4.02] b[+0.78, +3.99] | +1.00 t[−5.09, +7.09] b[−1.35, +3.34] | −3.34 b[−5.43, −1.30] | +0.30 |
| Δ vs reactive V7r | 0.30 | +1.26 t[−1.13, +3.64] b[−1.69, +4.21] | −2.86 t[−3.51, −2.22] b[−5.77, −0.04] | +1.74 | +0.04 |

Decision rule: an improvement over plan-on at 0.15 (clean b-interval above 0, contact point estimate −1.82); **not** at 0.30
(contact +1.00). Over reactive: clean and contact both better at 0.15; at 0.30 only contact is significantly lower. Groups
(3 seeds): cabinets 80.4 → 86.0% clean [+3.2, +8.1] with contact 17.3 → 5.9%; CAT courses and side gaps trade refusals for
contacts (CAT contact 67.8 → 75.2% at 0.15; side gaps 42.5 → 58.3% at 0.30).

I5 (expert-iteration planner, frozen trackers), seeds 1–2 (cluster 323469) vs their plan-on evaluations: clean +0.33
b[−1.37, +2.02] / +0.59 b[−1.17, +2.34], contact −0.85 b[−2.73, +1.04] / −0.98 b[−2.93, +0.98] at 0.15 / 0.30 — within noise;
the predicted ≥ 1.5 pp clean gain is not seen (seed 0 pending).

### I3 arbitration, I5 three seeds, I6 seeds 1–2 (07:50)

Seed 0, 384 scenes, against the same-code plan-on control (scene bootstrap, pp):

| Variant | clean 0.15 | contact 0.15 | clean 0.30 | contact 0.30 | refused 0.15 / 0.30 | replans rejected |
| --- | --- | --- | --- | --- | --- | --- |
| I3 value_safe + arbitration δ = −3.72 | +3.78 [+1.69, +5.86] | −2.73 [−5.08, −0.39] | +1.69 [−0.65, +4.17] | −2.21 [−4.69, +0.13] | −0.65 / +0.65 | 6.4% |
| I3 clearest + arbitration δ = 0 | +2.08 [−0.26, +4.56] | −3.26 [−6.51, 0.00] | +1.95 [−0.52, +4.43] | −2.08 [−5.21, +1.04] | +1.17 / +0.39 | 50.2% |

Absolute (δ = −3.72): 63.67 / 27.99 / 8.33 at 0.15 and 63.02 / 25.00 / 11.72 at 0.30 (clean / contact / refused).
Both arbitration rules pass the pilot gate (mean clean ≥ +1.5 pp, contact not up at either speed) — the first variants that
lower contact at both speeds. The pre-registered I3 prediction (refusal −2 pp vs value_safe alone) is not met: arbitration
lowers contact rather than refusal. Against the no-reference mode, δ = 0 lowers contact at 0.30 by 4.30 pp [−7.16, −1.56]
(side gaps 57.5 → 43.8%) at −0.52 clean, and costs 6 pp clean in cabinets (more refusals there). δ = −3.72 gains in cabinets
(80.3 → 87.2% clean) like value_safe and adds contact reductions in CAT courses at 0.30 (62.5 → 58.7%).
Seeds 1–2 for both rules: cluster 324627 (pending). E4 for all variants × 3 seeds: batch D2 (waiting for the dep-1 lock).

I5, 3 seeds (seed 0 on dep-1): clean +0.48 b[−0.83, +1.78] / +0.87 b[−0.56, +2.26], contact −1.22 b[−2.78, +0.35] /
−1.26 b[−2.73, +0.22] at 0.15 / 0.30 — consistent small contact reduction, no significant change; prediction not met.

I6 seeds 1–2 (324542): vs plan-on +1.82 b[0.00, +3.65] / +2.86 b[+1.04, +4.62] clean, −0.07 / +1.89 contact; vs the
no-reference mode +0.26 / +0.52 clean, +0.72 / +1.50 contact, −1.11 / −2.02 refusals. The stall-engaged plan behaves like the
no-reference mode; it does not recover plan-on's low contact in CAT courses/side gaps. Falsifier not triggered (clean is not
below no-reference by > 1 pp) but no gain either; seed 0 and E4 pending.

## Run ledger

| Id | Variant | Seed | Where | Output |
| --- | --- | --- | --- | --- |
| A | control (K=4, clearest), value_safe K=4, clear K=8, value_safe K=8, no reference (none=1) | 0 | dep-1, one lock hold, 5 in parallel, `92afe4e` | `~/robotics/cat-energy-local-runs/improve_<tag>_seed0_92afe4e` |
| E2 | refusal-pricing fine-tune (from the brief) | 0 | 322679_0 ws-ia | `overnight_E2_seed0_322679` |
| E2 | seeds 1–2: 322679_1/2 and 322686 stopped at the foreign-GPU-memory check on gpu nodes; 322708 with the check relaxed then ran out of GPU memory (other users' processes held 25–26 GB of the same GPU), so the check was right: gpu-partition GPUs are shared. Seeds 1–2 now wait for the seed-0 pilot (decision: run them on ws-ia/dep-1 only if seed 0 gains ≥ 1 pp clean without more contact) | 1–2 | — | — |
| I5 | expert iteration collect + fine-tune + seed-0 eval (384 and E4) | 0 | 322717 ran 38 min (one 50k-window round) and was cancelled by mistake when I resubmitted with eval stages; 322842 (`cd6d68b`) restarted it from scratch on ws-l3-021 at ~00:05 | `runs/wbp/planner_ei1`, `runs/improve_ei1{,_E4}_seed0_322842` |
| B | batch A hit CUDA OOM (full field bank ~7–13 GB per process); rerun as batch B with `--subset-fields` (`0772f2b`), 6 in parallel from 23:24; pure `value` K=4 stopped at step ~1000 to speed the rest (not pre-registered) | 0 | dep-1 | `improve_<tag>_seed0_0772f2b` |
| C | none seeds 1–2, ei1 seed 0, arbitration (δ = −3.72 on value_safe; δ = 0 on clearest), E4 control / none / ei1 seed 0 | 0–2 | dep-1, 8 parallel from 03:53 | `improve_<tag>_0772f2b` |
| I5 | 322842's eval stage failed to load the planner (`PosixPath` in saved args; fixed in 7e34461 and the checkpoint re-saved); seed-0 eval moved to batch C; seeds 1–2 queued as 323469 (cancel if seed 0 fails) | 1–2 | ws-ia | `improve_ei1_seed{1,2}_323469` |
| E2 | seed 0 done (see results); seeds 1–2 not run (pilot rule) | 0 | — | — |
| C/E4 | the three E4 runs in batch C ran out of GPU memory at start (E4 field bank ~6.4 GB per process); moved to batch D with `--subset-fields` | 0 | — | `failed_oom_improve_E4_*` |
| D | E4: control seeds 0–2, no reference seeds 0–2, ei1 seed 0 (`--subset-fields`) | 0–2 | dep-1, queued 03:58 behind batch C | `improve_E4_<tag>_0772f2b` |
| V | videos: 10 obstacle types × {no reference, planner–reflex}, seed-0 COMBO_FIX, 0.15 m/s, CPU MuJoCo recordings (`record_plan_reflex.py --device cpu`, contact checked once per control step), software-EGL renders, 720p H.264 yuv420p faststart with a silent AAC track | 0 | dep-1 CPU | `files/humanoid-protection/improve-videos/` |
| D2 | replaces the waiting batch D (stopped before it started): code `9f497ab`; I6 seed 0 on the 384 scenes; E4 control / no reference / I6 for seeds 0–2, ei1 seed 0 | 0–2 | dep-1, queued 05:22 | `improve_<tag>_9f497ab` |
| I6 | stall-triggered planning, 384 scenes | 1–2 | 324542 ws-ia (after 323469) | `improve_stallplan_seed{1,2}_324542` |
| I3 | arbitration δ = −3.72 (value_safe) and δ = 0 (clearest), 384 scenes, 2 in parallel per GPU | 1–2 | 324627 ws-ia (pending) | `improve_{arbvsafe,arb0}_seed{1,2}_324627` |
