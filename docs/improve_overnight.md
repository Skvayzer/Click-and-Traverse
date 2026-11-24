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

## Run ledger

| Id | Variant | Seed | Where | Output |
| --- | --- | --- | --- | --- |
