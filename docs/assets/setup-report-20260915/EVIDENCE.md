# Progress figure evidence

The progress snapshot was captured on **15 September 2026 at 14:47:33 UTC**.
It contains **1,086 complete updates**, ending at **1,138,753,536 training
transitions**, from the single W&B run
[skvayzer/CAT-wholebody/1d39c55c](https://wandb.ai/skvayzer/CAT-wholebody/runs/1d39c55c).
The status file was read just before the metrics and records one fewer saved
update; this is collection timing, not an absent metric update.

- Raw `metrics.jsonl` snapshot SHA-256:
  `7c85e48d853ab9f4595cb13078f167ca2438ec65ccd64a0a6c591d48e8c34d32`
- Numeric `progress-series.csv` SHA-256 (LF line endings):
  `c2f91218b252034cc22c942a4d268688226bbbc0d23a5e33c0b3a34a4d488f14`
- Running source commit: `40b4dd2682dd4e5d75a6b853c24333da99379cf3`.
- Released CAT model revision: `46ce4b57ba0639168d51741b661ff62f7ce6f045`.
  Both actor and critic were restored; initialization parity errors were zero.

`progress-series.csv` preserves the selected numeric metrics as round-trippable
floating-point values. It is sufficient to reconstruct the figure and comparison
windows without the 9 MB raw log, checkpoint files, machine launch records, or
credentials. Full raw evidence remains in `/tmp/cat-setup-report-20260915` on the
collection Mac and dep-0; `/tmp` is temporary storage.

## Reproduction

From the repository root, using its Python environment:

```bash
.venv/bin/python docs/assets/setup-report-20260915/reproduce_progress.py \
  --output-dir /tmp/cat-progress-reproduction
```

The analyzer requires only NumPy and Matplotlib. It reads its adjacent CSV by
default, accepts `--series` for another path, and writes the PNG, PDF, and a JSON
summary to `--output-dir`. It accesses no network, GPU, training process, or
checkpoint. Reproduction was checked against the report: window statistics
matched exactly, and the regenerated PNG was byte-identical in the same local
environment. PDF metadata timestamps may differ.

## Derivation

There are 1,048,576 transitions per update. Each selected rollout metric is a
mean over a growing buffer up to 1,000 updates, then a rolling 1,000-update buffer.
For logged mean `m[n]`, reconstruct the underlying update value `x[n]` as:

```text
n <= 1000: x[n] = n*m[n] - (n-1)*m[n-1], with m[0] = 0
n >  1000: x[n] = 1000*(m[n]-m[n-1]) + x[n-1000]
```

Episode counts and failure/timeout ratios are reconstructed independently for
the global aggregate, furniture scene PF37, and generic-clutter scene PF38.
Termination counts are the reconstructed ratio times reconstructed episode
count. Integer rounding is allowed only after numerical validation. Both clutter
scenes have positive reconstructed contributions in every update. Original CAT
counts are the global counts minus both clutter scenes; these 37 slots include
36 byte-verified original scenes and one explicitly reconstructed missing scene.

Validation checks contiguous updates, positive episode counts, integral failure
counts, failure/timeout partitions, nonnegative original residuals, and agreement
of global counts with independently reconstructed per-transition event rates and
cumulative completed-episode deltas. Maximum episode-count roundoff was
`2.33e-10`; maximum termination-count roundoff was `1.38e-10`.

Comparison windows are updates **21–70** and **1037–1086**. Failure curves and
window rates weight reconstructed counts by completed episodes. The top curves
use 20-update smoothing. Logged episode duration and episode reward are summaries
of the last 1,000 completed episodes; those values are averaged across the
comparison updates. Normalized hand/arm penalty is the mean of each logged
episode penalty divided by its corresponding logged episode length.

## Limits

These are stochastic training rollouts with an adaptive scene mix, not a matched
pretrained-versus-trained evaluation. The early window already includes
fine-tuning. Early termination combines falls, body/probe collisions, and invalid
states. A timeout establishes survival, not goal completion or successful room
traversal. The hand-clearance penalty measures the shaping objective, not a hand
collision count. The saved best model uses a single-update training-reward proxy,
not validated furniture traversal performance.
