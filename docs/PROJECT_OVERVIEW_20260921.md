# Project overview — 2026-09-21

## 1. Executive summary

The policy is better than released CAT on the fixed clutter evaluation, but worse on the broad CAT evaluation because it times out.
The apparent CAT improvement on three scenes was unrepresentative and is superseded by the 400-scene result.
The most important objective—raising/tucking hands inside passages to protect fingers—was structurally inactive for the audited 279M transitions.
Flat-ground posture and generic clearance did train; their improvements do not establish passage hand protection.
A successor bank restores active hand/heading zones, and its PPO run has started producing nonzero objective samples and costs.
The structural fix is verified; successful protected traversal is not yet demonstrated.
CAT timeout recovery remains the leading unresolved behavioral regression, subordinate to the user's finger-protection priority.
Disk capacity remains critical; this report used read-only inspection and did not interact with the training job or GPU.

## 2. Goal and priorities

CAT (Click-and-Traverse) whole-body control for Unitree G1 with fixed Dex3 grippers, 29 actuated joints, using mjlab/MuJoCo-Warp. Priorities are ordered:

1. **Protect the robot's fingers by raising/tucking hands in passages.** This is the primary objective.
2. Traverse narrow passages.
3. Preserve navigation and traversal in CAT's own scene bank.

Evidence convention: **verified** means supported by inspected source or saved artifacts; **session-reported** means supplied in the session brief but not independently reproduced here. Current file locations/line numbers supersede older session references. No simulations, evaluation jobs, or checkpoint tensor loads were run for this report.

## 3. What was changed this session

| Change | Reason and evidence | Status / limitation |
|---|---|---|
| SAPG (one leader, five followers) → plain PPO | Session-reported importance-sampling ESS collapse in the 29-dimensional action space made relabelling ineffective. Predecessor and successor run configuration confirm PPO. | Algorithm switch verified; historical ESS diagnosis not independently reconstructed. |
| Width curriculum | Session reports zero narrow success over 525M transitions before starting at 0.70 m instead of 0.40 m; six rungs descend to 0.40 m. Curriculum validation is in `cat_ppo/furniture/width_curriculum.py:15`. | Breakthrough after 14 updates is session-reported; later narrow success around 53% is verified. This does not establish success at every width. |
| Restore CAT reset mass | Original/procedural reset masses had both been zero, explaining a misleading 0.000 retention metric. Current retention family weights are 0.20/0.40 (`cat_ppo/furniture/width_curriculum.py:12`; `scripts/build_flat_balance_bank.py:63`). | Restored values verified; earlier zero-mass state is session-reported. These family weights are not the successor's final overall mixture. |
| Flat-ground posture reward | Added a walking/posture bonus; run override raises bonus scale from default 3 to 10, with region scale 0.40. | `cat_mjlab/balance.py:22`, `cat_ppo/furniture/balance_bank.py:10`; separate from in-passage contrast shaping. |
| Paired evaluation | `scripts/evaluate_paired_cat.py` compares frozen policies on matched scenes, certified reset poses and seeds. | Revealed clutter gains and CAT regression. Replay-equivalence verification remains incomplete; see §4. |
| Hand-objective bank and preflight | Restore protected/transition scenes and reject structurally empty hand/heading coverage. | Implemented and running; see §§5–6. |

The audited predecessor is `outputs/cat_flat_balance_ppo_37632_20260920`:

| Setting | Value |
|---|---:|
| Audited prefix | 232 updates / 279,379,968 transitions |
| Environments / unroll | 37,632 / 32 |
| Algorithm / discount | PPO / 0.98 |
| Maximum action standard deviation | 0.15 |
| Flat bonus / region scale | 10.0 / 0.40 |

**Snapshot correction:** the saved `metrics.jsonl` now contains 242 rows, ending at 290,217,984 steps. Rows 232 and 233 repeat step 279,379,968. Thus “232 updates” identifies the session's audited prefix, not the entire current file. Its `status.json` still says `running` at update 240; that stale label is not evidence that the predecessor is currently live.

The table below recomputes arithmetic means over five contiguous portions of the first 232 rows, with boundaries `floor(i × 232 / 5)`. These are means of logged update metrics, not pooled episode estimates.

| Metric | Q1 | Q2 | Q3 | Q4 | Q5 | Row 232 |
|---|---:|---:|---:|---:|---:|---:|
| Flat walking compliance | 0.0126 | 0.0388 | 0.0608 | 0.1150 | 0.1720 | 0.1942 |
| Left hand height p50, m | 0.7940 | 0.8148 | 0.8147 | 0.8298 | 0.8433 | 0.8487 |
| Left hand height p90, m | 0.8665 | 0.8782 | 0.8767 | 0.8879 | 0.8993 | 0.9049 |
| Flat fall rate per episode | 0.0512 | 0.0018 | 0.0015 | 0.0021 | 0.0018 | 0.0058 |
| CAT goal success | 0.1161 | 0.1111 | 0.1093 | 0.1194 | 0.1199 | 0.1242 |
| Ordinary clutter success | 0.4386 | 0.6807 | 0.6926 | 0.6240 | 0.5627 | 0.5670 |
| Narrow passage success | 0.2926 | 0.4860 | 0.5274 | 0.5327 | 0.5291 | 0.5375 |
| PPO ratio clip fraction | 0.3592 | 0.3575 | 0.3535 | 0.3558 | 0.3607 | — |
| Entropy loss | 0.0530 | 0.0516 | 0.0512 | 0.0518 | 0.0522 | — |

Source: [predecessor metrics](../outputs/cat_flat_balance_ppo_37632_20260920/metrics.jsonl) and [run configuration](../outputs/cat_flat_balance_ppo_37632_20260920/run.json).

The session's “15×” compliance improvement compares Q1 with the last update; Q5/Q1 is about 13.6×. Growth remains strong at the end, with no visible plateau; sustained future acceleration is not established. The median hand remains below the 0.870 m target while p90 exceeds it. Session-reported fall endpoints 0.0489→0.0016 and narrow start 0.2964 do not reproduce under the explicit aggregation above. Session-reported CAT peak 0.1495 in the first quintile and clutter peak 0.7137 refer to finer-grained peaks, not quintile means. None of these adaptive training rates substitutes for fixed-scene evaluation.

## 4. Results: the three paired evaluations

A is the released CAT generalist, converted/expanded into the mjlab evaluation setup; these are not measurements in its original runtime. B is our predecessor checkpoint. Both larger runs record deterministic actions and `resume.pt` as B's source, with frozen copies inside each evaluation directory. The successor instead starts from `best.pt`; do not assume its initialization is byte-identical to evaluated B.

| Evaluation / group | Scenes | Pairs | A clean-goal rate | B clean-goal rate | B − A |
|---|---:|---:|---:|---:|---:|
| Small CAT: original/published | 3 | 24 | 0.6875 | 0.8750 | +18.75 pp |
| Fixed clutter: furniture | 24 | 192 | 0.5573 | 0.7969 | +23.96 pp |
| Fixed clutter: generic clutter | 36 | 288 | 0.3819 | 0.7813 | +39.93 pp |
| Large CAT: all original/published | 64 | 512 | 0.2344 | 0.1934 | −4.10 pp |
| Large CAT: procedural subset | 336 | 2,688 | 0.2467 | 0.1957 | −5.10 pp |

The small evaluation and its exact reproduction in two independent processes are **session-reported**; the paired summary artifacts for those two processes were not verified here. Reproducibility did not make three scenes representative. The 400-scene evaluation reverses the optimistic CAT conclusion and is the appropriate evidence for broad CAT retention. It includes every original/published scene and a procedural sample, not necessarily every procedural scene in the training bank.

The clutter and CAT findings are compatible: they concern different scene populations. The fixed clutter test establishes a substantial advantage over A on those scenes. The session diagnosed the training clutter decline as **compositional**, caused by within-bucket sampling adapting toward scenes the policy survives; reported total-variation distances from initial uniform sampling were 0.400 and 0.481. The adaptive sampling mechanism is recorded in predecessor `run.json`; those TV calculations were not independently reproduced. A final-checkpoint comparison against A alone cannot prove that B never deteriorated relative to an earlier B checkpoint. It does invalidate interpreting the aggregate training decline as evidence that B is worse than released CAT on clutter.

| Group | Body-collision outcomes A → B | Unsuccessful timeouts A → B | Timeout rate A → B | Median first clean goal, steps A → B |
|---|---:|---:|---:|---:|
| Furniture | 8 → 6 | 0 → 0 | 0 → 0 | 435 → 553 |
| Generic clutter | 51 → 33 | 0 → 0 | 0 → 0 | 436 → 547 |
| Original/published CAT | 303 → 227 | 0 → 87 | 0 → 0.1699 | 112.5 → 198 |
| Procedural CAT | 1,557 → 1,333 | 0 → 308 | 0 → 0.1146 | 111 → 200.5 |

Clutter arrival medians are about 25–27% slower, not twice as slow. These medians condition on reaching a goal; they do not compare identical subsets of successful episodes. On CAT, B reduces collision outcomes but introduces many failures to arrive before timeout. Both policies remain weak in absolute clean-goal success under this evaluator.

| Original/published paired transition A → B | Count |
|---|---:|
| Body collision → clean goal | 49 |
| Clean goal → body collision | 44 |
| Clean goal → timeout | 26 |
| Body collision → timeout | 55 |
| Body collision → body collision | 177 |

This supports excessive caution/slowness as the main new failure mode; it does not by itself isolate which reward term caused it.

Evidence: [clutter summary](../outputs/paired_clutter_20260921/summary.json), [large CAT summary](../outputs/paired_cat400_det_20260921/summary.json), and their adjacent `run.json` files. Important qualifications:

- Both summaries explicitly say fixed-action replay verification was disabled and a known restore replay mismatch remains unverified. Matched intended inputs are stronger evidence than adaptive aggregates, but exact replay equivalence has not been certified.
- Initial out-of-bounds resets affect 45/512 original and 277/2,688 procedural pairs. For original scenes, conditioning on in-bounds resets gives A=0.2570 versus B=0.2120; the direction remains negative.
- Exploratory scene-cluster bootstrap 95% intervals for B−A are [−12.70, +4.69] pp for original/published and [−8.56, −1.86] pp for procedural. The original-bank point estimate is worse, but its interval crosses zero. Eight episodes per scene are not eight independent scene samples.

## 5. The dead hand objective

**The priority objective received no direct shaping signal throughout the audited 279M transitions. This was a bank-construction failure, not merely a plotting bug.**

The first 232 metric rows contain 37 `hand_contrast` keys; 33 are exactly zero throughout. Saved preflight evidence confirms that the old bank had contrast metadata but no active hand or heading objective zones. The session's detailed inspection reports every one of its 36 zones had `forward_weight=0`, `hand_active=[0,0]`, and `region_valid=[false,false]`; the saved audit independently verifies their aggregate inactivity.

The isolation bank kept only 12 narrow contrast scenes (`scripts/build_flat_balance_bank.py:31`; `cat_ppo/furniture/balance_bank.py:31`). Validation explicitly rejects hand table/shelf scenes (`balance_bank.py:35`). Protected/transition scenes carrying the hand objectives were excluded. Enabling the task's contrast flag therefore did not imply usable hand targets.

Reward assembly calls contrast terms at `cat_mjlab/task.py:408`; the actual −1.0 weights are configured at `cat_mjlab/config.py:50`. An enabled negative cost weight multiplied by an inactive zero cost still produces no learning signal. The older session citation placing those weights directly at task line 408 was imprecise.

Flat-ground posture bonus and generic hand/arm clearance penalties remained active (`cat_mjlab/task.py:404`, `:415`). They can explain posture and collision improvements without demonstrating raised/tucked passage behavior. The run was not wholly unproductive, but it did not train the user's principal objective.

| Left-hand lower-tail audit, first 232 rows | Verified result |
|---|---:|
| Updates with minimum height below 5 cm | 12 / 232 |
| Updates with minimum height below 10 cm | 40 / 232 |
| Worst minimum height | −0.010746 m |

These are counts of updates containing extreme samples, not fractions of all hand poses. They show a recurring lower-tail problem, including a measured hand point below the floor plane, despite improving medians. They are not a direct measurement of finger contact geometry.

## 6. Current setup now running

Successor bank: `data/furniture/cat_flat_hand_balance_v2_20260921`, with matching `_collision` and `_resets` directories. Combined size is session-reported as 458 MB; it was not remeasured here.

| Bank coverage | Old | Successor |
|---|---:|---:|
| Scenes | 2,351 | 2,375 |
| Contrast zones | 36 | 108 |
| Hand-active zones | 0 | 60 |
| Active hand-objective zones | 0 | 60 |
| Positive heading zones | 0 | 60 |
| Forward-protected / narrow / transition scenes | 0 / 12 / 0 | 12 / 12 / 12 |

| Successor reset category | Share |
|---|---:|
| Original CAT | 18% |
| Procedural CAT | 36% |
| Furniture | 5.625% |
| Generic clutter | 3.375% |
| Narrow | 4.5% |
| Flat | 22.5% |
| Forward protected | 7% |
| Transition | 3% |

CAT totals 54%, clutter 9%. Each pre-existing bucket keeps 90% of its prior share. The successor preserves all base scene records and appends 24 protected/transition records; narrow semantics are unchanged (`cat_ppo/furniture/hand_balance_bank.py:6`, `:35`, `:46`).

`cat_ppo/furniture/contrast_preflight.py:7` reads hash-pinned scene metadata and rejects requested hand-contrast training if active hand or heading coverage is zero. It runs during successor bank construction (`scripts/build_hand_balance_bank.py:70`) and training startup (`cat_mjlab/runner.py:379`). [Saved preflight results](assets/hand-balance-bank-20260921/preflight.json) show old-bank failure and successor-bank pass. The legacy isolation builder explicitly opts out (`scripts/build_flat_balance_bank.py:90`). This guard blocks the observed structural failure under the current training startup path; it does not prove runtime visitation, useful gradients, or successful behavior.

| Live successor configuration | Value |
|---|---|
| Launcher | `configs/pilots/flat_hand_balance_v2_20260921.sh` |
| Run directory | `outputs/cat_flat_hand_balance_ppo_30720_20260921` |
| Initialization | Predecessor `best.pt`, `--fresh-optimizer` |
| Required objective | `--require-hand-contrast` |
| Algorithm / environments | PPO / 30,720 |
| Batch / minibatches / unroll | 768 / 40 / 32 |
| Discount / maximum action std | 0.98 / 0.15 |
| Flat bonus / region scale | 10.0 / 0.40 |
| Checkpoint interval | 10 updates |

The session identifies tmux `cat_hand` as live; tmux was not accessed. Read-only inspection found three successor metric rows at 2,949,120 transitions. The latest row contains 169,272 hand-region cost samples, mean region cost 0.6367, heading compliance 0.0890, and hand-region compliance 0.0. This is evidence of active objective exposure, **not a hand-protection success claim**. The launcher still has a “Prepared only” comment; the produced metrics are better evidence of execution than that stale comment.

## 7. Open problems, ranked, with the proposed fix for each

Rank reflects user priorities; CAT timeouts are the leading regression after restoring the dead objective.

| Rank | Problem | Proposed next action / fix | Evidence needed |
|---|---|---|---|
| 1 | Passage finger protection remains unproven; dangerous hand-height tail persists. | Continue the active-objective experiment, track protected/transition exposure and both hands separately, and evaluate raised/tucked traversal on fixed scenes. Audit extreme low poses before adjusting protection rewards. | Nonzero runtime samples, improving region compliance, safe clearance and clean traversal together. |
| 2 | CAT timeout regression. | Smallest proposed code change: increase existing `tracking_root_field` velocity tracking on affected CAT rows. No dedicated CLI control currently exists. Test as a separate ablation; do not sacrifice fingers to recover speed. | Lower unsuccessful timeouts and restored fixed-bank success without increased hand/body collisions. |
| 3 | Exploration may be constrained. | Measure per-joint standard deviations and fraction at the 0.15 cap before testing a modest cap increase. | Better learning/coverage without instability. Clip fraction and entropy alone do not establish exploration starvation. |
| 4 | Reward flooring can weaken costs and encourage survival without progress. | Instrument unclamped totals and floor frequency; then test a targeted progress/time cost or revised clamp semantics if velocity tracking is insufficient. | Faster safe arrival and understood reward contributions, not merely higher return. |
| 5 | Teacher choice is domain-dependent. | Do not distil released CAT into clutter behavior where B already wins. Consider CAT-only teaching for retention, and inspect the older hand specialist before deciding whether it provides useful demonstrations. | Paired domain-specific gains and no loss of finger protection. |
| 6 | Evaluation confidence and disk headroom. | Resolve restore replay mismatch, retain fixed scenes/seeds and frozen checkpoint identities, and budget storage before further evaluation jobs. Coordinate cleanup of owned artifacts only. | Reproducible paired outcomes and enough space for checkpoint writes. |

The existing velocity reward is `exp(-4 * ||command_xy − velocity_xy||²)` (`cat_mjlab/task_math.py:227`). The inspected reward path has no explicit elapsed-time, distance-increment, or arrival-time penalty. The shaped sum is floored at zero by `(sum * dt).clamp(0,10000)` at `cat_mjlab/task.py:425`; the body-collision event penalty is added afterward at `:564`, providing the negative-reward path. Slow motion can still lose positive velocity-tracking reward, and discounting matters: “no explicit time penalty” does not mean speed has no incentive. It means dawdling is not directly charged a negative per-step cost in this design.

Exploration evidence is suggestive: clip fraction stays around 0.35–0.36, entropy loss barely changes, and quintile mean action std stays around 0.1465–0.1476 near the 0.15 cap. Binding on individual joints and causality remain hypotheses.

The session reports an unused `outputs/cat_hand_specialist_sapg_680` asset in the WholeBody repository, about 2.0 GB. Its location/content were not verified here; the obvious sibling path did not resolve. It should be located and evaluated before being treated as a viable teacher.

## 8. Methodological lessons

- **Adaptive aggregate success is not a stable ability measure.** Within-bucket scene probabilities move underneath it. The session reports hours lost interpreting a composition artifact as clutter regression. Fixed-scene paired evaluation is the reference measurement; comparisons between old and new B checkpoints are needed to establish temporal regression directly.
- **Small samples can reverse the conclusion.** Three scenes suggested +18.75 pp; broad original/published coverage gave −4.10 pp. Exact reproduction proves repeatability, not representativeness.
- **A permanently zero metric needs an exposure count.** Dead instrumentation concealed a dead objective through 279M transitions. Assert structural coverage at startup, then monitor nonzero runtime sample counts for the intended roles. A reported zero compliance with no samples must not be presented as measured failure or safety.
- **Timeout is not synonymous with failure to arrive.** Physical termination does not include goal crossing (`cat_mjlab/task.py:352`); outcome accounting is separate (`:427`). `unsuccessful_timeout` specifically means observed timeout with no prior clean goal, divided by all episodes. Both saved larger evaluations stop at first clean goal, so successful trajectories are censored for physical-duration/raw-timeout analysis.
- **Keep checkpoint and metric provenance explicit.** Evaluated `resume.pt`, warm-start `best.pt`, a 232-row audit, later appended rows, and stale status files are different objects. Do not silently combine them into one “final policy.”
- **Shared storage can stop training.** The session reports a full-disk training failure and another user's roughly 424 GB allocation; neither attribution nor crash cause was independently reconstructed. Current `df -h` confirms a 903 GB filesystem with 894 GB used and only 8.6 GB free, displayed as 100% after rounding. Each paired output freezes a session-reported roughly 743 MB checkpoint copy. Budget copies before launching work; do not remove another user's files.

## 9. What to watch next / success criteria

These are proposed acceptance checks, not achieved outcomes or pre-agreed numerical thresholds.

| Area | Watch next | Success criterion |
|---|---|---|
| Primary objective | Protected/transition sample counts, region costs, per-hand compliance and clearance | Demonstrate safe raised/tucked hands during successful fixed-scene passage traversal. Nonzero costs alone are insufficient. |
| Dangerous tail | Minimum and low-percentile hand heights; implicated trajectories and contacts | Explain and eliminate recurrent floor-level/through-floor hand excursions in the evaluated cases. |
| CAT retention | Same broad paired scene set; unsuccessful timeouts, first-goal times, collision transitions | Recover toward or beyond A's 0.2344/0.2467 group success while removing the observed timeout excess, without trading away finger safety. |
| Clutter | Reuse the fixed 60-scene set | Preserve the measured advantage over A; compare successor against frozen predecessor B as well. |
| Narrow passages | Fixed widths and per-width outcomes | Retain predecessor traversal ability and establish safe hand behavior; do not infer hardest-width competence from aggregate ~53% success. |
| Flat posture | Walking compliance, median and lower-tail hand height, falls | Continue compliance improvement and close the median gap to 0.870 m without reintroducing falls or dangerous tails. |
| Experiment integrity | Checkpoint hashes, runtime counts, replay check, disk space | No structurally dead objective, no ambiguous policy identity, and sufficient headroom to save the next checkpoint. |

The next useful result is a fixed-scene successor evaluation showing actual protected hand behavior alongside navigation outcomes. More training transitions or a higher blended selection score cannot substitute for that evidence.
