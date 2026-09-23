> **Historical record — superseded.** Native training now uses actor 222 / critic 310, no authored box features or rewards, and starts from scratch. Old expansion utilities are retired; no checkpoint trim is supported or requested. Commands and old test paths below describe the historical experiment, not the current workflow. See [current removal report](NATIVE_CLEARANCE_REMOVAL_20260922.md).

Implemented and verified on CPU; no training or pilot launch.

The opt-in `--lateral-corridor` replaces only forward-protected box shaping with
`-3 * mean((relu(abs(y_hand)-bound)/0.10)^2)`, multiplied by the existing approach/
zone phase and control dt. World-route coordinates include root drift. Actual
parallel cabinet faces define each scene's bound; nominal module width is not
used for the hand certificate. Bounds reproduce the audit's 184.793–187.393 mm.
Implementation: cat_mjlab/lateral_corridor.py:10, :31, :36;
cat_mjlab/acceptance.py:39; cat_mjlab/task.py:514.

Squared hinge is continuous with continuous first derivative at the boundary,
zero inside, and non-saturating outside. At 86.1 mm excess its cost is 0.741321
if both hands have that excess; each offending hand has cost gradient magnitude
8.61 / m because the cost averages two hands. At weight -3, dt .02 and phase 1,
reward gradient magnitude is 0.5166 / m per hand (1.0332 / m for moving both
inwards together). This is the geometric reward derivative, NOT a measured
policy/action gradient. The final reward-floor derivative further multiplies
it; the pilot retains the .2 soft floor and logs its slope. The median worst-hand
measurement does not imply both hands have the same excess.

Whole-arm protection belongs in BOTH shaping and the gate. All production arm
capsules, including shoulders, elbows and forearm/wrist links, are transformed
from current body poses. Their endpoints and radii define conservative lateral
swept clearance against the same cabinet faces. Minimum clearance must be
>=78.99 mm, with no acceptance tolerance. An additional weight -3 squared hinge
of the worst capsule's margin deficit / .10 supplies training pressure before
contact; a gate alone would reject failures without teaching their correction.
This is conservative across longitudinal cabinet gaps and certifies proxy
geometry at sampled poses, not continuous-time clearance or self-collision.
Implementation: cat_mjlab/lateral_corridor.py:41, :51; cat_mjlab/task.py:524.

S v2 requires the combined hand corridor, both pelvis/torso headings within
15 degrees of route forward, and whole-arm margin at every observed segment
that overlaps a required protected zone (including fades and exit steps).
Missing pose evidence fails closed. Existing ordered entries/exits, connecting
root corridors, crossing-average speed >=.2 m/s, deadlines, collision/fall
latches and all-assignment denominators remain active. Seeded-inside trials
remain separate diagnostics. Violations latch under `failure_other`.
Implementation: cat_mjlab/acceptance.py:152, :174, :294.

Training reward defaults OFF. S telemetry adopts the new criterion regardless
of the shaping flag. When enabled, protected checkpoint scoring also uses
combined pose plus instantaneous forward speed >=.2 m/s; protected posture
qualification uses combined pose. Other scene roles retain their shaping and
qualification. Legacy box cost/compliance stay logged under the existing
`hand_contrast_*region*` / `hand_contrast_hand_good` diagnostics. These are no
longer the protected selection objective with the flag enabled. Existing
heading and forward tracking rewards are unchanged and must have active scales
at configuration. See cat_mjlab/runner.py:314, cat_mjlab/task.py:567,
cat_mjlab/upper_control.py:19.

Saved-state rescore (outputs/lateral_corridor_combined_cpu/audit.json):

| Saved trace set | Corridor + heading + arm good core endpoints | Also >=.2 m/s over the 20 ms step | Mean full-core compliant distance, pose / speed-filtered | Full compliant first-core crossings with crossing-speed budget |
|---|---:|---:|---:|---:|
| Absolute reference-assisted, 12 scenes | 183/967 (18.92%) | 20/967 (2.07%) | 1.405% / 0.808% | 0/12 |
| Body-relative reference, 12 scenes | 135/280 (48.21%) | 0/280 (0%) | 0.308% / 0% | 0/12 |
| Older unmodified policy, 4 scenes | 32/800 (4.00%) | 1/800 (0.125%) | 1.166% / 0.0743% | 0/4 |

The speed-filtered endpoint count is a stricter dense diagnostic, NOT S's
average crossing-speed rule. `combined_gate` in the JSON uses pose compliance
and the remaining-distance crossing budget; `combined` is the dense diagnostic.
The saved trajectories begin inside the first core and do not record complete
upstream multi-zone trials. Therefore no complete S pass can be certified from
these recordings; these are first-core diagnostic rescoring numbers, not an
invented upstream evaluation. Missing initial 20 mm remains in the 1.35 m
distance denominator. Endpoint FK is used consistently for hands, headings,
and arms; old lagged hand sampling remains only in the baseline comparison.
The original 695/967 corridor-only count becomes 691/967 with endpoint FK.
Implementation/reproduction: scripts/verify_lateral_corridor_cpu.py:96.

All 12 independent nominal-arm 90-degree FK witnesses still pass the hand
corridor (hands ~5.374 mm off centre) and clear the arms by >=284.228 mm, but
FAIL the combined pose requirement. The gate test also gives such a witness
successful forward translation and clean goal; it still FAILS. Refuse-entry,
crawl, lateral bypass and fall-first all FAIL with protected geometry enabled;
a legitimate fast, aligned, arm-safe crossing PASSES. A 70 mm capsule clearance
fails despite compliant hands. Tests: tests/test_lateral_corridor.py:12, :28,
:48, :70, :143; existing S tests additionally preserve collision latches,
denominators, rotated routes and RNG/dynamics isolation.

Production `_rewards` was exercised using CPU MuJoCo FK and synthetic constant
fields (no policy or simulation rollout): opt-in replaces protected box reward,
retains box diagnostics and heading cost, and leaves every existing reward
component bit-exact on masked-out nonprotected roles. Existing flat, heading,
objective, acceptance and logging regressions were run as well. This verifies
code isolation, not future learned retention after shared-policy training.
The broader legacy task-reference suite could not run fully: missing jaxlie,
ml_collections and mujoco_playground caused five failures and two setup errors.
No packages were installed. Final targeted run: **93 passed in 16.39 s**.

Reproduce without CUDA or Python bytecode:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 15 .venv-mjlab/bin/python scripts/verify_lateral_corridor_cpu.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv-mjlab/bin/python -m pytest -q -p no:cacheprovider tests/test_lateral_corridor.py tests/test_mjlab_acceptance.py tests/test_upper_control.py tests/test_mjlab_passage_heading.py tests/test_protected_objective_fixes.py tests/test_flat_balance.py tests/test_mjlab_hand_objective.py tests/test_mjlab_hand_approach.py tests/test_mjlab_logging_diagnostics.py
```

Prepared launch command (NOT executed):

```sh
bash configs/pilots/lateral_corridor_50.sh
```

The script uses the requested live-run best.pt as a fresh weights-only warm
start, its v2 bank/reset/collision manifests, 30,720 environments, 50 updates,
gravity compensation, protected SDF margin, heading weight -5, soft floor .2,
and online CAT-wholebody / skvayzer. It preserves live flat bonus 10 and sigma
ceiling .15. No checkpoint was loaded or copied during this work. Do not run
alongside the current GPU job; GPU-memory fit and future checkpoint disk usage
have not been tested. The implementation rescore artifact is about 65 KiB;
free disk remained about 3.6 GiB. No tmux commands, signals, GPU tools, or writes
to the live output directory were issued. Read-only status reported running,
update 60, 58,982,400 env steps. Process visibility is namespaced; this is status
file evidence, not an independent GPU/process health check.

Expectation (inference, not validated): I would not expect reliable full
crossings in a 50-update pilot. The 8.6 cm median correction is substantially
smaller than the old box demand, but still not demonstrated dynamically by this
controller. It may be beyond the present action authority/control/exploration
combination; these measurements cannot establish that it is physically
unreachable. The combined rescore exposes severe heading/progress attrition,
so a larger hands-only score would not be persuasive.

In the first 20 updates, watch combined pose AND forward-moving compliance,
protected newly covered distance/zone entry and S failures, arm-margin tails,
heading compliance, the unchanged box diagnostic, falls, reward-floor slope,
and flat/CAT retention. Expect S to lag dense improvements. If the corridor
cost falls but motion, heading or arm safety worsens, that is not progress.
If combined moving compliance and covered distance remain flat by update 20,
do not assume another 30 updates will solve it; inspect controller authority
and the competing reward/PD terms before extending the run.
