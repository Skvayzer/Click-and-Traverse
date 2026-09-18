# Training from the original released checkpoint, 18 September 2026

The `cat_train_only` launcher profile supersedes the earlier recovery experiment.
It starts from the released generalist-v1 actor **and critic**, not a whole-body
checkpoint. The source is `Axian12138/Click-and-Traverse`, revision
`46ce4b57ba0639168d51741b661ff62f7ce6f045`, checkpoint
`logs_v1/generalist_v1/checkpoints/005033164800`. The checked-in native manifest
verifies every downloaded source file by SHA-256. `run.json` records these hashes
and numerical initialization parity.

## PPO and initialization

| Setting | New run |
|---|---:|
| Learning rate | 0.0003 |
| PPO clipping | 0.2 |
| Entropy coefficient | 0.003 |
| Discount / GAE | 0.98 / 0.95 |
| Unroll | 32 steps |
| Minibatches / passes | 64 / 4 |
| Gradient norm limit | 1 |
| Observation normalization | Off, as released |
| Initial capacity target | 24,576 environments, batch size 384 |

The three requested optimizer settings match `configs/cat_generalist_released.json`.
The capacity target gives 786,432 transitions per update, 50% more than the
previous 16,384-environment / batch-256 setup. Actual GPU capacity must be checked
during real training; 20,480 / batch-320 is the smaller fallback if needed.

The native independent NormalTanh action distribution trains the means and
scales of all 29 body actions. There is no frozen leg-noise actor, custom arm
temporal correlation, custom upper-noise clipping, or suppressed upper entropy.
Existing actor/critic weights and original leg action heads are copied by named
feature/joint mapping. The 60 added actor-input weights initially equal zero.
The 17 added action means start at zero and their latent Gaussian scales at
0.05; these scales are trainable, not clamped. Adam starts fresh once.

**No reference-policy penalty, retention gate, rollback, automatic learning-rate
reduction, or automatic episode evaluation is installed.** Initial parameter
parity checks do not simulate episodes. Training runs continuously until stopped.
An explicit exact resume of this same new experiment is supported after an
interruption; initializing from a fine-tuned best archive is rejected.

## Task retained

The scene bank, reset distribution, route guidance, collision geometry and task
rewards remain those of the approved whole-body implementation: 2,362 layouts,
including the available original/procedural scenes, 48 ordinary clutter rooms,
and 24 hand table/shelf layouts. Actor/critic dimensions remain 222/310 and the
action space remains 29. Fixed Dex3 fingers, two hand spheres and two elbow
observation points, 35 internal collision volumes, collision termination and
penalties, and hand/stabilization rewards remain enabled.

Hand tasks retain the easy-to-hard curriculum and its 60% / 64-outcome promotion
threshold. The new profile counts its outcomes when the navigation attempt is
resolved, rather than delaying every success until the physical episode ends.
This changes outcome accounting and promotion timing, not goal rewards or
physical termination rules.

## Training success and storage

One W&B run covers every scene. Four prominent charts show overall,
original/procedural, ordinary-clutter, and hand-protection success. Each physical
episode contributes at most one navigation outcome: first clean goal arrival
succeeds; a prior/simultaneous fault, or a timeout before arrival, fails. Goal
arrival itself does not reset the simulator. Later collisions still receive
their normal penalty and termination, but cannot rewrite an already recorded
navigation outcome.

Rates use attempts resolved during each PPO update; accompanying resolved and
success counts expose the denominator. An empty population has no rate point.
These are training statistics, not independent evaluation estimates, and should
not be directly compared with the old deterministic evaluation curves or old
end-of-episode success definition. Counters persist in the exact resume state.
No observations are added for logging. Thousands of per-scene W&B scalars are
omitted in this profile.

Storage remains one selected best model, ranked only by the training rollout
reward proxy, plus one atomically overwritten complete `resume.msgpack`. Saving
a best model never reloads it into the learner. The old stopped run and its
selected checkpoint remain preserved separately.
