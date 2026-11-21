#!/bin/bash
# Stationary hand protection (configs/pilots/stationary_hands_20261113.sh) on one RTX 5000 Ada 32 GB, ws-ia.
#   sbatch --job-name=hands_S1 --export=ALL,VARIANT=S1 hpc/stationary_hands.sbatch
#   smoke: sbatch --job-name=hands_smoke --time=00:40:00 --export=ALL,VARIANT=S1,SMOKE=1 hpc/stationary_hands.sbatch
# VARIANT  S1 stationary: feet planted (<= 5 cm, <= 5 steps off the floor) after settling, pelvis <= 0.2 m, else failure
#          S2 energy: cost 0.02 per watt of joint power in standing episodes, old non-terminal 0.3 m spot rule
#          S3 both;  S0 control: old rules only
#          S1b: S1 plus a dense cost on foot travel from the settled spot (weight -10, (travel/5 cm)^2)
#          S2far: S2 plus a far hand retreat in standing episodes (clearance target 0.20 m, anticipation 0.40 m,
#                 threatening objects need >= 0.15 m retreat); pass STUDENT=<frozen S2 checkpoint> (warm start)
#          S2far2: stronger still (target 0.30 m, anticipation 0.50 m, >= 0.25 m retreat); STUDENT=<frozen S2far checkpoint>
#          T1: S2far + table edges: face the route while walking (-3), leg + waist power -0.005/W (arms free), table group
#              mass 0.25, standing share 0.3, 30% raised-hand starts in forward_protected scenes, no anchor in table scenes
#          T1b: T1 + table-edge scenes keep the CAT weights of stall / goal_progress / goal_bonus (TABLE_PROGRESS=1)
#          T1c: T1b + facing -6 + hands-up bonus 2 (a hand near a table above the tabletop band)
#          T2: T1c on bank tablegap_v1 = sidegap_v2 + 64 table pairs with intermediate gaps 0.46-0.70 m (width curriculum)
#          T3: T2 + arm exploration (arm sigma 0.25 initial and ceiling; legs/waist stay 0.10)
#          T4: T2 + between a table pair: pelvis cross-track cost -2 ((d/5 cm)^2) and yaw cost -2 ((sin err/sin 10 deg)^2),
#              hands-up band 0.80 -> 0.92 m (narrow-pair failures: 4 cm off the route line, 9-15 deg yaw, palms sagging)
#          E1: energy only -- from S2far u200 (before any table stage) on the width curriculum with the anti-refusal terms,
#              an energy price on all 29 joints in table scenes (-0.02/W, as standing) and NO facing cost, hands-up bonus
#              or leg-power price: does pricing energy alone make the robot pass tables facing forward with raised hands?
#          E2: E1 + an energy price everywhere else (-0.01/W: traversal, walking; standing and tables keep -0.02):
#              does an energy objective in every environment give natural, efficient traversal without losing skills?
#          E2c: E2 continued from a checkpoint with a fresh optimizer (E2 could not resume: the old resume-contract path
#              bug), anchored to S2far u200 as E2 was -- the control for the stalling fix
#          E3: E2c + the stalling fix in room passages: anti-refusal terms (stall x1.5), body touches cost 0.5 per step
#              instead of ending the episode (hand contact still ends it), room-scaled hand margins between cabinets
#          E4: E2c + the stall gate: in room passages no positive reward is paid while the robot stalls under a go command
#              (E3's soft contacts made touching cost more per step than ending the episode and raised refusals to 99%)
#          J1: E4 with energy priced per metre instead of per second (--route-energy: P - q*v under a go command,
#              q = 234 J/m = E2's clutter-room energy per metre, cruise 0.6 m/s, tables x2 as in E2/E4)
#          V1: slow traversal from E2 u300 -- operator speed 0.1-0.6 m/s (fields give direction), slower near obstacles,
#              no stop latch, cut-out 0.05 m/s, rest starts in side gaps, clearance per metre, relative tracking/progress
#          V2: V1 + overspeed cost (-60/s above 1.2x command, outside the clip), goal bonus 500 (was 2000), relative progress /
#              stall in room passages, 35% near-nominal anchored episodes, safer rest starts (review of 2026-11-17)
#          V3a: V2 fixed (2026-11-18): guidance terms as progress along the field at the operator speed (the cosine of a slow,
#              swaying velocity read as misaligned near obstacles -- a reason to rush), tracking on the 0.4 s mean velocity,
#              goal bonus back to 2000, no overspeed cost. V3b: V3a + overspeed -10/s on the 0.4 s mean speed above 1.25x
#              the command + 0.05 m/s, inside the clip (V2's -60/s on the instantaneous speed, outside the clip, charged the
#              gait's sway: walking slowly paid less than standing, touching to end the episode paid most -> collapse)
#          V4: V3b + clearance / SDF costs at max(commanded, actual 0.4 s mean speed) (per metre when faster than told) +
#              side gaps / room passages free of the anchor: at u50 V3a / V3b still crossed the robot-side gap scenes at
#              ~0.6 m/s told 0.15 (E2 ~0.7) -- leaving a gap sooner cut its per-second costs, and the anchor is E2's lunge
#          V5: V4 + goal bonus x (1 - share of the last ~2 s above the speed limit / 0.25): with 0.98 discounting the
#              bonus paid ~20x more for arriving 1 s out than 4 s out, and every CAT / side-gap goal plane is 0.4-1.4 m
#              past its obstacle, so the policy sprinted through the gap for it
#          V6: V5 + overspeed -30/s (still inside the clip) + the guidance terms' goal-line constant (4 per point past x = 1.5
#              m, ~+16/s) gated by speed compliance like the goal bonus + all CAT (goal-line) scenes anchor-free; warm-started
#              from a V5 snapshot (STUDENT), anchor E2 u300 (rooms only)
#          V7: V6 + ground-projected goal (column-seed guidance cache, GOAL_SEED_CACHE) + success at the goal on the ground
#              (0.3 m) with the executor's arrival stop (0.25 / 0.45 m) and a 2 s hold, from a V6 snapshot; anchor E2 u300
#          TRACK1: V7 + tracker v1 (cat_mjlab/tracking.py): a time-indexed reference block (guidance path at the operator
#              speed with a clock that waits for a lagging robot / procedural paths with heading, crouch and arm targets /
#              none) in actor and critic, tracking terms, the command from the reference; warm-started from a V7 snapshot
#              with zero-weight reference inputs (actor and critic), anchor E2 u300 on reference-free worlds only,
#              50 critic-only updates; reactive standing episodes get no reference
#          COMBO1: planner with a field reflex -- TRACK1 plus planner v1 in the loop (the tracker's 'plan' source:
#              2 s whole-body chunks from the measured state toward a goal 2 s ahead on the guidance path, replanned
#              every 0.3-1.0 s, best of 4 by body clearance; cat_mjlab/planner/online.py), half the reactive standing
#              worlds on a stand-still plan, and the reflex rule (an arm whose hand is within 15 cm of an obstacle is not
#              tracked; the pelvis term tolerates up to 20 cm near obstacles). Sources plan 0.5, field 0.15, random 0.1,
#              none 0.25. Warm start from a V7r snapshot (STUDENT) with zero-weight reference inputs, 50 critic-only
#              updates, anchor E2 u300 on reference-free worlds. PLANNER=<planner v1 checkpoint>
#          COMBO1b: COMBO1 with route-following references in route worlds (rooms, cabinet passages; COMBO1's
#              references there followed the stored goal field, which stalls in rooms), warm-started from a COMBO1
#              snapshot (STUDENT), 10 critic-only updates
#          COMBO2: COMBO1b + commit (plan clocks do not wait for a lagging robot; the reflex deadband forgives only
#              sideways pelvis error): in COMBO1 a stalled robot kept 91% of track_pos because the reference restarted
#              at the robot and waited. Warm start from a COMBO1b snapshot (STUDENT), 10 critic-only updates
#          COMBO4: COMBO2 + plan continuity (--tracker-continuity 0.5: replans continue the executing plan from its own
#              state, so a stalled robot falls behind a reference that keeps moving; the robot re-anchors the plan only
#              beyond 0.5 m) + route go (--tracker-route-go: the command says go at the route speed whenever the route
#              does, so a hesitant plan can neither stop the gait nor switch off the stall cost). COMBO1b/2 refused in
#              27-30% of the 384 narrow scenes because replans restarted the slow-starting plans at the robot (a standing
#              robot kept 87-91% of track_pos) and the plan command said stop within 5 cm. STUDENT = a COMBO2 snapshot
#          COMBO5: COMBO4 + the beam/bar height curriculum (bank tablegap_bb_v1 = tablegap_v1 + 600 procedural beam and bar
#              scenes in 8 levels, scripts/generate_beam_bars.py; --height-curriculum 0.5 300: stage s samples levels <= s
#              and advances at 50% clean of 300 level-s episodes). Contact stays terminal (no obstacle forces in training).
#              No policy in the line passed CAT beams or bars cleanly (0-25%). STUDENT = a COMBO4 snapshot
#          COMBO3: COMBO2 + protection precision (--reactive-far-threat-only: the far retreat target only for threatening
#              objects; after the far stages the robot flinched at ~90% of pass-by objects). STUDENT = a COMBO2 snapshot
#          E1np / E1nc: RQ2 ablations of E1 -- without the anti-refusal terms (np) / without the width curriculum, on the
#              original 0.44 m and open table pairs only (nc)
#          T5: T2 + non-table scenes anchored to S2far u200 (the last policy before the table stages; ANCHOR) instead of the
#              start checkpoint + anti-refusal terms in the room passages (PASSAGE_PROGRESS): after T1-T2 the robot stopped
#              in front of narrow cabinet passages (< 0.60 m: refusals 10-33% -> 88-98%) instead of turning sideways
# CLONE=<git worktree pinned to a commit> runs that code instead of the live clone (a requeued job resumes only with the
# code it started with). On the gpu partition: sbatch -p gpu --time=08:00:00 --mem=64000M ...
# Data: dep-1's tree is mirrored under /l/users/$USER/cat_root and reached through CAT_PATH_MAP (cat_mjlab/pathmap.py);
# data/furniture/<bank> and data/motion_library in the clone are symlinks into it. cat_root, runs/ and checkpoints/ are
# pinned to healthy OSTs (13/14/10): OSTs 7 and 8 hung on writes on 2026-11-13. W&B offline (wandb sync later).
# Resubmitting the same VARIANT resumes from its run directory.
#SBATCH --job-name=hands
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=12
#SBATCH --mem=54000M
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
L=/l/users/$USER
CLONE=${CLONE:-$HOME/Click-and-Traverse-Mjlab}
# Run from a private copy of the code (node-local disk): later pulls in the clone cannot change a running job.
# data/furniture/*, data/motion_library and outputs/saved_checkpoints are symlinks and stay symlinks.
export REPO=$TMPDIR/code
rsync -a --exclude .git $CLONE/ $REPO/
VARIANT=${VARIANT:?set VARIANT=S0, S1, S2 or S3}
case $VARIANT in
  S0) ;;
  S1) export STATIONARY="0.05 5 30 0.2" ;;
  S2) export POWER=-0.02 ;;
  S3) export STATIONARY="0.05 5 30 0.2" POWER=-0.02 ;;
  S1b) export STATIONARY="0.05 5 30 0.2" FEET_HOLD=-10 ;;
  S2far) export POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15; : "${STUDENT:?S2far needs STUDENT=<S2 checkpoint>}" ;;
  S2far2) export POWER=-0.02 HAND_FAR="0.30 0.50" RETREAT_SUCCESS=0.25; : "${STUDENT:?S2far2 needs STUDENT=<S2far checkpoint>}" ;;
  T1) export POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_FACING=-3 TABLE_LEG_POWER=-0.005 \
        MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables.json; : "${STUDENT:?T1 needs STUDENT=<checkpoint>}" ;;
  T1b) export POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_FACING=-3 TABLE_LEG_POWER=-0.005 TABLE_PROGRESS=1 \
        MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables.json; : "${STUDENT:?T1b needs STUDENT=<T1 checkpoint>}" ;;
  T1c) export POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_FACING=-6 TABLE_LEG_POWER=-0.005 TABLE_PROGRESS=1 \
        TABLE_HANDS_UP=2 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables.json; : "${STUDENT:?T1c needs STUDENT=<T1b checkpoint>}" ;;
  T2) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_FACING=-6 \
        TABLE_LEG_POWER=-0.005 TABLE_PROGRESS=1 TABLE_HANDS_UP=2 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" \
        REACTIVE_SHARE=0.3 RAISED=0.3 ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json
      : "${STUDENT:?T2 needs STUDENT=<T1c checkpoint>}" ;;
  T3) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_FACING=-6 \
        TABLE_LEG_POWER=-0.005 TABLE_PROGRESS=1 TABLE_HANDS_UP=2 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" \
        REACTIVE_SHARE=0.3 RAISED=0.3 ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json ARM_STD=0.25
      : "${STUDENT:?T3 needs STUDENT=<T1c checkpoint>}" ;;
  T4) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_FACING=-6 \
        TABLE_LEG_POWER=-0.005 TABLE_PROGRESS=1 TABLE_HANDS_UP=2 TABLE_HANDS_UP_BAND="0.80 0.92" TABLE_CENTER=-2 \
        TABLE_ZONE_FACING=-2 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json
      : "${STUDENT:?T4 needs STUDENT=<T2 checkpoint>}" ;;
  T5) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_FACING=-6 \
        TABLE_LEG_POWER=-0.005 TABLE_PROGRESS=1 TABLE_HANDS_UP=2 PASSAGE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" \
        REACTIVE_SHARE=0.3 RAISED=0.3 ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 \
        ANCHOR=${ANCHOR:-$L/checkpoints/stationary_hands_S2far_u200.pt}
      : "${STUDENT:?T5 needs STUDENT=<T2 checkpoint, e.g. u410: the best measured table passages>}" ;;
  E1) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50
      : "${STUDENT:?E1 needs STUDENT=<S2far u200, the pre-table policy>}" ;;
  E2) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50
      : "${STUDENT:?E2 needs STUDENT=<S2far u200, the pre-table policy>}" ;;
  E1np) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50
      : "${STUDENT:?E1np needs STUDENT=<S2far u200>}" ;;
  E1nc) export BANK=data/furniture/sidegap_v2 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables.json SNAPSHOT_EVERY=50
      : "${STUDENT:?E1nc needs STUDENT=<S2far u200>}" ;;
  E2c) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 \
        ANCHOR=${ANCHOR:-$L/checkpoints/stationary_hands_S2far_u200.pt}
      : "${STUDENT:?E2c needs STUDENT=<E2 checkpoint to continue from>}" ;;
  E3) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 \
        PASSAGE_PROGRESS=1 PASSAGE_STALL=1.5 PASSAGE_SOFT=0.5 TIGHT_PASSAGES=1 \
        ANCHOR=${ANCHOR:-$L/checkpoints/stationary_hands_S2far_u200.pt}
      : "${STUDENT:?E3 needs STUDENT=<E2 checkpoint, the same as E2c>}" ;;
  E4) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 PASSAGE_GATE=1 \
        ANCHOR=${ANCHOR:-$L/checkpoints/stationary_hands_S2far_u200.pt}
      : "${STUDENT:?E4 needs STUDENT=<E2 checkpoint, the same as E2c>}" ;;
  J1) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 \
        ROUTE_ENERGY="-0.01 234 0.6 2" TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 \
        RAISED=0.3 ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 PASSAGE_GATE=1 \
        ANCHOR=${ANCHOR:-$L/checkpoints/stationary_hands_S2far_u200.pt}
      : "${STUDENT:?J1 needs STUDENT=<E2 checkpoint, the same as E4 and E2c>}" ;;
  V1) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 OPERATOR_SPEED="0.1 0.6" REST_STARTS=0.25 \
        ANCHOR=${ANCHOR:-$STUDENT}
      : "${STUDENT:?V1 needs STUDENT=<E2 u300>}" ;;
  V2) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 OPERATOR_SPEED="0.1 0.6" REST_STARTS=0.25 \
        OPERATOR_V2=1 OVERSPEED=-60 GOAL_BONUS=500 PASSAGE_PROGRESS=1 ANCHOR=${ANCHOR:-$STUDENT}
      : "${STUDENT:?V2 needs STUDENT=<E2 u300>}" ;;
  V3a) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 OPERATOR_SPEED="0.1 0.6" REST_STARTS=0.25 \
        OPERATOR_V3=1 PASSAGE_PROGRESS=1 ANCHOR=${ANCHOR:-$STUDENT}
      : "${STUDENT:?V3a needs STUDENT=<E2 u300>}" ;;
  V3b) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 OPERATOR_SPEED="0.1 0.6" REST_STARTS=0.25 \
        OPERATOR_V3=1 OVERSPEED=-10 PASSAGE_PROGRESS=1 ANCHOR=${ANCHOR:-$STUDENT}
      : "${STUDENT:?V3b needs STUDENT=<E2 u300>}" ;;
  V4) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 OPERATOR_SPEED="0.1 0.6" REST_STARTS=0.25 \
        OPERATOR_V3=1 OPERATOR_V4=1 OVERSPEED=-10 PASSAGE_PROGRESS=1 ANCHOR=${ANCHOR:-$STUDENT}
      : "${STUDENT:?V4 needs STUDENT=<E2 u300>}" ;;
  V5) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 OPERATOR_SPEED="0.1 0.6" REST_STARTS=0.25 \
        OPERATOR_V3=1 OPERATOR_V4=1 OPERATOR_V5=1 OVERSPEED=-10 PASSAGE_PROGRESS=1 ANCHOR=${ANCHOR:-$STUDENT}
      : "${STUDENT:?V5 needs STUDENT=<E2 u300>}" ;;
  V6) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 OPERATOR_SPEED="0.1 0.6" REST_STARTS=0.25 \
        OPERATOR_V3=1 OPERATOR_V4=1 OPERATOR_V5=1 OPERATOR_V6=1 OVERSPEED=-30 PASSAGE_PROGRESS=1 \
        ANCHOR=${ANCHOR:-$L/checkpoints/stationary_hands_E2_u300.pt}
      : "${STUDENT:?V6 needs STUDENT=<a V5 snapshot>}" ;;
  V7) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 OPERATOR_SPEED="0.1 0.6" REST_STARTS=0.25 \
        OPERATOR_V3=1 OPERATOR_V4=1 OPERATOR_V5=1 OPERATOR_V6=1 OVERSPEED=-30 PASSAGE_PROGRESS=1 \
        GOAL_SEED_CACHE=${GOAL_SEED_CACHE:-data/furniture/tablegap_v1_goal_column} GOAL_GROUND=1 \
        ANCHOR=${ANCHOR:-$L/checkpoints/stationary_hands_E2_u300.pt}
      : "${STUDENT:?V7 needs STUDENT=<a V6 snapshot>}" ;;
  TRACK1) export BANK=data/furniture/tablegap_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 OPERATOR_SPEED="0.1 0.6" REST_STARTS=0.25 \
        OPERATOR_V3=1 OPERATOR_V4=1 OPERATOR_V5=1 OPERATOR_V6=1 OVERSPEED=-30 PASSAGE_PROGRESS=1 \
        GOAL_SEED_CACHE=${GOAL_SEED_CACHE:-data/furniture/tablegap_v1_goal_column} GOAL_GROUND=1 \
        TRACKER=1 TRACKER_MIX=${TRACKER_MIX:-field=0.5,random=0.3,none=0.2} WARMUP=${WARMUP:-50} \
        ANCHOR=${ANCHOR:-$L/checkpoints/stationary_hands_E2_u300.pt}
      : "${STUDENT:?TRACK1 needs STUDENT=<a V7 snapshot>}" ;;
  COMBO1|COMBO1b|COMBO2|COMBO3|COMBO4|COMBO5) export BANK=data/furniture/tablegap$([ "$VARIANT" = COMBO5 ] && echo _bb)_v1 POWER=-0.02 HAND_FAR="0.20 0.40" RETREAT_SUCCESS=0.15 TABLE_POWER=-0.02 \
        GLOBAL_POWER=-0.01 TABLE_PROGRESS=1 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.25 0.03" REACTIVE_SHARE=0.3 RAISED=0.3 \
        ROUTE_FILE=hpc/route_anchor_no_reactive_no_tables_v2.json SNAPSHOT_EVERY=50 OPERATOR_SPEED="0.1 0.6" REST_STARTS=0.25 \
        OPERATOR_V3=1 OPERATOR_V4=1 OPERATOR_V5=1 OPERATOR_V6=1 OVERSPEED=-30 PASSAGE_PROGRESS=1 \
        GOAL_SEED_CACHE=${GOAL_SEED_CACHE:-data/furniture/tablegap$([ "$VARIANT" = COMBO5 ] && echo _bb)_v1_goal_column} GOAL_GROUND=1 \
        TRACKER=1 TRACKER_MIX=${TRACKER_MIX:-plan=0.5,field=0.15,random=0.1,none=0.25} \
        WARMUP=${WARMUP:-$([ "$VARIANT" = COMBO1 ] && echo 50 || ([ "$VARIANT" = COMBO4 ] && echo 20 || echo 10))} \
        TRACKER_COMMIT=$([ "$VARIANT" = COMBO2 ] || [ "$VARIANT" = COMBO3 ] || [ "$VARIANT" = COMBO4 ] || [ "$VARIANT" = COMBO5 ] && echo 1) \
        TRACKER_CONTINUITY=$([ "$VARIANT" = COMBO4 ] || [ "$VARIANT" = COMBO5 ] && echo 0.5) \
        TRACKER_ROUTE_GO=$([ "$VARIANT" = COMBO4 ] || [ "$VARIANT" = COMBO5 ] && echo 1) \
        HEIGHT_CURRICULUM=$([ "$VARIANT" = COMBO5 ] && echo "0.5 300") \
        FAR_THREAT_ONLY=$([ "$VARIANT" = COMBO3 ] && echo 1) \
        TRACKER_PLANNER=${PLANNER:-$L/runs/wbp/planner_v1/train/planner.pt} TRACKER_HOLD_PLAN=${HOLD_PLAN:-0.5} TRACKER_REFLEX=1 \
        ANCHOR=${ANCHOR:-$L/checkpoints/stationary_hands_E2_u300.pt}
      : "${STUDENT:?COMBO1 needs STUDENT=<a V7r snapshot> (COMBO1b: <a COMBO1 snapshot>)}" ;;
  *) echo "unknown VARIANT $VARIANT"; exit 2 ;;
esac
if [[ -n "${SMOKE:-}" ]]; then
  RUN=$L/runs/smoke_hands_${VARIANT}_${SLURM_JOB_ID}; export EXTRA="--max-updates ${SMOKE_UPDATES:-3}" WANDB_MODE=disabled
else
  RUN=$L/runs/stationary_hands_20261113_$VARIANT; export WANDB_MODE=${WANDB_MODE:-offline}
fi
mkdir -p $RUN
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
export STUDENT=${STUDENT:-$L/checkpoints/generalist_robust_delay_v1_u150.pt} RUN_DIR=$RUN/run WANDB_DIR=$RUN
export NUM_ENVS=${NUM_ENVS:-12288} BATCH_SIZE=1024    # 16384 peaked at 31.2 of 32.8 GB (bank fields ~20 GB fixed)
export NUM_MINIBATCHES=$(( NUM_ENVS / 1024 ))          # 32768-transition minibatches, as 40960 envs / 40 on dep-1
if [[ -f $RUN/run/status.json ]]; then export EXTRA="--resume ${EXTRA:-}" FRESH=""; fi
{ echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) variant $VARIANT on $(hostname) at $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "commit $(git -C $CLONE rev-parse HEAD) $(git -C $CLONE status --porcelain --untracked-files=no | wc -l) modified tracked files (copied to $REPO)"
  echo "STATIONARY='${STATIONARY:-}' POWER='${POWER:-}' FEET_HOLD='${FEET_HOLD:-}' BANK='${BANK:-}' ARM_STD='${ARM_STD:-}' HAND_FAR='${HAND_FAR:-}' RETREAT_SUCCESS='${RETREAT_SUCCESS:-}' TABLE_FACING='${TABLE_FACING:-}' TABLE_LEG_POWER='${TABLE_LEG_POWER:-}' TABLE_PROGRESS='${TABLE_PROGRESS:-}' TABLE_HANDS_UP='${TABLE_HANDS_UP:-}' TABLE_HANDS_UP_BAND='${TABLE_HANDS_UP_BAND:-}' TABLE_CENTER='${TABLE_CENTER:-}' TABLE_ZONE_FACING='${TABLE_ZONE_FACING:-}' PASSAGE_PROGRESS='${PASSAGE_PROGRESS:-}' TABLE_POWER='${TABLE_POWER:-}' GLOBAL_POWER='${GLOBAL_POWER:-}' PASSAGE_STALL='${PASSAGE_STALL:-}' PASSAGE_SOFT='${PASSAGE_SOFT:-}' PASSAGE_GATE='${PASSAGE_GATE:-}' ROUTE_ENERGY='${ROUTE_ENERGY:-}' OPERATOR_SPEED='${OPERATOR_SPEED:-}' REST_STARTS='${REST_STARTS:-}' OPERATOR_V2='${OPERATOR_V2:-}' OPERATOR_V3='${OPERATOR_V3:-}' OPERATOR_V4='${OPERATOR_V4:-}' OPERATOR_V5='${OPERATOR_V5:-}' OPERATOR_V6='${OPERATOR_V6:-}' GOAL_SEED_CACHE='${GOAL_SEED_CACHE:-}' GOAL_GROUND='${GOAL_GROUND:-}' OVERSPEED='${OVERSPEED:-}' GOAL_BONUS='${GOAL_BONUS:-}' TIGHT_PASSAGES='${TIGHT_PASSAGES:-}' ANCHOR='${ANCHOR:-}' SNAPSHOT_EVERY='${SNAPSHOT_EVERY:-}' MASSES='${MASSES:-}' REACTIVE_SHARE='${REACTIVE_SHARE:-}' RAISED='${RAISED:-}' ROUTE_FILE='${ROUTE_FILE:-}' NUM_ENVS=$NUM_ENVS NUM_MINIBATCHES=$NUM_MINIBATCHES EXTRA='${EXTRA:-}'"
  echo "STUDENT=$STUDENT CAT_PATH_MAP=$CAT_PATH_MAP"
  echo "config: configs/pilots/stationary_hands_20261113.sh"; } >> $RUN/run_info.txt
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l ${GPU_LOG_SECONDS:-60} >> $RUN/gpu.csv &
( while sleep 60; do echo "$(date +%T) $(ps -u $USER -o rss= | awk '{s+=$1} END {printf "%.0f", s/1024}') MB"; done ) >> $RUN/host_mem.txt &
status=0
bash $REPO/configs/pilots/stationary_hands_20261113.sh || status=$?
echo "== exit $status at $(date -Is); peak GPU MiB $(cut -d, -f2 $RUN/gpu.csv | tr -dc '0-9\n' | sort -n | tail -1); peak host $(awk '{print $2}' $RUN/host_mem.txt | sort -n | tail -1) MB" | tee -a $RUN/run_info.txt
exit $status
