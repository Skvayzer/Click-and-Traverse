# Corrected CAT training setup

[WHOLE_BODY.md](../WHOLE_BODY.md) contains the current setup and commands. [CAT_SETUP_CORRECTION.md](CAT_SETUP_CORRECTION.md) records the correction, checks and limitations.

The former sequential curriculum was retired after the September 15 audit. It changed CAT's task, reset Adam between scenes, rolled live weights back to selected snapshots and opened separate W&B runs. The corrected `train_cat_wholebody.py` uses one continuous mixed-scene learner, the released configuration, one experiment ID and exact recovery.

After the user authorized restart, the 8,192-environment / 512-trajectory-minibatch configuration passed native GPU compiler capacity validation, and continuous training restarted from the released CAT actor and critic. [Active run, observations and stop command](CORRECTED_RUN_20260915.md). Tests and startup checks establish code behavior and initial operation, not learned traversal quality.
