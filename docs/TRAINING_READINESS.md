# Corrected CAT training setup

[WHOLE_BODY.md](../WHOLE_BODY.md) contains the current setup and commands. [CAT_SETUP_CORRECTION.md](CAT_SETUP_CORRECTION.md) records the correction, checks and limitations.

The former sequential curriculum was retired after the September 15 audit. It changed CAT's task, reset Adam between scenes, rolled live weights back to selected snapshots and opened separate W&B runs. The corrected `train_cat_wholebody.py` uses one continuous mixed-scene learner, the released configuration, one experiment ID and exact recovery.

Training was stopped for the correction and has not been restarted by this implementation work. The GPU resource profile remains subject to capacity validation. Tests establish code behavior and compatibility, not learned traversal quality.
