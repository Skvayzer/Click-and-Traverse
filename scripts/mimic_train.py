#!/usr/bin/env python3
"""Train the full-body motion tracker: mjlab's training CLI with the task Cat-Mimic-G1 registered (cat_mjlab/mimic.py).

    python scripts/mimic_train.py Cat-Mimic-G1 --env.commands.motion.motion-file LIB/train.npz \\
        --env.scene.num-envs 4096 --agent.max-iterations 6000 --log-root RUNS/mimic_v1/logs
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cat_mjlab.mimic  # noqa: E402,F401  (registers Cat-Mimic-G1)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
    main()
