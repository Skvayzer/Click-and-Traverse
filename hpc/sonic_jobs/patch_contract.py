"""Remove the latent adaptor's soft-contact phase fields from a saved run's contracts (run.json, resume.pt, best.pt).
They record the wrapper's initial soft-contact state (contact_penalty 2.0, contact_soft_hands True), which the schedule
ended at update 200; a resume builds the environment without them. Nothing else is changed; backups kept. Idempotent."""
import json, shutil, sys
from pathlib import Path
import torch
run = Path(sys.argv[1])
KEYS = {"contact_penalty": 2.0, "contact_soft_hands": True}

def strip(env):
    for k, v in KEYS.items():
        if k in env:
            assert env[k] == v, (k, env[k])
            del env[k]

for name in ("run.json", "resume.pt", "best.pt"):
    if not (run / (name + ".pre_resume_patch")).exists():
        shutil.copy2(run / name, run / (name + ".pre_resume_patch"))
r = json.loads((run / "run.json").read_text())
strip(r["contract"]["environment_config"])
(run / "run.json").write_text(json.dumps(r, indent=2, sort_keys=True))
for name in ("resume.pt", "best.pt"):
    snap = torch.load(run / name, map_location="cpu", weights_only=False)
    strip(snap["contract"]["environment_config"])
    assert snap["contract"] == r["contract"], f"{name} contract differs beyond the patched fields"
    torch.save(snap, run / name)
print("patched", run)
