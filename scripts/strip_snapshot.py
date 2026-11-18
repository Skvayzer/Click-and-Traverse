#!/usr/bin/env python3
"""Strip a training snapshot (runs/<run>/run/snapshots/update_NNNN.pt, ~325 MB with the optimizer, simulator and task
state of every env) to what scripts/record_mjlab_rollout.load_policy and scripts/export_deploy_bundle.py read: schema,
contract, observation contract and the learner without its optimizer (~9 MB). Not resumable.

  python scripts/strip_snapshot.py runs/.../snapshots/update_0150.pt out/update_0150.pt
"""
import sys

import torch

d = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
keep = {k: d[k] for k in ('schema', 'contract', 'observation_contract') if k in d}
learner = dict(d['learner']); learner.pop('optimizer', None); keep['learner'] = learner
torch.save(keep, sys.argv[2])
print(sys.argv[2], learner.get('updates'), learner.get('env_steps'))
