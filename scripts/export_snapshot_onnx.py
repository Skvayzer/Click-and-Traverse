#!/usr/bin/env python3
"""Export a training snapshot's actor to a sim-to-sim bundle directory (policy.onnx + links to robot.xml/assets of a
reference bundle). Same observation contract; for quick robot-stack checks of intermediate checkpoints.
  export_snapshot_onnx.py SNAPSHOT.pt OUT_DIR [--reference-bundle outputs/deploy/generalist_finetune_v1_u596]"""
import argparse, os, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from cat_mjlab.distill import load_expert
from export_deploy_bundle import Actor
p = argparse.ArgumentParser(); p.add_argument('snapshot'); p.add_argument('out')
p.add_argument('--reference-bundle', default=str(ROOT/'outputs/deploy/generalist_finetune_v1_u596'))
a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model = load_expert(a.snapshot, 'cpu'); actor = Actor(model).eval()
torch.onnx.export(actor, torch.zeros(1, model.config.actor_obs), str(out/'policy.onnx'), input_names=['obs'], output_names=['action', 'mean'],
                  dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}, 'mean': {0: 'batch'}}, opset_version=17, dynamo=False)
for name in ('robot.xml', 'assets'):
    if not (out/name).exists(): os.symlink(Path(a.reference_bundle).resolve()/name, out/name)
print(out)
