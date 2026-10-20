#!/usr/bin/env python3
"""Function-preserving widening of the actor MLP (Net2WiderNet, Chen et al. 2016) of a native checkpoint.

Each hidden layer grows to the requested width: new units copy a random existing unit's incoming weights and bias,
and every copied unit's OUTGOING weights are divided by its number of copies, so the network computes the same
function (exact for any elementwise activation, here SiLU). A small noise on the copies' incoming weights
(--noise, relative to the layer's weight std) breaks the symmetry so the copies can learn different features; it
changes the outputs only slightly (reported). Only the actor changes; critic, style critic and environment contract
are kept. Use the result with --checkpoint-native --fresh-optimizer.

  widen_checkpoint.py --checkpoint IN.pt --hidden 1024 512 256 128 --out OUT.pt [--check reference_rollouts.npz]
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def actor_forward(weights, x):
    n = len([k for k in weights if k.startswith('actor.layers.') and k.endswith('.weight')])
    for i in range(n):
        x = x @ weights[f'actor.layers.{i}.weight'].T+weights[f'actor.layers.{i}.bias']
        if i < n-1:
            x = F.silu(x)
    return x


def widen(model, hidden, noise, seed=0):
    g = torch.Generator().manual_seed(seed)
    model = {k: v.clone() for k, v in model.items()}
    n = len([k for k in model if k.startswith('actor.layers.') and k.endswith('.weight')])
    if len(hidden) != n-1:
        raise ValueError(f"expected {n-1} hidden widths, got {len(hidden)}")
    for i, width in enumerate(hidden):
        W, b = model[f'actor.layers.{i}.weight'], model[f'actor.layers.{i}.bias']
        old = W.shape[0]
        if width < old:
            raise ValueError("widening only")
        mapping = torch.cat((torch.arange(old), torch.randint(0, old, (width-old,), generator=g)))
        counts = torch.bincount(mapping, minlength=old).float()
        W_new, b_new = W[mapping].clone(), b[mapping].clone()
        if noise > 0 and width > old:
            W_new[old:] += torch.randn(W_new[old:].shape, generator=g)*noise*W.std()
        nxt = model[f'actor.layers.{i+1}.weight']
        model[f'actor.layers.{i}.weight'], model[f'actor.layers.{i}.bias'] = W_new, b_new
        model[f'actor.layers.{i+1}.weight'] = nxt[:, mapping]/counts[mapping][None]
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--hidden', type=int, nargs='+', required=True)
    p.add_argument('--noise', type=float, default=.01)
    p.add_argument('--out', required=True)
    p.add_argument('--check', help='npz with obs (e.g. a deploy bundle reference_rollouts.npz) to compare outputs')
    args = p.parse_args()
    snapshot = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    learner = snapshot['learner']
    old_model = learner['model']
    new_model = widen(old_model, args.hidden, args.noise)
    out = copy.copy(snapshot)
    out['learner'] = dict(learner, model=new_model, config=dict(learner['config'], actor_hidden=tuple(args.hidden)))
    out['learner'].pop('optimizer', None)                    # shapes changed: a widened net needs a fresh optimizer
    if 'contract' in out and 'learner_config' in out['contract']:
        out['contract'] = dict(out['contract'], learner_config=dict(out['contract']['learner_config'], actor_hidden=list(args.hidden)))
    old_params = sum(v.numel() for k, v in old_model.items() if k.startswith('actor.'))
    new_params = sum(v.numel() for k, v in new_model.items() if k.startswith('actor.'))
    report = dict(actor_params=(old_params, new_params), hidden=args.hidden)
    if args.check:
        obs = torch.tensor(np.load(args.check)['obs'], dtype=torch.float32)
        a, b = actor_forward(old_model, obs), actor_forward(new_model, obs)
        exact = actor_forward(widen(old_model, args.hidden, 0.), obs)
        report.update(max_abs_diff_with_noise=float((a-b).abs().max()), max_abs_diff_exact=float((a-exact).abs().max()),
                      action_max_abs_diff=float((torch.tanh(a[:, :29])-torch.tanh(b[:, :29])).abs().max()))
    torch.save(out, args.out)
    print(report)


if __name__ == '__main__':
    main()
