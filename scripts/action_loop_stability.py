#!/usr/bin/env python3
"""Self-feedback stability of a policy's action path (the 25 Hz waist sawtooth on the real G1, 2026-10-21).

The actor observes its own last action and its own motor targets. Linearised around recorded states, one control
step maps (targets T, previous action a) to the next:
  a_t     = Dtanh (J_T dT_t + J_a da_{t-1})                     (J = d mean / d observation, network gradients)
  T_{t+1} = legs: T_t + 0.5 a_t (incremental)  |  upper: nominal + 0.8 a_t   [optionally averaged: (T + T_prev)/2]
The joints themselves cannot follow a 25 Hz flip, so measured positions/velocities are left out (worst case for
the dither mode). Reports per joint the single-joint loop gain (upper body: below -1 = period-2 limit cycle) and the
eigenvalues of the full multi-joint map (|lambda| > 1 = self-excited oscillation; negative real = 25 Hz).
  action_loop_stability.py CHECKPOINT [--obs reference_rollouts.npz] [--upper-average-obs]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from cat_mjlab.distill import load_expert
from cat_mjlab.learning import gaussian_parameters

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint'); p.add_argument('--obs', default=str(ROOT/'outputs/deploy/generalist_finetune_v1_u596/reference_rollouts.npz'))
    p.add_argument('--contract', default='/home/konstantinsmirnov/robotics/humanoid_navigation/cat/wholebody/policy_contract.json')
    p.add_argument('--samples', type=int, default=400)
    p.add_argument('--upper-average-obs', action='store_true', help='the observed upper targets are the 2-step average (the fix)')
    p.add_argument('--exclude', nargs='*', default=[], help='joints held fixed (e.g. waist_roll waist_pitch) to test the rest')
    a = p.parse_args()
    c = json.load(open(a.contract)); f = c['feature_names']; names = [n.replace('_joint', '') for n in c['joint_order']]
    tg = [f.index('motor_target.'+n) for n in c['joint_order']]; la = [f.index('last_action.'+n) for n in c['joint_order']]
    obs = torch.tensor(np.load(a.obs)['obs'], dtype=torch.float32)
    obs = obs[torch.linspace(0, len(obs)-1, min(a.samples, len(obs))).long()]
    m = load_expert(a.checkpoint, 'cpu'); n = m.config.actor_obs
    leg = np.arange(29) < 12
    diag, radius, worst = [], [], []
    for x in obs:
        x = x[:n].clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(lambda v: gaussian_parameters(m.logits(v[None], 0))[0][0], x).numpy()
        mean = gaussian_parameters(m.logits(x[None], 0))[0][0].detach().numpy()
        D = np.diag(1-np.tanh(mean)**2)
        KT, Ka = D@J[:, tg], D@J[:, la]
        B = np.diag(np.where(leg, .5, .8)); S = np.diag(leg.astype(float))
        if not a.upper_average_obs:
            A = np.block([[S+B@KT, B@Ka], [KT, Ka]])                       # state (T, a_prev)
        else:
            # observed upper target = (T_u + T_u_prev)/2: state (T, T_prev, a_prev)
            U = np.diag((~leg).astype(float)); L = np.diag(leg.astype(float))
            KTo = KT@(L+.5*U); KTp = KT@(.5*U)
            A = np.block([[S+B@KTo, B@KTp, B@Ka], [np.eye(29), np.zeros((29, 29)), np.zeros((29, 29))], [KTo, KTp, Ka]])
        if a.exclude:
            keep = np.array([nm not in a.exclude for nm in names]); k = np.tile(keep, A.shape[0]//29)
            A = A[np.ix_(k, k)]
        ev = np.linalg.eigvals(A)
        radius.append(np.abs(ev).max()); worst.append(ev[np.argmax(np.abs(ev))])
        diag.append(np.where(leg, np.nan, np.diag(Ka)+.8*np.diag(KT)))
    diag = np.array(diag)
    print(f"{a.checkpoint}  ({len(obs)} states{', averaged upper-target observation' if a.upper_average_obs else ''}{', without '+' '.join(a.exclude) if a.exclude else ''})")
    print("single-joint loop gain (upper body; < -1 = 25 Hz limit cycle):")
    for j in range(12, 29):
        g = diag[:, j]; print(f"  {names[j]:22s} median {np.median(g):+6.2f}   share < -1: {np.mean(g < -1):4.0%}")
    r = np.array(radius); w = np.array(worst)
    print(f"full closed loop: spectral radius median {np.median(r):.2f}, > 1 in {np.mean(r > 1):.0%} of states; "
          f"dominant eigenvalue negative-real (25 Hz) in {np.mean((np.abs(w.imag) < 1e-6) & (w.real < 0)):.0%}")

if __name__ == '__main__':
    main()
