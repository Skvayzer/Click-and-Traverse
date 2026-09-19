#!/usr/bin/env python3
"""Backend migration check, without policy evaluation or W&B logging."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=24)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import mujoco
    import numpy as np
    import torch
    from cat_mjlab.sim import CATSimulation, OBS_FIELDS
    from cat_mjlab import constants
    start = time.perf_counter()
    sim = CATSimulation(args.num_envs, device=args.device)
    initial = torch.as_tensor(constants.DEFAULT_QPOS, dtype=torch.float32, device=args.device)
    sim.data.qpos[:] = initial
    sim.data.qvel.zero_()
    sim.forward()
    reference = mujoco.MjData(sim.model)
    reference.qpos[:] = initial.cpu().numpy()
    mujoco.mj_forward(sim.model, reference)
    fk = sim.final_collision_data()
    np.testing.assert_allclose(fk.xpos[0].cpu().numpy(), reference.xpos, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(fk.xmat[0].cpu().numpy(), reference.xmat.reshape(-1, 3, 3), atol=2e-6, rtol=2e-6)
    before = {name: getattr(sim.data, name).clone() for name in OBS_FIELDS}
    sim.final_collision_data()
    for name, value in before.items():
        torch.testing.assert_close(getattr(sim.data, name), value, rtol=0, atol=0)
    kps = torch.as_tensor(constants.KPs, dtype=torch.float32, device=args.device)
    kds = torch.as_tensor(constants.KDs, dtype=torch.float32, device=args.device)
    limit = torch.as_tensor(constants.TORQUE_LIMIT, dtype=torch.float32, device=args.device)
    native_max_error = 0.
    # Compare one identical physics step against native MuJoCo on the same XML.
    torque = (kps * (initial[7:] - sim.data.qpos[:, 7:]) - kds * sim.data.qvel[:, 6:]).clamp(-limit, limit)
    sim.data.ctrl.copy_(torque)
    reference.ctrl[:] = torque[0].cpu().numpy()
    sim.step(); mujoco.mj_step(sim.model, reference)
    native_max_error = float(np.max(np.abs(sim.data.qpos[0].cpu().numpy() - reference.qpos)))
    if native_max_error > 2e-4:
        raise AssertionError(f'One-step Warp/native MuJoCo qpos mismatch: {native_max_error}')
    # Selective reset must not refresh the other world's observation timing.
    if args.num_envs > 1:
        old = {name: getattr(sim.data, name)[1:].clone() for name in OBS_FIELDS}
        ids = torch.tensor([0], device=args.device)
        sim.reset_data(ids); sim.data.qpos[ids] = initial; sim.forward(ids)
        for name, value in old.items():
            torch.testing.assert_close(getattr(sim.data, name)[1:], value, atol=0, rtol=0)
    if args.device.startswith('cuda'): torch.cuda.synchronize()
    startup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(args.steps):
        sim.data.ctrl.copy_((kps * (initial[7:] - sim.data.qpos[:, 7:]) - kds * sim.data.qvel[:, 6:]).clamp(-limit, limit))
        sim.step()
    if args.device.startswith('cuda'): torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    assert torch.isfinite(sim.data.qpos).all() and torch.isfinite(sim.data.qvel).all()
    report = dict(backend='mjlab/MuJoCo-Warp', worlds=args.num_envs, steps=args.steps,
        physics_steps_per_second=args.steps * args.num_envs / seconds,
        startup_seconds=startup_seconds, options=sim.option_contract(), capacity=sim.capacity_report(),
        native_one_step_qpos_max_error=native_max_error,
        final_fk_preserves_observations=True, selective_reset_preserves_observations=True,
        policy_evaluation=False, training=False,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__': main()
