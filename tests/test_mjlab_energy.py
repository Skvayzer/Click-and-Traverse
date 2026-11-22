"""CPU regressions for legacy pricing and substep-integrated G1 energy."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cat_mjlab.constants import TORQUE_LIMIT
from cat_mjlab.energy import EnergyMeter
from cat_mjlab.task_math import pd_torques
from scripts.standing_event_energy import StandingEventEnergy, classify
from train_cat_mjlab import parser


def test_default_legacy_price_has_exact_reduction_and_nonfinite_handling():
    args = parser().parse_args(["run", "--from-scratch", "--bank-manifest", "bank.json",
                                "--body-collision-bank", "collision.json", "--body-collision-resets",
                                "resets.json", "--run-dir", "run"])
    assert args.energy_model is None
    rng = torch.Generator().manual_seed(0)
    torque = torch.randn((4, 29), generator=rng)
    velocity = torch.randn((4, 29), generator=rng)
    velocity[0, 28] = float("nan")
    meter = EnergyMeter(4, torch.tensor(TORQUE_LIMIT), {})
    meter.record(torch.zeros_like(torque), torque, velocity, .002)
    full, lower = meter.reward_power(torque, velocity, .02)

    def old(ctrl, qvel):
        return torch.nan_to_num((ctrl * qvel).abs().sum(-1), nan=0., posinf=0., neginf=0.)

    assert torch.equal(full, old(torque, velocity))
    assert torch.equal(lower, old(torque[:, :15], velocity[:, :15]))


def test_control_step_integrates_all_ten_substeps_and_resets():
    meter = EnergyMeter(1, torch.ones(29), {"model": "mech"})
    torque = torch.zeros((1, 29))
    torque[0, 0] = 1.
    meter.begin_step()
    for speed in range(1, 11):
        meter.record(torque, torque, torch.full_like(torque, speed), .002)
    power, _ = meter.reward_power(torque, torch.full_like(torque, 10.), .02)
    torch.testing.assert_close(power, torch.tensor([5.5]))
    torch.testing.assert_close(meter.step_j[0, 0, 0, 0], torch.tensor(.11))
    meter.begin_step()
    assert not meter.step_j.any()
    assert not meter.price_j.any()


def test_reward_uses_clipped_pd_and_gravity_without_rfi_or_offsets():
    limits = torch.ones(29)
    shape = (1, 29)
    pos = torch.zeros(shape)
    vel = torch.ones(shape)
    targets = torch.full(shape, .75)
    scale = torch.ones(1)
    gravity = torch.full(shape, .35)
    offset = torch.full(shape, -.6)
    rfi = torch.full(shape, .2)
    noise = -torch.ones(shape)
    clean, applied = pd_torques(pos, vel, targets, limits, limits * 0, scale, scale, rfi, noise, limits,
                                feedforward=gravity, torque_offset=offset)
    legacy = (targets + (gravity + offset) + rfi * noise).clamp(-limits, limits)
    assert torch.equal(applied, legacy)
    assert torch.equal(clean, torch.ones(shape))
    meter = EnergyMeter(1, limits, {"model": "mech"})
    meter.record(clean, applied, vel, .002)
    price, _ = meter.reward_power(applied, vel, .002)
    torch.testing.assert_close(price, torch.tensor([29.]))
    torch.testing.assert_close(meter.step_j[0, 1, 0].sum(), legacy.abs().sum() * .002)


def test_static_hold_has_heat_cost_at_real_motor_limits():
    limits = torch.tensor(TORQUE_LIMIT, dtype=torch.float32)
    torque = (.25 * limits)[None]
    meter = EnergyMeter(1, limits, {"model": "heat"})
    meter.record(torque, torque, torch.zeros_like(torque), .002)
    price, _ = meter.reward_power(torque, torch.zeros_like(torque), .002)
    assert not meter.step_j[0, 0, 0].any()
    torch.testing.assert_close(price, torch.tensor([500 * 29 * .25 ** 2]))


@pytest.mark.parametrize("model, expected", [("mech", [12, 3, 14]), ("heat", [12, 3, 14]),
                                           ("metab", [48, 12, 56])])
def test_group_prices_only_affect_the_selected_group(model, expected):
    torque = torch.ones((1, 29))
    all_prices = []
    for arm, leg in ((1., 1.), (3., 1.), (1., 2.)):
        meter = EnergyMeter(1, torch.ones(29), {"model": model, "alpha": 0., "arm_mult": arm, "leg_mult": leg})
        meter.record(torque, torque, torque, .002)
        all_prices.append(torch.stack(meter.reward_power(torque, torque, .002))[:, 0])
    legs, waist, arms = expected
    torch.testing.assert_close(torch.stack(all_prices), torch.tensor([
        [legs + waist + arms, legs + waist],
        [legs + waist + 3 * arms, legs + waist],
        [2 * legs + waist + arms, 2 * legs + waist],
    ], dtype=torch.float32))


def test_heat_clamps_net_work_and_metabolic_prices_negative_work():
    torque = torch.zeros((1, 29))
    torque[0, 0] = -3.
    torque[0, 15] = 2.
    velocity = torch.ones_like(torque)
    limits = torch.full((29,), 10.)
    heat = EnergyMeter(1, limits, {"model": "heat", "alpha": 100.})
    heat.record(torque, torque, velocity, .002)
    power, _ = heat.reward_power(torque, velocity, .002)
    torch.testing.assert_close(power, torch.tensor([13.]))
    metabolism = EnergyMeter(1, limits, {"model": "metab"})
    metabolism.record(torque, torque, velocity, .002)
    power, _ = metabolism.reward_power(torque, velocity, .002)
    torch.testing.assert_close(power, torch.tensor([2 / .25 + 3 / 1.2]))


def test_approach_classification_thresholds_and_body_priority():
    assert classify(torch.tensor([.10, .11, .11, .09, .11, .11]),
                    torch.tensor([.04, .05, .04, .02, .04, .04]),
                    torch.tensor([.04, .04, .05, .10, .04, .04]),
                    torch.tensor([False, False, False, False, True, False])).tolist() == [1, 0, 0, 2, 2, 1]


def test_cli_config_route_baseline_and_zero_all_prices():
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.energy import PRICE_TERMS

    cfg = wholebody_config(reactive_power_weight=-.02, table_leg_power_weight=-.01,
                           route_energy=(-.01, 234, .6, 2.), route_energy_p0=11., energy_model="heat",
                           energy_alpha=150., arm_mult=.5, leg_mult=2.)
    assert cfg["route_energy"] == dict(q=234, cap=.6, table_factor=2., p0=11.)
    assert cfg["energy_config"] == dict(model="heat", alpha=150., arm_mult=.5, leg_mult=2.)
    cfg["reward_group_scales"] = {"room": {"route_energy": -.05}}
    free = wholebody_config(cfg, no_energy_price=True)
    assert "route_energy" not in free
    assert not (set(free["reward_config"]["scales"]) & set(PRICE_TERMS))
    assert not free["reward_group_scales"]["room"]


@pytest.mark.parametrize("model", [None, "mech"])
def test_task_step_prices_expected_torque_and_retains_episode_joules(model):
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.energy import GROUPS
    from cat_mjlab.task import CATTask
    from test_mjlab_task import _CPUSimulation, _tiny_bank

    simulation = _CPUSimulation(2)
    cfg = wholebody_config(energy_model=model, global_power_weight=-.01)
    cfg["randomize_initial_episode_steps"] = False
    bank = _tiny_bank()
    bank.manifest = {"scenes": [{} for _ in range(bank.count)]}
    bank.weights = torch.ones(bank.count)
    task = CATTask(simulation, bank, cfg, seed=33)
    samples = []
    integrate = simulation.step

    def step():
        samples.append((simulation.data.ctrl * simulation.data.qvel[:, 6:]).abs())
        integrate()

    simulation.step = step
    out = task.step(torch.full((2, 29), .05))
    assert not out["done"].any()
    assert len(samples) == 10
    mean = torch.stack(samples).mean(0)
    metrics = out["metrics"]
    expected = torch.stack([mean[:, joints].sum(-1) for joints in GROUPS.values()], -1)
    torch.testing.assert_close(metrics["energy/step_w"][:, 1, 0], expected)
    torch.testing.assert_close(metrics["energy/episode_j"][:, 1, 0], expected * task.dt)
    if model is None:
        old = (simulation.data.ctrl * simulation.data.qvel[:, 6:]).abs().sum(-1) * -.01
        assert torch.equal(metrics["reward/global_power"], old)
    else:
        torch.testing.assert_close(metrics["reward/global_power"], metrics["energy/step_w"][:, 0, 0].sum(-1) * -.01)


def test_event_uses_torso_frame_and_counts_only_active_substeps():
    model = SimpleNamespace(site=lambda name: SimpleNamespace(id={"left": 1, "right": 2}[name]),
                            site_bodyid=np.arange(3), site_pos=np.zeros((3, 3)))
    task = SimpleNamespace(model=model, torso_site=0, site_names=["head", "root", "torso", "left", "right"],
                           sim_dt=.002, num_envs=1, device="cpu",
                           data=SimpleNamespace(qpos=torch.zeros((1, 36))))
    event = StandingEventEnergy(task)
    data = SimpleNamespace(xmat=torch.eye(3).expand(1, 3, 3, 3).clone(), xpos=torch.zeros((1, 3, 3)))
    row = torch.tensor([0])
    hand = torch.zeros((1, 2, 3))
    ray = torch.tensor([[1., 0., 0.]])
    root = torch.zeros((1, 3))
    sample = torch.full((1, 2, 2, 3), .002)
    event.observe(data, row, torch.tensor([-.02]), row, ray, hand, root)
    event.record(sample)
    event.observe(data, row, torch.zeros(1), row, ray, hand, root)
    event.record(sample)
    data.xpos[0, 0, 0] = .2
    hand[0, 0, 0] = .32
    event.observe(data, row, torch.tensor([.002]), row, ray, hand, root)
    event.record(sample)
    event.finish(row, row)
    summary, details = event.summary([17], ["danger"])
    assert summary["arm_first"]["count"] == 1
    assert details[0]["row"] == 17
    assert details[0]["retreat_m"] == .12
    assert details[0]["energy"]["policy"]["heat"]["waist"] == {"energy_j": .004, "mean_w": 1.}
