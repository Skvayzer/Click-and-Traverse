import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from energy_optimal_reference import (
    BatchedReplayCPU, InverseReference, integrate_cost, minimum_jerk, segment_distance, supported_zmp,
)
from cat_mjlab.constants import DEFAULT_QPOS
from cat_mjlab.sim_cpu import CPUSimulation


def test_reference_geometry_and_motor_energy():
    assert segment_distance(np.array([.5, .2, 0.]), np.zeros(3), np.array([1., 0., 0.])) == .2
    assert_allclose(minimum_jerk(np.array([0., .5, 1.])), [0., .5, 1.])
    torque = np.array([[1., -1.], [1., -1.]])
    energy = integrate_cost(torque, np.ones_like(torque), np.array([0., 2.]),
                            limits=np.array([2., 2.]), weights=np.array([1., .25]))
    assert energy == {"mech": 4., "heat": 312.5}
    feet = np.array([[0., -.1, 0.], [0., .1, 0.]])
    rotations = np.stack((np.eye(3), np.eye(3)))
    assert supported_zmp(np.array([0., 0., 100.]), np.zeros(3), feet, rotations)
    assert not supported_zmp(np.array([0., 0., 100.]), np.array([40., 0., 0.]), feet, rotations)


def test_inverse_model_has_active_foot_support_and_holding_cost():
    reference = InverseReference(samples=9)
    home = np.asarray(DEFAULT_QPOS)
    power = reference.inverse(np.tile(home, (9, 1)), 1., None)
    assert power is not None
    assert power["mech"] == 0.
    assert power["heat"] > 0.


def test_batched_replay_agrees_with_scalar_bridge_for_one_step():
    single = CPUSimulation()
    batch = BatchedReplayCPU(2, model=single.model)
    single.reset_data()
    batch.reset_data()
    single.step()
    batch.step()
    assert_allclose(batch.data.qpos[0], single.data.qpos[0])
    assert_allclose(batch.final_collision_data().xpos[0], single.final_collision_data().xpos[0])



def test_scripted_step_moves_a_foot_and_can_balance_with_a_long_deadline():
    reference = InverseReference(samples=25)
    home = np.asarray(DEFAULT_QPOS, dtype=float)
    reference.kine(home)
    feet, orientations = reference.frames()
    paths = list(reference.step_paths(home, np.linspace(0., 1., 25), feet, orientations))
    assert paths
    feasible = [(path, support) for path, support in paths if reference.inverse(path, 3., support)]
    assert feasible
    path, support = feasible[0]
    assert (support.sum(axis=1) == 1).any()
    reference.kine(path[-1])
    landed, _ = reference.frames()
    assert np.linalg.norm(landed-feet, axis=1).max() >= .075
