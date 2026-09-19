"""Capacity loss must be detected before optimization, including across reset."""
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')

from cat_mjlab.sim import CATSimulation


def simulation():
    sim=object.__new__(CATSimulation)
    sim.device='cpu';sim.num_envs=3
    sim.wp=SimpleNamespace(to_torch=lambda value:value)
    sim.backend=SimpleNamespace(wp_data=SimpleNamespace(
        nacon=torch.tensor([7],dtype=torch.int32),ncollision=torch.tensor([12],dtype=torch.int32),
        nefc=torch.tensor([6,12,2],dtype=torch.int32),overflow=torch.zeros(3,dtype=torch.int32),
        naconmax=32,njmax=24))
    sim._initialize_capacity_guard()
    return sim


def test_transient_constraint_overflow_survives_world_reset():
    sim=simulation();raw=sim.backend.wp_data
    raw.overflow[1]=1;raw.nefc[1]=31
    sim._latch_capacity()
    # A later reset/forward erases live flags and produces ordinary counts.
    raw.overflow.zero_();raw.nefc[:]=2;raw.nacon[:]=2
    with pytest.raises(RuntimeError,match='overflow latched.*bits=1'):
        sim.capacity_report()


def test_broadphase_overflow_is_detected_even_without_narrowphase_overflow():
    sim=simulation();raw=sim.backend.wp_data
    raw.ncollision[:]=45
    sim._latch_capacity()
    raw.ncollision[:]=3
    assert raw.nacon.item()<raw.naconmax
    with pytest.raises(RuntimeError,match='bits=4'):
        sim.capacity_report()


def test_sparse_constraint_overflow_bit_is_not_lost():
    sim=simulation();raw=sim.backend.wp_data
    raw.overflow[2]=2
    sim._latch_capacity();raw.overflow.zero_()
    with pytest.raises(RuntimeError,match='bits=2'):
        sim.capacity_report()


def test_capacity_report_retains_peaks_without_clearing_valid_history():
    sim=simulation();raw=sim.backend.wp_data
    sim._latch_capacity();raw.nacon[:]=1;raw.ncollision[:]=2;raw.nefc[:]=1
    report=sim.capacity_report()
    assert report['contacts']==7 and report['broadphase_candidates']==12
    assert report['max_constraints']==12 and report['overflow_worlds']==0
