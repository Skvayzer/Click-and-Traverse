import mujoco
import numpy as np
import pytest
import torch

from cat_mjlab.plan_blind import BlindObstacle, LEAD, OBSTACLE, obstacle_xml, replay_assignments


def test_schedule_hides_object_and_sweeps_before_intercept():
    guide = dict(start=[[.3, 0., 1.]], end=[[0., 0., 1.]], reveal=[1.], valid=[True])
    ids = torch.tensor([0])
    points = torch.tensor([[[0., 0., 1.]]])
    base = torch.ones(1, 1, 1)
    for variant in ('static', 'moving'):
        probe = BlindObstacle(guide, variant, 'cpu')
        gf = torch.ones_like(points)
        invisible = probe.merge(points, ids, torch.tensor([.9]), gf, gf, base, 0.)
        assert all(torch.equal(a, b) for a, b in zip(invisible, (gf, gf, base)))
        center, active = probe.locate(ids, torch.tensor([1. + LEAD / 2]))
        assert active.item()
        expected = .15 if variant == 'moving' else 0.
        torch.testing.assert_close(center[0, 0], torch.tensor(expected))
        assert probe.merge(points, ids, torch.tensor([1. + LEAD]), gf, gf, base, 0.)[-1].item() < 0


def test_mocap_box_is_a_physical_contact_with_the_approved_palm_proxy():
    model = mujoco.MjModel.from_xml_string(obstacle_xml())
    data = mujoco.MjData(model)
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    geom = model.geom(OBSTACLE).id
    mocap = model.body_mocapid[model.body(OBSTACLE).id]
    mujoco.mj_forward(model, data)
    assert all(geom not in touch.geom for touch in data.contact)
    data.mocap_pos[mocap] = data.site('left_palm').xpos
    mujoco.mj_forward(model, data)
    contacts = [touch for touch in data.contact if geom in touch.geom and touch.dist < 0]
    assert len(contacts) > 0
    assert any(model.geom(g).name.startswith('blind_proxy_hands') for hit in contacts for g in hit.geom)


def test_guide_replay_preserves_order_and_rejects_changed_trials():
    guide = dict(scenes=np.array(["B", "A", "A", "B"]), pose=np.zeros((4, 36)), velocity=np.zeros((4, 35)))
    ids = {"A": 7, "B": 9}
    assert replay_assignments(guide, ids, 2) == [9, 7, 7, 9]
    with pytest.raises(ValueError, match="scene repeats"):
        replay_assignments({**guide, "scenes": np.array(["A", "B", "B", "B"])}, ids, 2)
    with pytest.raises(ValueError, match="Invalid blind guide pose"):
        replay_assignments({**guide, "pose": np.full((4, 36), np.nan)}, ids, 2)
