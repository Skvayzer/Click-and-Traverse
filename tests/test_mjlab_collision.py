import numpy as np
import pytest

torch = pytest.importorskip("torch")
from cat_mjlab import collision as port


@pytest.mark.parametrize("kind", ["sphere", "capsule", "box"])
def test_primitive_tests_match_jax_for_rotated_degenerate_and_parallel_cases(kind):
    import jax.numpy as jp
    from scipy.spatial.transform import Rotation
    from cat_ppo.furniture import body_collision_geometry as reference
    rng = np.random.default_rng(936)
    c, bc = [rng.normal(size=(97, 3)).astype(np.float32) for _ in range(2)]
    rotation = Rotation.random(97, random_state=rng).as_matrix().astype(np.float32)
    rotation[:4] = np.eye(3)
    br = Rotation.random(97, random_state=rng).as_matrix().astype(np.float32)
    br[:4] = np.eye(3)
    half, bh = [rng.uniform(.01, .5, size=(97, 3)).astype(np.float32) for _ in range(2)]
    radius = rng.uniform(.01, .3, 97).astype(np.float32)
    if kind == "sphere":
        args = (c, radius, bc, br, bh)
    elif kind == "capsule":
        end = c + rng.normal(size=c.shape).astype(np.float32)
        end[:3] = c[:3]
        args = (c, end, radius, bc, br, bh)
    else:
        args = (c, rotation, half, bc, br, bh)
    fn = kind + "_box_separation"
    expected = getattr(reference, fn)(*[jp.array(a) for a in args])
    actual = getattr(port, fn)(*[torch.tensor(a) for a in args]).numpy()
    np.testing.assert_allclose(actual, np.asarray(expected), atol=8e-7, rtol=1e-5)


def test_box_touching_and_capsule_crossing():
    z = torch.zeros(3); eye = torch.eye(3); half = torch.ones(3)
    assert port.box_box_separation(z, eye, half, torch.tensor([2., 0., 0.]), eye, half) == 0
    assert port.capsule_box_separation(torch.tensor([-3., 0., 0.]), torch.tensor([3., 0., 0.]),
                                     torch.tensor(.1), z, eye, half) == pytest.approx(-.1)


def test_proposal_matches_original_model_compilation():
    import json
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    from cat_ppo.furniture.body_collision_geometry import compile_proposal
    proposal = json.loads(port.PROPOSAL.read_text())
    model = mujoco.MjModel.from_xml_string(assemble_training_xml())
    original = compile_proposal(proposal, model)
    new = port.compile_proposal(proposal, model, "cpu")
    for name in ("body_ids", "local_centers", "local_rotations", "local_endpoints", "half_sizes", "radii"):
        np.testing.assert_allclose(new[name].numpy(), original[name], atol=0, rtol=0)
    assert len(new["body_ids"]) == 35
