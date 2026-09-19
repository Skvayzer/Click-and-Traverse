import numpy as np
import pytest

torch = pytest.importorskip("torch")
from cat_mjlab.fields import sample_ragged_field


def test_field_matches_released_cat_at_fractional_and_boundary_positions():
    import jax.numpy as jp
    from cat_ppo.furniture.generalist_fields import sample_ragged_field as reference
    rng = np.random.default_rng(918)
    shapes = np.array([[5, 7, 4], [4, 3, 6]])
    offsets = np.array([0, np.prod(shapes[0])])
    origin = rng.normal(size=(2, 3)).astype(np.float32)
    dx = np.array([.04, .07], np.float32)
    field = rng.normal(size=(sum(np.prod(shapes, axis=1)), 3)).astype(np.float32)
    positions = origin[:, None] + rng.uniform(-1, 7, size=(2, 19, 3)).astype(np.float32) * dx[:, None, None]
    positions[:, 0] = origin
    positions[:, 1] = origin + (shapes - 1) * dx[:, None]
    out = sample_ragged_field(torch.tensor(field), torch.tensor(positions), origin=torch.tensor(origin),
                            dx=torch.tensor(dx), shape=torch.tensor(shapes), offset=torch.tensor(offsets)).numpy()
    for b in range(2):
        expected = reference(jp.array(field), jp.array(positions[b]), origin=jp.array(origin[b]),
                             dx=jp.array(dx[b]), shape=jp.array(shapes[b]), offset=jp.array(offsets[b]))
        np.testing.assert_allclose(out[b], np.asarray(expected), atol=4e-7, rtol=2e-6)


def test_single_scene_interface_is_identical_to_batched():
    field = torch.arange(60., dtype=torch.float32).reshape(20, 3)
    pos = torch.tensor([[.2, .6, .4]])
    args = dict(origin=torch.zeros(3), dx=torch.tensor(1.), shape=torch.tensor([2, 2, 5]), offset=torch.tensor(0))
    actual = sample_ragged_field(field, pos, **args)
    expected = sample_ragged_field(field, pos[None], **{k: v[None] for k, v in args.items()})
    torch.testing.assert_close(actual, expected[0])
