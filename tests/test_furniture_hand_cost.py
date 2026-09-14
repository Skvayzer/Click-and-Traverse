import jax
import jax.numpy as jnp
import pytest

from cat_ppo.furniture.perception import hand_protection_cost


def test_hand_cost_penalizes_approach_and_predicted_collision():
    features = jnp.array([[.10, .10, .10, 1., 0., 0., 0., 0., 0.]])
    approaching = jnp.array([[-.2, 0., 0.]])
    retreating = -approaching
    cost = jax.jit(hand_protection_cost)
    assert float(cost(features, approaching)) > float(cost(features, retreating))
    forecast = features.at[0, 2].set(-.02)
    assert float(cost(forecast, approaching)) > float(cost(features, approaching))
    assert float(cost(features.at[:, :3].set(.5), approaching)) == 0.
    assert float(hand_protection_cost(features, approaching, urgency_gain=0)) == pytest.approx(float(cost(features, retreating)))
