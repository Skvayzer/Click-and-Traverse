"""Native observation dimensions, derived from the named physical feature list."""
from cat_ppo.furniture.control import mjlab_observation_contract

CONTRACT = mjlab_observation_contract()
ACTOR_SIZE = len(CONTRACT['actor_features'])
CRITIC_SIZE = len(CONTRACT['critic_features'])
assert (ACTOR_SIZE, CRITIC_SIZE) == (222, 310)
SDF_RATE_SIZE = 4


def actor_size(sdf_rate=False):
    """Actor input width: the baseline contract plus the optional hand/elbow distance-rate features."""
    return ACTOR_SIZE + (SDF_RATE_SIZE if sdf_rate else 0)
