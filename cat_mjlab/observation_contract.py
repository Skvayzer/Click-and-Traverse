"""Native observation dimensions, derived from the named physical feature list."""
from cat_ppo.furniture.control import mjlab_observation_contract

CONTRACT = mjlab_observation_contract()
ACTOR_SIZE = len(CONTRACT['actor_features'])
CRITIC_SIZE = len(CONTRACT['critic_features'])
assert (ACTOR_SIZE, CRITIC_SIZE) == (222, 310)
