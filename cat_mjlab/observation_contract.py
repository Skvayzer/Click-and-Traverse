"""Native observation dimensions, derived from the named physical feature list."""
from cat_ppo.furniture.control import mjlab_observation_contract

CONTRACT = mjlab_observation_contract()
ACTOR_SIZE = len(CONTRACT['actor_features'])
CRITIC_SIZE = len(CONTRACT['critic_features'])
assert (ACTOR_SIZE, CRITIC_SIZE) == (222, 310)
SDF_RATE_SIZE = 4


def actor_size(sdf_rate=False, tracker=False):
    """Actor input width: the baseline contract plus the optional hand/elbow distance-rate features and the tracker's
    reference block (cat_mjlab/tracking.py)."""
    from .tracking import FEATURES
    return ACTOR_SIZE + (SDF_RATE_SIZE if sdf_rate else 0) + (FEATURES if tracker else 0)


def critic_size(tracker=False):
    from .tracking import FEATURES
    return CRITIC_SIZE + (FEATURES if tracker else 0)


def observation_contract(sdf_rate=False, tracker=False):
    """The feature contract, with the tracker's reference block appended to both observations when enabled."""
    contract = mjlab_observation_contract(sdf_rate)
    if tracker:
        from .tracking import feature_names
        names = feature_names()
        contract = dict(contract, actor_features=list(contract['actor_features']) + names,
                        critic_features=list(contract['critic_features']) + ['critic.' + n for n in names], tracker_features=names)
    return contract
