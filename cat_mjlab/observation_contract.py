"""Native observation dimensions, derived from the named physical feature list."""
from cat_ppo.furniture.control import mjlab_observation_contract

CONTRACT = mjlab_observation_contract()
ACTOR_SIZE = len(CONTRACT['actor_features'])
CRITIC_SIZE = len(CONTRACT['critic_features'])
assert (ACTOR_SIZE, CRITIC_SIZE) == (222, 310)
SDF_RATE_SIZE = 4


CADENCE_FEATURES = ('gait.stride_frequency_hz',)


def actor_size(sdf_rate=False, tracker=False, cadence=False):
    """Actor input width: the baseline contract plus the optional hand/elbow distance-rate features, the tracker's
    reference block (cat_mjlab/tracking.py) and the commanded stride frequency (speed-dependent cadence)."""
    from .tracking import FEATURES
    return ACTOR_SIZE + (SDF_RATE_SIZE if sdf_rate else 0) + (FEATURES if tracker else 0) + (len(CADENCE_FEATURES) if cadence else 0)


def critic_size(tracker=False, cadence=False):
    from .tracking import FEATURES
    return CRITIC_SIZE + (FEATURES if tracker else 0) + (len(CADENCE_FEATURES) if cadence else 0)


def observation_contract(sdf_rate=False, tracker=False, cadence=False):
    """The feature contract, with the tracker's reference block and then the stride frequency appended to both
    observations when enabled."""
    contract = mjlab_observation_contract(sdf_rate)
    if tracker:
        from .tracking import feature_names
        names = feature_names()
        contract = dict(contract, actor_features=list(contract['actor_features']) + names,
                        critic_features=list(contract['critic_features']) + ['critic.' + n for n in names], tracker_features=names)
    if cadence:
        names = list(CADENCE_FEATURES)
        contract = dict(contract, actor_features=list(contract['actor_features']) + names,
                        critic_features=list(contract['critic_features']) + ['critic.' + n for n in names], cadence_features=names)
    return contract
