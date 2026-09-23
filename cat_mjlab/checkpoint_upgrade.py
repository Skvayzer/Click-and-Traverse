"""Retired observation-expansion API. No checkpoint conversion is performed."""
RETIRED = 'Observation expansion is retired. Train --from-scratch on native 222/310; no checkpoint trimming is supported.'


def upgrade_snapshot(*args, **kwargs):
    raise ValueError(RETIRED)


def upgrade_file(*args, **kwargs):
    # Fail before opening source or destination.
    raise ValueError(RETIRED)


def validate_upgrade(snapshot):
    if 'upgrade' in snapshot:
        raise ValueError(RETIRED)
