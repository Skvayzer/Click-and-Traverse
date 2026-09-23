"""Retired expansion must fail before any checkpoint file IO."""
import builtins
import pytest
from cat_mjlab.checkpoint_upgrade import upgrade_file, upgrade_snapshot, validate_upgrade

def test_no_expansion_or_trim_and_no_checkpoint_access(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('Checkpoint IO is forbidden')
    monkeypatch.setattr(builtins,'open',forbidden)
    with pytest.raises(ValueError,match='retired'):upgrade_file('unused','unused')
    with pytest.raises(ValueError,match='retired'):upgrade_snapshot({})
    with pytest.raises(ValueError,match='retired'):validate_upgrade({'upgrade':{}})
    validate_upgrade({})
