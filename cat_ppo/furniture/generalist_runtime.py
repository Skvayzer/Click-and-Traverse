"""One atomic, complete CAT learner resume file, independent of best-model selection."""
from pathlib import Path
import os
import tempfile

from flax import serialization


def atomic_save_runtime(path, snapshot):
    """Replace one msgpack file only after all runtime state is safely written."""
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Runtime destination must be a regular file")
    if snapshot.get("schema") != "cat-ppo-runtime-v1":
        raise ValueError("Unrecognized runtime snapshot schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        payload = serialization.msgpack_serialize(snapshot, in_place=False)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_runtime(path):
    """Read arrays and primitive metadata without executing serialized Python."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Runtime snapshot must be a regular file")
    snapshot = serialization.msgpack_restore(path.read_bytes())
    if not isinstance(snapshot, dict) or snapshot.get("schema") != "cat-ppo-runtime-v1":
        raise ValueError("Unrecognized runtime snapshot schema")
    return snapshot
