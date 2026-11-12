"""Path-prefix remapping for file ACCESS, for running on a machine whose data mirror lives elsewhere.

Scene-bank manifests pin their parent banks, storage roots and provenance files by the absolute path of the
machine that built them (dep-1: /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab/...). On the HPC
cluster the same files are mirrored under another root. With

    CAT_PATH_MAP="/old/prefix=/new/prefix[;/old2=/new2]"

every open / stat / listdir of a path starting with an old prefix is redirected to the new prefix. Path STRINGS
are not rewritten: manifests, pins and their sha256 digests are unchanged and still verify the same bytes;
Path.resolve() of an old path returns the old string, so chain comparisons between pinned paths keep working.
Only Python-level file access is covered (open/io.open/os.open/os.stat/os.listdir/os.scandir; numpy np.load and
np.memmap and torch.load(path) go through open). install() is a no-op when the variable is unset.
"""
import builtins
import io
import os

_MAP = []
_INSTALLED = False


def remap(path):
    """The redirected path (same type as given) if it starts with a mapped prefix, else the path unchanged."""
    if not _MAP or isinstance(path, int):
        return path
    try:
        text = os.fspath(path)
    except TypeError:
        return path
    raw = isinstance(text, bytes)
    if raw:
        text = os.fsdecode(text)
    for old, new in _MAP:
        if text == old or text.startswith(old + '/'):
            mapped = new + text[len(old):]
            return os.fsencode(mapped) if raw else mapped
    return path


def install(spec=None):
    """Read CAT_PATH_MAP (or `spec`) and wrap the file-access functions once. Returns the active map."""
    global _INSTALLED
    spec = os.environ.get('CAT_PATH_MAP', '') if spec is None else spec
    pairs = []
    for item in filter(None, (s.strip() for s in spec.split(';'))):
        old, sep, new = item.partition('=')
        if not sep or not old.startswith('/') or not new.startswith('/'):
            raise ValueError(f'CAT_PATH_MAP entries must be /old=/new, got {item!r}')
        pairs.append((old.rstrip('/'), new.rstrip('/')))
    _MAP[:] = pairs
    if not pairs or _INSTALLED:
        return list(_MAP)
    _INSTALLED = True
    original_open, original_io_open = builtins.open, io.open
    original_os_open, original_stat, original_listdir, original_scandir = os.open, os.stat, os.listdir, os.scandir

    def open_(file, *args, **kwargs):
        return original_open(remap(file), *args, **kwargs)

    def io_open(file, *args, **kwargs):
        return original_io_open(remap(file), *args, **kwargs)

    def os_open(path, *args, **kwargs):
        return original_os_open(remap(path), *args, **kwargs)

    def stat(path, *args, **kwargs):
        return original_stat(remap(path), *args, **kwargs)

    def listdir(path='.'):
        return original_listdir(remap(path))

    def scandir(path='.'):
        return original_scandir(remap(path))

    builtins.open, io.open, os.open, os.stat, os.listdir, os.scandir = open_, io_open, os_open, stat, listdir, scandir
    return list(_MAP)
