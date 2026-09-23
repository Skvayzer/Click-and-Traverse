"""Compact on-disk/in-memory encoding for scene fields.

Three observations make this lossless where it matters:

* ``bf`` and ``gf`` are renormalized before they reach the policy
  (cat_mjlab/task.py:308-309 and :433-434), so their magnitude is discarded.
  Only the direction carries information, and a direction needs two numbers,
  not three floats. Octahedral encoding in two int16 costs 4 bytes instead of
  12 with a worst-case angular error near 0.005 degrees.
* ``sdf`` is a distance in metres on a 0.04 m grid. float16 reproduces it to
  0.000488 m, a tenth of a percent of one cell.
* ``obs`` is binary occupancy stored one byte per voxel.

Zero/undefined direction vectors are preserved exactly through a validity mask
rather than a sentinel code, so decoding never invents a direction.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "cat-packed-fields-v1"


def _nonzero_sign(values):
    """numpy's sign() returns 0 at 0, which folds the octahedron incorrectly."""
    return np.where(values >= 0., 1., -1.)


def _octahedral_encode(vectors):
    """Map unit vectors to the [-1, 1]^2 octahedral square."""
    absolute = np.abs(vectors).sum(-1, keepdims=True)
    projected = vectors / np.maximum(absolute, 1e-20)
    x, y, z = projected[..., 0], projected[..., 1], projected[..., 2]
    lower = z < 0.
    u = np.where(lower, (1. - np.abs(y)) * _nonzero_sign(x), x)
    v = np.where(lower, (1. - np.abs(x)) * _nonzero_sign(y), y)
    return np.stack((u, v), -1)


def _octahedral_decode(square):
    u, v = square[..., 0], square[..., 1]
    z = 1. - np.abs(u) - np.abs(v)
    x = np.where(z < 0., (1. - np.abs(v)) * _nonzero_sign(u), u)
    y = np.where(z < 0., (1. - np.abs(u)) * _nonzero_sign(v), v)
    vectors = np.stack((x, y, z), -1)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-20)


def pack_direction(field):
    """Encode a (..., 3) direction field. Returns (int16 codes, packed validity bits)."""
    field = np.asarray(field, dtype=np.float32)
    norm = np.linalg.norm(field, axis=-1, keepdims=True)
    valid = (norm[..., 0] > 0.)
    unit = field / np.maximum(norm, 1e-20)
    square = _octahedral_encode(unit)
    codes = np.clip(np.rint(square * 32767.), -32767, 32767).astype(np.int16)
    codes[~valid] = 0
    return codes, np.packbits(valid.reshape(-1))


def unpack_direction(codes, valid_bits, shape):
    """Inverse of pack_direction; zeros stay exactly zero."""
    codes = np.asarray(codes, dtype=np.int16)
    count = int(np.prod(shape))
    valid = np.unpackbits(np.asarray(valid_bits, dtype=np.uint8), count=count).astype(bool)
    vectors = _octahedral_decode(codes.astype(np.float32) / 32767.)
    vectors[~valid.reshape(shape)] = 0.
    return vectors.astype(np.float32)


def pack_scalar(field):
    """Distances survive float16 far below one grid cell."""
    return np.asarray(field, dtype=np.float32).astype(np.float16)


def unpack_scalar(field):
    return np.asarray(field, dtype=np.float16).astype(np.float32)


def pack_occupancy(field):
    field = np.asarray(field)
    if field.max(initial=0) > 1:
        raise ValueError("Occupancy must be binary to pack as bits")
    return np.packbits(field.reshape(-1).astype(bool))


def unpack_occupancy(bits, shape):
    count = int(np.prod(shape))
    return np.unpackbits(np.asarray(bits, dtype=np.uint8), count=count).reshape(shape).astype(np.uint8)


def packed_bytes(shape):
    """Bytes per scene under this encoding, for capacity planning."""
    voxels = int(np.prod(shape))
    return voxels * (4 + 4 + 2) + 3 * ((voxels + 7) // 8)


def raw_bytes(shape):
    voxels = int(np.prod(shape))
    return voxels * (12 + 12 + 4 + 1)
