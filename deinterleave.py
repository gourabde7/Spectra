"""De-interleaving: Block, Convolutional, Diagonal, Pseudo-random."""

from __future__ import annotations

import numpy as np


def block_deinterleave(bits: np.ndarray, rows: int = 8, cols: int = 16) -> np.ndarray:
    n = rows * cols
    pad = (-len(bits)) % n
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    blocks = bits.reshape(-1, n)
    out = blocks.reshape(-1, rows, cols).transpose(0, 2, 1).reshape(-1)
    return out[: len(bits) - pad]


def convolutional_deinterleave(bits: np.ndarray, depth: int = 8) -> np.ndarray:
    streams = [bits[i::depth] for i in range(depth)]
    max_len = max(len(s) for s in streams)
    out = []
    for j in range(max_len):
        for s in streams:
            if j < len(s):
                out.append(s[j])
    return np.array(out, dtype=bits.dtype)


def diagonal_deinterleave(bits: np.ndarray, size: int = 16) -> np.ndarray:
    pad = (-len(bits)) % (size * size)
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    mat = bits.reshape(-1, size, size)
    out = []
    for m in mat:
        for d in range(2 * size - 1):
            for i in range(size):
                j = d - i
                if 0 <= j < size:
                    out.append(m[i, j])
    return np.array(out, dtype=bits.dtype)[: len(bits) - pad]


def prbs_deinterleave(bits: np.ndarray, seed: int = 0x1D0F) -> np.ndarray:
    n = len(bits)
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random(n))
    inv = np.empty(n, dtype=int)
    inv[perm] = np.arange(n)
    return bits[inv]


def deinterleave(bits: np.ndarray, mode: str, **kwargs) -> np.ndarray:
    mode = mode.lower()
    if "block" in mode:
        return block_deinterleave(bits, kwargs.get("rows", 8), kwargs.get("cols", 16))
    if "conv" in mode:
        return convolutional_deinterleave(bits, kwargs.get("depth", 8))
    if "diag" in mode:
        return diagonal_deinterleave(bits, kwargs.get("size", 16))
    if "pseudo" in mode or "prbs" in mode:
        return prbs_deinterleave(bits, kwargs.get("seed", 0x1D0F))
    return bits.copy()
