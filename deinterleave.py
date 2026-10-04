"""Interleavers and their inverses (Block, Convolutional/Forney, Diagonal, Pseudo-random).

Every function works on any 1-D array (hard bits OR soft LLRs), so soft information
survives de-interleaving.  Only whole blocks are permuted; a trailing partial block is
passed through unchanged so that the output length always equals the input length
(except the Forney convolutional pair, which are delay-based - see below).
"""

from __future__ import annotations

import numpy as np


# ----------------------------- block ----------------------------------------
def block_interleave(x, rows: int = 16, cols: int = 16):
    x = np.asarray(x)
    n = rows * cols
    nb = len(x) // n
    if nb == 0:
        return x.copy()
    body = x[: nb * n].reshape(nb, rows, cols).transpose(0, 2, 1).reshape(-1)
    return np.concatenate([body, x[nb * n :]])


def block_deinterleave(x, rows: int = 16, cols: int = 16):
    x = np.asarray(x)
    n = rows * cols
    nb = len(x) // n
    if nb == 0:
        return x.copy()
    body = x[: nb * n].reshape(nb, cols, rows).transpose(0, 2, 1).reshape(-1)
    return np.concatenate([body, x[nb * n :]])


# ----------------------------- diagonal -------------------------------------
def _diag_index(rows: int, cols: int) -> np.ndarray:
    idx = []
    for d in range(cols):
        for r in range(rows):
            idx.append(r * cols + (r + d) % cols)
    return np.array(idx, dtype=np.int64)


def diagonal_interleave(x, rows: int = 16, cols: int = 16):
    x = np.asarray(x)
    n = rows * cols
    nb = len(x) // n
    if nb == 0:
        return x.copy()
    idx = _diag_index(rows, cols)
    body = x[: nb * n].reshape(nb, n)[:, idx].reshape(-1)
    return np.concatenate([body, x[nb * n :]])


def diagonal_deinterleave(x, rows: int = 16, cols: int = 16):
    x = np.asarray(x)
    n = rows * cols
    nb = len(x) // n
    if nb == 0:
        return x.copy()
    idx = _diag_index(rows, cols)
    blocks = x[: nb * n].reshape(nb, n)
    out = np.empty_like(blocks)
    out[:, idx] = blocks
    return np.concatenate([out.reshape(-1), x[nb * n :]])


# ----------------------------- pseudo-random --------------------------------
def prbs_permutation(n: int, seed: int = 0x1D0F) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)


def prbs_interleave(x, block: int = 1024, seed: int = 0x1D0F):
    x = np.asarray(x)
    nb = len(x) // block
    if nb == 0:
        return x.copy()
    p = prbs_permutation(block, seed)
    body = x[: nb * block].reshape(nb, block)[:, p].reshape(-1)
    return np.concatenate([body, x[nb * block :]])


def prbs_deinterleave(x, block: int = 1024, seed: int = 0x1D0F):
    x = np.asarray(x)
    nb = len(x) // block
    if nb == 0:
        return x.copy()
    p = prbs_permutation(block, seed)
    blocks = x[: nb * block].reshape(nb, block)
    out = np.empty_like(blocks)
    out[:, p] = blocks
    return np.concatenate([out.reshape(-1), x[nb * block :]])


# ----------------------------- convolutional (Forney) -----------------------
def conv_delay(depth: int, unit: int = 1) -> int:
    """Total end-to-end delay (in symbols) of a Forney interleaver/deinterleaver pair."""
    return depth * (depth - 1) * unit


def conv_interleave(x, depth: int = 8, unit: int = 1):
    """Forney convolutional interleaver, B=depth branches, branch i delays i*unit.

    The input is zero-padded so that the last symbols are flushed out; the output is
    therefore longer than the input by about (depth-1)*unit*depth symbols.
    """
    x = np.asarray(x)
    B = int(depth)
    steps = -(-len(x) // B) + (B - 1) * unit
    y = np.zeros(steps * B, dtype=x.dtype)
    for i in range(B):
        s = x[i::B]
        lane = np.zeros(steps, dtype=x.dtype)
        lane[i * unit : i * unit + len(s)] = s
        y[i::B] = lane
    return y


def conv_deinterleave(y, depth: int = 8, unit: int = 1):
    """Inverse of conv_interleave (branch i delays (B-1-i)*unit); the known start-up delay is removed."""
    y = np.asarray(y)
    B = int(depth)
    steps = len(y) // B
    z = np.zeros(steps * B, dtype=y.dtype)
    for i in range(B):
        r = y[i : steps * B : B]
        d = (B - 1 - i) * unit
        lane = np.zeros(steps, dtype=y.dtype)
        if d < steps:
            lane[d:] = r[: steps - d]
        z[i::B] = lane
    D = conv_delay(B, unit)
    return z[D:]


# ----------------------------- dispatcher -----------------------------------
def deinterleave(bits, mode: str, **kw):
    """GUI dispatcher. Parameters (all optional): rows, cols, depth, unit, block, seed."""
    m = mode.lower()
    rows = int(kw.get("rows", 16))
    cols = int(kw.get("cols", 16))
    if m.startswith("none") or m == "":
        return np.asarray(bits).copy()
    if "block" in m:
        return block_deinterleave(bits, rows, cols)
    if "conv" in m:
        return conv_deinterleave(bits, int(kw.get("depth", rows)), int(kw.get("unit", 1)))
    if "diag" in m:
        return diagonal_deinterleave(bits, rows, cols)
    if "pseudo" in m or "prbs" in m or "random" in m:
        return prbs_deinterleave(bits, int(kw.get("block", rows * cols)), int(kw.get("seed", 0x1D0F)))
    return np.asarray(bits).copy()


def interleave(bits, mode: str, **kw):
    m = mode.lower()
    rows = int(kw.get("rows", 16))
    cols = int(kw.get("cols", 16))
    if m.startswith("none") or m == "":
        return np.asarray(bits).copy()
    if "block" in m:
        return block_interleave(bits, rows, cols)
    if "conv" in m:
        return conv_interleave(bits, int(kw.get("depth", rows)), int(kw.get("unit", 1)))
    if "diag" in m:
        return diagonal_interleave(bits, rows, cols)
    if "pseudo" in m or "prbs" in m or "random" in m:
        return prbs_interleave(bits, int(kw.get("block", rows * cols)), int(kw.get("seed", 0x1D0F)))
    return np.asarray(bits).copy()
