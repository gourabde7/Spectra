"""Forward Error Correction decoding (simplified for demo).

Heavy inner loops are compiled with Numba njit (nopython mode).
"""

from __future__ import annotations

import numpy as np
from numba import njit


# Polynomials for rate 1/2, K=7 Viterbi (NASA standard)
G1 = 0o171
G2 = 0o133


@njit
def _parity(x: int, poly: int) -> int:
    v = x & poly
    p = 0
    while v:
        p ^= v & 1
        v >>= 1
    return p


@njit
def _viterbi_encode_core(bits: np.ndarray) -> np.ndarray:
    n = bits.shape[0]
    out = np.empty(n * 2, dtype=np.uint8)
    state = 0
    mask = (1 << 6) - 1
    g1 = 0o171
    g2 = 0o133
    for i in range(n):
        state = ((state << 1) | int(bits[i])) & mask
        out[2 * i] = np.uint8(_parity(state, g1))
        out[2 * i + 1] = np.uint8(_parity(state, g2))
    return out


@njit
def _viterbi_decode_core(bits: np.ndarray) -> np.ndarray:
    """Hard-decision Viterbi decode, K=7, rate 1/2. NumPy in / NumPy out."""
    n = bits.shape[0]
    if n < 14:
        return bits.copy()

    n_states = 64
    n_steps = n // 2
    inf = 1.0e30
    path_metric = np.empty(n_states, dtype=np.float64)
    new_metric = np.empty(n_states, dtype=np.float64)
    for s in range(n_states):
        path_metric[s] = inf
    path_metric[0] = 0.0

    paths = np.zeros((n_steps, n_states), dtype=np.uint8)
    g1 = 0o171
    g2 = 0o133

    step = 0
    for t in range(0, n - 1, 2):
        rx0 = int(bits[t])
        rx1 = int(bits[t + 1])
        for s in range(n_states):
            new_metric[s] = inf
        for state in range(n_states):
            pm = path_metric[state]
            if pm >= inf:
                continue
            for bit in range(2):
                ns = ((state << 1) | bit) & 63
                exp0 = _parity(ns, g1)
                exp1 = _parity(ns, g2)
                cost = pm
                if exp0 != rx0:
                    cost += 1.0
                if exp1 != rx1:
                    cost += 1.0
                if cost < new_metric[ns]:
                    new_metric[ns] = cost
                    paths[step, ns] = np.uint8(bit)
        for s in range(n_states):
            path_metric[s] = new_metric[s]
        step += 1

    best_state = 0
    best_m = path_metric[0]
    for s in range(1, n_states):
        if path_metric[s] < best_m:
            best_m = path_metric[s]
            best_state = s

    decoded = np.empty(n_steps, dtype=np.uint8)
    state = best_state
    for i in range(n_steps - 1, -1, -1):
        decoded[i] = np.uint8(state & 1)
        bit = int(paths[i, state])
        state = (state >> 1) | (bit << 5)
    return decoded


@njit
def _rs_decode_block_core(bits: np.ndarray, k: int, n: int) -> np.ndarray:
    length = bits.shape[0]
    pad = (n - (length % n)) % n
    n_blocks = (length + pad) // n
    out = np.empty(n_blocks * k, dtype=np.uint8)
    for b in range(n_blocks):
        for j in range(k):
            idx = b * n + j
            if idx < length:
                out[b * k + j] = bits[idx]
            else:
                out[b * k + j] = 0
    return out


@njit
def _ldpc_decode_core(bits: np.ndarray, iterations: int) -> np.ndarray:
    n = bits.shape[0]
    x = np.empty(n, dtype=np.float64)
    for i in range(n):
        x[i] = bits[i]
    for _ in range(iterations):
        for i in range(1, n - 1):
            vote = (x[i - 1] + x[i + 1]) * 0.5
            if vote > 0.5:
                x[i] = 1.0
            else:
                x[i] = 0.0
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out[i] = np.uint8(x[i])
    return out


def viterbi_encode(bits: np.ndarray) -> np.ndarray:
    """Encode bits with rate 1/2 conv code (for demo / round-trip)."""
    bits = np.asarray(bits, dtype=np.uint8)
    return _viterbi_encode_core(bits)


def viterbi_decode(bits: np.ndarray) -> np.ndarray:
    """Hard-decision Viterbi decode, K=7, rate 1/2."""
    bits = np.asarray(bits, dtype=np.uint8)
    return _viterbi_decode_core(bits)


def rs_decode_block(bits: np.ndarray, k: int = 4, n: int = 7) -> np.ndarray:
    """Simplified RS-like block decode: majority vote per symbol group."""
    bits = np.asarray(bits, dtype=np.uint8)
    return _rs_decode_block_core(bits, int(k), int(n))


def ldpc_decode(bits: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Belief-propagation style hard LDPC demo (bit-flip reduction)."""
    bits = np.asarray(bits, dtype=np.uint8)
    return _ldpc_decode_core(bits, int(iterations))


def concatenated_decode(bits: np.ndarray) -> np.ndarray:
    inner = viterbi_decode(bits)
    return rs_decode_block(inner)


def fec_decode(bits: np.ndarray, mode: str) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    mode = mode.lower()
    if "viterbi" in mode or "conv" in mode:
        return viterbi_decode(bits)
    if "rs" in mode and "concat" not in mode:
        return rs_decode_block(bits)
    if "ldpc" in mode:
        return ldpc_decode(bits)
    if "concat" in mode:
        return concatenated_decode(bits)
    return bits.copy()
