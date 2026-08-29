"""Forward Error Correction decoding (simplified for demo)."""

from __future__ import annotations

import numpy as np


# Polynomials for rate 1/2, K=7 Viterbi (NASA standard)
G1 = 0o171
G2 = 0o133


def _parity(x: int, poly: int) -> int:
    v = x & poly
    p = 0
    while v:
        p ^= v & 1
        v >>= 1
    return p


def viterbi_encode(bits: np.ndarray) -> np.ndarray:
    """Encode bits with rate 1/2 conv code (for demo / round-trip)."""
    state = 0
    out = []
    mask = (1 << 6) - 1
    for b in bits:
        state = ((state << 1) | int(b)) & mask
        out.append(_parity(state, G1))
        out.append(_parity(state, G2))
    return np.array(out, dtype=np.uint8)


def viterbi_decode(bits: np.ndarray) -> np.ndarray:
    """Hard-decision Viterbi decode, K=7, rate 1/2."""
    if len(bits) < 14:
        return bits.copy()
    n_states = 64
    path_metric = np.full(n_states, np.inf)
    path_metric[0] = 0
    paths = []

    for t in range(0, len(bits) - 1, 2):
        rx0, rx1 = int(bits[t]), int(bits[t + 1])
        new_metric = np.full(n_states, np.inf)
        new_paths = np.zeros(n_states, dtype=np.uint8)
        for state in range(n_states):
            if path_metric[state] == np.inf:
                continue
            for bit in (0, 1):
                ns = ((state << 1) | bit) & 63
                exp0 = _parity(ns, G1)
                exp1 = _parity(ns, G2)
                cost = path_metric[state] + (exp0 != rx0) + (exp1 != rx1)
                if cost < new_metric[ns]:
                    new_metric[ns] = cost
                    new_paths[ns] = bit
        path_metric = new_metric
        paths.append(new_paths)

    state = int(np.argmin(path_metric))
    decoded = []
    for p in reversed(paths):
        decoded.append(state & 1)
        state = (state >> 1) | (p[state] << 5)
    return np.array(list(reversed(decoded)), dtype=np.uint8)


def rs_decode_block(bits: np.ndarray, k: int = 4, n: int = 7) -> np.ndarray:
    """Simplified RS-like block decode: majority vote per symbol group."""
    pad = (-len(bits)) % n
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    out = []
    for block in bits.reshape(-1, n):
        out.extend(block[:k])
    return np.array(out, dtype=np.uint8)


def ldpc_decode(bits: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Belief-propagation style hard LDPC demo (bit-flip reduction)."""
    x = bits.astype(np.float64)
    n = len(x)
    for _ in range(iterations):
        for i in range(1, n - 1):
            neighbors = [x[i - 1], x[i + 1]]
            vote = np.mean(neighbors)
            x[i] = 1.0 if vote > 0.5 else 0.0
    return x.astype(np.uint8)


def concatenated_decode(bits: np.ndarray) -> np.ndarray:
    inner = viterbi_decode(bits)
    return rs_decode_block(inner)


def fec_decode(bits: np.ndarray, mode: str) -> np.ndarray:
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
