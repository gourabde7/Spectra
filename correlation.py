"""Bit stream correlation for header / preamble detection."""

from __future__ import annotations

import numpy as np


def correlate_bits(stream: np.ndarray, pattern: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Find positions where pattern best matches stream (XOR mismatch score)."""
    m = len(pattern)
    if len(stream) < m:
        return np.array([]), []
    scores = np.array(
        [np.sum(stream[i : i + m] != pattern) for i in range(len(stream) - m + 1)]
    )
    best = int(np.argmin(scores))
    hits = [i for i, s in enumerate(scores) if s <= max(1, m // 8)]
    return scores, hits


def pattern_from_hex(hex_str: str) -> np.ndarray:
    hex_str = hex_str.replace(" ", "").replace("0x", "")
    val = int(hex_str, 16)
    bits = []
    for i in range(len(hex_str) * 4):
        bits.append((val >> i) & 1)
    return np.array(list(reversed(bits)), dtype=np.uint8)


def format_bit_preview(bits: np.ndarray, max_bits: int = 128) -> str:
    chunk = bits[:max_bits]
    s = "".join(str(int(b)) for b in chunk)
    if len(bits) > max_bits:
        s += f"... ({len(bits)} total)"
    return s
