"""Demodulation: FSK, PSK, QAM."""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt


def _slice_symbols(samples: np.ndarray, sps: int) -> np.ndarray:
    sps = max(1, int(sps))
    idx = np.arange(sps // 2, len(samples), sps)
    return samples[idx]


def demod_bpsk(samples: np.ndarray, sps: int = 8) -> np.ndarray:
    syms = _slice_symbols(samples, sps)
    bits = (np.real(syms) > 0).astype(np.uint8)
    return bits


def demod_qpsk(samples: np.ndarray, sps: int = 8) -> np.ndarray:
    syms = _slice_symbols(samples, sps)
    syms = syms / (np.mean(np.abs(syms)) + 1e-12)
    bits = np.empty(len(syms) * 2, dtype=np.uint8)
    bits[0::2] = (np.real(syms) > 0).astype(np.uint8)
    bits[1::2] = (np.imag(syms) > 0).astype(np.uint8)
    return bits


def demod_16qam(samples: np.ndarray, sps: int = 8) -> np.ndarray:
    levels = np.array([-3, -1, 1, 3], dtype=np.float64)
    syms = _slice_symbols(samples, sps)
    syms = syms / (np.mean(np.abs(syms)) + 1e-12) * 3

    def level_bits(v):
        idx = int(np.argmin(np.abs(levels - v)))
        return [(idx >> 1) & 1, idx & 1]

    bits = []
    for s in syms:
        bits.extend(level_bits(np.real(s)))
        bits.extend(level_bits(np.imag(s)))
    return np.array(bits, dtype=np.uint8)


def demod_fsk(samples: np.ndarray, sps: int = 20, h: float = 0.5) -> np.ndarray:
    phase = np.unwrap(np.angle(samples))
    dphi = np.diff(phase)
    dphi = medfilt(dphi, kernel_size=5)
    syms = dphi[:: max(1, sps // 2)]
    bits = (syms > 0).astype(np.uint8)
    return bits


def demodulate(samples: np.ndarray, modulation: str, sps: int = 8) -> np.ndarray:
    mod = modulation.upper()
    if "FSK" in mod:
        return demod_fsk(samples, sps=sps)
    if "BPSK" in mod:
        return demod_bpsk(samples, sps=sps)
    if "QPSK" in mod or "PSK" in mod:
        return demod_qpsk(samples, sps=sps)
    if "QAM" in mod:
        return demod_16qam(samples, sps=sps)
    return demod_qpsk(samples, sps=sps)
