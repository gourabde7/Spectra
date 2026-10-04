"""Demodulation front-end (thin wrapper over dsp_core)."""

from __future__ import annotations

import numpy as np

import dsp_core as D


def demodulate_full(samples, fs, mod=None, sps=None, baud=None) -> D.RxResult:
    """Blind demodulation with timing/carrier recovery. mod=None -> auto-classify."""
    return D.receive(np.asarray(samples, dtype=np.complex128), fs, mod=mod, baud=baud, sps=sps)


def demodulate(samples, mod: str, sps: float | None = None, fs: float = 1.0) -> np.ndarray:
    """Backward-compatible helper: returns hard bits (normal polarity)."""
    r = demodulate_full(samples, fs if sps else 1.0, mod=mod, sps=sps)
    return r.variants[0]["bits"]
