"""Generate synthetic test signals for demo."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import soundfile as sf


def generate_qpsk_wav(path: str, fs: int = 48000, duration: float = 0.5) -> str:
    n = int(fs * duration)
    sps = 16
    bits = np.random.randint(0, 2, n // sps * 2)
    syms = (2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)
    syms = syms / np.sqrt(2)
    pulse = np.zeros(n, dtype=np.complex128)
    pulse[::sps][: len(syms)] = syms
    from scipy.signal import resample_poly

    bb = resample_poly(pulse, 1, 1)
    i = np.real(bb).astype(np.float32)
    q = np.imag(bb).astype(np.float32)
    sf.write(path, np.column_stack([i, q]), fs)
    return path


def generate_fsk_iq(path: str, fs: int = 1_000_000, duration: float = 0.2) -> str:
    n = int(fs * duration)
    t = np.arange(n) / fs
    bits = np.random.randint(0, 2, n // 200)
    freq = np.repeat(np.where(bits, 50000.0, 30000.0), 200)[:n]
    phase = 2 * np.pi * np.cumsum(freq) / fs
    sig = np.exp(1j * phase).astype(np.complex64)
    interleaved = np.empty(n * 2, dtype=np.float32)
    interleaved[0::2] = sig.real
    interleaved[1::2] = sig.imag
    Path(path).write_bytes(struct.pack(f"<{len(interleaved)}f", *interleaved))
    return path
