"""Load .wav and .IQ signal files into complex baseband samples."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import soundfile as sf


def load_wav(path: str) -> tuple[np.ndarray, float]:
    """Load WAV as analytic-like complex signal (Hilbert on real audio)."""
    data, fs = sf.read(path, always_2d=True)
    if data.shape[1] >= 2:
        # Stereo treated as I/Q if two channels present
        i = data[:, 0].astype(np.float64)
        q = data[:, 1].astype(np.float64)
        samples = i + 1j * q
    else:
        x = data[:, 0].astype(np.float64)
        from scipy.signal import hilbert

        samples = hilbert(x)
    return samples, float(fs)


def load_iq(
    path: str,
    sample_rate: float,
    dtype: str = "cf32",
    offset: int = 0,
) -> tuple[np.ndarray, float]:
    """Load raw IQ file. dtype: cf32, ci16, cu8."""
    raw = Path(path).read_bytes()
    if offset:
        raw = raw[offset:]

    if dtype == "cf32":
        count = len(raw) // 4
        floats = struct.unpack(f"<{count}f", raw[: count * 4])
        arr = np.array(floats, dtype=np.float64).reshape(-1, 2)
        samples = arr[:, 0] + 1j * arr[:, 1]
    elif dtype == "ci16":
        count = len(raw) // 2
        ints = struct.unpack(f"<{count}h", raw[: count * 2])
        arr = np.array(ints, dtype=np.float64).reshape(-1, 2) / 32768.0
        samples = arr[:, 0] + 1j * arr[:, 1]
    elif dtype == "cu8":
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        arr = (arr - 127.5) / 127.5
        samples = arr[0::2] + 1j * arr[1::2]
    else:
        raise ValueError(f"Unsupported IQ dtype: {dtype}")

    return samples, float(sample_rate)


def load_signal(
    path: str,
    iq_sample_rate: float = 2_000_000,
    iq_dtype: str = "cf32",
    iq_offset: int = 0,
) -> tuple[np.ndarray, float, str]:
    """Auto-detect by extension and load."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".wav":
        samples, fs = load_wav(str(p))
        return samples, fs, "wav"
    if ext in (".iq", ".bin", ".raw", ".dat"):
        samples, fs = load_iq(str(p), iq_sample_rate, iq_dtype, iq_offset)
        return samples, fs, "iq"
    raise ValueError(f"Unsupported file type: {ext}")
