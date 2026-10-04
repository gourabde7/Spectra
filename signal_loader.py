"""Load .wav (mono real / stereo I-Q) and raw .iq/.bin/.raw/.dat captures."""

from __future__ import annotations

import os

import numpy as np
from scipy.signal import hilbert

MAX_SAMPLES = 1_500_000

IQ_FORMATS = {
    "cf32": (np.float32, 2, None),
    "ci16": (np.int16, 2, 32768.0),
    "ci8": (np.int8, 2, 128.0),
    "cu8": (np.uint8, 2, None),
}


def load_wav(path: str):
    import soundfile as sf

    data, fs = sf.read(path, always_2d=True, dtype="float64")
    if data.shape[1] >= 2:
        x = data[:, 0] + 1j * data[:, 1]
    else:
        x = hilbert(data[:, 0])  # real audio -> analytic signal
    return x[:MAX_SAMPLES].astype(np.complex128), float(fs)


def load_iq(path: str, fs: float, dtype: str = "cf32"):
    dtype = dtype.lower()
    if dtype not in IQ_FORMATS:
        raise ValueError(f"Unknown IQ format '{dtype}'. Use one of {list(IQ_FORMATS)}")
    npt, _, scale = IQ_FORMATS[dtype]
    raw = np.fromfile(path, dtype=npt, count=MAX_SAMPLES * 2)
    raw = raw[: (len(raw) // 2) * 2].astype(np.float64)
    if dtype == "cu8":
        raw = (raw - 127.5) / 127.5
    elif scale:
        raw = raw / scale
    return (raw[0::2] + 1j * raw[1::2]), float(fs)


def load_signal(path: str, iq_sample_rate: float = 2_000_000.0, iq_dtype: str = "cf32"):
    """Returns (complex samples, sample rate, 'wav' | 'iq')."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        x, fs = load_wav(path)
        return x, fs, "wav"
    x, fs = load_iq(path, iq_sample_rate, iq_dtype)
    if len(x) == 0:
        raise ValueError("File is empty or the IQ format does not match.")
    return x, fs, "iq"
