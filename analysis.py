"""Signal parameter estimation and spectral analysis."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq


def estimate_bandwidth(samples: np.ndarray, fs: float) -> float:
    n = min(len(samples), 65536)
    x = samples[:n]
    spec = np.abs(fft(x * np.hanning(n))) ** 2
    freqs = fftfreq(n, 1 / fs)
    pos = freqs >= 0
    spec = spec[pos]
    freqs = freqs[pos]
    threshold = 0.01 * np.max(spec)
    mask = spec > threshold
    if not np.any(mask):
        return fs * 0.1
    return float(freqs[mask].max() - freqs[mask].min())


def estimate_snr_db(samples: np.ndarray) -> float:
    power = np.mean(np.abs(samples) ** 2)
    noise_floor = np.percentile(np.abs(samples) ** 2, 10)
    if noise_floor <= 0:
        return 0.0
    return float(10 * np.log10(power / noise_floor))


def estimate_modulation(samples: np.ndarray, fs: float) -> str:
    """Heuristic modulation classifier for demo."""
    n = min(len(samples), 8192)
    x = samples[:n]
    x = x / (np.std(x) + 1e-12)

    # Instantaneous frequency variation
    phase = np.unwrap(np.angle(x))
    dphi = np.diff(phase)
    freq_var = np.var(dphi)

    # Amplitude stats
    amp = np.abs(x)
    amp_cv = np.std(amp) / (np.mean(amp) + 1e-12)

    # Constellation cluster count (rough)
    dec = x[:: max(1, len(x) // 512)]
    re = np.round(dec.real * 4) / 4
    im = np.round(dec.imag * 4) / 4
    clusters = len(set(zip(re.tolist(), im.tolist())))

    if freq_var > 0.5 and amp_cv < 0.15:
        return "FSK"
    if clusters <= 6:
        return "BPSK/QPSK"
    if clusters <= 20:
        return "8-PSK / 16-QAM"
    return "64-QAM / OFDM-like"


def guess_fec_and_interleaving(modulation: str) -> tuple[str, str]:
    """Demo metadata inference from modulation family."""
    mapping = {
        "FSK": ("Convolutional (K=7, rate 1/2)", "Block interleaver"),
        "BPSK/QPSK": ("RS(255,223) + Conv K=7", "Convolutional interleaver"),
        "8-PSK / 16-QAM": ("LDPC (DVB-S2-like)", "Diagonal interleaver"),
        "64-QAM / OFDM-like": ("Concatenated RS + Conv", "Pseudo-random interleaver"),
    }
    return mapping.get(modulation, ("Unknown FEC", "Unknown interleaving"))


def compute_spectrum(samples: np.ndarray, fs: float, nfft: int = 4096):
    n = min(len(samples), nfft)
    x = samples[:n] * np.hanning(n)
    spec = np.abs(fft(x, n=nfft))
    freqs = fftfreq(nfft, 1 / fs)
    pos = freqs >= 0
    return freqs[pos], 20 * np.log10(spec[pos] + 1e-12)


def compute_waterfall(samples: np.ndarray, fs: float, nperseg: int = 256):
    n = min(len(samples), 65536)
    f, t, sxx = sp_signal.spectrogram(
        samples[:n], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, mode="magnitude"
    )
    return f, t, 10 * np.log10(sxx + 1e-12)


def constellation_points(samples: np.ndarray, max_points: int = 2000) -> np.ndarray:
    step = max(1, len(samples) // max_points)
    pts = samples[::step][:max_points]
    power = np.mean(np.abs(pts) ** 2)
    if power > 0:
        pts = pts / np.sqrt(power)
    return pts


def extract_parameters(samples: np.ndarray, fs: float) -> dict:
    mod = estimate_modulation(samples, fs)
    fec, interleave = guess_fec_and_interleaving(mod)
    return {
        "Sampling Frequency (Hz)": f"{fs:,.0f}",
        "Estimated Bandwidth (Hz)": f"{estimate_bandwidth(samples, fs):,.0f}",
        "SNR Estimate (dB)": f"{estimate_snr_db(samples):.1f}",
        "Modulation (estimated)": mod,
        "FEC (inferred)": fec,
        "Interleaving (inferred)": interleave,
        "Sample Count": f"{len(samples):,}",
        "Duration (s)": f"{len(samples) / fs:.4f}",
    }
