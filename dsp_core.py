"""Receiver DSP core: blind parameter estimation, timing + carrier recovery, soft demapping.

Everything here is measured from the samples - nothing is looked up from a table.

Pipeline for linear modulations (BPSK / QPSK / 16-QAM):
    burst trim -> band-centre (coarse CFO) -> cyclostationary baud estimate ->
    matched-filter selection -> block-wise maximum-energy timing tracking ->
    M-th-power fine CFO + Viterbi&Viterbi carrier-phase tracking -> soft demapping
Pipeline for 2-FSK:
    instantaneous frequency -> baud estimate -> 2-means tone detection -> timing tracking -> slicer
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp
from scipy.signal import find_peaks, fftconvolve, resample_poly
from scipy.ndimage import uniform_filter1d

MAX_PROC = 1_500_000  # hard cap on samples processed in one go
SEL_N = 80_000  # samples used for filter selection / classification
LLR_CLIP = 30.0


# --------------------------------------------------------------------------- #
# basic spectral estimates
# --------------------------------------------------------------------------- #
def psd_welch(x: np.ndarray, fs: float, nperseg: int = 1024):
    x = np.asarray(x, dtype=np.complex128)
    nper = int(min(nperseg, max(16, len(x) // 4)))
    f, P = sp.welch(x, fs=fs, nperseg=nper, return_onesided=False, detrend=False, scaling="density")
    o = np.argsort(f)
    return f[o], P[o]


def band_estimate(x: np.ndarray, fs: float) -> dict:
    """Noise-floor-referenced occupied band: bandwidth, centre, floor, in-band SNR."""
    f, P = psd_welch(x, fs)
    Ps = uniform_filter1d(P, size=5, mode="nearest")
    floor = float(np.percentile(Ps, 15)) + 1e-30
    peak = float(Ps.max())
    if peak / floor < 4.0:  # >= 6 dB above floor nowhere: wideband / noise-like
        return {"bw": float(fs), "fc": 0.0, "floor": floor, "snr_psd": 0.0, "wideband": True, "f": f, "P": P}
    mask = Ps > 2.0 * floor
    on = np.flatnonzero(mask)
    lo, hi = int(on[0]), int(on[-1])
    df = f[1] - f[0]
    bw = (hi - lo + 1) * df
    seg = P[lo : hi + 1]
    fc = float(np.sum(f[lo : hi + 1] * seg) / np.sum(seg))
    sig = float(np.sum(seg - floor).clip(min=0))
    noi = float(floor * len(seg))
    snr = 10 * np.log10(max(sig / noi, 1e-6)) if noi > 0 else 60.0
    return {
        "bw": float(bw),
        "fc": fc,
        "floor": floor,
        "snr_psd": float(snr),
        "wideband": bool(bw > 0.85 * fs),
        "f": f,
        "P": P,
    }


def trim_to_burst(x: np.ndarray, fs: float):
    """Crop leading/trailing noise-only parts if the capture contains an obvious burst.

    Two-level (2-means) split of the smoothed log-power; the longest above-threshold run is the burst.
    """
    n = len(x)
    w = max(16, min(512, n // 64))
    xd = np.asarray(x, dtype=np.complex128)
    try:  # band-limit to the signal for detection (raises burst-to-noise ratio by fs/bw)
        be = band_estimate(xd, fs)
        if not be["wideband"] and n > 1024:
            cutoff = float(np.clip(0.6 * be["bw"], fs * 0.005, fs * 0.45))
            h = sp.firwin(129, cutoff, fs=fs)
            xd = fftconvolve(xd * np.exp(-2j * np.pi * be["fc"] * np.arange(n) / fs), h, mode="same")
    except Exception:
        pass
    p = np.convolve(np.abs(xd) ** 2, np.ones(w) / w, mode="same")
    lp = 10 * np.log10(p + 1e-30)
    c0, c1 = np.percentile(lp, 3), np.percentile(lp, 97)
    for _ in range(25):
        thr = 0.5 * (c0 + c1)
        lo, hi = lp[lp <= thr], lp[lp > thr]
        if len(lo) == 0 or len(hi) == 0:
            break
        c0, c1 = lo.mean(), hi.mean()
    thr = 0.5 * (c0 + c1)
    frac_lo = float(np.mean(lp <= thr))
    if (c1 - c0) < 6.0 or frac_lo < 0.01 or frac_lo > 0.99:
        return x, (0, n), False
    above = lp > thr
    # longest contiguous run (bridging gaps shorter than the smoothing window)
    d = np.diff(np.r_[0, above.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = [(s, e) for s, e in zip(starts, ends)]
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= w:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    s, e = max(merged, key=lambda r: r[1] - r[0])
    s = max(0, s - w)
    e = min(n, e + w)
    if e - s < 256 or (e - s) > 0.97 * n:
        return x, (0, n), False
    return x[s:e], (int(s), int(e)), True


# --------------------------------------------------------------------------- #
# cyclostationary baud-rate estimation
# --------------------------------------------------------------------------- #
def _cyclic_peaks(v: np.ndarray, fs: float, fmin: float, fmax: float):
    v = np.asarray(v, dtype=np.float64)
    v = v - v.mean()
    n = len(v)
    if n < 64:
        return []
    nfft = int(2 ** np.ceil(np.log2(max(n, 1024) * 4)))
    S = np.abs(np.fft.rfft(v * np.hanning(n), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    lo, hi = np.searchsorted(f, fmin), np.searchsorted(f, fmax)
    seg = S[lo:hi]
    if len(seg) < 16 or seg.max() <= 0:
        return []
    pk, _ = find_peaks(seg, height=0.25 * seg.max(), distance=max(4, int(8 * nfft / n)))
    out = []
    w = max(100, int(40 * nfft / n))
    for p in pk:
        k = p + lo
        a, b = max(0, k - w), min(len(S), k + w)
        med = float(np.median(S[a:b])) + 1e-30
        prom = float(S[k] / med)
        if 1 <= k < len(S) - 1 and S[k - 1] > 0 and S[k + 1] > 0:
            la, lb, lc = np.log(S[k - 1] + 1e-300), np.log(S[k] + 1e-300), np.log(S[k + 1] + 1e-300)
            den = la - 2 * lb + lc
            d = 0.5 * (la - lc) / den if den != 0 else 0.0
        else:
            d = 0.0
        out.append((float((k + d) * fs / nfft), prom, float(S[k])))
    return out


def estimate_baud(x: np.ndarray, fs: float) -> dict:
    """Blind symbol-rate estimate from cyclic spectral lines of |x|^2, |dx|^2 and |d(freq)|."""
    x = np.asarray(x[:MAX_PROC], dtype=np.complex128)
    n = len(x)
    fmin = max(30.0 * fs / n, fs / 400.0)
    fmax = fs / 2.2
    cands = {"envelope": np.abs(x) ** 2}
    dx = np.diff(x)
    cands["difference"] = np.abs(dx) ** 2
    ph = np.angle(x[1:] * np.conj(x[:-1]))
    fi = uniform_filter1d(ph, size=3)
    cands["freq-edges"] = np.abs(np.diff(fi))
    best = None
    for name, v in cands.items():
        pk = _cyclic_peaks(v, fs, fmin, fmax)
        pk = [p for p in pk if p[1] >= 20.0]
        if not pk:
            continue
        top = max(p[2] for p in pk)
        strong = [p for p in pk if p[2] >= 0.3 * top]
        fund = min(strong, key=lambda p: p[0])  # lowest strong line = fundamental
        score = fund[1]
        if best is None or score > best["prominence"]:
            best = {"baud": fund[0], "prominence": score, "source": name}
    if best is None:
        return {"baud": None, "prominence": 0.0, "source": "none"}
    return best


# --------------------------------------------------------------------------- #
# filters / timing
# --------------------------------------------------------------------------- #
def rrc_taps(beta: float, sps: float, span: int = 8) -> np.ndarray:
    half = int(np.ceil(span * sps / 2))
    t = np.arange(-half, half + 1) / sps
    h = np.zeros_like(t)
    t0 = np.abs(t) < 1e-9
    tb = (np.abs(np.abs(t) - 1.0 / (4 * beta)) < 1e-9) if beta > 0 else np.zeros_like(t0)
    oth = ~(t0 | tb)
    h[t0] = 1.0 - beta + 4 * beta / np.pi
    if beta > 0:
        h[tb] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    to = t[oth]
    h[oth] = (np.sin(np.pi * to * (1 - beta)) + 4 * beta * to * np.cos(np.pi * to * (1 + beta))) / (
        np.pi * to * (1 - (4 * beta * to) ** 2)
    )
    return h / np.sqrt(np.sum(h**2))


def matched_filter(x: np.ndarray, fs: float, baud: float, kind: str) -> np.ndarray:
    sps = fs / baud
    if kind == "none":
        return x
    if kind == "box":
        L = max(1, int(round(sps)))
        return fftconvolve(x, np.ones(L) / L, mode="same")
    if kind.startswith("rrc"):
        beta = float(kind[3:])
        return fftconvolve(x, rrc_taps(beta, sps), mode="same")
    raise ValueError(kind)


def _upsample(z: np.ndarray, sps: float):
    U = 1 if sps >= 16 else int(np.ceil(16.0 / sps))
    if U > 1:
        z = resample_poly(z, U, 1)
    return z, U


def timing_sample(vals: np.ndarray, metric: np.ndarray, Tu: float, block: int = 256):
    """Block-wise tracking of the sampling phase (maximum-metric criterion).

    vals   : complex or real sequence to sample (already matched filtered / upsampled)
    metric : non-negative sequence whose mean is maximal at the ideal sampling instants
    Tu     : (fractional) samples per symbol of `vals`
    Returns (samples, positions).
    """
    N = len(vals)
    nsym = int((N - 2 - 2 * Tu) // Tu)
    if nsym < 8:
        raise ValueError("signal too short for the estimated symbol rate")
    idx = np.arange(N)
    k = np.arange(nsym)
    nb = int(np.ceil(nsym / block))
    pos = np.empty(nsym)
    delta = None

    def energy(cands, ks):
        T = cands[:, None] + ks[None, :] * Tu
        T = np.clip(T, 0, N - 1)
        return np.interp(T.ravel(), idx, metric).reshape(T.shape).mean(axis=1)

    for b in range(nb):
        ks = k[b * block : (b + 1) * block]
        if delta is None:
            ks_i = k[: min(512, nsym)]
            cands = np.linspace(0, Tu, 33)[:-1]
            E = energy(cands, ks_i)
            Es = np.convolve(np.r_[E[-2:], E, E[:2]], np.ones(5) / 5, mode="valid")
            j = int(np.argmax(Es))
            step = Tu / 32.0
            c2 = cands[j] + np.linspace(-step, step, 9)
            E2 = energy(c2, ks_i)
            delta = float(c2[int(np.argmax(E2))])
        else:
            cands = delta + np.linspace(-Tu / 3.0, Tu / 3.0, 17)
            E = energy(cands, ks)
            Es = np.convolve(E, np.ones(3) / 3, mode="same")
            Es[0], Es[-1] = E[0], E[-1]
            j = int(np.argmax(Es))
            step = (2 * Tu / 3.0) / 16.0
            c2 = cands[j] + np.linspace(-step, step, 7)
            E2 = energy(c2, ks)
            delta = float(c2[int(np.argmax(E2))])
        pos[b * block : b * block + len(ks)] = delta + ks * Tu
    keep = pos <= N - 1
    pos = pos[keep]
    if np.iscomplexobj(vals):
        s = np.interp(pos, idx, vals.real) + 1j * np.interp(pos, idx, vals.imag)
    else:
        s = np.interp(pos, idx, vals)
    return s, pos


# --------------------------------------------------------------------------- #
# constellations / demapping
# --------------------------------------------------------------------------- #
_Q16 = np.sqrt(10.0)
_GRAY_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0]) / _Q16
_GRAY_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)  # per level: [MSB, LSB]


def _norm_mod(mod: str) -> str:
    m = mod.upper()
    if "FSK" in m:
        return "FSK"
    if "QAM" in m:
        return "16-QAM"
    if "QPSK" in m:
        return "QPSK"
    if "BPSK" in m:
        return "BPSK"
    return "QPSK"


def nearest_points(y: np.ndarray, mod: str) -> np.ndarray:
    mod = _norm_mod(mod)
    if mod == "BPSK":
        return np.where(y.real >= 0, 1.0, -1.0).astype(np.complex128)
    if mod == "QPSK":
        return (np.where(y.real >= 0, 1.0, -1.0) + 1j * np.where(y.imag >= 0, 1.0, -1.0)) / np.sqrt(2.0)
    di = _GRAY_LEVELS[np.argmin(np.abs(y.real[:, None] - _GRAY_LEVELS[None, :]), axis=1)]
    dq = _GRAY_LEVELS[np.argmin(np.abs(y.imag[:, None] - _GRAY_LEVELS[None, :]), axis=1)]
    return di + 1j * dq


def evm_stats(y: np.ndarray, mod: str) -> dict:
    d = nearest_points(y, mod)
    mse = float(np.mean(np.abs(y - d) ** 2)) + 1e-12
    snr = 10 * np.log10(float(np.mean(np.abs(d) ** 2)) / mse)
    return {"mse": mse, "evm_pct": 100 * np.sqrt(mse / float(np.mean(np.abs(d) ** 2))), "snr_db": float(snr)}


def _axis_llr_16(r: np.ndarray, sigma2: float):
    """Max-log LLR for the two Gray bits of one 16-QAM axis. Returns (msb_llr, lsb_llr)."""
    d2 = (r[:, None] - _GRAY_LEVELS[None, :]) ** 2  # (n,4)
    out = []
    for bit in (0, 1):
        m1 = np.min(d2[:, _GRAY_BITS[:, bit] == 1], axis=1)
        m0 = np.min(d2[:, _GRAY_BITS[:, bit] == 0], axis=1)
        out.append((m0 - m1) / (2.0 * sigma2))
    return out[0], out[1]


def demap(y: np.ndarray, mod: str, rot: int = 0, conj: bool = False):
    """Symbols -> (hard bits uint8, LLR float). rot = k means y*(1j)**k; conj applies first."""
    mod = _norm_mod(mod)
    y = np.asarray(y, dtype=np.complex128)
    if conj:
        y = np.conj(y)
    if rot:
        y = y * (1j) ** (rot % 4)
    d = nearest_points(y, mod)
    mse = float(np.mean(np.abs(y - d) ** 2))
    s2 = max(mse / 2.0, 1e-4)  # noise variance per real dimension
    if mod == "BPSK":
        llr = 2.0 * y.real / max(mse / 2.0, 1e-4)
        bits = (y.real > 0).astype(np.uint8)
    elif mod == "QPSK":
        a = 1.0 / np.sqrt(2.0)
        bits = np.empty(2 * len(y), dtype=np.uint8)
        llr = np.empty(2 * len(y))
        bits[0::2] = y.real > 0
        bits[1::2] = y.imag > 0
        llr[0::2] = 2 * a * y.real / s2
        llr[1::2] = 2 * a * y.imag / s2
    else:  # 16-QAM
        bits = np.empty(4 * len(y), dtype=np.uint8)
        llr = np.empty(4 * len(y))
        li1, li0 = _axis_llr_16(y.real, s2)
        lq1, lq0 = _axis_llr_16(y.imag, s2)
        # max-log returns (m0-m1)/(2s2) which is positive when bit==1 is closer
        for j, l in enumerate((li1, li0, lq1, lq0)):
            llr[j::4] = l
        for j, (axis, bitpos) in enumerate(((y.real, 0), (y.real, 1), (y.imag, 0), (y.imag, 1))):
            lev = _GRAY_LEVELS[np.argmin(np.abs(axis[:, None] - _GRAY_LEVELS[None, :]), axis=1)]
            li = np.searchsorted(_GRAY_LEVELS, lev - 1e-9)
            bits[j::4] = _GRAY_BITS[np.clip(li, 0, 3), bitpos]
    return bits, np.clip(llr, -LLR_CLIP, LLR_CLIP)


def map_bits(bits: np.ndarray, mod: str) -> np.ndarray:
    """Bits -> unit-power symbols (the exact inverse of demap with rot=0, conj=False)."""
    mod = _norm_mod(mod)
    b = np.asarray(bits, dtype=np.uint8)
    if mod == "BPSK":
        return (2.0 * b - 1.0).astype(np.complex128)
    if mod == "QPSK":
        n = (len(b) // 2) * 2
        return ((2.0 * b[0:n:2] - 1) + 1j * (2.0 * b[1:n:2] - 1)) / np.sqrt(2.0)
    n = (len(b) // 4) * 4
    b = b[:n].reshape(-1, 4)
    lut = {tuple(r): i for i, r in enumerate(_GRAY_BITS)}
    I = np.array([_GRAY_LEVELS[lut[(r[0], r[1])]] for r in b])
    Q = np.array([_GRAY_LEVELS[lut[(r[2], r[3])]] for r in b])
    return I + 1j * Q


# --------------------------------------------------------------------------- #
# carrier recovery
# --------------------------------------------------------------------------- #
def carrier_recover(y: np.ndarray, mod: str, win: int = 128):
    """Symbol-rate CFO removal (M-th power FFT) + Viterbi&Viterbi phase tracking.

    Returns (corrected symbols with unit RMS, residual CFO in cycles/symbol).
    The M-fold phase ambiguity is NOT resolved here (it is resolved by frame sync).
    """
    mod = _norm_mod(mod)
    M = 2 if mod == "BPSK" else 4
    off = 0.0 if mod == "BPSK" else np.pi
    y = np.asarray(y, dtype=np.complex128)
    y = y / (np.sqrt(np.mean(np.abs(y) ** 2)) + 1e-12)
    n = len(y)
    yM = y**M
    nfft = int(2 ** np.ceil(np.log2(max(n, 64) * 8)))
    S = np.abs(np.fft.fft(yM * np.hanning(n), nfft))
    k = int(np.argmax(S))
    a, b, c = np.log(S[(k - 1) % nfft] + 1e-300), np.log(S[k] + 1e-300), np.log(S[(k + 1) % nfft] + 1e-300)
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if den != 0 else 0.0
    f = (k + d) / nfft
    if f > 0.5:
        f -= 1.0
    cfo = f / M
    y = y * np.exp(-2j * np.pi * cfo * np.arange(n))
    w = int(max(8, min(win, n // 4)))
    s = np.convolve(y**M, np.ones(w), mode="same")
    ph = np.unwrap(np.angle(s))
    phi = (ph - off) / M
    y = y * np.exp(-1j * phi)
    y = y / (np.sqrt(np.mean(np.abs(y) ** 2)) + 1e-12)
    return y, float(cfo)


# --------------------------------------------------------------------------- #
# receivers
# --------------------------------------------------------------------------- #
FILTER_CANDIDATES = ("none", "box", "rrc0.35", "rrc0.25", "rrc0.5")


def _front_end(x: np.ndarray, fs: float, baud: float, kind: str):
    sps = fs / baud
    z = matched_filter(x, fs, baud, kind)
    zu, U = _upsample(z, sps)
    pw = zu.real**2 + zu.imag**2
    sym, pos = timing_sample(zu, pw, sps * U)
    return sym, pos / U


def recover_linear(x: np.ndarray, fs: float, mod: str, baud: float, filters=FILTER_CANDIDATES) -> dict:
    mod = _norm_mod(mod)
    x = np.asarray(x[:MAX_PROC], dtype=np.complex128)
    be = band_estimate(x, fs)
    fc = 0.0 if be["wideband"] else be["fc"]
    n = np.arange(len(x))
    x0 = x * np.exp(-2j * np.pi * fc * n / fs)
    sel = x0[:SEL_N]
    best = None
    for kind in filters:
        try:
            sym, _ = _front_end(sel, fs, baud, kind)
            y, _ = carrier_recover(sym, mod)
            st = evm_stats(y, mod)
        except Exception:
            continue
        if best is None or st["mse"] < best[0]:
            best = (st["mse"], kind)
    if best is None:
        raise RuntimeError("receiver could not lock (try giving samples/symbol manually)")
    kind = best[1]
    sym, pos = _front_end(x0, fs, baud, kind)
    y, cfo_sym = carrier_recover(sym, mod)
    st = evm_stats(y, mod)
    return {
        "mod": mod,
        "baud": float(baud),
        "sps": fs / baud,
        "filter": kind,
        "symbols": y,
        "positions": pos,
        "cfo_hz": float(fc + cfo_sym * baud),
        "evm_pct": st["evm_pct"],
        "snr_db": st["snr_db"],
        "mse": st["mse"],
        "locked": bool(st["snr_db"] > 3.0),
        "bw_hz": be["bw"],
    }


def _two_means(v: np.ndarray, iters: int = 30):
    c0, c1 = np.percentile(v, 10), np.percentile(v, 90)
    for _ in range(iters):
        thr = 0.5 * (c0 + c1)
        lo, hi = v[v <= thr], v[v > thr]
        if len(lo) == 0 or len(hi) == 0:
            break
        n0, n1 = lo.mean(), hi.mean()
        if abs(n0 - c0) + abs(n1 - c1) < 1e-9 * (abs(c0) + abs(c1) + 1):
            break
        c0, c1 = n0, n1
    thr = 0.5 * (c0 + c1)
    lo, hi = v[v <= thr], v[v > thr]
    s0 = lo.std() if len(lo) > 1 else 1e-9
    s1 = hi.std() if len(hi) > 1 else 1e-9
    sep = (c1 - c0) / (0.5 * (s0 + s1) + 1e-12)
    mass = min(len(lo), len(hi)) / len(v)
    return float(c0), float(c1), float(sep), float(mass), float(max(s0, s1))


def _inst_freq(x: np.ndarray, fs: float, smooth: int) -> np.ndarray:
    f = np.angle(x[1:] * np.conj(x[:-1])) * fs / (2 * np.pi)
    if smooth > 1:
        f = uniform_filter1d(f, size=smooth, mode="nearest")
    return f


def recover_fsk(x: np.ndarray, fs: float, baud: float) -> dict:
    x = np.asarray(x[:MAX_PROC], dtype=np.complex128)
    sps = fs / baud
    fl = _inst_freq(x, fs, max(1, int(sps / 3)))
    c0, c1, sep, mass, sd = _two_means(fl)
    thr = 0.5 * (c0 + c1)
    metric = (fl - thr) ** 2
    vals, pos = timing_sample(fl, metric, sps)
    bits = (vals > thr).astype(np.uint8)
    sigma2 = max(sd**2, 1e-6)
    llr = np.clip((c1 - c0) * (vals - thr) / sigma2, -LLR_CLIP, LLR_CLIP)
    return {
        "mod": "FSK",
        "baud": float(baud),
        "sps": sps,
        "tones_hz": (c0, c1),
        "centre_hz": thr,
        "deviation_h": (c1 - c0) / baud,
        "separation": sep,
        "bits": bits,
        "llr": llr,
        "snr_db": float(10 * np.log10(max(((c1 - c0) / 2) ** 2 / sigma2, 1e-3))),
        "locked": bool(sep > 3.0 and mass > 0.1),
        "positions": pos,
    }


# --------------------------------------------------------------------------- #
# modulation classification
# --------------------------------------------------------------------------- #
def _line_prominence(y: np.ndarray, M: int) -> float:
    n = len(y)
    if n < 64:
        return 0.0
    nfft = int(2 ** np.ceil(np.log2(n * 4)))
    S = np.abs(np.fft.fft((y**M) * np.hanning(n), nfft)) ** 2
    return float(S.max() / (np.median(S) + 1e-30))


def _build_qpsk_to_qam_table():
    """Expected 16-QAM-lattice residual for QPSK data at a given QPSK residual (calibration table)."""
    from scipy.special import ndtri

    u = (np.arange(4000) + 0.5) / 4000.0
    z = ndtri(u)
    a = 1.0 / np.sqrt(2.0)
    mq = np.logspace(-3.5, 0.3, 60)
    g = []
    for m in mq:
        sd = np.sqrt(m / 2.0)
        r = a + sd * z
        d2 = np.min((r[:, None] - _GRAY_LEVELS[None, :]) ** 2, axis=1)
        g.append(2.0 * d2.mean())
    return np.log(mq), np.log(np.array(g))


_QT_X, _QT_Y = _build_qpsk_to_qam_table()


def _qam_residual_if_qpsk(m_q: float) -> float:
    return float(np.exp(np.interp(np.log(max(m_q, 1e-4)), _QT_X, _QT_Y)))


def classify_modulation(x: np.ndarray, fs: float, baud: float | None = None) -> dict:
    x = np.asarray(x[:SEL_N * 2], dtype=np.complex128)
    if baud is None:
        be = estimate_baud(x, fs)
        baud = be["baud"]
    if baud is None or fs / baud < 2.0:
        return {"label": "Unknown", "confidence": 0.0, "baud": baud, "detail": "no cyclic symbol-rate line found"}
    # --- FSK test (bimodal instantaneous frequency) ---
    sps = fs / baud
    fl = _inst_freq(x, fs, max(1, int(sps / 3)))
    c0, c1, sep, mass, _ = _two_means(fl)
    env = np.abs(x)
    cv = float(env.std() / (env.mean() + 1e-12))
    if sep > 5.0 and mass > 0.15 and cv < 0.5:
        conf = float(np.clip((sep - 5.0) / 10.0 + 0.6, 0.0, 1.0))
        return {"label": "2-FSK", "confidence": conf, "baud": baud, "detail": f"tone separation {sep:.1f} sigma, tones {c0:.0f}/{c1:.0f} Hz"}
    # --- linear modulations ---
    be = band_estimate(x, fs)
    fc = 0.0 if be["wideband"] else be["fc"]
    x0 = x * np.exp(-2j * np.pi * fc * np.arange(len(x)) / fs)
    best = None
    for kind in ("none", "box", "rrc0.35"):
        try:
            sym, _ = _front_end(x0, fs, baud, kind)
        except Exception:
            continue
        y = sym / (np.sqrt(np.mean(np.abs(sym) ** 2)) + 1e-12)
        L = {M: _line_prominence(y, M) for M in (2, 4, 8)}
        score = max(L.values())
        if best is None or score > best[0]:
            best = (score, L, y, kind)
    if best is None:
        return {"label": "Unknown", "confidence": 0.0, "baud": baud, "detail": "timing recovery failed"}
    _, L, y, kind = best
    thr = float(np.clip(0.04 * len(y) + 12.0, 20.0, 60.0))
    bpsk_ok = False
    if L[2] > thr and L[2] > 0.5 * L[4]:
        yb, _ = carrier_recover(y, "BPSK")
        bpsk_ok = float(np.mean(yb.imag**2) / (np.mean(yb.real**2) + 1e-12)) < 0.2  # BPSK has no quadrature energy
    if bpsk_ok:
        conf = float(np.clip(np.log10(L[2] / 12.0) / 2.0, 0.0, 1.0))
        return {"label": "BPSK", "confidence": conf, "baud": baud, "detail": f"2nd-power line {L[2]:.0f}x"}
    if L[4] > thr:
        yq, _ = carrier_recover(y, "QPSK")
        m_q = evm_stats(yq, "QPSK")["mse"]
        m_16 = evm_stats(yq, "16-QAM")["mse"]
        g = _qam_residual_if_qpsk(m_q)
        if len(yq) >= 300 and m_16 < 0.8 * g:
            conf = float(np.clip(1.0 - m_16 / g, 0.2, 1.0))
            return {"label": "16-QAM", "confidence": conf, "baud": baud, "detail": f"4th-power line {L[4]:.0f}x, 3-ring structure (residual {m_16:.3f} vs {g:.3f} expected for QPSK)"}
        conf = float(np.clip(np.log10(L[4] / 12.0) / 2.0, 0.0, 1.0))
        return {"label": "QPSK", "confidence": conf, "baud": baud, "detail": f"4th-power line {L[4]:.0f}x"}
    if L[8] > thr:
        conf = float(np.clip(np.log10(L[8] / 12.0) / 2.0, 0.0, 1.0))
        return {"label": "8-PSK (detected; demodulator not available)", "confidence": conf, "baud": baud, "detail": f"8th-power line {L[8]:.0f}x"}
    return {"label": "Unknown", "confidence": 0.1, "baud": baud, "detail": "no M-th power line (noise-like, or an unsupported scheme)"}


# --------------------------------------------------------------------------- #
# top-level receive()
# --------------------------------------------------------------------------- #
@dataclass
class RxResult:
    mod: str
    baud: float
    sps: float
    rec: dict
    burst: tuple
    cls: dict | None
    variants: list = field(default_factory=list)  # list of dicts: name, rot, conj, bits, llr


def make_variants(rec: dict) -> list:
    mod = rec["mod"]
    out = []
    if mod == "FSK":
        b, l = rec["bits"], rec["llr"]
        out.append({"name": "normal", "rot": 0, "conj": False, "bits": b, "llr": l})
        out.append({"name": "inverted tones", "rot": 0, "conj": False, "bits": 1 - b, "llr": -l})
        return out
    y = rec["symbols"]
    if mod == "BPSK":
        combos = [(0, False), (2, False)]
    else:
        combos = [(r, c) for c in (False, True) for r in range(4)]
    for rot, conj in combos:
        b, l = demap(y, mod, rot, conj)
        name = f"rot {90 * rot} deg" + (" + conj" if conj else "")
        out.append({"name": name, "rot": rot, "conj": conj, "bits": b, "llr": l})
    return out


def center_and_filter(x: np.ndarray, fs: float):
    """Mix the occupied band to DC and low-pass it (removes out-of-band noise). Returns (x', fc, band_info)."""
    be = band_estimate(x, fs)
    if be["wideband"] or len(x) < 1024:
        return x, 0.0, be
    fc = be["fc"]
    xm = x * np.exp(-2j * np.pi * fc * np.arange(len(x)) / fs)
    cutoff = float(np.clip(0.5 * be["bw"] * 1.3, fs * 0.002, fs * 0.48))
    ntap = int(np.clip(3.3 * fs / (0.5 * cutoff), 65, 2047)) | 1
    h = sp.firwin(ntap, cutoff, fs=fs)
    return fftconvolve(xm, h, mode="same"), float(fc), be


def receive(x: np.ndarray, fs: float, mod: str | None = None, baud: float | None = None, sps: float | None = None) -> RxResult:
    x = np.asarray(x, dtype=np.complex128)
    xt, burst, trimmed = trim_to_burst(x, fs)
    xt, fc0, band0 = center_and_filter(xt, fs)
    if sps is not None and sps > 0:
        baud = fs / float(sps)
    est = None
    if baud is None:
        est = estimate_baud(xt, fs)
        baud = est["baud"]
    if baud is None:
        raise RuntimeError("could not estimate the symbol rate (no cyclic line); enter samples/symbol manually")
    cls = None
    if mod is None or str(mod).lower().startswith("auto"):
        cls = classify_modulation(xt, fs, baud)
        mod = cls["label"]
        if mod.startswith("Unknown") or "8-PSK" in mod:
            raise RuntimeError(f"modulation not supported / not recognised: {mod}")
    mod = "FSK" if "FSK" in str(mod).upper() else _norm_mod(mod)
    rec = recover_fsk(xt, fs, baud) if mod == "FSK" else recover_linear(xt, fs, mod, baud)
    if "cfo_hz" in rec:
        rec["cfo_hz"] += fc0
    if mod == "FSK":
        rec["centre_hz"] += fc0
        rec["tones_hz"] = (rec["tones_hz"][0] + fc0, rec["tones_hz"][1] + fc0)
    rec["bw_hz"] = band0["bw"]
    rec["band_centre_hz"] = fc0
    res = RxResult(mod=mod, baud=float(baud), sps=fs / baud, rec=rec, burst=burst, cls=cls)
    res.variants = make_variants(rec)
    return res
