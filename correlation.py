"""Bit-stream correlation: sync-word / preamble search, frame parsing, structure statistics."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

# Well-known synchronisation markers (searched in both polarities)
SYNC_LIBRARY = {
    "CCSDS ASM (1ACFFC1D)": "1ACFFC1D",
    "HDLC/AX.25 flag (7E)": "7E7E",
    "GSM-style / generic alt (AAAA)": "AAAAAAAA",
    "Barker-13 (1F35)": "1F35",
    "Iridium-like UW (789)": "0789",
}


def pattern_from_hex(hex_str: str) -> np.ndarray:
    h = hex_str.strip().replace(" ", "").replace("0x", "").replace("0X", "")
    if not h:
        raise ValueError("Empty hex pattern")
    if len(h) % 2:
        h = "0" + h
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)).astype(np.uint8)


def mismatch_profile(stream: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Number of mismatching bits for every alignment (fast FFT correlation)."""
    stream = np.asarray(stream, dtype=np.uint8)
    pattern = np.asarray(pattern, dtype=np.uint8)
    m = len(pattern)
    if len(stream) < m:
        return np.zeros(0)
    a = 2.0 * stream - 1.0
    b = 2.0 * pattern[::-1] - 1.0
    corr = fftconvolve(a, b, mode="valid")
    return np.rint((m - corr) / 2.0)


def correlate_bits(stream: np.ndarray, pattern: np.ndarray, max_err: int | None = None):
    """Return (mismatch_scores, hit_positions). A hit is an alignment with <= max_err mismatches."""
    scores = mismatch_profile(stream, pattern)
    if len(scores) == 0:
        return scores, []
    if max_err is None:
        max_err = max(1, len(pattern) // 8)
    hits = [int(i) for i in np.flatnonzero(scores <= max_err)]
    return scores, hits


def search_sync(stream: np.ndarray, patterns: dict[str, np.ndarray] | None = None, max_err_frac: float = 0.1):
    """Search every library pattern (normal + inverted). Returns best-first list of dicts.

    The allowed mismatch count is lowered automatically until the expected number of
    false alarms on random data of this length is below 0.05, so a reported hit is
    statistically significant (short markers therefore need an exact match).
    """
    from scipy.stats import binom

    if patterns is None:
        patterns = {k: pattern_from_hex(v) for k, v in SYNC_LIBRARY.items()}
    stream = np.asarray(stream, dtype=np.uint8)
    found = []
    for name, pat in patterns.items():
        m = len(pat)
        positions = max(1, len(stream) - m + 1)
        max_err = int(m * max_err_frac)
        while max_err >= 0 and 2 * positions * binom.cdf(max_err, m, 0.5) > 0.05:
            max_err -= 1
        if max_err < 0:
            continue
        for inv in (False, True):
            p = (1 - pat) if inv else pat
            scores = mismatch_profile(stream, p)
            if len(scores) == 0:
                continue
            idx = np.flatnonzero(scores <= max_err)
            if len(idx) == 0:
                continue
            best = int(idx[np.argmin(scores[idx])])
            found.append(
                {
                    "pattern": name,
                    "inverted": inv,
                    "position": best,
                    "mismatches": int(scores[best]),
                    "length": m,
                    "n_hits": int(len(idx)),
                    "positions": [int(i) for i in idx[:50]],
                }
            )
    found.sort(key=lambda d: (d["mismatches"] / d["length"], -d["length"], d["position"]))
    return found


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def bits_to_bytes(bits: np.ndarray, max_bytes: int | None = None) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    n = (len(bits) // 8) * 8
    if max_bytes is not None:
        n = min(n, max_bytes * 8)
    return np.packbits(bits[:n]).tobytes()


def parse_frame(bits: np.ndarray) -> dict | None:
    """Parse   [length:16][payload:length bytes][CRC-16-CCITT:16]   from the start of `bits`.

    Returns a dict (valid flag, payload bytes ...) or None if the stream is too short / length absurd.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) < 16 + 16 + 8:
        return None
    length = int("".join(map(str, bits[:16])), 2)
    need = 16 + 8 * length + 16
    if length == 0 or need > len(bits):
        return None
    data = bits_to_bytes(bits[16 : 16 + 8 * length])
    rx_crc = int("".join(map(str, bits[16 + 8 * length : need])), 2)
    calc = crc16_ccitt(data)
    return {
        "length": length,
        "payload": data,
        "crc_rx": rx_crc,
        "crc_calc": calc,
        "crc_ok": rx_crc == calc,
        "frame_bits": need,
    }


def find_frames(bits: np.ndarray, max_frames: int = 8):
    """Brute-force search for CRC-valid frames at any bit offset (used when no sync word exists)."""
    bits = np.asarray(bits, dtype=np.uint8)
    out = []
    # cheap pre-filter: lengths 1..512 bytes -> test every offset
    for off in range(0, max(0, len(bits) - 56)):
        length = int("".join(map(str, bits[off : off + 16])), 2)
        if length == 0 or length > 512:
            continue
        need = 16 + 8 * length + 16
        if off + need > len(bits):
            continue
        fr = parse_frame(bits[off : off + need])
        if fr and fr["crc_ok"]:
            fr["offset"] = off
            out.append(fr)
            if len(out) >= max_frames:
                break
    return out


def hex_ascii(data: bytes, max_bytes: int = 32) -> tuple[str, str]:
    d = data[:max_bytes]
    if not d:
        return "(insufficient bits)", "(insufficient bits)"
    return " ".join(f"{b:02X}" for b in d), "".join(chr(b) if 32 <= b < 127 else "." for b in d)


def format_bit_preview(bits: np.ndarray, max_bits: int = 128) -> str:
    chunk = np.asarray(bits)[:max_bits]
    s = "".join(str(int(b)) for b in chunk)
    if len(bits) > max_bits:
        s += f"... ({len(bits)} total)"
    return s


def bit_entropy(bits: np.ndarray, word: int = 8) -> float:
    """Shannon entropy per byte-word, normalised to 0..1 (1.0 = indistinguishable from random)."""
    bits = np.asarray(bits, dtype=np.uint8)
    n = (len(bits) // word) * word
    if n < word * 16:
        return 1.0
    vals = np.packbits(bits[:n]).astype(np.int64) if word == 8 else None
    if vals is None:
        return 1.0
    p = np.bincount(vals, minlength=256).astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum() / 8.0)


def estimate_frame_period(bits: np.ndarray, max_lag: int = 4096):
    """Strongest periodicity in the bit-stream autocorrelation -> (lag, strength) or (None, 0)."""
    b = np.asarray(bits, dtype=float) * 2 - 1
    n = len(b)
    if n < 256:
        return None, 0.0
    max_lag = min(max_lag, n // 2)
    f = np.fft.rfft(b, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:max_lag]
    ac = ac / ac[0]
    lag = int(np.argmax(ac[16:]) + 16)
    strength = float(ac[lag])
    thr = 6.0 / np.sqrt(n)
    return (lag, strength) if strength > thr * 2 else (None, strength)
