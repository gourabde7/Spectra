"""Ground-truth demo signal generator.

Frame:  [ ASM 1ACFFC1D (uncoded) ][ interleaver( FEC( [len:16][payload][CRC16] ) ) ]
Every file has realistic impairments: RRC pulse shaping, carrier offset, random phase,
fractional timing offset, AWGN, and noise-only lead-in / lead-out.

Run:  python demo_signals.py          (writes demo files next to this script)
"""

from __future__ import annotations

import json
import os
import struct
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

import correlation as C
import deinterleave as DI
import dsp_core as D
import fec as F

ASM = C.pattern_from_hex("1ACFFC1D")
LONG_MESSAGE = " ".join(f"[{i:02d}] SPECTRA 16-QAM link, report {(i * 7919) % 1000:03d}: {chr(65 + (i * 11) % 26) * (3 + i % 5)} ok." for i in range(9))
MESSAGE = "SPECTRA DEMO: COMINT workbench decoded this message end-to-end. CRC verified."


def build_frame(msg: str) -> np.ndarray:
    data = msg.encode("ascii")
    raw = struct.pack(">H", len(data)) + data + struct.pack(">H", C.crc16_ccitt(data))
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


def _pad223(raw: bytes) -> bytes:
    """Fill the RS information field with pseudo-random bytes (like a CCSDS randomiser would)."""
    fill = np.random.default_rng(99).integers(0, 256, 223 - len(raw)).astype(np.uint8).tobytes()
    return raw + fill


def encode_payload(frame_bits, fec="conv", inter="Block", **kw):
    b = frame_bits
    if fec == "conv":
        b = F.conv_encode(b, terminate=True)
    elif fec == "rs":
        data = _pad223(np.packbits(b).tobytes())
        b = F.bytes_to_bits(F.rs_encode(data))
    elif fec == "concat":
        data = _pad223(np.packbits(b).tobytes())
        b = F.conv_encode(F.bytes_to_bits(F.rs_encode(data)), terminate=True)
    if inter != "None":
        blk = kw.get("rows", 16) * kw.get("cols", 16)
        if inter in ("Block", "Diagonal"):
            b = np.concatenate([b, np.zeros((-len(b)) % blk, dtype=np.uint8)])
        b = DI.interleave(b, inter, **kw)
    return b.astype(np.uint8)


def _shape_linear(bits, mod, fs, baud, beta=0.35):
    syms = D.map_bits(bits, mod)
    K = 16
    up = np.zeros(len(syms) * K, dtype=complex)
    up[::K] = syms
    w = np.convolve(up, D.rrc_taps(beta, K, span=10), mode="same")
    fr = Fraction(fs / baud / K).limit_denominator(4000)
    return resample_poly(w, fr.numerator, fr.denominator)


def make_signal(bits, mod, fs, baud, snr_db=14.0, cfo=0.0, seed=1, lead=2500, trail=2000, f0=None, f1=None):
    rng = np.random.default_rng(seed)
    sps = fs / baud
    if mod == "FSK":
        n = int(len(bits) * sps)
        idx = np.clip(np.floor(np.arange(n) / sps).astype(int), 0, len(bits) - 1)
        f = np.where(bits[idx] == 1, f1, f0).astype(float)
        w = np.exp(2j * np.pi * np.cumsum(f) / fs)
        sig2 = 10 ** (-snr_db / 10)
    else:
        w = _shape_linear(bits, mod, fs, baud)
        w = w / np.sqrt(np.mean(np.abs(w) ** 2)) * np.sqrt(1.0 / sps)
        sig2 = 10 ** (-snr_db / 10)
    nn = np.arange(len(w))
    w = w * np.exp(1j * (2 * np.pi * cfo * nn / fs + rng.uniform(0, 2 * np.pi)))
    noise = lambda m: (rng.normal(size=m) + 1j * rng.normal(size=m)) * np.sqrt(sig2 / 2)
    return np.concatenate([noise(lead), w + noise(len(w)), noise(trail)])


def write_wav(path, x, fs):
    import soundfile as sf

    pk = np.max(np.abs(np.r_[x.real, x.imag])) * 1.05
    sf.write(path, np.column_stack([x.real / pk, x.imag / pk]).astype(np.float32), int(fs), subtype="FLOAT")


def write_iq(path, x):
    pk = np.max(np.abs(np.r_[x.real, x.imag])) * 1.05
    z = np.empty(2 * len(x), dtype=np.float32)
    z[0::2], z[1::2] = x.real / pk, x.imag / pk
    z.tofile(path)


# name: (mod, fec, interleaver, kw, fs, baud, snr, cfo, container)
CATALOG = {
    "demo": ("QPSK", "conv", "Block", {"rows": 16, "cols": 16}, 48000.0, 2400.0, 14.0, 137.0, "wav"),
    "demo_bpsk_rs": ("BPSK", "rs", "None", {}, 48000.0, 1200.0, 12.0, -220.0, "wav"),
    "demo_qpsk_concat": ("QPSK", "concat", "Convolutional", {"depth": 12, "unit": 1}, 250000.0, 20000.0, 13.0, 1800.0, "iq"),
    "demo_fsk": ("FSK", "conv", "Diagonal", {"rows": 8, "cols": 32}, 48000.0, 1200.0, 14.0, 0.0, "wav"),
    "demo_16qam_uncoded": ("16-QAM", "none", "None", {}, 48000.0, 2400.0, 26.0, -90.0, "wav"),
}


def generate_all(folder: str = ".", seed: int = 7) -> dict:
    os.makedirs(folder, exist_ok=True)
    truth_all = {}
    for i, (name, (mod, fec, inter, kw, fs, baud, snr, cfo, box)) in enumerate(CATALOG.items()):
        msg = LONG_MESSAGE if mod == "16-QAM" else MESSAGE
        frame = build_frame(msg)
        body = encode_payload(frame, fec, inter, **kw)
        bits = np.concatenate([ASM, body]).astype(np.uint8)
        pad = (-len(bits)) % 4
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        x = make_signal(bits, mod, fs, baud, snr, cfo, seed + i, f0=-1000.0 if mod == "FSK" else None, f1=1000.0 if mod == "FSK" else None)
        if box == "iq":
            path = os.path.join(folder, name + ".iq")
            write_iq(path, x)
        else:
            if mod == "FSK":
                x = x * 1.0
            path = os.path.join(folder, name + ".wav")
            write_wav(path, x, fs)
        truth = {"file": os.path.basename(path), "modulation": mod, "fec": fec, "interleaver": inter, "interleaver_params": kw,
                 "sample_rate": fs, "baud": baud, "snr_db": snr, "cfo_hz": cfo, "iq_format": "cf32" if box == "iq" else None,
                 "message": msg}
        with open(os.path.join(folder, name + ".truth.json"), "w") as fh:
            json.dump(truth, fh, indent=2)
        truth_all[name] = truth
    return truth_all


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    t = generate_all(here)
    for k, v in t.items():
        print(f"{v['file']:26s} {v['modulation']:7s} fec={v['fec']:6s} interleaver={v['interleaver']:14s} fs={v['sample_rate']:.0f} baud={v['baud']:.0f}")
