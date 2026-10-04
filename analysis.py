"""Signal analysis + end-to-end receive chain used by the GUI.

analyze_signal() runs the complete blind chain:
    burst trim -> baud/modulation estimation -> timing+carrier recovery -> soft demap ->
    frame-sync search (all 4/8 phase ambiguities) -> blind interleaver/FEC detection ->
    decode -> CRC-checked frame parse
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp

import blind as BL
import correlation as C
import dsp_core as D
import fec as F
from demod import demodulate_full

# ------------------------------------------------------------------ plots ----
def compute_spectrum(samples, fs, nfft: int = 2048):
    f, P = D.psd_welch(samples, fs, nperseg=nfft)
    return f, 10 * np.log10(P + 1e-20)


def compute_waterfall(samples, fs, nperseg: int = 256):
    x = np.asarray(samples, dtype=np.complex128)
    nper = int(min(nperseg, max(16, len(x) // 8)))
    f, t, S = sp.spectrogram(x, fs=fs, nperseg=nper, noverlap=nper // 2, return_onesided=False, mode="psd")
    f = np.fft.fftshift(f)
    S = np.fft.fftshift(S, axes=0)
    return f, t, 10 * np.log10(S + 1e-20)


def constellation_points(samples, max_points: int = 3000, symbols=None):
    pts = np.asarray(symbols if symbols is not None else samples)
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
    return pts


# ------------------------------------------------------------ receive chain ---
def sync_stage(rx: D.RxResult):
    """Search all phase-ambiguity variants for a known sync word. Returns dict."""
    best = None
    for v in rx.variants:
        hits = C.search_sync(v["bits"])
        if hits:
            h = hits[0]
            key = (h["mismatches"] / h["length"], -h["length"])
            if best is None or key < best[0]:
                best = (key, v, h)
    if best is None:
        v = rx.variants[0]
        return {"found": False, "variant": v, "bits": v["bits"], "llr": v["llr"], "hit": None}
    _, v, h = best
    start = h["position"] + h["length"]
    bits, llr = v["bits"], v["llr"]
    if h["inverted"]:
        bits, llr = 1 - bits, -llr
    return {"found": True, "variant": v, "hit": h, "bits": bits[start:], "llr": llr[start:], "raw_bits": bits, "start": start}


def analyze_signal(samples, fs, mod=None, sps=None, baud=None) -> dict:
    out = {"error": None, "rx": None, "sync": None, "chain": None, "decoded_bits": None, "fec_info": None, "frame": None}
    x = np.asarray(samples, dtype=np.complex128)
    be = D.band_estimate(x[: D.SEL_N * 4], fs)
    out["band"] = be
    try:
        rx = demodulate_full(x, fs, mod=mod, sps=sps, baud=baud)
    except Exception as e:
        out["error"] = str(e)
        return out
    out["rx"] = rx
    sy = sync_stage(rx)
    out["sync"] = sy
    chain = BL.detect_chain(sy["bits"], sy["llr"], aligned=sy["found"])
    out["chain"] = chain
    bits, info = BL.apply_chain(sy["bits"], sy["llr"], chain)
    out["decoded_bits"], out["fec_info"] = bits, info
    fr = C.parse_frame(bits) if sy["found"] or chain.get("fec_mode") else None
    out["frame"] = fr if (fr and fr["crc_ok"]) else None
    out["entropy"] = C.bit_entropy(sy["bits"])
    return out


def flatten_params(res: dict, fs: float, n: int, path: str = "") -> dict:
    be = res["band"]
    p = {}
    p["Sampling Frequency (Hz)"] = f"{fs:,.0f}"
    rx = res["rx"]
    if rx is None:
        p["Estimated Bandwidth (Hz)"] = f"{be['bw']:,.0f}"
        p["SNR Estimate (dB)"] = f"{be['snr_psd']:.1f} (PSD method)"
        p["Modulation (estimated)"] = "Not recognised"
        p["Receiver status"] = f"FAILED: {res['error']}"
        p["Sample Count"] = f"{n:,}"
        p["Duration (s)"] = f"{n / fs:.4f}"
        return p
    r = rx.rec
    p["Estimated Bandwidth (Hz)"] = f"{r.get('bw_hz', be['bw']):,.0f}" + ("  (wide-band / unshaped)" if be["wideband"] else "")
    snr = r["snr_db"]
    p["SNR Estimate (dB)"] = (">40 (noiseless)" if snr > 40 else f"{snr:.1f}") + "  (decision-directed, per symbol)"
    conf = f"  [confidence {rx.cls['confidence'] * 100:.0f}%]" if rx.cls else "  [user-selected]"
    p["Modulation (estimated)"] = rx.mod + conf
    p["Symbol Rate (baud)"] = f"{rx.baud:,.2f}  ({rx.sps:.2f} samples/symbol)"
    if rx.mod == "FSK":
        t0, t1 = r["tones_hz"]
        p["FSK Tones (Hz)"] = f"{t0:,.0f} / {t1:,.0f}  (h = {r['deviation_h']:.2f})"
    else:
        p["Carrier Offset (Hz)"] = f"{r['cfo_hz']:+.1f}"
        p["Pulse / Matched Filter"] = r["filter"]
        p["EVM"] = f"{r['evm_pct']:.1f} %"
    p["Receiver Lock"] = "LOCKED" if r["locked"] else "WEAK / NOT LOCKED"
    sy = res["sync"]
    if sy["found"]:
        h = sy["hit"]
        p["Frame Sync"] = f"{h['pattern']} at bit {h['position']} ({sy['variant']['name']}{', inverted' if h['inverted'] else ''}, {h['mismatches']} bit errors)"
    else:
        p["Frame Sync"] = "No known sync word found"
    ch = res["chain"]
    p["FEC (inferred)"] = ch["fec"]
    p["Interleaving (inferred)"] = ch["interleaver"]
    fr = res["frame"]
    if fr:
        p["Decoded Payload"] = f"CRC OK  |  {fr['length']} bytes  |  \"{fr['payload'].decode('ascii', 'replace')[:60]}\""
    elif not sy["found"]:
        p["Stream Statistics"] = f"entropy {res['entropy']:.3f} (1.000 = random) - no framing / coding structure detected"
    p["Sample Count"] = f"{n:,}"
    p["Duration (s)"] = f"{n / fs:.4f}"
    return p


def extract_parameters(samples, fs):
    res = analyze_signal(samples, fs)
    return flatten_params(res, fs, len(samples))


# ------------------------------------------------------- offline analyst ------
def offline_analyst(query: str, params: dict, status: str = "") -> str:
    lines = ["[Offline analyst - rule-based summary of measured values]"]
    for k in ("Modulation (estimated)", "Symbol Rate (baud)", "SNR Estimate (dB)", "Estimated Bandwidth (Hz)",
              "Carrier Offset (Hz)", "Frame Sync", "FEC (inferred)", "Interleaving (inferred)", "Decoded Payload", "Stream Statistics"):
        if k in params:
            lines.append(f"  {k}: {params[k]}")
    q = query.lower()
    if "tactical" in q or "assessment" in q:
        if "Decoded Payload" in params:
            lines.append("Assessment: transmission fully recovered (CRC-verified). Archive the payload and log carrier/baud for emitter fingerprinting.")
        elif "Stream Statistics" in params:
            lines.append("Assessment: digital modulation recovered but content has no recognisable framing or coding - possibly encrypted/scrambled or an unknown protocol. Recommend longer capture and scrambler search.")
        else:
            lines.append("Assessment: partial recovery. Re-capture at higher SNR or supply known sync word / interleaver parameters.")
    elif "explain" in q or "pipeline" in q:
        lines.append("All pipeline settings above were measured from the signal (cyclic baud line, M-th power carrier line, trial decoding of interleaver/FEC candidates) - none are assumed.")
    return "\n".join(lines)
