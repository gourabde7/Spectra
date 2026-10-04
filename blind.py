"""Blind detection of interleaver + FEC by trial decoding.

Score = fraction of received code bits that disagree with the re-encoded Viterbi output
(~0 for the right interleaver/code, ~0.3 for a wrong hypothesis).
"""

from __future__ import annotations

import numpy as np

import deinterleave as DI
import fec as F

SIZES = (4, 8, 12, 16, 24, 32, 64)
DEPTHS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 17, 24, 32)
CONV_THRESHOLD = 0.08
N_TRY = 8192


def conv_score(llr: np.ndarray):
    """Best (score, offset) over bit-pair alignment 0/1 for a soft stream."""
    best = (1.0, 0)
    for off in (0, 1):
        seg = llr[off:]
        n = (len(seg) // 2) * 2
        if n < 200:
            continue
        _, info = F.viterbi_decode_ex(seg[:n], soft=True, terminated=False)
        if info["mismatch_frac"] < best[0]:
            best = (info["mismatch_frac"], off)
    return best


def _candidates(n_avail: int):
    yield ("None", {})
    for r in SIZES:
        for c in SIZES:
            if r * c * 2 <= n_avail:
                yield ("Block", {"rows": r, "cols": c})
                yield ("Diagonal", {"rows": r, "cols": c})
    for B in DEPTHS:
        for u in (1, 2):
            if B * (B - 1) * u + 400 < n_avail:
                yield ("Convolutional", {"depth": B, "unit": u})


def detect_conv_and_interleaver(llr: np.ndarray, search_interleaver: bool = True):
    """Returns ranked list of dicts and the best accepted hypothesis (or None)."""
    llr = np.asarray(llr, dtype=np.float64)
    n_avail = min(len(llr), N_TRY * 2)
    results = []
    cands = _candidates(n_avail) if search_interleaver else iter([("None", {})])
    for mode, kw in cands:
        if mode == "Block" or mode == "Diagonal":
            blk = kw["rows"] * kw["cols"]
            use = (n_avail // blk) * blk
            seg = DI.deinterleave(llr[:use], mode, **kw)
        elif mode == "Convolutional":
            seg = DI.deinterleave(llr[:n_avail], mode, **kw)
        else:
            seg = llr[:n_avail]
        if len(seg) < 1000:
            continue
        sc, off = conv_score(seg)
        results.append({"mode": mode, "params": kw, "score": float(sc), "offset": off})
    results.sort(key=lambda d: d["score"])
    best = None
    if results and results[0]["score"] < CONV_THRESHOLD:
        if len(results) > 3:  # need clear contrast against the wrong hypotheses
            med = float(np.median([r["score"] for r in results]))
            if results[0]["score"] < 0.5 * med:
                best = results[0]
        elif results[0]["score"] < 0.05:
            best = results[0]
    return results, best


def detect_rs(bits: np.ndarray):
    """True if RS(255,223) codewords decode at offset 0 of this byte stream."""
    data = F.bits_to_bytes(bits)
    if len(data) < 255:
        return False, {}
    try:
        out, st = F.rs_decode_bytes(data[:255], 255, 32)
    except Exception:
        return False, {}
    return (st["blocks_ok"] == 1 and st["symbols_corrected"] <= 16), st


def detect_chain(bits: np.ndarray, llr: np.ndarray, aligned: bool):
    """Full blind search. `aligned` = stream starts exactly at the code-block start (after sync)."""
    info = {"interleaver": "None detected", "fec": "None detected (stream looks uncoded)", "ranking": [], "rs": False}
    if aligned:
        ok, st = detect_rs(bits)
        if ok:
            info.update({"rs": True, "fec_mode": "RS Block Code", "interleaver_mode": "None", "interleaver": "None",
                         "fec": f"Reed-Solomon RS(255,223)  [{st['symbols_corrected']} symbol errors corrected]"})
            return info
    res, best = detect_conv_and_interleaver(llr, search_interleaver=aligned)
    info["ranking"] = res[:5]
    if best is not None:
        mode, kw = best["mode"], best["params"]
        info["interleaver_mode"], info["interleaver_kw"] = mode, kw
        info["interleaver"] = "None" if mode == "None" else f"{mode} {kw}"
        info["conv_offset"] = best["offset"]
        info["conv_score"] = best["score"]
        info["fec_mode"] = "Viterbi (Conv K=7)"
        info["fec"] = f"Convolutional K=7 r=1/2 (171,133)  [mismatch {best['score'] * 100:.1f}%]"
        # outer RS?
        seg = DI.deinterleave(llr, mode, **kw) if mode != "None" else llr
        seg = seg[best["offset"]:]
        dec, _ = F.viterbi_decode_ex(seg, soft=True)
        ok, st = detect_rs(dec)
        if ok:
            info["rs"] = True
            info["fec_mode"] = "Concatenated (RS + Conv)"
            info["fec"] = "RS(255,223) + Convolutional K=7 r=1/2"
        return info
    ok, st = detect_rs(bits)
    if ok:
        info["rs"] = True
        info["fec_mode"] = "RS Block Code"
        info["fec"] = f"Reed-Solomon RS(255,223)  [{st['symbols_corrected']} symbol errors corrected]"
        info["interleaver_mode"] = "None"
        info["interleaver"] = "None"
    return info


def apply_chain(bits, llr, info):
    """Deinterleave + decode according to detect_chain() result. Returns (bits, text-info dict)."""
    mode = info.get("interleaver_mode", "None")
    kw = info.get("interleaver_kw", {})
    b, l = np.asarray(bits), np.asarray(llr, dtype=np.float64)
    if mode != "None":
        b = DI.deinterleave(b, mode, **kw)
        l = DI.deinterleave(l, mode, **kw)
    fm = info.get("fec_mode")
    if fm is None:
        return b, {"mode": "none"}
    off = info.get("conv_offset", 0) if "Conv" in fm or "Viterbi" in fm else 0
    b, l = b[off:], l[off:]
    return F.fec_decode_ex(b, fm, llr=l)
