"""Forward-error-correction: real decoders (no placeholders).

* Convolutional code  K=7, rate 1/2, polynomials (171,133) octal
    - encoder (optional zero-tail termination)
    - hard- and soft-decision Viterbi (numba), automatic termination detection
* Reed-Solomon        RS(255,223) / any (n, n-nsym) via the `reedsolo` library
* Concatenated        Viterbi (inner)  ->  RS (outer)
* LDPC                built-in rate-~1/2 regular (3,6) code, n = 1020,
                      systematic GF(2) encoder, normalised min-sum decoder (numba)

Soft-bit convention used everywhere in this project:
    llr > 0  <=>  bit 1 is more likely,   llr < 0  <=>  bit 0,   llr == 0  <=>  erasure.
"""

from __future__ import annotations

import numpy as np
from numba import njit

try:  # Reed-Solomon is optional at import time; a clear error is raised on use.
    from reedsolo import RSCodec, ReedSolomonError

    _HAVE_RS = True
except Exception:  # pragma: no cover
    _HAVE_RS = False

G1 = 0o171
G2 = 0o133
CONV_TAIL = 6


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n]).tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8)).astype(np.uint8)


def hard_to_llr(bits: np.ndarray, mag: float = 1.0) -> np.ndarray:
    return (np.asarray(bits, dtype=np.float64) * 2.0 - 1.0) * mag


def _parity_py(x: int) -> int:
    return bin(x).count("1") & 1


def _build_trellis():
    out = np.zeros((64, 2, 2), dtype=np.uint8)  # [state, input bit, which output]
    nxt = np.zeros((64, 2), dtype=np.int64)
    for s in range(64):
        for b in range(2):
            r = ((s << 1) | b) & 127
            out[s, b, 0] = _parity_py(r & G1)
            out[s, b, 1] = _parity_py(r & G2)
            nxt[s, b] = r & 63
    return out, nxt


_OUT, _NXT = _build_trellis()


# --------------------------------------------------------------------------- #
# convolutional code
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _conv_encode_core(bits, out_tab, tail):
    n = bits.shape[0]
    total = n + tail
    out = np.empty(2 * total, dtype=np.uint8)
    state = 0
    for i in range(total):
        b = 0
        if i < n:
            b = int(bits[i])
        out[2 * i] = out_tab[state, b, 0]
        out[2 * i + 1] = out_tab[state, b, 1]
        state = ((state << 1) | b) & 63
    return out


def conv_encode(bits: np.ndarray, terminate: bool = True) -> np.ndarray:
    """Rate-1/2 K=7 encoder. With terminate=True, 6 zero tail bits are flushed."""
    bits = np.ascontiguousarray(bits, dtype=np.uint8)
    return _conv_encode_core(bits, _OUT, CONV_TAIL if terminate else 0)


@njit(cache=True)
def _viterbi_core(llr, out_tab):
    """Maximum-correlation Viterbi. Returns (survivors, final path metrics)."""
    n_steps = llr.shape[0] // 2
    neg = -1.0e30
    pm = np.full(64, neg)
    pm[0] = 0.0
    new = np.empty(64)
    surv = np.zeros((n_steps, 64), dtype=np.uint8)
    # expected-output sign table for the 64x2 branches
    for t in range(n_steps):
        l0 = llr[2 * t]
        l1 = llr[2 * t + 1]
        for ns in range(64):
            best = neg
            bx = 0
            b = ns & 1
            for x in range(2):
                ps = (ns >> 1) | (x << 5)
                m = pm[ps]
                if m <= neg:
                    continue
                e0 = 1.0 if out_tab[ps, b, 0] == 1 else -1.0
                e1 = 1.0 if out_tab[ps, b, 1] == 1 else -1.0
                m = m + e0 * l0 + e1 * l1
                if m > best:
                    best = m
                    bx = x
            new[ns] = best
            surv[t, ns] = bx
        mx = new[0]
        for s in range(1, 64):
            if new[s] > mx:
                mx = new[s]
        for s in range(64):  # normalise to avoid drift on very long streams
            pm[s] = new[s] - mx if new[s] > neg else neg
    return surv, pm


@njit(cache=True)
def _traceback(surv, start_state):
    n_steps = surv.shape[0]
    dec = np.empty(n_steps, dtype=np.uint8)
    state = start_state
    for t in range(n_steps - 1, -1, -1):
        dec[t] = state & 1
        x = surv[t, state]
        state = (state >> 1) | (int(x) << 5)
    return dec


def viterbi_decode_ex(
    soft_or_hard: np.ndarray,
    soft: bool = False,
    terminated: str | bool = "auto",
    strip_tail: bool = False,
):
    """Viterbi decode. `soft_or_hard` is LLRs if soft=True, else 0/1 bits.

    Returns (decoded_bits, info). info has: errors_corrected, terminated_used,
    n_in, n_out. Stream length is truncated to an even number of bits.
    """
    x = np.asarray(soft_or_hard, dtype=np.float64)
    if not soft:
        x = hard_to_llr(x)
    n = (len(x) // 2) * 2
    if n < 2:
        return np.zeros(0, dtype=np.uint8), {"errors_corrected": 0, "terminated_used": False}
    x = np.ascontiguousarray(x[:n])
    surv, pm = _viterbi_core(x, _OUT)
    best = int(np.argmax(pm))
    scale = float(np.mean(np.abs(x))) or 1.0
    if terminated == "auto":
        use_term = bool(pm[0] >= pm[best] - 2.0 * scale - 1e-9)
    else:
        use_term = bool(terminated)
    dec = _traceback(surv, 0 if use_term else best)
    if use_term and strip_tail and len(dec) > CONV_TAIL:
        dec = dec[:-CONV_TAIL]
    # estimate channel errors by re-encoding
    reenc = conv_encode(dec[: len(dec) if not strip_tail else len(dec)], terminate=False)
    m = min(len(reenc), n)
    valid = x[:m] != 0
    hard_in = (x[:m] > 0).astype(np.uint8)
    errs = int(np.sum((reenc[:m] != hard_in) & valid))
    return dec, {
        "errors_corrected": errs,
        "terminated_used": use_term,
        "n_in": n,
        "n_out": len(dec),
        "mismatch_frac": errs / max(1, int(np.sum(valid))),
    }


def viterbi_decode(bits: np.ndarray, llr: np.ndarray | None = None) -> np.ndarray:
    """Convenience wrapper (hard bits, or soft if llr given)."""
    if llr is not None:
        return viterbi_decode_ex(llr, soft=True)[0]
    return viterbi_decode_ex(bits, soft=False)[0]


# --------------------------------------------------------------------------- #
# Reed-Solomon
# --------------------------------------------------------------------------- #
def _require_rs():
    if not _HAVE_RS:
        raise RuntimeError("Reed-Solomon needs the 'reedsolo' package:  pip install reedsolo")


def rs_encode(data: bytes, nsym: int = 32) -> bytes:
    _require_rs()
    return bytes(RSCodec(nsym).encode(bytes(data)))


def rs_decode_bytes(raw: bytes, n: int = 255, nsym: int = 32):
    """Decode consecutive RS(n, n-nsym) codewords. Returns (data, stats)."""
    _require_rs()
    rsc = RSCodec(nsym)
    out = bytearray()
    ok = bad = corrected = 0
    pos = 0
    while pos + nsym + 1 <= len(raw):
        blk = raw[pos : pos + n]
        pos += n
        try:
            res = rsc.decode(bytes(blk))
            msg = bytes(res[0])
            errata = res[2] if len(res) > 2 else []
            corrected += len(errata)
            out += msg
            ok += 1
        except ReedSolomonError:
            out += blk[: max(0, len(blk) - nsym)]
            bad += 1
    return bytes(out), {"blocks_ok": ok, "blocks_failed": bad, "symbols_corrected": corrected}


def rs_decode_block(bits: np.ndarray, k: int = 223, n: int = 255):
    """Bits -> RS(255,223) decode -> bits. Returns (bits, stats)."""
    data, st = rs_decode_bytes(bits_to_bytes(bits), n=n, nsym=n - k)
    return bytes_to_bits(data), st


# --------------------------------------------------------------------------- #
# LDPC (built-in code)
# --------------------------------------------------------------------------- #
class _LDPC:
    def __init__(self, n: int = 1020, wc: int = 3, wr: int = 6, seed: int = 20260):
        assert n % wr == 0
        rng = np.random.default_rng(seed)
        m0 = n // wr
        blocks = []
        base = np.zeros((m0, n), dtype=np.uint8)
        for i in range(m0):
            base[i, i * wr : (i + 1) * wr] = 1
        blocks.append(base)
        for _ in range(wc - 1):
            blocks.append(base[:, rng.permutation(n)])
        H = np.vstack(blocks)
        self.n = n
        self.m = H.shape[0]
        self.wr = wr
        self.H = H
        self._gauss(H.copy())
        self.chk_vars = np.array([np.flatnonzero(H[r]) for r in range(self.m)], dtype=np.int32)

    def _gauss(self, A: np.ndarray):
        m, n = A.shape
        piv_cols = []
        r = 0
        for c in range(n):
            if r >= m:
                break
            rows = np.flatnonzero(A[r:, c])
            if len(rows) == 0:
                continue
            p = r + rows[0]
            if p != r:
                A[[r, p]] = A[[p, r]]
            others = np.flatnonzero(A[:, c])
            others = others[others != r]
            A[others] ^= A[r]
            piv_cols.append(c)
            r += 1
        self.rank = r
        self.piv_cols = np.array(piv_cols, dtype=np.int64)
        self.info_cols = np.array([c for c in range(n) if c not in set(piv_cols)], dtype=np.int64)
        self.k = len(self.info_cols)
        self.R = A[: self.rank]

    def encode_block(self, info: np.ndarray) -> np.ndarray:
        x = np.zeros(self.n, dtype=np.uint8)
        x[self.info_cols] = info
        # pivot row r: x[piv_r] = XOR_{c in info with R[r,c]=1} x[c]
        par = (self.R[:, self.info_cols].astype(np.int64) @ info.astype(np.int64)) & 1
        x[self.piv_cols] = par.astype(np.uint8)
        return x


_LDPC_CODE: _LDPC | None = None


def ldpc_code() -> _LDPC:
    global _LDPC_CODE
    if _LDPC_CODE is None:
        _LDPC_CODE = _LDPC()
    return _LDPC_CODE


def ldpc_encode(info_bits: np.ndarray) -> np.ndarray:
    code = ldpc_code()
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    pad = (-len(info_bits)) % code.k
    if pad:
        info_bits = np.concatenate([info_bits, np.zeros(pad, dtype=np.uint8)])
    return np.concatenate([code.encode_block(b) for b in info_bits.reshape(-1, code.k)])


@njit(cache=True)
def _minsum(llr, chk_vars, iters, alpha):
    m, wr = chk_vars.shape
    n = llr.shape[0]
    c2v = np.zeros((m, wr))
    total = np.empty(n)
    hard = np.zeros(n, dtype=np.uint8)
    converged = False
    it_used = 0
    for it in range(iters):
        it_used = it + 1
        for v in range(n):
            total[v] = llr[v]
        for c in range(m):
            for j in range(wr):
                total[chk_vars[c, j]] += c2v[c, j]
        for v in range(n):
            hard[v] = 1 if total[v] > 0 else 0
        ok = True
        for c in range(m):
            s = 0
            for j in range(wr):
                s ^= hard[chk_vars[c, j]]
            if s:
                ok = False
                break
        if ok:
            converged = True
            break
        for c in range(m):
            min1 = 1.0e30
            min2 = 1.0e30
            imin = -1
            sgn = 1.0
            vals = np.empty(wr)
            for j in range(wr):
                v = chk_vars[c, j]
                t = total[v] - c2v[c, j]
                vals[j] = t
                a = abs(t)
                if t < 0:
                    sgn = -sgn
                if a < min1:
                    min2 = min1
                    min1 = a
                    imin = j
                elif a < min2:
                    min2 = a
            for j in range(wr):
                t = vals[j]
                s = sgn * (-1.0 if t < 0 else 1.0)
                mag = min2 if j == imin else min1
                c2v[c, j] = alpha * s * mag
    return hard, converged, it_used


def ldpc_decode_ex(llr: np.ndarray, iters: int = 60):
    code = ldpc_code()
    llr = np.asarray(llr, dtype=np.float64)
    nb = len(llr) // code.n
    out = []
    conv = 0
    for b in range(nb):
        blk = np.ascontiguousarray(np.clip(llr[b * code.n : (b + 1) * code.n], -30, 30))
        hard, ok, _ = _minsum(blk, code.chk_vars, iters, 0.8)
        conv += int(ok)
        out.append(hard[code.info_cols])
    dec = np.concatenate(out) if out else np.zeros(0, dtype=np.uint8)
    return dec.astype(np.uint8), {"blocks": nb, "blocks_converged": conv, "k": code.k, "n": code.n}


# --------------------------------------------------------------------------- #
# concatenated RS + convolutional
# --------------------------------------------------------------------------- #
def concatenated_decode_ex(bits=None, llr=None, nsym: int = 32, n: int = 255):
    if llr is not None:
        inner, iinfo = viterbi_decode_ex(llr, soft=True, strip_tail=False)
    else:
        inner, iinfo = viterbi_decode_ex(bits, soft=False, strip_tail=False)
    data, rinfo = rs_decode_bytes(bits_to_bytes(inner), n=n, nsym=nsym)
    info = {"viterbi_errors_corrected": iinfo["errors_corrected"]}
    info.update(rinfo)
    return bytes_to_bits(data), info


# --------------------------------------------------------------------------- #
# GUI-facing dispatchers
# --------------------------------------------------------------------------- #
def fec_decode_ex(bits: np.ndarray, mode: str, llr: np.ndarray | None = None):
    """Decode according to a GUI mode string. Returns (bits, info_dict)."""
    bits = np.asarray(bits, dtype=np.uint8)
    m = mode.lower()
    if m.startswith("none") or m == "":
        return bits.copy(), {"mode": "none"}
    if "concat" in m:
        out, info = concatenated_decode_ex(bits=bits if llr is None else None, llr=llr)
        info["mode"] = "RS(255,223) + Viterbi K=7"
        return out, info
    if "ldpc" in m:
        L = llr if llr is not None else hard_to_llr(bits, 4.0)
        out, info = ldpc_decode_ex(L)
        info["mode"] = "LDPC (built-in n=1020 rate~1/2, min-sum)"
        return out, info
    if "rs" in m and "conv" not in m and "viterbi" not in m:
        out, info = rs_decode_block(bits)
        info["mode"] = "Reed-Solomon RS(255,223)"
        return out, info
    if "viterbi" in m or "conv" in m:
        if llr is not None:
            out, info = viterbi_decode_ex(llr, soft=True)
            info["input"] = "soft-decision"
        else:
            out, info = viterbi_decode_ex(bits, soft=False)
            info["input"] = "hard-decision"
        info["mode"] = "Convolutional K=7 r=1/2 (171,133)"
        return out, info
    return bits.copy(), {"mode": "unknown mode - passthrough"}


def fec_decode(bits: np.ndarray, mode: str, llr: np.ndarray | None = None) -> np.ndarray:
    return fec_decode_ex(bits, mode, llr)[0]


def warm_up() -> None:
    """Trigger numba compilation once (so the first GUI click is not slow)."""
    b = np.random.randint(0, 2, 64).astype(np.uint8)
    c = conv_encode(b)
    viterbi_decode_ex(c, soft=False)
    ldpc_decode_ex(np.zeros(ldpc_code().n))
