"""BER-vs-Eb/N0 benchmark that runs through the REAL receiver chain.

For each Eb/N0 point:  random info bits -> FEC encode -> QPSK (RRC, carrier offset, timing offset, AWGN)
-> dsp_core.receive (timing + carrier recovery + soft demap) -> FEC decode -> compare with the info bits.

Curves: uncoded QPSK (theory + measured), Conv K=7 r=1/2 hard-decision, Conv soft-decision, LDPC soft-decision.
Alignment note: the sync-word position/phase variant is chosen by best match to the KNOWN preamble
(genie alignment), so the curves isolate demodulator + decoder performance from sync-miss events.

CLI:  python benchmark.py [--quick]      -> results/ber_vs_snr.png and .csv
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.special import erfc

import correlation as C
import demo_signals as DS
import dsp_core as D
import fec as F

FS, BAUD = 48000.0, 2400.0
ASM = DS.ASM
POINTS_FULL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
POINTS_QUICK = [1, 3, 5, 7, 9]


def theory_qpsk(ebn0_db):
    return 0.5 * erfc(np.sqrt(10 ** (np.asarray(ebn0_db, dtype=float) / 10)))


def _receive_after_asm(x):
    """Receive blindly-timed QPSK; return variants' (bits, llr) aligned to the known ASM (genie alignment)."""
    rx = D.receive(x, FS, mod="QPSK", baud=BAUD)
    best = None
    for v in rx.variants:
        prof = C.mismatch_profile(v["bits"], ASM)
        if len(prof) == 0:
            continue
        k = int(np.argmin(prof))
        if best is None or prof[k] < best[0]:
            best = (prof[k], v, k)
    _, v, k = best
    s = k + len(ASM)
    return v["bits"][s:], v["llr"][s:]


def _tx(bits, esn0_db, seed):
    bits = np.concatenate([ASM, bits]).astype(np.uint8)
    bits = np.concatenate([bits, np.zeros((-len(bits)) % 2, dtype=np.uint8)])
    return DS.make_signal(bits, "QPSK", FS, BAUD, snr_db=esn0_db, cfo=120.0, seed=seed, lead=1200, trail=800)


LOSS_THRESHOLD = 0.3  # a frame with >30 % bit errors = receiver lost lock / sync: counted as a lost frame, not in BER


def _tally(acc, key, errors, n):
    acc[key][3] += 1
    if n <= 0 or errors / n > LOSS_THRESHOLD:
        acc[key][2] += 1
    else:
        acc[key][0] += errors
        acc[key][1] += n


def run_benchmark(points=POINTS_FULL, n_info=2400, min_errors=40, max_trials=8, min_trials=3, progress=None, seed=1234):
    rng = np.random.default_rng(seed)
    F.warm_up()
    code = F.ldpc_code()
    r_ldpc = code.k / code.n
    curves = {k: {"ebn0": [], "ber": [], "bits": [], "lost": [], "frames": []} for k in ("Uncoded QPSK", "Conv K=7 hard", "Conv K=7 soft", "LDPC soft")}
    total = len(points)
    for pi, eb in enumerate(points):
        acc = {k: [0, 0, 0, 0] for k in curves}  # errors, bits, lost frames, frames
        for t in range(max_trials):
            sd = int(rng.integers(1, 1 << 30))
            # ---- uncoded
            info = rng.integers(0, 2, n_info).astype(np.uint8)
            try:
                b, _ = _receive_after_asm(_tx(info, eb + 10 * np.log10(2), sd))
                n = min(len(b), len(info))
                _tally(acc, "Uncoded QPSK", int(np.sum(b[:n] != info[:n])), n)
            except Exception:
                acc["Uncoded QPSK"][3] += 1
                acc["Uncoded QPSK"][2] += 1
            # ---- conv (one transmission, decoded twice)
            info = rng.integers(0, 2, n_info // 2).astype(np.uint8)
            coded = F.conv_encode(info, terminate=True)
            try:
                b, l = _receive_after_asm(_tx(coded, eb + 10 * np.log10(2 * 0.5), sd + 1))
                m = min(len(b), len(coded))
                hd, _ = F.viterbi_decode_ex(b[:m], soft=False, terminated=True, strip_tail=True)
                sd_, _ = F.viterbi_decode_ex(l[:m], soft=True, terminated=True, strip_tail=True)
                n = len(info)
                for key, dec in (("Conv K=7 hard", hd), ("Conv K=7 soft", sd_)):
                    k_ = min(len(dec), n)
                    e = int(np.sum(dec[:k_] != info[:k_])) + (n - k_)
                    _tally(acc, key, e, n)
            except Exception:
                for key in ("Conv K=7 hard", "Conv K=7 soft"):
                    acc[key][3] += 1
                    acc[key][2] += 1
            # ---- LDPC
            info = rng.integers(0, 2, code.k * 2).astype(np.uint8)
            cw = F.ldpc_encode(info)
            try:
                b, l = _receive_after_asm(_tx(cw, eb + 10 * np.log10(2 * r_ldpc), sd + 2))
                m = (min(len(l), len(cw)) // code.n) * code.n
                dec, _ = F.ldpc_decode_ex(l[:m])
                n = len(info)
                k_ = min(len(dec), n)
                _tally(acc, "LDPC soft", int(np.sum(dec[:k_] != info[:k_])) + (n - k_), n)
            except Exception:
                acc["LDPC soft"][3] += 1
                acc["LDPC soft"][2] += 1
            if t + 1 >= min_trials and all(acc[k][0] >= min_errors or acc[k][2] >= 2 for k in acc):
                break
        for k in curves:
            curves[k]["ebn0"].append(eb)
            curves[k]["ber"].append(acc[k][0] / acc[k][1] if acc[k][1] > 0 else float("nan"))
            curves[k]["bits"].append(acc[k][1])
            curves[k]["lost"].append(acc[k][2])
            curves[k]["frames"].append(acc[k][3])
        if progress:
            progress((pi + 1) / total, f"Eb/N0 = {eb} dB done")
    return curves


STYLE = {
    "Uncoded QPSK": ("#ff5555", "o", "-"),
    "Conv K=7 hard": ("#ffaa00", "s", "-"),
    "Conv K=7 soft": ("#00e5ff", "D", "-"),
    "LDPC soft": ("#00ff66", "^", "-"),
}


def plot_curves(fig, curves, dark=True):
    fig.clear()
    ax = fig.add_subplot(111)
    fg = "white" if dark else "black"
    if dark:
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")
    xs = np.linspace(0, 10, 200)
    ax.semilogy(xs, theory_qpsk(xs), "--", color="#888888", lw=1.2, label="Uncoded QPSK (theory)")
    floor = 1.0
    for name, c in curves.items():
        col, mk, ls = STYLE[name]
        rows = [(e, b, n, lo) for e, b, n, lo in zip(c["ebn0"], c["ber"], c["bits"], c["lost"]) if n > 0 and b == b]
        pts = [r for r in rows if r[1] > 0]
        zero = [(r[0], r[2]) for r in rows if r[1] == 0]
        if pts:
            ax.semilogy([p[0] for p in pts], [p[1] for p in pts], color=col, ls=ls, lw=1.6, label=name)
            for e, b, n, lo in pts:  # hollow marker = some frames lost at this point
                ax.semilogy([e], [b], marker=mk, ms=7, color=col, markerfacecolor="none" if lo else col, ls="")
            floor = min(floor, min(p[1] for p in pts))
        for e, n in zero:  # error-free: down-triangle at the 1/N detection floor
            ax.semilogy([e], [1.0 / n], marker="v", ms=8, color=col, markerfacecolor="none", ls="")
            floor = min(floor, 1.0 / n)
    ax.set_ylim(max(floor / 3, 1e-6), 1.0)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Eb/N0 (dB)", color=fg)
    ax.set_ylabel("Bit error rate", color=fg)
    ax.set_title("BER vs Eb/N0 - measured through the full SPECTRA receiver", color=fg, fontweight="bold")
    ax.grid(True, which="both", alpha=0.2)
    ax.tick_params(colors=fg)
    for sp_ in ax.spines.values():
        sp_.set_edgecolor("#555555" if dark else "black")
    leg = ax.legend(loc="lower left", facecolor="#242424" if dark else "white", labelcolor=fg, fontsize=9)
    fig.text(0.99, 0.01, "v = no errors seen (plotted at 1/bits tested);  hollow marker = some frames lost (receiver lock failure, excluded from BER)", ha="right", va="bottom", fontsize=7, color="#aaaaaa" if dark else "#555555")
    fig.tight_layout()


def save_results(curves, folder="results"):
    from matplotlib.figure import Figure

    os.makedirs(folder, exist_ok=True)
    fig = Figure(figsize=(8.5, 5.5), dpi=150)
    plot_curves(fig, curves, dark=False)
    png = os.path.join(folder, "ber_vs_snr.png")
    fig.savefig(png)
    fig2 = Figure(figsize=(8.5, 5.5), dpi=150)
    plot_curves(fig2, curves, dark=True)
    fig2.savefig(os.path.join(folder, "ber_vs_snr_dark.png"), facecolor=fig2.get_facecolor())
    with open(os.path.join(folder, "ber_vs_snr.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["curve", "ebn0_db", "ber", "bits_tested", "frames_lost", "frames"])
        for name, c in curves.items():
            for e, b, n, lo, fr in zip(c["ebn0"], c["ber"], c["bits"], c["lost"], c["frames"]):
                w.writerow([name, e, f"{b:.3e}", n, lo, fr])
    return png


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    cv = run_benchmark(POINTS_QUICK if quick else POINTS_FULL, max_trials=3 if quick else 8,
                       progress=lambda f, m: print(f"[{f * 100:3.0f}%] {m}", flush=True))
    print("saved", save_results(cv, os.path.join(here, "results")))
