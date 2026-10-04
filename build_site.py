"""Builds the static web showcase in ./web (data.json) from the bundled demo captures.

Usage:  python build_site.py        then deploy the `web/` folder (Vercel: Root Directory = web).
The desktop app (main.py, Tkinter) is the live tool; the website shows its measured results.
"""
import csv, json, os
import numpy as np
from analysis import analyze_signal, flatten_params, compute_spectrum
from signal_loader import load_signal

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = ["demo", "demo_bpsk_rs", "demo_qpsk_concat", "demo_fsk", "demo_16qam_uncoded"]


def build():
    out = {"demos": [], "ber": {}}
    for name in DEMOS:
        truth = json.load(open(os.path.join(HERE, name + ".truth.json")))
        path = os.path.join(HERE, truth["file"])
        x, fs, ft = load_signal(path, truth["sample_rate"], truth.get("iq_format") or "cf32")
        res = analyze_signal(x, fs)
        rx, ch, sy, fr = res["rx"], res["chain"], res["sync"], res["frame"]
        params = flatten_params(res, fs, len(x))
        pts = rx.rec.get("symbols")
        if pts is not None:
            pts = pts[np.linspace(0, len(pts) - 1, min(500, len(pts))).astype(int)]
            const = [[round(float(p.real), 3), round(float(p.imag), 3)] for p in pts]
        else:
            const = []
        f, db = compute_spectrum(x, fs, 1024)
        idx = np.linspace(0, len(f) - 1, 128).astype(int)
        spec = {"f_khz": [round(float(f[i] / 1e3), 3) for i in idx], "db": [round(float(db[i]), 1) for i in idx]}
        det_mod = rx.mod
        match = {
            "modulation": (truth["modulation"].replace("-", "").upper() in det_mod.replace("-", "").upper()),
            "baud": abs(rx.baud - truth["baud"]) / truth["baud"] < 0.005,
            "message": bool(fr and fr["payload"].decode() == truth["message"]),
        }
        out["demos"].append({
            "id": name, "file": truth["file"], "fs": fs, "truth": truth, "params": params,
            "detected": {"mod": det_mod, "baud": round(rx.baud, 2), "fec": ch["fec"], "interleaver": ch["interleaver"],
                         "sync": params.get("Frame Sync", ""), "snr": round(rx.rec["snr_db"], 1)},
            "ranking": [{"mode": r["mode"], "params": str(r["params"]), "score": round(r["score"] * 100, 1)} for r in ch.get("ranking", [])[:5]],
            "payload": fr["payload"].decode("ascii", "replace") if fr else None,
            "crc_ok": bool(fr and fr["crc_ok"]), "match": match, "const": const, "spec": spec,
        })
        print(name, match, flush=True)
    with open(os.path.join(HERE, "results", "ber_vs_snr.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            c = out["ber"].setdefault(r["curve"], [])
            c.append({"x": float(r["ebn0_db"]), "ber": None if r["ber"] == "nan" else float(r["ber"]),
                      "bits": int(r["bits_tested"]), "lost": int(r["frames_lost"]), "frames": int(r["frames"])})
    os.makedirs(os.path.join(HERE, "web"), exist_ok=True)
    with open(os.path.join(HERE, "web", "data.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print("wrote web/data.json", os.path.getsize(os.path.join(HERE, "web", "data.json")), "bytes")


if __name__ == "__main__":
    build()
