"""SPECTRA - live web version of the COMINT Signal Analysis Workbench (same backend as the desktop app).

Run locally:   streamlit run streamlit_app.py
Deploy:        Hugging Face Spaces (Docker SDK) or Render - see Dockerfile / README.
"""
import csv
import os
import tempfile

import numpy as np
import streamlit as st
from matplotlib.figure import Figure

import benchmark as BM
import blind as BL
import correlation as COR
import deinterleave as DI
from analysis import (analyze_signal, compute_spectrum, compute_waterfall, constellation_points,
                      flatten_params, offline_analyst)
from signal_loader import load_signal

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = {
    "demo.wav - QPSK + Conv K=7 + Block interleaver": ("demo.wav", 48000, "cf32"),
    "demo_bpsk_rs.wav - BPSK + RS(255,223)": ("demo_bpsk_rs.wav", 48000, "cf32"),
    "demo_qpsk_concat.iq - QPSK + RS+Conv + Forney interleaver": ("demo_qpsk_concat.iq", 250000, "cf32"),
    "demo_fsk.wav - 2-FSK + Conv + Diagonal interleaver": ("demo_fsk.wav", 48000, "cf32"),
    "demo_16qam_uncoded.wav - 16-QAM uncoded": ("demo_16qam_uncoded.wav", 48000, "cf32"),
}

st.set_page_config(page_title="SPECTRA - COMINT Signal Analysis", page_icon="📡", layout="wide")
st.title("📡 SPECTRA - COMINT Signal Analysis Workbench")
st.caption("SIH 2026 · PS 26147 · blind analysis of .wav / .iq intercepts - every parameter is measured from the samples")


@st.cache_data(show_spinner=False)
def run(path_or_bytes, name, fs_iq, fmt, mod, sps):
    if isinstance(path_or_bytes, bytes):
        suffix = os.path.splitext(name)[1] or ".iq"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
            fh.write(path_or_bytes)
            path = fh.name
    else:
        path = path_or_bytes
    x, fs, ftype = load_signal(path, fs_iq, fmt)
    x = x[:600000]
    res = analyze_signal(x, fs, mod=None if mod == "Auto" else mod, sps=sps)
    return x, fs, ftype, res


# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.header("Input")
    src = st.radio("Signal source", ["Bundled demo capture", "Upload my own file"])
    fs_iq, fmt, path, data, name = 2_000_000.0, "cf32", None, None, ""
    if src == "Bundled demo capture":
        label = st.selectbox("Capture", list(DEMOS))
        fn, fs_iq, fmt = DEMOS[label]
        path, name = os.path.join(HERE, fn), fn
    else:
        up = st.file_uploader("Upload .wav / .iq / .bin", type=["wav", "iq", "bin", "raw", "dat"])
        fs_iq = st.number_input("Sample rate for raw IQ (Hz)", value=2_000_000.0, step=1000.0)
        fmt = st.selectbox("IQ format", ["cf32", "ci16", "ci8", "cu8"])
        if up is not None:
            data, name = up.getvalue(), up.name
    with st.expander("Manual override (optional)"):
        mod = st.selectbox("Modulation", ["Auto", "BPSK", "QPSK", "16-QAM", "FSK"])
        sps_txt = st.text_input("Samples/symbol (auto or number)", "auto")
    go = st.button("▶ Run full pipeline", type="primary", use_container_width=True)

sps = None if sps_txt.strip().lower() in ("", "auto") else float(sps_txt)
if "ran" not in st.session_state:
    st.session_state.ran = False
if go:
    st.session_state.ran = True
src_ready = (path is not None) or (data is not None)

tabs = st.tabs(["1. Parameters", "2. Spectrum / Waterfall / Constellation", "3. Demodulation", "4. De-interleaving",
                "5. FEC", "6. Bit Correlation", "7. AI Analyst", "8. BER Benchmark"])

if st.session_state.ran and src_ready:
    try:
        with st.spinner("Running blind receiver (baud → modulation → timing/carrier → sync → interleaver/FEC detection)..."):
            x, fs, ftype, res = run(data if data is not None else path, name, fs_iq, fmt, mod, sps)
        params = flatten_params(res, fs, len(x))
        rx, sy, ch = res["rx"], res["sync"], res["chain"]
    except Exception as e:
        st.error(f"Could not process this file: {e}")
        st.stop()

    with tabs[0]:
        if res["frame"]:
            st.success(f"Payload recovered, CRC-16 OK: “{res['frame']['payload'].decode('ascii', 'replace')}”")
        elif rx is None:
            st.error(f"Receiver could not lock: {res['error']}")
        else:
            st.warning("Receiver locked but no CRC-valid frame was found (stream may be random/encrypted).")
        c1, c2 = st.columns(2)
        items = list(params.items())
        half = (len(items) + 1) // 2
        for col, part in ((c1, items[:half]), (c2, items[half:])):
            col.table({k: [v] for k, v in part}) if False else col.markdown("\n".join(f"**{k}**: {v}  " for k, v in part))

    with tabs[1]:
        f, db = compute_spectrum(x, fs)
        fw, tw, Sw = compute_waterfall(x, fs)
        fig = Figure(figsize=(11, 7))
        ax = fig.subplots(2, 2)
        ax[0, 0].plot(f / 1e3, db, color="#00aa44", lw=0.8)
        ax[0, 0].set(title="Spectrum (PSD)", xlabel="Frequency (kHz)", ylabel="dB")
        ax[0, 1].pcolormesh(tw, fw / 1e3, Sw, shading="auto", cmap="viridis")
        ax[0, 1].set(title="Waterfall", xlabel="Time (s)", ylabel="Frequency (kHz)")
        pts = constellation_points(None, symbols=rx.rec["symbols"]) if rx is not None and rx.rec.get("symbols") is not None else constellation_points(x)
        if rx is not None and rx.mod == "FSK":
            ax[1, 0].plot(rx.rec["llr"][:400], lw=0.8)
            ax[1, 0].set(title="FSK soft decisions", xlabel="Symbol", ylabel="LLR")
        else:
            ax[1, 0].scatter(pts.real, pts.imag, s=3, color="#00bcd4", alpha=0.6)
            ax[1, 0].set(title="Recovered constellation", xlabel="I", ylabel="Q", aspect="equal")
            ax[1, 0].grid(alpha=0.3)
        seg = x[:800]
        ax[1, 1].plot(seg.real, label="I", lw=0.8)
        ax[1, 1].plot(seg.imag, label="Q", lw=0.8)
        ax[1, 1].set(title="Time domain (first 800 samples)", xlabel="Sample")
        ax[1, 1].legend()
        fig.tight_layout()
        st.pyplot(fig)

    with tabs[2]:
        if rx is None:
            st.error(res["error"])
        else:
            r = rx.rec
            st.write(f"**Modulation:** {rx.mod}  ·  **Symbol rate:** {rx.baud:,.2f} baud ({rx.sps:.2f} samples/symbol)  ·  "
                     f"**SNR:** {r['snr_db']:.1f} dB  ·  **Lock:** {'LOCKED' if r['locked'] else 'WEAK'}")
            if sy["found"]:
                h = sy["hit"]
                st.write(f"**Frame sync:** {h['pattern']} at raw bit {h['position']} ({sy['variant']['name']}"
                         f"{', inverted' if h['inverted'] else ''}, {h['mismatches']} bit errors) - phase ambiguity resolved.")
            else:
                st.write("**Frame sync:** no known sync word found (raw stream shown).")
            st.code(COR.format_bit_preview(sy["bits"], 512), language=None)

    with tabs[3]:
        if rx is not None:
            st.write(f"**Blind interleaver search** - chosen: `{ch['interleaver']}`")
            if ch.get("ranking"):
                st.table([{"Hypothesis": r["mode"], "Parameters": str(r["params"]), "Code mismatch %": round(r["score"] * 100, 1)} for r in ch["ranking"]])
            mode, kw = ch.get("interleaver_mode", "None"), ch.get("interleaver_kw", {})
            out = DI.deinterleave(sy["bits"], mode, **kw) if mode != "None" else sy["bits"]
            st.code(COR.format_bit_preview(out, 512), language=None)

    with tabs[4]:
        if rx is not None:
            st.write(f"**Blind FEC detection:** {ch['fec']}")
            st.json({k: (v if not isinstance(v, (np.generic,)) else v.item()) for k, v in (res["fec_info"] or {}).items()})
            st.code(COR.format_bit_preview(res["decoded_bits"], 512), language=None)

    with tabs[5]:
        if rx is not None:
            bits = res["decoded_bits"]
            fr = COR.parse_frame(bits)
            lib = COR.search_sync(sy.get("raw_bits", sy["bits"]))
            st.write("**Sync-word library matches (raw stream):** " + (", ".join(f"{h['pattern']} @ {h['position']} ({h['mismatches']}/{h['length']} errors)" for h in lib[:3]) or "none"))
            if fr and fr["crc_ok"]:
                hx, asc = COR.hex_ascii(fr["payload"], 48)
                st.success(f"Frame found: length field {fr['length']} bytes, CRC-16 OK (0x{fr['crc_rx']:04X})")
                st.code(f"HEX   : {hx}\nASCII : {fr['payload'].decode('ascii', 'replace')}", language=None)
            else:
                hx, asc = COR.hex_ascii(COR.bits_to_bytes(bits, 32), 32)
                st.info("No CRC-valid frame. First bytes of the stream:")
                st.code(f"HEX   : {hx}\nASCII : {asc}", language=None)
            lag, strength = COR.estimate_frame_period(bits)
            st.write(f"Stream entropy: **{COR.bit_entropy(bits):.3f}** (1.000 = random) · periodicity: "
                     f"{'frame period ~%d bits' % lag if lag else 'none detected'}")

    with tabs[6]:
        st.caption("Offline rule-based analyst (the desktop app can additionally use Gemini with your own key).")
        for label in ("Summarize Signal", "Explain Pipeline Settings", "Tactical Assessment"):
            if st.button(label):
                st.code(offline_analyst(label, params), language=None)
else:
    with tabs[0]:
        st.info("Choose a bundled capture (or upload your own) in the sidebar and press **Run full pipeline**.")

with tabs[7]:
    st.write("BER vs Eb/N0 measured through the real receiver chain (RRC pulse, carrier offset, timing offset, AWGN).")
    csv_path = os.path.join(HERE, "results", "ber_vs_snr.csv")
    curves = None
    if st.button("Run quick benchmark now (seconds)"):
        with st.spinner("Simulating..."):
            curves = BM.run_benchmark(BM.POINTS_QUICK, max_trials=3)
    elif os.path.isfile(csv_path):
        curves = {}
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                c = curves.setdefault(r["curve"], {"ebn0": [], "ber": [], "bits": [], "lost": [], "frames": []})
                c["ebn0"].append(float(r["ebn0_db"])); c["ber"].append(float(r["ber"])); c["bits"].append(int(r["bits_tested"]))
                c["lost"].append(int(r["frames_lost"])); c["frames"].append(int(r["frames"]))
        st.caption("Showing stored full benchmark results.")
    if curves:
        fig = Figure(figsize=(9, 5.5))
        BM.plot_curves(fig, curves, dark=False)
        st.pyplot(fig)
