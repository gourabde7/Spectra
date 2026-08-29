"""
NTRO Signal Analyzer — GUI for .IQ / .wav analysis, demodulation, FEC, correlation.
Simple ideation-round demo application.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import numpy as np

from analysis import (
    compute_spectrum,
    compute_waterfall,
    constellation_points,
    extract_parameters,
)
from correlation import correlate_bits, format_bit_preview, pattern_from_hex
from deinterleave import deinterleave
from demod import demodulate
from fec import fec_decode
from signal_loader import load_signal


class SignalAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Signal Analyzer — IQ / WAV Processing")
        self.geometry("1200x780")
        self.minsize(1000, 650)

        self.samples: np.ndarray | None = None
        self.fs: float = 0.0
        self.file_type: str = ""
        self.file_path: str = ""
        self.bitstream: np.ndarray | None = None

        self._build_ui()

    def _build_ui(self):
        header = ttk.Frame(self, padding=8)
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Automated Signal Analysis — HF/VHF/UHF (.IQ & .WAV)",
            font=("Segoe UI", 14, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Button(header, text="Load File", command=self.load_file).pack(side=tk.RIGHT, padx=4)
        ttk.Button(header, text="Run Full Pipeline", command=self.run_pipeline).pack(
            side=tk.RIGHT, padx=4
        )

        opts = ttk.LabelFrame(self, text="Input Options (for .IQ files)", padding=6)
        opts.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(opts, text="Sample Rate (Hz):").grid(row=0, column=0, sticky=tk.W)
        self.iq_fs_var = tk.StringVar(value="2000000")
        ttk.Entry(opts, textvariable=self.iq_fs_var, width=14).grid(row=0, column=1, padx=4)
        ttk.Label(opts, text="IQ Format:").grid(row=0, column=2, sticky=tk.W, padx=(12, 0))
        self.iq_dtype_var = tk.StringVar(value="cf32")
        ttk.Combobox(
            opts, textvariable=self.iq_dtype_var, values=["cf32", "ci16", "cu8"], width=8
        ).grid(row=0, column=3, padx=4)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.tab_params = ttk.Frame(nb)
        self.tab_plots = ttk.Frame(nb)
        self.tab_demod = ttk.Frame(nb)
        self.tab_deint = ttk.Frame(nb)
        self.tab_fec = ttk.Frame(nb)
        self.tab_corr = ttk.Frame(nb)

        nb.add(self.tab_params, text="1. Parameters")
        nb.add(self.tab_plots, text="2. Spectral / Waterfall / Constellation")
        nb.add(self.tab_demod, text="3. Demodulation")
        nb.add(self.tab_deint, text="4. De-interleaving")
        nb.add(self.tab_fec, text="5. FEC")
        nb.add(self.tab_corr, text="6. Bit Correlation")

        self._build_params_tab()
        self._build_plots_tab()
        self._build_demod_tab()
        self._build_deint_tab()
        self._build_fec_tab()
        self._build_corr_tab()

        self.status = ttk.Label(self, text="Ready — load a .wav or .IQ file", relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_params_tab(self):
        f = ttk.Frame(self.tab_params, padding=8)
        f.pack(fill=tk.BOTH, expand=True)
        ttk.Button(f, text="Extract Parameters", command=self.extract_params).pack(anchor=tk.W)
        self.params_text = tk.Text(f, height=18, font=("Consolas", 11))
        self.params_text.pack(fill=tk.BOTH, expand=True, pady=8)

    def _build_plots_tab(self):
        f = ttk.Frame(self.tab_plots)
        f.pack(fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax_spec = self.fig.add_subplot(221)
        self.ax_wf = self.fig.add_subplot(222)
        self.ax_const = self.fig.add_subplot(223)
        self.ax_time = self.fig.add_subplot(224)
        self.fig.tight_layout()
        canvas_frame = ttk.Frame(f)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, canvas_frame)
        ttk.Button(f, text="Refresh Plots", command=self.update_plots).pack(pady=4)

    def _build_demod_tab(self):
        f = ttk.Frame(self.tab_demod, padding=8)
        f.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Modulation:").pack(side=tk.LEFT)
        self.mod_var = tk.StringVar(value="QPSK")
        ttk.Combobox(
            row,
            textvariable=self.mod_var,
            values=["BPSK", "QPSK", "16-QAM", "FSK"],
            width=12,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Samples/Symbol:").pack(side=tk.LEFT)
        self.sps_var = tk.StringVar(value="8")
        ttk.Entry(row, textvariable=self.sps_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Demodulate", command=self.run_demod).pack(side=tk.LEFT, padx=8)
        self.demod_text = tk.Text(f, height=20, font=("Consolas", 10))
        self.demod_text.pack(fill=tk.BOTH, expand=True, pady=8)

    def _build_deint_tab(self):
        f = ttk.Frame(self.tab_deint, padding=8)
        f.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Mode:").pack(side=tk.LEFT)
        self.deint_var = tk.StringVar(value="Block")
        ttk.Combobox(
            row,
            textvariable=self.deint_var,
            values=["Block", "Convolutional", "Diagonal", "Pseudo-random"],
            width=18,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="De-interleave", command=self.run_deinterleave).pack(
            side=tk.LEFT, padx=8
        )
        self.deint_text = tk.Text(f, height=20, font=("Consolas", 10))
        self.deint_text.pack(fill=tk.BOTH, expand=True, pady=8)

    def _build_fec_tab(self):
        f = ttk.Frame(self.tab_fec, padding=8)
        f.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        ttk.Label(row, text="FEC Decoder:").pack(side=tk.LEFT)
        self.fec_var = tk.StringVar(value="Viterbi (Conv K=7)")
        ttk.Combobox(
            row,
            textvariable=self.fec_var,
            values=[
                "Viterbi (Conv K=7)",
                "RS Block Code",
                "LDPC",
                "Concatenated (RS + Conv)",
            ],
            width=24,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Apply FEC", command=self.run_fec).pack(side=tk.LEFT, padx=8)
        self.fec_text = tk.Text(f, height=20, font=("Consolas", 10))
        self.fec_text.pack(fill=tk.BOTH, expand=True, pady=8)

    def _build_corr_tab(self):
        f = ttk.Frame(self.tab_corr, padding=8)
        f.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Preamble (hex):").pack(side=tk.LEFT)
        self.pattern_var = tk.StringVar(value="7E7E7E")
        ttk.Entry(row, textvariable=self.pattern_var, width=20).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Correlate", command=self.run_correlation).pack(side=tk.LEFT, padx=8)
        self.corr_text = tk.Text(f, height=20, font=("Consolas", 10))
        self.corr_text.pack(fill=tk.BOTH, expand=True, pady=8)

    def _set_status(self, msg: str):
        self.status.config(text=msg)
        self.update_idletasks()

    def load_file(self):
        path = filedialog.askopenfilename(
            title="Select .wav or .IQ file",
            filetypes=[
                ("Signal files", "*.wav *.iq *.bin *.raw *.dat"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            fs_iq = float(self.iq_fs_var.get())
            samples, fs, ftype = load_signal(
                path, iq_sample_rate=fs_iq, iq_dtype=self.iq_dtype_var.get()
            )
            self.samples = samples
            self.fs = fs
            self.file_type = ftype
            self.file_path = path
            self.bitstream = None
            self._set_status(f"Loaded {ftype.upper()}: {path} | fs={fs:,.0f} Hz | N={len(samples):,}")
            self.extract_params()
            self.update_plots()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _require_signal(self) -> bool:
        if self.samples is None:
            messagebox.showwarning("No Data", "Please load a signal file first.")
            return False
        return True

    def extract_params(self):
        if not self._require_signal():
            return
        params = extract_parameters(self.samples, self.fs)
        self.params_text.delete("1.0", tk.END)
        self.params_text.insert(tk.END, "=== Extracted Signal Parameters ===\n\n")
        for k, v in params.items():
            self.params_text.insert(tk.END, f"  {k:30s}: {v}\n")
        self.params_text.insert(tk.END, f"\n  File Type                     : {self.file_type.upper()}\n")
        self.params_text.insert(tk.END, f"  Source                        : {self.file_path}\n")
        self._set_status("Parameters extracted")

    def update_plots(self):
        if not self._require_signal():
            return
        freqs, spec = compute_spectrum(self.samples, self.fs)
        f, t, sxx = compute_waterfall(self.samples, self.fs)
        pts = constellation_points(self.samples)

        self.ax_spec.clear()
        self.ax_spec.plot(freqs / 1e3, spec, color="#1f77b4", linewidth=0.8)
        self.ax_spec.set_title("Spectrum")
        self.ax_spec.set_xlabel("Frequency (kHz)")
        self.ax_spec.set_ylabel("Magnitude (dB)")
        self.ax_spec.grid(True, alpha=0.3)

        self.ax_wf.clear()
        self.ax_wf.pcolormesh(t, f / 1e3, sxx, shading="auto", cmap="viridis")
        self.ax_wf.set_title("Waterfall (Time-Frequency)")
        self.ax_wf.set_xlabel("Time (s)")
        self.ax_wf.set_ylabel("Frequency (kHz)")

        self.ax_const.clear()
        self.ax_const.scatter(np.real(pts), np.imag(pts), s=4, alpha=0.5, c="#d62728")
        self.ax_const.set_title("Constellation")
        self.ax_const.set_xlabel("I")
        self.ax_const.set_ylabel("Q")
        self.ax_const.grid(True, alpha=0.3)
        self.ax_const.set_aspect("equal", adjustable="box")

        n_show = min(2000, len(self.samples))
        t_ax = np.arange(n_show) / self.fs
        self.ax_time.clear()
        self.ax_time.plot(t_ax, np.real(self.samples[:n_show]), label="I", linewidth=0.6)
        self.ax_time.plot(t_ax, np.imag(self.samples[:n_show]), label="Q", linewidth=0.6, alpha=0.7)
        self.ax_time.set_title("Time Domain (I/Q)")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.legend(loc="upper right", fontsize=8)
        self.ax_time.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()
        self._set_status("Plots updated")

    def run_demod(self):
        if not self._require_signal():
            return
        try:
            sps = int(self.sps_var.get())
            bits = demodulate(self.samples, self.mod_var.get(), sps=sps)
            self.bitstream = bits
            self.demod_text.delete("1.0", tk.END)
            self.demod_text.insert(tk.END, f"Demodulation: {self.mod_var.get()}\n")
            self.demod_text.insert(tk.END, f"Samples/symbol: {sps}\n")
            self.demod_text.insert(tk.END, f"Bit count: {len(bits)}\n\n")
            self.demod_text.insert(tk.END, format_bit_preview(bits, 256))
            self._set_status(f"Demodulated — {len(bits)} bits")
        except Exception as e:
            messagebox.showerror("Demod Error", str(e))

    def run_deinterleave(self):
        if self.bitstream is None:
            messagebox.showwarning("No Bits", "Run demodulation first.")
            return
        mode = self.deint_var.get()
        out = deinterleave(self.bitstream, mode)
        self.bitstream = out
        self.deint_text.delete("1.0", tk.END)
        self.deint_text.insert(tk.END, f"De-interleaving: {mode}\n")
        self.deint_text.insert(tk.END, f"Output bits: {len(out)}\n\n")
        self.deint_text.insert(tk.END, format_bit_preview(out, 256))
        self._set_status(f"De-interleaved ({mode})")

    def run_fec(self):
        if self.bitstream is None:
            messagebox.showwarning("No Bits", "Run demodulation first.")
            return
        out = fec_decode(self.bitstream, self.fec_var.get())
        self.bitstream = out
        self.fec_text.delete("1.0", tk.END)
        self.fec_text.insert(tk.END, f"FEC Decoder: {self.fec_var.get()}\n")
        self.fec_text.insert(tk.END, f"Decoded bits: {len(out)}\n\n")
        self.fec_text.insert(tk.END, format_bit_preview(out, 256))
        self._set_status(f"FEC applied ({self.fec_var.get()})")

    def run_correlation(self):
        if self.bitstream is None:
            messagebox.showwarning("No Bits", "Run demodulation first.")
            return
        try:
            pattern = pattern_from_hex(self.pattern_var.get())
            scores, hits = correlate_bits(self.bitstream, pattern)
            self.corr_text.delete("1.0", tk.END)
            self.corr_text.insert(tk.END, f"Preamble pattern: {self.pattern_var.get()}\n")
            self.corr_text.insert(tk.END, f"Pattern length: {len(pattern)} bits\n")
            self.corr_text.insert(tk.END, f"Match positions: {hits[:20]}\n")
            if len(hits) > 20:
                self.corr_text.insert(tk.END, f"  ... and {len(hits) - 20} more\n")
            if len(scores):
                best = int(np.argmin(scores))
                self.corr_text.insert(
                    tk.END, f"\nBest offset: {best} (mismatches: {int(scores[best])})\n"
                )
                payload_start = best + len(pattern)
                self.corr_text.insert(
                    tk.END,
                    f"Payload preview @ {payload_start}: "
                    f"{format_bit_preview(self.bitstream[payload_start:], 64)}\n",
                )
            self._set_status(f"Correlation complete — {len(hits)} hits")
        except Exception as e:
            messagebox.showerror("Correlation Error", str(e))

    def run_pipeline(self):
        if not self._require_signal():
            return
        self.extract_params()
        self.update_plots()
        self.run_demod()
        self.run_deinterleave()
        self.run_fec()
        self.run_correlation()
        self._set_status("Full analysis pipeline completed")


def main():
    app = SignalAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
