"""
NTRO Signal Analyzer — GUI for .IQ / .wav analysis, demodulation, FEC, correlation.
Simple ideation-round demo application.
"""

from __future__ import annotations
import threading
from google import genai

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import numpy as np

# --- Backend Connections ---
from analysis import (
    compute_spectrum,
    compute_waterfall,
    constellation_points,
    extract_parameters,
)
# Note: If you used the 10-second adapter fix earlier, change this back to 'mcorrelation'
from correlation import correlate_bits, format_bit_preview, pattern_from_hex
from deinterleave import deinterleave
from demod import demodulate
from fec import fec_decode
from signal_loader import load_signal

# --- Global UI Theme ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SignalAnalyzerApp(ctk.CTk):
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
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            header,
            text="Automated Signal Analysis — HF/VHF/UHF (.IQ & .WAV)",
            font=("Segoe UI", 18, "bold"),
        ).pack(side=tk.LEFT)
        
        ctk.CTkButton(header, text="Run Full Pipeline", command=self.run_pipeline, corner_radius=8, font=("Segoe UI", 12, "bold")).pack(side=tk.RIGHT, padx=4)
        ctk.CTkButton(header, text="Load File", command=self.load_file, corner_radius=8, font=("Segoe UI", 12, "bold")).pack(side=tk.RIGHT, padx=4)

        # Input Options
        opts = ctk.CTkFrame(self, corner_radius=8, border_width=2, border_color="#005500") 
# Gives a dark green glowing edge that pops against the dark gray background
        opts.pack(fill=tk.X, padx=10, pady=4)
        
        ctk.CTkLabel(opts, text="Input Options (.IQ):", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=10, pady=10)
        ctk.CTkLabel(opts, text="Sample Rate (Hz):", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=(10, 5))
        
        self.iq_fs_var = ctk.StringVar(value="2000000")
        ctk.CTkEntry(opts, textvariable=self.iq_fs_var, width=120).pack(side=tk.LEFT, padx=4)
        
        ctk.CTkLabel(opts, text="IQ Format:", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=(15, 5))
        self.iq_dtype_var = ctk.StringVar(value="cf32")
        ctk.CTkComboBox(
            opts, variable=self.iq_dtype_var, values=["cf32", "ci16", "cu8"], width=100
        ).pack(side=tk.LEFT, padx=4)

        # Notebook / Tabs
        self.nb = ctk.CTkTabview(self, corner_radius=8)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_params = self.nb.add("1. Parameters")
        self.tab_plots = self.nb.add("2. Spectral / Waterfall / Constellation")
        self.tab_demod = self.nb.add("3. Demodulation")
        self.tab_deint = self.nb.add("4. De-interleaving")
        self.tab_fec = self.nb.add("5. FEC")
        self.tab_corr = self.nb.add("6. Bit Correlation")
        self.tab_ai = self.nb.add("7. AI Analyst")

        self._build_params_tab()
        self._build_plots_tab()
        self._build_demod_tab()
        self._build_deint_tab()
        self._build_fec_tab()
        self._build_corr_tab()
        self._build_ai_tab()
        self.status = ctk.CTkLabel(self, text="Ready — load a .wav or .IQ file", font=("Segoe UI", 12, "italic"), anchor="w", fg_color="#1f1f1f", corner_radius=4)
        self.status.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10, ipady=5)

    def _build_params_tab(self):
        f = ctk.CTkFrame(self.tab_params, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ctk.CTkButton(f, text="Extract Parameters", command=self.extract_params, corner_radius=8, font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        self.params_text = ctk.CTkTextbox(f, font=("Consolas", 14), corner_radius=8)
        self.params_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def _build_plots_tab(self):
        f = ctk.CTkFrame(self.tab_plots, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True)
        # Force Matplotlib to have a dark background to match the UI
        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor='#242424')
        self.ax_spec = self.fig.add_subplot(221)
        self.ax_wf = self.fig.add_subplot(222)
        self.ax_const = self.fig.add_subplot(223)
        self.ax_time = self.fig.add_subplot(224)
        
        # Color the axes for dark mode
        for ax in [self.ax_spec, self.ax_wf, self.ax_const, self.ax_time]:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            
        self.fig.tight_layout()
        canvas_frame = ctk.CTkFrame(f)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()
        ctk.CTkButton(f, text="Refresh Plots", command=self.update_plots, corner_radius=8, font=("Segoe UI", 12, "bold")).pack(pady=10)

    def _build_demod_tab(self):
        f = ctk.CTkFrame(self.tab_demod, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill=tk.X)
        
        ctk.CTkLabel(row, text="Modulation:", font=("Segoe UI", 12)).pack(side=tk.LEFT)
        self.mod_var = ctk.StringVar(value="QPSK")
        ctk.CTkComboBox(
            row,
            variable=self.mod_var,
            values=["BPSK", "QPSK", "16-QAM", "FSK"],
            width=120,
        ).pack(side=tk.LEFT, padx=6)
        
        ctk.CTkLabel(row, text="Samples/Symbol:", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=(10,0))
        self.sps_var = ctk.StringVar(value="8")
        ctk.CTkEntry(row, textvariable=self.sps_var, width=60).pack(side=tk.LEFT, padx=4)
        
        ctk.CTkButton(row, text="Demodulate", command=self.run_demod, corner_radius=8, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=15)

        self.demod_text = ctk.CTkTextbox(f, font=("Consolas", 12), corner_radius=8)
        self.demod_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def _build_deint_tab(self):
        f = ctk.CTkFrame(self.tab_deint, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill=tk.X)
        
        ctk.CTkLabel(row, text="Mode:", font=("Segoe UI", 12)).pack(side=tk.LEFT)
        self.deint_var = ctk.StringVar(value="Block")
        ctk.CTkComboBox(
            row,
            variable=self.deint_var,
            values=["Block", "Convolutional", "Diagonal", "Pseudo-random"],
            width=180,
        ).pack(side=tk.LEFT, padx=6)
        
        ctk.CTkButton(row, text="De-interleave", command=self.run_deinterleave, corner_radius=8, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=15)
        self.deint_text = ctk.CTkTextbox(f, font=("Consolas", 12), corner_radius=8)
        self.deint_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def _build_fec_tab(self):
        f = ctk.CTkFrame(self.tab_fec, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill=tk.X)
        
        ctk.CTkLabel(row, text="FEC Decoder:", font=("Segoe UI", 12)).pack(side=tk.LEFT)
        self.fec_var = ctk.StringVar(value="Viterbi (Conv K=7)")
        ctk.CTkComboBox(
            row,
            variable=self.fec_var,
            values=[
                "Viterbi (Conv K=7)",
                "RS Block Code",
                "LDPC",
                "Concatenated (RS + Conv)",
            ],
            width=240,
        ).pack(side=tk.LEFT, padx=6)
        
        ctk.CTkButton(row, text="Apply FEC", command=self.run_fec, corner_radius=8, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=15)
        self.fec_text = ctk.CTkTextbox(f, font=("Consolas", 12), corner_radius=8)
        self.fec_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def _build_corr_tab(self):
        f = ctk.CTkFrame(self.tab_corr, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill=tk.X)
        
        ctk.CTkLabel(row, text="Preamble (hex):", font=("Segoe UI", 12)).pack(side=tk.LEFT)
        self.pattern_var = ctk.StringVar(value="7E7E7E")
        ctk.CTkEntry(row, textvariable=self.pattern_var, width=180).pack(side=tk.LEFT, padx=6)
        
        ctk.CTkButton(row, text="Correlate", command=self.run_correlation, corner_radius=8, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=15)
        self.corr_text = ctk.CTkTextbox(f, font=("Consolas", 12), corner_radius=8)
        self.corr_text.pack(fill=tk.BOTH, expand=True, pady=10)


    def _build_ai_tab(self):
        f = ctk.CTkFrame(self.tab_ai, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.ai_chat_log = ctk.CTkTextbox(f, font=("Roboto Mono", 13), corner_radius=8, state=tk.DISABLED)
        self.ai_chat_log.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        input_frame = ctk.CTkFrame(f, fg_color="transparent")
        input_frame.pack(fill=tk.X)
        
        self.ai_input = ctk.CTkEntry(input_frame, placeholder_text="Ask the AI Analyst about the current signal...", font=("Segoe UI", 13))
        self.ai_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.ai_input.bind("<Return>", lambda e: self.send_ai_query()) 
        
        self.ai_button = ctk.CTkButton(
            input_frame, text="Send", command=self.send_ai_query, 
            corner_radius=8, font=("Segoe UI", 12, "bold"),
            fg_color="#1a1a1a", border_width=2, border_color="#00ff00", hover_color="#004400"
        )
        self.ai_button.pack(side=tk.RIGHT)

    def send_ai_query(self):
        query = self.ai_input.get()
        if not query.strip():
            return
            
        self.ai_input.delete(0, tk.END)
        self._append_to_chat(f"OPERATOR: {query}\n\n")
        threading.Thread(target=self._process_ai_query, args=(query,), daemon=True).start()
        
    def _process_ai_query(self, query):
        try:
            sig_info = "No signal loaded."
            if self.samples is not None:
                sig_info = (
                    f"File: {self.file_type.upper()} at {self.fs:,.0f} Hz. "
                    f"Length: {len(self.samples)} samples. "
                    f"Selected Demodulator: {self.mod_var.get()}. "
                    f"Selected FEC: {self.fec_var.get()}."
                )
                if self.bitstream is not None:
                    sig_info += f" Demodulated Bit Count: {len(self.bitstream)}."
            
            system_prompt = (
                "You are an expert Signals Intelligence (SIGINT) AI embedded in a software called SPECTRA. "
                "Keep your answers brief, highly technical, and directly related to the user's query. "
                "Do not use markdown formatting like bolding or headers, just plain text. "
                f"Current live system state: {sig_info}"
            )
            
            client = genai.Client(api_key= "AQ.Ab8RN6JTaa07pKJqKVKwf_2ziSyntpaA6PxF741IcJUIv552-A")
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=f"{system_prompt}\n\nUser Question: {query}"
            )
            self._append_to_chat(f"SPECTRA AI: {response.text}\n\n{'='*50}\n\n")
            
        except Exception as e:
            self._append_to_chat(f"SYSTEM ERROR: AI connection failed. {str(e)}\n\n")

    def _append_to_chat(self, text):
        def update():
            self.ai_chat_log.configure(state=tk.NORMAL)
            self.ai_chat_log.insert(tk.END, text)
            self.ai_chat_log.see(tk.END)
            self.ai_chat_log.configure(state=tk.DISABLED)
        self.after(0, update)

    def _set_status(self, msg: str):
        self.status.configure(text=f"  {msg}")
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
        self.ax_spec.plot(freqs / 1e3, spec, color="#00ff00", linewidth=0.8) # Changed to neon green for cyber theme
        self.ax_spec.set_title("Spectrum")
        self.ax_spec.set_xlabel("Frequency (kHz)")
        self.ax_spec.set_ylabel("Magnitude (dB)")
        self.ax_spec.grid(True, alpha=0.1)

        self.ax_wf.clear()
        self.ax_wf.pcolormesh(t, f / 1e3, sxx, shading="auto", cmap="viridis")
        self.ax_wf.set_title("Waterfall (Time-Frequency)")
        self.ax_wf.set_xlabel("Time (s)")
        self.ax_wf.set_ylabel("Frequency (kHz)")

        self.ax_const.clear()
        self.ax_const.scatter(np.real(pts), np.imag(pts), s=4, alpha=0.5, c="#00ffff") # Cyan constellation
        self.ax_const.set_title("Constellation")
        self.ax_const.set_xlabel("I")
        self.ax_const.set_ylabel("Q")
        self.ax_const.grid(True, alpha=0.1)
        self.ax_const.set_aspect("equal", adjustable="box")

        n_show = min(2000, len(self.samples))
        t_ax = np.arange(n_show) / self.fs
        self.ax_time.clear()
        self.ax_time.plot(t_ax, np.real(self.samples[:n_show]), label="I", linewidth=0.8, color="#00ff00")
        self.ax_time.plot(t_ax, np.imag(self.samples[:n_show]), label="Q", linewidth=0.8, alpha=0.7, color="#ff00ff")
        self.ax_time.set_title("Time Domain (I/Q)")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.legend(loc="upper right", fontsize=8, facecolor='#242424', labelcolor='white')
        self.ax_time.grid(True, alpha=0.1)
        # --- Re-apply the dark theme colors after clearing ---
        for ax in [self.ax_spec, self.ax_wf, self.ax_const, self.ax_time]:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            # Make the graph borders gray instead of black
            for spine in ax.spines.values():
                spine.set_edgecolor('#555555')

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
