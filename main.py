"""
NTRO Signal Analyzer — GUI for .IQ / .wav analysis, demodulation, FEC, correlation.
Simple ideation-round demo application.
"""

from __future__ import annotations
from datetime import datetime
import html
import io
import os
import textwrap
import time
import threading
import zipfile
from google import genai

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np

# --- Backend Connections ---
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
        self.extracted_params: dict = {}
        self.last_pipeline_time: float = 0.0

        # Cached plot arrays so exports and view switches are instant
        self._cached_freqs: np.ndarray | None = None
        self._cached_spec: np.ndarray | None = None
        self._cached_f: np.ndarray | None = None
        self._cached_t: np.ndarray | None = None
        self._cached_sxx: np.ndarray | None = None
        self._cached_pts: np.ndarray | None = None

        # Interactive pan state for plot dragging
        self._pan_ax = None
        self._pan_start = None

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

        # Export Mission Report Button + Format Selector (PDF / DOCX / TXT)
        ctk.CTkButton(
            header,
            text="Export Mission Report",
            command=self.export_report,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
            fg_color="#006666",
            hover_color="#004d4d",
        ).pack(side=tk.RIGHT, padx=4)

        self.export_fmt_var = ctk.StringVar(value="PDF (.pdf)")
        self.export_fmt_combo = ctk.CTkComboBox(
            header,
            variable=self.export_fmt_var,
            values=["PDF (.pdf)", "Word (.docx)", "Text (.txt)"],
            width=125,
            font=("Segoe UI", 12),
        )
        self.export_fmt_combo.pack(side=tk.RIGHT, padx=4)

        ctk.CTkButton(
            header,
            text="Run Full Pipeline",
            command=self.run_pipeline,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.RIGHT, padx=4)

        ctk.CTkButton(
            header,
            text="Load File",
            command=self.load_file,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.RIGHT, padx=4)

        # Input Options
        opts = ctk.CTkFrame(self, corner_radius=8, border_width=2, border_color="#005500")
        opts.pack(fill=tk.X, padx=10, pady=4)

        ctk.CTkLabel(opts, text="Input Options (.IQ):", font=("Segoe UI", 12, "bold")).pack(
            side=tk.LEFT, padx=10, pady=10
        )
        ctk.CTkLabel(opts, text="Sample Rate (Hz):", font=("Segoe UI", 12)).pack(
            side=tk.LEFT, padx=(10, 5)
        )

        self.iq_fs_var = ctk.StringVar(value="2000000")
        ctk.CTkEntry(opts, textvariable=self.iq_fs_var, width=120).pack(side=tk.LEFT, padx=4)

        ctk.CTkLabel(opts, text="IQ Format:", font=("Segoe UI", 12)).pack(
            side=tk.LEFT, padx=(15, 5)
        )
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

        self.status = ctk.CTkLabel(
            self,
            text="Ready — load a .wav or .IQ file",
            font=("Segoe UI", 12, "italic"),
            anchor="w",
            fg_color="#1f1f1f",
            corner_radius=4,
        )
        self.status.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10, ipady=5)

    def _build_params_tab(self):
        f = ctk.CTkFrame(self.tab_params, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_row = ctk.CTkFrame(f, fg_color="transparent")
        btn_row.pack(fill=tk.X)

        ctk.CTkButton(
            btn_row,
            text="Extract Parameters",
            command=self.extract_params,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT)
        ctk.CTkButton(
            btn_row,
            text="Apply Recommended Params & Run Full Pipeline",
            command=self.run_pipeline,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
            fg_color="#00aa44",
            hover_color="#008833",
        ).pack(side=tk.LEFT, padx=10)

        self.params_text = ctk.CTkTextbox(f, font=("Consolas", 14), corner_radius=8)
        self.params_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def _build_plots_tab(self):
        f = ctk.CTkFrame(self.tab_plots, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True)

        top_ctrl = ctk.CTkFrame(f, fg_color="transparent")
        top_ctrl.pack(fill=tk.X, padx=6, pady=(4, 6))

        ctk.CTkLabel(top_ctrl, text="View:", font=("Segoe UI", 12, "bold")).pack(
            side=tk.LEFT, padx=(4, 6)
        )
        self.plot_view_var = ctk.StringVar(value="All 4 (Grid)")
        self.plot_selector = ctk.CTkSegmentedButton(
            top_ctrl,
            values=[
                "All 4 (Grid)",
                "Spectrum",
                "Waterfall",
                "Constellation",
                "Time Domain",
            ],
            variable=self.plot_view_var,
            command=lambda _: self._draw_cached_plots() if self.samples is not None else None,
            font=("Segoe UI", 12, "bold"),
        )
        self.plot_selector.pack(side=tk.LEFT, padx=4)

        ctk.CTkButton(
            top_ctrl,
            text="Reset Zoom",
            command=self._draw_cached_plots,
            corner_radius=8,
            width=100,
            font=("Segoe UI", 12, "bold"),
            fg_color="#333333",
            hover_color="#555555",
        ).pack(side=tk.RIGHT, padx=4)

        self.cmap_var = ctk.StringVar(value="viridis")
        self.cmap_combo = ctk.CTkComboBox(
            top_ctrl,
            variable=self.cmap_var,
            values=["viridis", "inferno", "plasma", "magma", "bone"],
            width=110,
            command=lambda _: self._draw_cached_plots() if self.samples is not None else None,
        )
        self.cmap_combo.pack(side=tk.RIGHT, padx=4)
        ctk.CTkLabel(top_ctrl, text="Theme:", font=("Segoe UI", 12)).pack(
            side=tk.RIGHT, padx=(8, 2)
        )

        self.fig = Figure(figsize=(10, 6.5), dpi=100, facecolor="#242424")
        canvas_frame = ctk.CTkFrame(f)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)

        bot_ctrl = ctk.CTkFrame(f, fg_color="transparent")
        bot_ctrl.pack(fill=tk.X, padx=10, pady=6)

        ctk.CTkLabel(
            bot_ctrl,
            text="Tip: Scroll Wheel = Zoom In/Out   |   Left-Click & Drag = Pan   |   Double-Click = Reset View",
            font=("Segoe UI", 12, "italic"),
            text_color="#00ffcc",
        ).pack(side=tk.LEFT)

        self.time_samples_var = ctk.IntVar(value=800)
        self.time_slider = ctk.CTkSlider(
            bot_ctrl,
            from_=100,
            to=4000,
            number_of_steps=39,
            variable=self.time_samples_var,
            width=160,
            command=lambda _: self._draw_cached_plots() if self.samples is not None else None,
        )
        self.time_slider.pack(side=tk.RIGHT, padx=6)
        ctk.CTkLabel(
            bot_ctrl, text="Time Wave Detail:", font=("Segoe UI", 12)
        ).pack(side=tk.RIGHT)

    def _on_plot_scroll(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        scale = 0.75 if event.button == "up" else 1.33
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(
            [
                event.xdata - (event.xdata - xlim[0]) * scale,
                event.xdata + (xlim[1] - event.xdata) * scale,
            ]
        )
        ax.set_ylim(
            [
                event.ydata - (event.ydata - ylim[0]) * scale,
                event.ydata + (ylim[1] - event.ydata) * scale,
            ]
        )
        self.canvas.draw_idle()

    def _on_plot_press(self, event):
        if event.inaxes is None:
            return
        if event.dblclick:
            self._draw_cached_plots()
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self._pan_ax = event.inaxes
            self._pan_start = (event.xdata, event.ydata)

    def _on_plot_release(self, _event):
        self._pan_ax = None
        self._pan_start = None

    def _on_plot_motion(self, event):
        if self._pan_ax is None or event.inaxes != self._pan_ax:
            return
        if event.xdata is None or event.ydata is None or self._pan_start is None:
            return
        dx = event.xdata - self._pan_start[0]
        dy = event.ydata - self._pan_start[1]
        xlim = self._pan_ax.get_xlim()
        ylim = self._pan_ax.get_ylim()
        self._pan_ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self._pan_ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self.canvas.draw_idle()

    def _build_demod_tab(self):
        f = ctk.CTkFrame(self.tab_demod, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill=tk.X)

        ctk.CTkLabel(row, text="Modulation:", font=("Segoe UI", 12)).pack(side=tk.LEFT)
        self.mod_var = ctk.StringVar(value="QPSK")
        self.mod_combo = ctk.CTkComboBox(
            row,
            variable=self.mod_var,
            values=["BPSK", "QPSK", "16-QAM", "FSK"],
            width=120,
        )
        self.mod_combo.pack(side=tk.LEFT, padx=6)

        ctk.CTkLabel(row, text="Samples/Symbol:", font=("Segoe UI", 12)).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.sps_var = ctk.StringVar(value="8")
        self.sps_entry = ctk.CTkEntry(row, textvariable=self.sps_var, width=60)
        self.sps_entry.pack(side=tk.LEFT, padx=4)

        ctk.CTkButton(
            row,
            text="Demodulate",
            command=self.run_demod,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT, padx=15)

        self.demod_text = ctk.CTkTextbox(f, font=("Consolas", 12), corner_radius=8)
        self.demod_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def _build_deint_tab(self):
        f = ctk.CTkFrame(self.tab_deint, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill=tk.X)

        ctk.CTkLabel(row, text="Mode:", font=("Segoe UI", 12)).pack(side=tk.LEFT)
        self.deint_var = ctk.StringVar(value="Block")
        self.deint_combo = ctk.CTkComboBox(
            row,
            variable=self.deint_var,
            values=["Block", "Convolutional", "Diagonal", "Pseudo-random"],
            width=180,
        )
        self.deint_combo.pack(side=tk.LEFT, padx=6)

        ctk.CTkButton(
            row,
            text="De-interleave",
            command=self.run_deinterleave,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT, padx=15)
        self.deint_text = ctk.CTkTextbox(f, font=("Consolas", 12), corner_radius=8)
        self.deint_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def _build_fec_tab(self):
        f = ctk.CTkFrame(self.tab_fec, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill=tk.X)

        ctk.CTkLabel(row, text="FEC Decoder:", font=("Segoe UI", 12)).pack(side=tk.LEFT)
        self.fec_var = ctk.StringVar(value="Viterbi (Conv K=7)")
        self.fec_combo = ctk.CTkComboBox(
            row,
            variable=self.fec_var,
            values=[
                "Viterbi (Conv K=7)",
                "RS Block Code",
                "LDPC",
                "Concatenated (RS + Conv)",
            ],
            width=240,
        )
        self.fec_combo.pack(side=tk.LEFT, padx=6)

        ctk.CTkButton(
            row,
            text="Apply FEC",
            command=self.run_fec,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT, padx=15)
        self.fec_text = ctk.CTkTextbox(f, font=("Consolas", 12), corner_radius=8)
        self.fec_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def _build_corr_tab(self):
        f = ctk.CTkFrame(self.tab_corr, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill=tk.X)

        ctk.CTkLabel(row, text="Preamble (hex):", font=("Segoe UI", 12)).pack(side=tk.LEFT)
        self.pattern_var = ctk.StringVar(value="7E7E7E")
        self.pattern_entry = ctk.CTkEntry(row, textvariable=self.pattern_var, width=180)
        self.pattern_entry.pack(side=tk.LEFT, padx=6)

        ctk.CTkButton(
            row,
            text="Correlate",
            command=self.run_correlation,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT, padx=15)
        self.corr_text = ctk.CTkTextbox(f, font=("Consolas", 12), corner_radius=8)
        self.corr_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def _build_ai_tab(self):
        f = ctk.CTkFrame(self.tab_ai, fg_color="transparent")
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        quick_frame = ctk.CTkFrame(f, fg_color="transparent")
        quick_frame.pack(fill=tk.X, pady=(0, 8))

        ctk.CTkLabel(
            quick_frame, text="Quick Actions:", font=("Segoe UI", 12, "bold")
        ).pack(side=tk.LEFT, padx=(0, 8))

        ctk.CTkButton(
            quick_frame,
            text="Summarize Signal",
            command=lambda: self._send_preset_ai_query(
                "Summarize the current signal parameters, SNR, bandwidth, and modulation profile."
            ),
            corner_radius=6,
            height=28,
            font=("Segoe UI", 11, "bold"),
            fg_color="#2b2b2b",
            hover_color="#005500",
        ).pack(side=tk.LEFT, padx=4)

        ctk.CTkButton(
            quick_frame,
            text="Explain Pipeline Settings",
            command=lambda: self._send_preset_ai_query(
                "Explain why the current Demodulation, De-interleaving, and FEC settings are suitable for this signal."
            ),
            corner_radius=6,
            height=28,
            font=("Segoe UI", 11, "bold"),
            fg_color="#2b2b2b",
            hover_color="#005500",
        ).pack(side=tk.LEFT, padx=4)

        ctk.CTkButton(
            quick_frame,
            text="Tactical Assessment",
            command=lambda: self._send_preset_ai_query(
                "Provide a brief tactical SIGINT assessment of this transmission and recommended operator actions."
            ),
            corner_radius=6,
            height=28,
            font=("Segoe UI", 11, "bold"),
            fg_color="#2b2b2b",
            hover_color="#005500",
        ).pack(side=tk.LEFT, padx=4)

        self.ai_chat_log = ctk.CTkTextbox(
            f, font=("Roboto Mono", 13), corner_radius=8, state=tk.DISABLED
        )
        self.ai_chat_log.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        input_frame = ctk.CTkFrame(f, fg_color="transparent")
        input_frame.pack(fill=tk.X)

        self.ai_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="Ask the AI Analyst about the current signal...",
            font=("Segoe UI", 13),
        )
        self.ai_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.ai_input.bind("<Return>", lambda e: self.send_ai_query())

        self.ai_button = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_ai_query,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
            fg_color="#1a1a1a",
            border_width=2,
            border_color="#00ff00",
            hover_color="#004400",
        )
        self.ai_button.pack(side=tk.RIGHT)

    def _send_preset_ai_query(self, preset_text: str):
        self.ai_input.delete(0, tk.END)
        self.ai_input.insert(0, preset_text)
        self.send_ai_query()

    def send_ai_query(self):
        query = self.ai_input.get()
        if not query.strip():
            return

        self.ai_input.delete(0, tk.END)
        self._append_to_chat(f"YOU: {query}\n\n")
        threading.Thread(target=self._process_ai_query, args=(query,), daemon=True).start()

    def _process_ai_query(self, query):
        try:
            sig_info = "No signal loaded."
            if self.samples is not None:
                params_summary = ", ".join(
                    f"{k}: {v}" for k, v in self.extracted_params.items()
                )
                sig_info = (
                    f"File: {self.file_type.upper()} at {self.fs:,.0f} Hz. "
                    f"Length: {len(self.samples)} samples. "
                    f"Extracted Parameters: [{params_summary}]. "
                    f"Selected Demodulator: {self.mod_combo.get()}. "
                    f"Selected De-interleaver: {self.deint_combo.get()}. "
                    f"Selected FEC: {self.fec_combo.get()}."
                )
                if self.bitstream is not None:
                    sig_info += f" Demodulated Bit Count: {len(self.bitstream)}."

            system_prompt = (
                "You are an expert Signals Intelligence (SIGINT) AI embedded in a software called SPECTRA. "
                "Keep your answers brief, highly technical, and directly related to the user's query. "
                "Do not use markdown formatting like bolding or headers, just plain text. "
                f"Current live system state: {sig_info}"
            )

            client = genai.Client(
                api_key="AQ.Ab8RN6JTaa07pKJqKVKwf_2ziSyntpaA6PxF741IcJUIv552-A"
            )
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=f"{system_prompt}\n\nUser Question: {query}",
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
            self._cached_freqs = None
            self._set_status(
                f"Loaded {ftype.upper()}: {path} | fs={fs:,.0f} Hz | N={len(samples):,}"
            )
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
        self.extracted_params = params if isinstance(params, dict) else {}
        self.params_text.delete("1.0", tk.END)
        self.params_text.insert(tk.END, "=== Extracted Signal Parameters ===\n\n")
        for k, v in self.extracted_params.items():
            self.params_text.insert(tk.END, f"  {k:30s}: {v}\n")
        self.params_text.insert(
            tk.END, f"\n  File Type                     : {self.file_type.upper()}\n"
        )
        self.params_text.insert(
            tk.END, f"  Source                        : {self.file_path}\n"
        )
        self._set_status("Parameters extracted")

    def _apply_recommended_params(self):
        """Maps Page 1 'Modulation (estimated)', 'FEC (inferred)', and 'Interleaving (inferred)' to Tabs 3, 4, and 5."""
        for k, v in self.extracted_params.items():
            k_low = str(k).lower()
            v_up = str(v).upper()

            if "modulation" in k_low:
                if "QAM" in v_up:
                    self.mod_var.set("16-QAM")
                    self.mod_combo.set("16-QAM")
                elif "QPSK" in v_up:
                    self.mod_var.set("QPSK")
                    self.mod_combo.set("QPSK")
                elif "BPSK" in v_up:
                    self.mod_var.set("BPSK")
                    self.mod_combo.set("BPSK")
                elif "FSK" in v_up:
                    self.mod_var.set("FSK")
                    self.mod_combo.set("FSK")

            elif "interleav" in k_low:
                if "PSEUDO" in v_up or "RANDOM" in v_up:
                    self.deint_var.set("Pseudo-random")
                    self.deint_combo.set("Pseudo-random")
                elif "CONV" in v_up:
                    self.deint_var.set("Convolutional")
                    self.deint_combo.set("Convolutional")
                elif "DIAG" in v_up:
                    self.deint_var.set("Diagonal")
                    self.deint_combo.set("Diagonal")
                elif "BLOCK" in v_up:
                    self.deint_var.set("Block")
                    self.deint_combo.set("Block")

            elif "fec" in k_low:
                if "CONCAT" in v_up or ("RS" in v_up and "CONV" in v_up):
                    self.fec_var.set("Concatenated (RS + Conv)")
                    self.fec_combo.set("Concatenated (RS + Conv)")
                elif "LDPC" in v_up:
                    self.fec_var.set("LDPC")
                    self.fec_combo.set("LDPC")
                elif "RS" in v_up or "REED" in v_up:
                    self.fec_var.set("RS Block Code")
                    self.fec_combo.set("RS Block Code")
                elif "VITERBI" in v_up or "CONV" in v_up:
                    self.fec_var.set("Viterbi (Conv K=7)")
                    self.fec_combo.set("Viterbi (Conv K=7)")

        self.params_text.insert(
            tk.END,
            f"\n=== Auto-Applied Settings to Pipeline ===\n"
            f"  Tab 3 Demodulation            : {self.mod_combo.get()}\n"
            f"  Tab 4 De-interleaving         : {self.deint_combo.get()}\n"
            f"  Tab 5 FEC Decoder             : {self.fec_combo.get()}\n",
        )

    def _style_axis(self, ax):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

    def update_plots(self):
        if not self._require_signal():
            return
        plot_samples = (
            self.samples[:150000] if len(self.samples) > 150000 else self.samples
        )
        self._cached_freqs, self._cached_spec = compute_spectrum(plot_samples, self.fs)
        self._cached_f, self._cached_t, self._cached_sxx = compute_waterfall(
            plot_samples, self.fs
        )
        self._cached_pts = constellation_points(plot_samples)
        self._draw_cached_plots()

    def _draw_cached_plots(self):
        if self.samples is None:
            return
        if self._cached_freqs is None:
            self.update_plots()
            return

        freqs, spec = self._cached_freqs, self._cached_spec
        f, t, sxx = self._cached_f, self._cached_t, self._cached_sxx
        pts = self._cached_pts

        selected_cmap = self.cmap_combo.get() if hasattr(self, "cmap_combo") else "viridis"
        view_mode = self.plot_view_var.get() if hasattr(self, "plot_view_var") else "All 4 (Grid)"
        n_show = min(int(self.time_samples_var.get()), len(self.samples))
        t_ax = np.arange(n_show) / self.fs

        self.fig.clear()

        def draw_spectrum(ax):
            ax.plot(freqs / 1e3, spec, color="#00ff00", linewidth=1.0)
            ax.set_title("Spectrum (Power Spectral Density)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Frequency (kHz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, alpha=0.15)
            self._style_axis(ax)

        def draw_waterfall(ax):
            ax.pcolormesh(t, f / 1e3, sxx, shading="auto", cmap=selected_cmap)
            ax.set_title(f"Waterfall — Time-Frequency ({selected_cmap})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (kHz)")
            self._style_axis(ax)

        def draw_constellation(ax):
            ax.scatter(np.real(pts), np.imag(pts), s=8, alpha=0.6, c="#00ffff")
            ax.set_title("IQ Constellation Diagram", fontsize=12, fontweight="bold")
            ax.set_xlabel("In-Phase (I)")
            ax.set_ylabel("Quadrature (Q)")
            ax.grid(True, alpha=0.15)
            ax.set_aspect("equal", adjustable="datalim")
            self._style_axis(ax)

        def draw_time(ax):
            ax.plot(t_ax, np.real(self.samples[:n_show]), label="I (In-Phase)", linewidth=1.0, color="#00ff00")
            ax.plot(t_ax, np.imag(self.samples[:n_show]), label="Q (Quadrature)", linewidth=1.0, alpha=0.75, color="#ff00ff")
            ax.set_title(f"Time Domain Waveform (First {n_show} Samples)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right", fontsize=9, facecolor="#242424", labelcolor="white")
            ax.grid(True, alpha=0.15)
            self._style_axis(ax)

        if view_mode == "Spectrum":
            draw_spectrum(self.fig.add_subplot(111))
        elif view_mode == "Waterfall":
            draw_waterfall(self.fig.add_subplot(111))
        elif view_mode == "Constellation":
            draw_constellation(self.fig.add_subplot(111))
        elif view_mode == "Time Domain":
            draw_time(self.fig.add_subplot(111))
        else:
            draw_spectrum(self.fig.add_subplot(221))
            draw_waterfall(self.fig.add_subplot(222))
            draw_constellation(self.fig.add_subplot(223))
            draw_time(self.fig.add_subplot(224))

        self.fig.tight_layout()
        self.canvas.draw()
        self._set_status(f"Plots updated — View: {view_mode} | Theme: {selected_cmap}")

    def run_demod(self):
        if not self._require_signal():
            return
        try:
            sps = int(self.sps_entry.get())
            mod_choice = self.mod_combo.get()
            active_samples = (
                self.samples[:500000] if len(self.samples) > 500000 else self.samples
            )
            bits = demodulate(active_samples, mod_choice, sps=sps)
            self.bitstream = bits
            self.demod_text.delete("1.0", tk.END)
            self.demod_text.insert(tk.END, f"Demodulation: {mod_choice}\n")
            self.demod_text.insert(tk.END, f"Samples/symbol: {sps}\n")
            self.demod_text.insert(tk.END, f"Bit count: {len(bits)}\n\n")
            self.demod_text.insert(tk.END, format_bit_preview(bits, 256))
            self._set_status(f"Demodulated ({mod_choice}) — {len(bits)} bits")
        except Exception as e:
            messagebox.showerror("Demod Error", str(e))

    def run_deinterleave(self):
        if self.bitstream is None:
            messagebox.showwarning("No Bits", "Run demodulation first.")
            return
        mode = self.deint_combo.get()
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
        fec_choice = self.fec_combo.get()
        out = fec_decode(self.bitstream, fec_choice)
        self.bitstream = out
        self.fec_text.delete("1.0", tk.END)
        self.fec_text.insert(tk.END, f"FEC Decoder: {fec_choice}\n")
        self.fec_text.insert(tk.END, f"Decoded bits: {len(out)}\n\n")
        self.fec_text.insert(tk.END, format_bit_preview(out, 256))
        self._set_status(f"FEC applied ({fec_choice})")

    @staticmethod
    def _bits_to_hex_and_ascii(bits: np.ndarray, max_bytes: int = 32) -> tuple[str, str]:
        n_bits = min(len(bits), max_bytes * 8)
        n_bits = (n_bits // 8) * 8
        if n_bits == 0:
            return "(insufficient bits)", "(insufficient bits)"

        byte_vals = []
        for i in range(0, n_bits, 8):
            chunk = bits[i : i + 8]
            val = 0
            for b in chunk:
                val = (val << 1) | (1 if int(b) != 0 else 0)
            byte_vals.append(val)

        hex_str = " ".join(f"{b:02X}" for b in byte_vals)
        ascii_str = "".join(chr(b) if 32 <= b <= 126 else "." for b in byte_vals)
        return hex_str, ascii_str

    def run_correlation(self):
        if self.bitstream is None:
            messagebox.showwarning("No Bits", "Run demodulation first.")
            return
        try:
            pattern = pattern_from_hex(self.pattern_entry.get())
            scores, hits = correlate_bits(self.bitstream, pattern)
            self.corr_text.delete("1.0", tk.END)
            self.corr_text.insert(tk.END, f"Preamble pattern: {self.pattern_entry.get()}\n")
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
                payload_slice = self.bitstream[payload_start:]
                self.corr_text.insert(
                    tk.END,
                    f"Payload preview @ {payload_start}: "
                    f"{format_bit_preview(payload_slice, 64)}\n",
                )

                hex_out, ascii_out = self._bits_to_hex_and_ascii(payload_slice, max_bytes=32)
                self.corr_text.insert(
                    tk.END,
                    f"\n=== Decoded Payload Inspector (First 32 Bytes) ===\n"
                    f"HEX   : {hex_out}\n"
                    f"ASCII : {ascii_out}\n",
                )
            self._set_status(f"Correlation complete — {len(hits)} hits")
        except Exception as e:
            messagebox.showerror("Correlation Error", str(e))

    def _save_as_docx(self, filepath: str, timestamp: str):
        """Builds a styled Microsoft Word (.docx) report using built-in zipfile + XML."""
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        )
        rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>"
        )

        def section_heading(title_str: str) -> str:
            t = html.escape(title_str)
            return (
                '<w:p><w:pPr>'
                '<w:shd w:val="clear" w:color="auto" w:fill="0F2C44"/>'
                '<w:spacing w:before="220" w:after="100"/>'
                '</w:pPr>'
                '<w:r><w:rPr>'
                '<w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/>'
                '<w:b/><w:color w:val="FFFFFF"/><w:sz w:val="22"/>'
                '</w:rPr>'
                f'<w:t xml:space="preserve">  {t}</w:t>'
                '</w:r></w:p>'
            )

        def make_table(headers: tuple[str, str], rows: list[tuple[str, str]]) -> str:
            h1, h2 = html.escape(headers[0]), html.escape(headers[1])
            xml_rows = [
                '<w:tr>'
                '<w:tc><w:tcPr><w:tcW w:w="4200" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="006666"/></w:tcPr>'
                '<w:p><w:r><w:rPr><w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/><w:b/><w:color w:val="FFFFFF"/><w:sz w:val="20"/></w:rPr>'
                f'<w:t xml:space="preserve"> {h1}</w:t></w:r></w:p></w:tc>'
                '<w:tc><w:tcPr><w:tcW w:w="5300" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="006666"/></w:tcPr>'
                '<w:p><w:r><w:rPr><w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/><w:b/><w:color w:val="FFFFFF"/><w:sz w:val="20"/></w:rPr>'
                f'<w:t xml:space="preserve"> {h2}</w:t></w:r></w:p></w:tc>'
                '</w:tr>'
            ]
            for idx, (k, v) in enumerate(rows):
                bg = "F2F7FA" if idx % 2 == 0 else "FFFFFF"
                ke, ve = html.escape(str(k)), html.escape(str(v))
                xml_rows.append(
                    '<w:tr>'
                    f'<w:tc><w:tcPr><w:tcW w:w="4200" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="{bg}"/></w:tcPr>'
                    '<w:p><w:r><w:rPr><w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/><w:b/><w:color w:val="1B263B"/><w:sz w:val="20"/></w:rPr>'
                    f'<w:t xml:space="preserve"> {ke}</w:t></w:r></w:p></w:tc>'
                    f'<w:tc><w:tcPr><w:tcW w:w="5300" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="{bg}"/></w:tcPr>'
                    '<w:p><w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:color w:val="0A192F"/><w:sz w:val="20"/></w:rPr>'
                    f'<w:t xml:space="preserve"> {ve}</w:t></w:r></w:p></w:tc>'
                    '</w:tr>'
                )
            return (
                '<w:tbl>'
                '<w:tblPr>'
                '<w:tblW w:w="9500" w:type="dxa"/>'
                '<w:tblBorders>'
                '<w:top w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>'
                '<w:left w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>'
                '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>'
                '<w:right w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>'
                '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
                '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
                '</w:tblBorders>'
                '</w:tblPr>'
                f"{''.join(xml_rows)}"
                '</w:tbl>'
            )

        def callout_box(raw_text: str) -> str:
            text_clean = raw_text.strip() or "(Stage not executed yet)"
            lines_xml = []
            for line in text_clean.splitlines():
                for wrapped in textwrap.wrap(line, width=88) or [""]:
                    esc = html.escape(wrapped)
                    lines_xml.append(
                        '<w:p><w:pPr>'
                        '<w:pBdr><w:left w:val="single" w:sz="24" w:space="8" w:color="00AA66"/></w:pBdr>'
                        '<w:shd w:val="clear" w:color="auto" w:fill="F4F6F9"/>'
                        '<w:spacing w:before="20" w:after="20"/>'
                        '</w:pPr>'
                        '<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:color w:val="112233"/><w:sz w:val="18"/></w:rPr>'
                        f'<w:t xml:space="preserve">{esc}</w:t></w:r></w:p>'
                    )
            return "".join(lines_xml)

        banner_xml = (
            '<w:p><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="0B192C"/><w:spacing w:before="120" w:after="40"/></w:pPr>'
            '<w:r><w:rPr><w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/><w:b/><w:color w:val="00FFCC"/><w:sz w:val="32"/></w:rPr>'
            '<w:t xml:space="preserve">  SPECTRA // AUTOMATED SIGINT MISSION DOSSIER</w:t></w:r></w:p>'
            '<w:p><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="0B192C"/><w:spacing w:before="0" w:after="160"/></w:pPr>'
            '<w:r><w:rPr><w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/><w:color w:val="D0E1F9"/><w:sz w:val="18"/></w:rPr>'
            f'<w:t xml:space="preserve">  Generated: {html.escape(timestamp)}   |   Pipeline Execution: {self.last_pipeline_time:.2f}s   |   Format: {html.escape(self.file_type.upper())}</w:t></w:r></w:p>'
        )

        param_rows = [(str(k), str(v)) for k, v in self.extracted_params.items()]
        param_rows.append(("Signal File Type", self.file_type.upper()))
        param_rows.append(("Signal Source Path", self.file_path))

        config_rows = [
            ("Tab 3 — Active Demodulator", self.mod_combo.get()),
            ("Tab 3 — Samples Per Symbol (SPS)", self.sps_entry.get()),
            ("Tab 4 — De-interleaving Mode", self.deint_combo.get()),
            ("Tab 5 — FEC Decoder", self.fec_combo.get()),
            ("Tab 6 — Correlation Preamble (Hex)", self.pattern_entry.get()),
            ("Total Pipeline Execution Time", f"{self.last_pipeline_time:.2f} seconds"),
        ]

        body_parts = [
            banner_xml,
            section_heading("1. EXTRACTED SIGNAL PARAMETERS (PAGE 1 TELEMETRY)"),
            make_table(("Signal Parameter", "Extracted Value"), param_rows),
            section_heading("2. AUTO-APPLIED DSP PIPELINE CONFIGURATION"),
            make_table(("Pipeline Stage / Setting", "Configured Value"), config_rows),
            section_heading("3. DEMODULATION TELEMETRY & BITSTREAM PREVIEW"),
            callout_box(self.demod_text.get("1.0", tk.END)),
            section_heading("4. DE-INTERLEAVING OUTPUT SUMMARY"),
            callout_box(self.deint_text.get("1.0", tk.END)),
            section_heading("5. FORWARD ERROR CORRECTION (FEC) SUMMARY"),
            callout_box(self.fec_text.get("1.0", tk.END)),
            section_heading("6. BIT CORRELATION & DECODED PAYLOAD INSPECTOR"),
            callout_box(self.corr_text.get("1.0", tk.END)),
        ]

        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{''.join(body_parts)}</w:body>"
            "</w:document>"
        )

        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", rels)
            zf.writestr("word/document.xml", doc_xml)

    def _save_as_pdf(self, filepath: str, timestamp: str):
        """
        100% Pure-Python Vector PDF 1.4 Generator (5 Pages).
        Zero Matplotlib, zero PIL, zero C-extensions — writes vector PDF commands directly to disk.
        """
        if self._cached_freqs is None:
            plot_samples = self.samples[:150000] if len(self.samples) > 150000 else self.samples
            self._cached_freqs, self._cached_spec = compute_spectrum(plot_samples, self.fs)
            self._cached_f, self._cached_t, self._cached_sxx = compute_waterfall(plot_samples, self.fs)
            self._cached_pts = constellation_points(plot_samples)

        W, H = 842.0, 595.0  # Landscape A4 in PDF points

        def hex_rgb(h: str) -> tuple[float, float, float]:
            h = h.lstrip("#")
            return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)

        def pdf_esc(s: str) -> str:
            clean = "".join(ch if 32 <= ord(ch) <= 126 else "?" for ch in str(s))
            return clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        def rect_cmd(x: float, y_top: float, w: float, h: float, fill_hex: str | None = None, stroke_hex: str | None = None, lw: float = 1.0) -> str:
            py = H - y_top - h
            cmds = []
            if fill_hex and stroke_hex:
                fr, fg, fb = hex_rgb(fill_hex)
                sr, sg, sb = hex_rgb(stroke_hex)
                cmds.append(f"{lw:.2f} w {fr:.3f} {fg:.3f} {fb:.3f} rg {sr:.3f} {sg:.3f} {sb:.3f} RG {x:.2f} {py:.2f} {w:.2f} {h:.2f} re B")
            elif fill_hex:
                fr, fg, fb = hex_rgb(fill_hex)
                cmds.append(f"{fr:.3f} {fg:.3f} {fb:.3f} rg {x:.2f} {py:.2f} {w:.2f} {h:.2f} re f")
            elif stroke_hex:
                sr, sg, sb = hex_rgb(stroke_hex)
                cmds.append(f"{lw:.2f} w {sr:.3f} {sg:.3f} {sb:.3f} RG {x:.2f} {py:.2f} {w:.2f} {h:.2f} re S")
            return "\n".join(cmds)

        def text_cmd(x: float, y_top: float, txt: str, font: str = "F1", size: float = 10.0, color_hex: str = "#FFFFFF") -> str:
            py = H - y_top
            r, g, b = hex_rgb(color_hex)
            return f"BT /{font} {size:.1f} Tf {r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {py:.2f} Td ({pdf_esc(txt)}) Tj ET"

        def line_cmd(x1: float, y1_top: float, x2: float, y2_top: float, color_hex: str = "#FFFFFF", lw: float = 1.0) -> str:
            py1, py2 = H - y1_top, H - y2_top
            r, g, b = hex_rgb(color_hex)
            return f"{lw:.2f} w {r:.3f} {g:.3f} {b:.3f} RG {x1:.2f} {py1:.2f} m {x2:.2f} {py2:.2f} l S"

        def polyline_cmd(xs: np.ndarray, ys_top: np.ndarray, color_hex: str = "#00FF66", lw: float = 1.2) -> str:
            if len(xs) < 2:
                return ""
            r, g, b = hex_rgb(color_hex)
            parts = [f"{lw:.2f} w {r:.3f} {g:.3f} {b:.3f} RG {float(xs[0]):.2f} {H - float(ys_top[0]):.2f} m"]
            for x_v, y_v in zip(xs[1:], ys_top[1:]):
                parts.append(f"{float(x_v):.2f} {H - float(y_v):.2f} l")
            parts.append("S")
            return " ".join(parts)

        page_streams: list[str] = []

        # =========================================================================
        # PAGE 1: EXECUTIVE SIGINT TELEMETRY DOSSIER
        # =========================================================================
        p1: list[str] = [
            rect_cmd(0, 0, W, H, fill_hex="#0B192C"),
            rect_cmd(0, 0, W, 64, fill_hex="#050C17"),
            rect_cmd(0, 64, W, 3.5, fill_hex="#00FFCC"),
            text_cmd(28, 28, "SPECTRA // AUTOMATED SIGINT MISSION DOSSIER", "F2", 16, "#00FFCC"),
            text_cmd(
                28, 49,
                f"Generated: {timestamp}   |   Format: {self.file_type.upper()} ({self.fs:,.0f} Hz)   |   Runtime: {self.last_pipeline_time:.2f}s",
                "F1", 9.5, "#A0C4E2"
            ),
            # Left Card
            rect_cmd(28, 82, 382, 472, fill_hex="#101D30", stroke_hex="#1E3A5F", lw=1.2),
            rect_cmd(28, 82, 382, 26, fill_hex="#004D40"),
            text_cmd(38, 99, "1. EXTRACTED SIGNAL PARAMETERS (TAB 1)", "F2", 10.5, "#00FFCC"),
        ]

        y_cursor = 122.0
        param_items = list(self.extracted_params.items())
        param_items.append(("File Type", self.file_type.upper()))
        for idx, (k, v) in enumerate(param_items[:9]):
            bg = "#15263F" if idx % 2 == 0 else "#101D30"
            p1.append(rect_cmd(34, y_cursor - 12, 370, 18, fill_hex=bg))
            p1.append(text_cmd(40, y_cursor, str(k)[:28], "F2", 8.5, "#90B4D4"))
            p1.append(text_cmd(225, y_cursor, str(v)[:28], "F3", 8.5, "#FFFFFF"))
            y_cursor += 20.0

        p1.append(rect_cmd(28, 316, 382, 26, fill_hex="#004D40"))
        p1.append(text_cmd(38, 333, "2. AUTO-APPLIED PIPELINE CONFIGURATION", "F2", 10.5, "#00FFCC"))

        cfg_items = [
            ("Tab 3 — Demodulator", f"{self.mod_combo.get()} (SPS = {self.sps_entry.get()})"),
            ("Tab 4 — De-interleaver", self.deint_combo.get()),
            ("Tab 5 — FEC Decoder", self.fec_combo.get()),
            ("Tab 6 — Preamble (Hex)", self.pattern_entry.get()),
            ("Execution Time", f"{self.last_pipeline_time:.2f} seconds"),
            ("Source File", os.path.basename(self.file_path)),
        ]
        y_cursor = 358.0
        for idx, (k, v) in enumerate(cfg_items):
            bg = "#15263F" if idx % 2 == 0 else "#101D30"
            p1.append(rect_cmd(34, y_cursor - 12, 370, 20, fill_hex=bg))
            p1.append(text_cmd(40, y_cursor, k, "F2", 8.5, "#90B4D4"))
            p1.append(text_cmd(205, y_cursor, str(v)[:32], "F4", 8.5, "#00FFAA"))
            y_cursor += 23.0

        # Right Card
        p1.extend([
            rect_cmd(426, 82, 388, 472, fill_hex="#101D30", stroke_hex="#1E3A5F", lw=1.2),
            rect_cmd(426, 82, 388, 26, fill_hex="#003852"),
            text_cmd(436, 99, "3. STAGE TELEMETRY & DECODED PAYLOAD INSPECTOR", "F2", 10.5, "#00FFCC"),
        ])

        demod_summary = (self.demod_text.get("1.0", tk.END).strip() or "Demodulation not run").splitlines()[:4]
        deint_summary = (self.deint_text.get("1.0", tk.END).strip() or "De-interleaving not run").splitlines()[:3]
        fec_summary = (self.fec_text.get("1.0", tk.END).strip() or "FEC not run").splitlines()[:3]
        corr_summary = (self.corr_text.get("1.0", tk.END).strip() or "Correlation not run").splitlines()[:14]

        telemetry_lines = (
            ["=== [TAB 3] DEMODULATION ==="]
            + demod_summary
            + ["", "=== [TAB 4] DE-INTERLEAVING ==="]
            + deint_summary
            + ["", "=== [TAB 5] FEC DECODER ==="]
            + fec_summary
            + ["", "=== [TAB 6] CORRELATION & PAYLOAD ==="]
            + corr_summary
        )

        ty = 124.0
        for raw_ln in telemetry_lines:
            for wrapped_ln in textwrap.wrap(raw_ln, width=60) or [""]:
                if ty > 540.0:
                    break
                is_hdr = wrapped_ln.startswith("===")
                p1.append(
                    text_cmd(
                        436, ty, wrapped_ln,
                        "F4" if is_hdr else "F3",
                        8.2,
                        "#00FFCC" if is_hdr else "#E2EEF8"
                    )
                )
                ty += 13.0

        p1.append(
            text_cmd(235, 578, "SPECTRA SIGINT Automated Mission Report  —  Page 1 of 5 (Executive Telemetry Dossier)", "F1", 8.5, "#6C8EAD")
        )
        page_streams.append("\n".join(p1))

        # Helper for Plot Pages 2 to 5
        def build_plot_frame(page_no: int, title_str: str, x_lbl: str, y_lbl: str, box=(80.0, 92.0, 710.0, 430.0)) -> list[str]:
            bx, by, bw, bh = box
            cmds = [
                rect_cmd(0, 0, W, H, fill_hex="#0B192C"),
                rect_cmd(0, 0, W, 60, fill_hex="#050C17"),
                rect_cmd(0, 60, W, 3.0, fill_hex="#00FFCC"),
                text_cmd(28, 26, f"SPECTRA SIGINT // {title_str}", "F2", 14.5, "#00FFCC"),
                text_cmd(
                    28, 46,
                    f"Source: {os.path.basename(self.file_path)}   |   Sample Rate: {self.fs:,.0f} Hz   |   Active Mod: {self.mod_combo.get()}",
                    "F1", 9.0, "#A0C4E2"
                ),
                rect_cmd(bx, by, bw, bh, fill_hex="#121A24", stroke_hex="#00FFCC", lw=1.4),
            ]
            for i in range(1, 5):
                gx = bx + bw * i / 5.0
                gy = by + bh * i / 5.0
                cmds.append(line_cmd(gx, by, gx, by + bh, "#1F2E40", 0.7))
                cmds.append(line_cmd(bx, gy, bx + bw, gy, "#1F2E40", 0.7))
            cmds.append(text_cmd(bx + bw / 2.0 - 45, by + bh + 26, x_lbl, "F2", 10.5, "#00FFCC"))
            cmds.append(text_cmd(18, by + bh / 2.0, y_lbl, "F2", 10.0, "#00FFCC"))
            cmds.append(
                text_cmd(235, 578, f"SPECTRA SIGINT Automated Mission Report  -  Page {page_no} of 5 ({title_str})", "F1", 8.5, "#6C8EAD")
            )
            return cmds

        # =========================================================================
        # PAGE 2: DEDICATED SPECTRUM PLOT (VECTOR POLYLINE)
        # =========================================================================
        bx, by, bw, bh = 80.0, 92.0, 710.0, 430.0
        p2 = build_plot_frame(2, "PLOT 1 OF 4 - POWER SPECTRAL DENSITY (SPECTRUM)", "Frequency (kHz)", "Mag (dB)")
        freqs_k = np.nan_to_num(self._cached_freqs / 1e3, nan=0.0, posinf=0.0, neginf=0.0)
        spec_arr = np.nan_to_num(self._cached_spec, nan=0.0, posinf=0.0, neginf=0.0)
        if len(freqs_k) > 1:
            step = max(1, len(freqs_k) // 600)
            fk = freqs_k[::step]
            sp = spec_arr[::step]
            xmin, xmax = float(np.min(fk)), float(np.max(fk))
            ymin, ymax = float(np.min(sp)), float(np.max(sp))
            if xmax <= xmin:
                xmax = xmin + 1.0
            if ymax <= ymin:
                ymax = ymin + 1.0
            xs = bx + 8.0 + ((fk - xmin) / (xmax - xmin)) * (bw - 16.0)
            ys_top = by + bh - 8.0 - ((sp - ymin) / (ymax - ymin)) * (bh - 16.0)
            p2.append(polyline_cmd(xs, ys_top, "#00FF66", 1.2))
            p2.append(text_cmd(bx, by + bh + 14, f"{xmin:.1f} kHz", "F3", 8.5, "#A0C4E2"))
            p2.append(text_cmd(bx + bw - 55, by + bh + 14, f"{xmax:.1f} kHz", "F3", 8.5, "#A0C4E2"))
            p2.append(text_cmd(bx - 48, by + 10, f"{ymax:.1f}", "F3", 8.5, "#A0C4E2"))
            p2.append(text_cmd(bx - 48, by + bh, f"{ymin:.1f}", "F3", 8.5, "#A0C4E2"))
        page_streams.append("\n".join(p2))

        # =========================================================================
        # PAGE 3: DEDICATED WATERFALL SPECTROGRAM (VECTOR HEATMAP GRID)
        # =========================================================================
        p3 = build_plot_frame(3, "PLOT 2 OF 4 - TIME-FREQUENCY WATERFALL SPECTROGRAM", "Time (s)", "Freq (kHz)")
        sxx = np.nan_to_num(self._cached_sxx, nan=0.0, posinf=0.0, neginf=0.0)
        if sxx.ndim == 2 and sxx.size > 0:
            n_f, n_t = sxx.shape
            grid_r, grid_c = min(42, n_f), min(75, n_t)
            r_idx = np.linspace(0, n_f - 1, grid_r).astype(int)
            c_idx = np.linspace(0, n_t - 1, grid_c).astype(int)
            sub = sxx[np.ix_(r_idx, c_idx)]
            smin, smax = float(np.min(sub)), float(np.max(sub))
            norm = (sub - smin) / (smax - smin + 1e-12)
            cell_w = (bw - 8.0) / grid_c
            cell_h = (bh - 8.0) / grid_r
            for ri in range(grid_r):
                y_cell = by + 4.0 + (grid_r - 1 - ri) * cell_h
                for ci in range(grid_c):
                    v = float(norm[ri, ci])
                    # Viridis-style color interpolation
                    cr = int(np.clip(np.interp(v, [0.0, 0.35, 0.7, 1.0], [20, 33, 68, 253]), 0, 255))
                    cg = int(np.clip(np.interp(v, [0.0, 0.35, 0.7, 1.0], [12, 118, 200, 231]), 0, 255))
                    cb = int(np.clip(np.interp(v, [0.0, 0.35, 0.7, 1.0], [84, 142, 95, 37]), 0, 255))
                    chex = f"#{cr:02X}{cg:02X}{cb:02X}"
                    x_cell = bx + 4.0 + ci * cell_w
                    p3.append(rect_cmd(x_cell, y_cell, cell_w + 0.4, cell_h + 0.4, fill_hex=chex))
        page_streams.append("\n".join(p3))

        # =========================================================================
        # PAGE 4: DEDICATED IQ CONSTELLATION DIAGRAM (SQUARE ASPECT RATIO)
        # =========================================================================
        cbx, cby, cbw, cbh = 210.0, 92.0, 430.0, 430.0
        p4 = build_plot_frame(
            4, "PLOT 3 OF 4 - IQ CONSTELLATION DIAGRAM", "In-Phase (I)", "Quad (Q)", box=(cbx, cby, cbw, cbh)
        )
        cx, cy = cbx + cbw / 2.0, cby + cbh / 2.0
        p4.append(line_cmd(cx, cby, cx, cby + cbh, "#3A506B", 1.2))
        p4.append(line_cmd(cbx, cy, cbx + cbw, cy, "#3A506B", 1.2))
        pts = self._cached_pts
        if pts is not None and len(pts) > 0:
            re_p = np.nan_to_num(np.real(pts), nan=0.0, posinf=0.0, neginf=0.0)
            im_p = np.nan_to_num(np.imag(pts), nan=0.0, posinf=0.0, neginf=0.0)
            max_abs = float(max(np.max(np.abs(re_p)), np.max(np.abs(im_p)), 1e-6))
            rad = (cbw * 0.43) / max_abs
            pxs = cx + re_p[:700] * rad
            pys = cy - im_p[:700] * rad
            p4.append("0.000 1.000 1.000 rg")
            dot_cmds = []
            for x_pt, y_pt in zip(pxs, pys):
                xf, yf = float(x_pt), float(y_pt)
                if cbx + 4 < xf < cbx + cbw - 4 and cby + 4 < yf < cby + cbh - 4:
                    dot_cmds.append(f"{xf - 1.6:.2f} {H - yf - 1.6:.2f} 3.2 3.2 re f")
            p4.append(" ".join(dot_cmds))
        page_streams.append("\n".join(p4))

        # =========================================================================
        # PAGE 5: DEDICATED TIME-DOMAIN WAVEFORM (VECTOR I/Q CURVES)
        # =========================================================================
        n_show = min(int(self.time_samples_var.get()), len(self.samples), 600)
        p5 = build_plot_frame(
            5, f"PLOT 4 OF 4 - TIME DOMAIN I/Q WAVEFORM (FIRST {n_show} SAMPLES)", "Time (s)", "Amplitude"
        )
        sig_slice = self.samples[:n_show]
        if len(sig_slice) > 1:
            i_wave = np.nan_to_num(np.real(sig_slice), nan=0.0, posinf=0.0, neginf=0.0)
            q_wave = np.nan_to_num(np.imag(sig_slice), nan=0.0, posinf=0.0, neginf=0.0)
            wmin = float(min(np.min(i_wave), np.min(q_wave)))
            wmax = float(max(np.max(i_wave), np.max(q_wave)))
            if wmax <= wmin:
                wmax = wmin + 1.0
            xs = np.linspace(bx + 8.0, bx + bw - 8.0, len(sig_slice))
            yi_top = by + bh - 10.0 - ((i_wave - wmin) / (wmax - wmin)) * (bh - 20.0)
            yq_top = by + bh - 10.0 - ((q_wave - wmin) / (wmax - wmin)) * (bh - 20.0)
            p5.append(polyline_cmd(xs, yq_top, "#FF00FF", 1.1))
            p5.append(polyline_cmd(xs, yi_top, "#00FF66", 1.1))
            p5.append(rect_cmd(bx + bw - 185, by + 10, 175, 36, fill_hex="#0B192C", stroke_hex="#3A506B", lw=0.8))
            p5.append(text_cmd(bx + bw - 175, by + 24, "I (In-Phase) — Neon Green", "F2", 8.5, "#00FF66"))
            p5.append(text_cmd(bx + bw - 175, by + 39, "Q (Quadrature) — Magenta", "F2", 8.5, "#FF00FF"))
        page_streams.append("\n".join(p5))

        # =========================================================================
        # ASSEMBLE VALID MULTI-PAGE PDF 1.4 BINARY
        # =========================================================================
        objects: list[bytes] = []

        def add_obj(data: bytes) -> int:
            objects.append(data)
            return len(objects)

        add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")  # 1: Catalog
        add_obj(b"")                                    # 2: Pages placeholder
        add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")       # 3: F1
        add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")  # 4: F2
        add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")         # 5: F3
        add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold /Encoding /WinAnsiEncoding >>")    # 6: F4

        page_ids: list[int] = []
        for stream_str in page_streams:
            s_bytes = stream_str.encode("latin-1", errors="replace")
            c_id = add_obj(f"<< /Length {len(s_bytes)} >>\nstream\n".encode("ascii") + s_bytes + b"\nendstream")
            p_id = add_obj(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 842 595] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R /F4 6 0 R >> >> "
                f"/Contents {c_id} 0 R >>".encode("ascii")
            )
            page_ids.append(p_id)

        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

        out = io.BytesIO()
        out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: list[int] = []
        for idx, obj_bytes in enumerate(objects, start=1):
            offsets.append(out.tell())
            out.write(f"{idx} 0 obj\n".encode("ascii"))
            out.write(obj_bytes)
            out.write(b"\nendobj\n")

        xref_pos = out.tell()
        out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
        for off in offsets:
            out.write(f"{off:010d} 00000 n \n".encode("ascii"))
        out.write(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
        )

        with open(filepath, "wb") as f:
            f.write(out.getvalue())

    def export_report(self):
        if not self._require_signal():
            return

        from pathlib import Path

        # Create a crash-proof CustomTkinter Export Window (never calls Windows Explorer)
        popup = ctk.CTkToplevel(self)
        popup.title("Export SPECTRA Mission Report")
        popup.geometry("520x320")
        popup.resizable(False, False)
        popup.attributes("-topmost", True)
        popup.grab_set()

        ctk.CTkLabel(
            popup,
            text="Export Automated SIGINT Mission Report",
            font=("Segoe UI", 16, "bold"),
            text_color="#00ffcc",
        ).pack(pady=(15, 10))

        form = ctk.CTkFrame(popup, fg_color="transparent")
        form.pack(fill=tk.BOTH, expand=True, padx=25)

        # 1. Portable Folder Selector (works automatically on any Windows/Mac/Linux computer)
        ctk.CTkLabel(form, text="Save Location (Auto-Detected):", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(4, 2))
        loc_var = ctk.StringVar(value="Downloads Folder")
        loc_combo = ctk.CTkComboBox(
            form,
            variable=loc_var,
            values=[
                "Downloads Folder",
                "Desktop Folder",
                "Documents Folder",
                "App Folder (SPECTRA_Reports)",
            ],
            width=460,
        )
        loc_combo.pack(pady=(0, 10))

        # 2. Editable File Name
        selected_fmt = self.export_fmt_combo.get()
        if ".pdf" in selected_fmt:
            ext = ".pdf"
        elif ".docx" in selected_fmt:
            ext = ".docx"
        else:
            ext = ".txt"

        stamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        ctk.CTkLabel(form, text="File Name:", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(4, 2))
        name_var = ctk.StringVar(value=f"SPECTRA_Mission_Report_{stamp_file}{ext}")
        name_entry = ctk.CTkEntry(form, textvariable=name_var, width=460)
        name_entry.pack(pady=(0, 15))

        def do_save():
            try:
                home = Path.home()  # Automatically finds the user folder on ANY computer
                choice = loc_combo.get()
                if choice == "Downloads Folder":
                    target_dir = home / "Downloads"
                elif choice == "Desktop Folder":
                    target_dir = home / "Desktop"
                elif choice == "Documents Folder":
                    target_dir = home / "Documents"
                else:
                    target_dir = Path(os.path.abspath(__file__)).parent / "SPECTRA_Reports"

                # Fallback if a folder doesn't exist on someone's machine
                if not target_dir.exists():
                    target_dir = home / "SPECTRA_Reports"
                target_dir.mkdir(parents=True, exist_ok=True)

                fname = name_entry.get().strip() or f"SPECTRA_Mission_Report_{stamp_file}{ext}"
                if not fname.lower().endswith((".pdf", ".docx", ".txt")):
                    fname += ext

                save_path = str(target_dir / fname)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                popup.destroy()
                self.update_idletasks()
                self._set_status(f"Saving report to {save_path}...")

                lower_path = save_path.lower()
                if lower_path.endswith(".pdf"):
                    self._save_as_pdf(save_path, timestamp)
                elif lower_path.endswith(".docx"):
                    self._save_as_docx(save_path, timestamp)
                else:
                    report_content = (
                        f"==========================================================\n"
                        f"          SPECTRA SIGINT AUTOMATED MISSION REPORT         \n"
                        f"==========================================================\n"
                        f"Generated At       : {timestamp}\n"
                        f"Signal Source      : {self.file_path}\n"
                        f"Pipeline Runtime   : {self.last_pipeline_time:.2f} seconds\n\n"
                        f"--- [1] PARAMETER EXTRACTION & PIPELINE CONFIGURATION ---\n"
                        f"{self.params_text.get('1.0', tk.END).strip()}\n\n"
                        f"--- [2] DEMODULATION SUMMARY ---\n"
                        f"{self.demod_text.get('1.0', tk.END).strip()}\n\n"
                        f"--- [3] DE-INTERLEAVING SUMMARY ---\n"
                        f"{self.deint_text.get('1.0', tk.END).strip()}\n\n"
                        f"--- [4] FORWARD ERROR CORRECTION (FEC) ---\n"
                        f"{self.fec_text.get('1.0', tk.END).strip()}\n\n"
                        f"--- [5] BIT CORRELATION & PAYLOAD EXTRACTION ---\n"
                        f"{self.corr_text.get('1.0', tk.END).strip()}\n"
                        f"==========================================================\n"
                    )
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(report_content)

                self._set_status(f"Report saved: {save_path}")
                messagebox.showinfo(
                    "Mission Report Saved",
                    f"Saved successfully to {choice}!\n\nFile:\n{save_path}",
                )
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

        btn_row = ctk.CTkFrame(popup, fg_color="transparent")
        btn_row.pack(pady=10)

        ctk.CTkButton(
            btn_row,
            text="Save Report Now",
            command=do_save,
            fg_color="#00aa44",
            hover_color="#008833",
            font=("Segoe UI", 13, "bold"),
            width=180,
        ).pack(side=tk.LEFT, padx=8)

        ctk.CTkButton(
            btn_row,
            text="Cancel",
            command=popup.destroy,
            fg_color="#444444",
            hover_color="#666666",
            font=("Segoe UI", 13),
            width=100,
        ).pack(side=tk.LEFT, padx=8)

    def run_pipeline(self):
        if not self._require_signal():
            return
        t_start = time.perf_counter()

        if not self.extracted_params:
            self.extract_params()
        self._apply_recommended_params()
        self.update_idletasks()
        if self._cached_freqs is None:
            self.update_plots()
        self.update_idletasks()
        self.run_demod()
        self.update_idletasks()
        self.run_deinterleave()
        self.update_idletasks()
        self.run_fec()
        self.update_idletasks()
        self.run_correlation()

        self.last_pipeline_time = time.perf_counter() - t_start
        self.params_text.insert(
            tk.END, f"  Pipeline Execution Time       : {self.last_pipeline_time:.2f} s\n"
        )
        self._set_status(
            f"Pipeline Done in {self.last_pipeline_time:.2f}s — "
            f"Mod: {self.mod_combo.get()} | De-int: {self.deint_combo.get()} | FEC: {self.fec_combo.get()}"
        )


def main():
    app = SignalAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
