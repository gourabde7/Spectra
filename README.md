# NTRO Signal Analyzer

GUI-based automated analysis tool for **.IQ** and **.wav** terrestrial signals (HF/VHF/UHF).

## Features

1. **Parameter extraction** — sampling rate, bandwidth, SNR, modulation estimate, FEC/interleaving inference
2. **Visualizations** — spectrum, waterfall, constellation, I/Q time domain
3. **Demodulation** — BPSK, QPSK, 16-QAM, FSK
4. **De-interleaving** — Block, Convolutional, Diagonal, Pseudo-random
5. **FEC** — Viterbi (K=7), RS block, LDPC, Concatenated
6. **Bit stream correlation** — preamble/header search and payload preview

## Setup

```bash
cd signal-analyzer
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Demo

1. Click **Load File** and select a `.wav` or `.IQ` file.
2. For `.IQ` files, set sample rate and format (`cf32`, `ci16`, `cu8`).
3. Use tabs 1–6 or click **Run Full Pipeline** for end-to-end analysis.

**Architecture diagram (one slide):** open `docs/architecture-slide.html` in a browser and screenshot, or use the Mermaid diagram in `docs/architecture-slide.md`.

Generate sample files (optional):

```bash
python -c "from demo_signals import *; generate_qpsk_wav('demo.wav'); generate_fsk_iq('demo.iq'); print('Created demo.wav and demo.iq')"
```

## Note

This is an **ideation-round prototype**. Parameter/FEC inference uses heuristics suitable for demonstration; production systems would use trained classifiers and full protocol stacks (e.g. GNU Radio).
