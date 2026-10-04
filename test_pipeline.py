"""End-to-end self-test.  Run:  python test_pipeline.py"""
import json, os, sys, tempfile
import numpy as np
import demo_signals as DS
from signal_loader import load_signal
from analysis import analyze_signal, flatten_params

def main():
    tmp = tempfile.mkdtemp()
    truth = DS.generate_all(tmp)
    ok_all = True
    for name, t in truth.items():
        path = os.path.join(tmp, t["file"])
        x, fs, _ = load_signal(path, t["sample_rate"], "cf32")
        res = analyze_signal(x, fs)
        fr = res["frame"]
        ok = bool(fr and fr["crc_ok"] and fr["payload"].decode() == t["message"])
        ok_all &= ok
        rx = res["rx"]
        print(f"{'PASS' if ok else 'FAIL'}  {t['file']:26s} detected={rx.mod if rx else res['error']}  "
              f"fec={res['chain']['fec'][:38] if res['chain'] else '-'}  inter={res['chain']['interleaver'] if res['chain'] else '-'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1

if __name__ == "__main__":
    sys.exit(main())
