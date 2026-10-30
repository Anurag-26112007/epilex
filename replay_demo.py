"""Replay a real EEG file as a live stream: model -> decision logic -> alerts.

Usage:
    python replay_demo.py --edf data/chbmit/chb02/chb02_16.edf --speed 10 [--start 2900]
(--speed 10 means 10x real time; --start skips ahead so the seizure shows up quickly)
"""
import argparse, json, time
import numpy as np
import joblib
from eeg import load_edf, features_from_signal, FS, WIN, STRIDE
from alerts import send_alert, caregiver_note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf", required=True)
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    ap.add_argument("--lang", default="en", choices=["en", "hi"])
    a = ap.parse_args()

    clf = joblib.load("models/rf_baseline.joblib")
    cfg = json.load(open("models/decision.json"))
    thr, k, cooldown = cfg["thr"], cfg["k"], cfg["cooldown"]

    sig = load_edf(a.edf)
    run, last_alarm, t = 0, -1e9, a.start
    print(f"Streaming {a.edf} at {a.speed}x from t={a.start}s  (thr={thr}, k={k})")
    while int((t + 2) * FS) <= sig.shape[1]:
        chunk = sig[:, int(t * FS): int(t * FS) + WIN]
        f, _ = features_from_signal(chunk)                  # exactly one window
        p = float(clf.predict_proba(f)[0, 1])
        run = run + 1 if p >= thr else 0
        flag = "SEIZURE?" if p >= thr else "normal  "
        print(f"t={t:7.0f}s  p={p:.2f}  {flag}  run={run}")
        if run >= k and t - last_alarm >= cooldown:
            when = time.strftime("%H:%M:%S")
            msg = caregiver_note(p, when, a.lang)
            print(f"\n*** ALERT *** {msg}")
            print("delivery:", send_alert(msg, confidence=p, source="eeg-replay"), "\n")
            last_alarm, run = t, 0
        t += 1.0
        time.sleep(1.0 / a.speed)


if __name__ == "__main__":
    main()
