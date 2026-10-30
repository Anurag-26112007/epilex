"""EPILEX dashboard.   Run:  streamlit run app.py
Mode A: replay a real CHB-MIT EEG file as a live stream (random-forest baseline model).
Mode B: webcam shaking detector (MediaPipe).  Both feed the same confirm -> alert engine (alerts.py).
"""
import json, time, pathlib
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from eeg import load_edf, features_from_signal, parse_summary, FS
from alerts import send_alert, caregiver_note

st.set_page_config(page_title="EPILEX", layout="wide")
st.title("EPILEX")
st.caption("A Signal Detected, A Life Protected  |  Assistive research prototype, not a medical device")

try:
    cfg = json.load(open("models/decision.json"))
except Exception:
    cfg = {"thr": 0.7, "k": 3, "cooldown": 60.0}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    mode = st.radio("Input mode", ["Mode A: EEG replay", "Mode B: Webcam"])
    lang = st.selectbox("Caregiver note language", ["en", "hi"], format_func=lambda x: {"en": "English", "hi": "Hindi"}[x])
    k = st.slider("Confirm: consecutive positives (k)", 1, 8, int(cfg["k"]))
    cooldown = st.slider("Cooldown after alert (s)", 5, 120, int(cfg["cooldown"]))
    fire = st.checkbox("Send real alerts (siren, push, email)", value=True)
    if st.button("Send test alert"):
        st.write(send_alert("EPILEX test alert. Please ignore.", title="EPILEX TEST", source="test"))


def fire_alert(conf, source):
    when = time.strftime("%H:%M:%S")
    msg = caregiver_note(conf, when, lang)
    status = send_alert(msg, confidence=conf, source=source) if fire else {"alerts": "disabled in sidebar"}
    return msg, status


def show_alert(box, msg, status):
    box.error(f"SEIZURE ALERT\n\n{msg}")
    box.json(status)


# ------------------------------------------------------------------ Mode A
if mode.startswith("Mode A"):
    st.subheader("Mode A: EEG replay (public CHB-MIT data)")
    edfs = sorted(pathlib.Path("data/chbmit").rglob("*.edf"))
    if not edfs:
        st.warning("No EDF files found in data/chbmit. Run python get_data.py first.")
        st.stop()
    if not pathlib.Path("models/rf_baseline.joblib").exists():
        st.warning("No trained model yet. Run: python train_baseline.py --root data/chbmit --train chb01 --test chb02")
        st.stop()

    names = [str(p) for p in edfs]
    c1, c2, c3 = st.columns([3, 1, 1])
    edf = c1.selectbox("EEG file", names)
    p = pathlib.Path(edf)
    iv = []
    try:
        iv = parse_summary(p.parent / f"{p.parent.name}-summary.txt").get(p.name, [])
    except Exception:
        pass
    default_start = max(0, iv[0][0] - 30) if iv else 0
    start = c2.number_input("Start at (s)", 0, 100000, int(default_start), step=10)
    speed = c3.slider("Speed (x real time)", 1, 60, 20)
    if iv:
        st.info(f"Doctor-marked seizure in this file: {iv[0][0]}s to {iv[0][1]}s (start set 30 s before it).")
    else:
        st.info("No seizure in this file: use it to show the system stays quiet.")

    @st.cache_resource
    def get_clf():
        return joblib.load("models/rf_baseline.joblib")

    @st.cache_data(show_spinner="Reading and filtering EEG ...")
    def prep(path, start_s):
        sig = load_edf(path)
        if sig is None:
            return None
        X, c = features_from_signal(sig[:, int(start_s * FS):])
        return X, c + start_s

    if st.button("Start replay", type="primary"):
        data = prep(edf, start)
        if data is None:
            st.error("This file is missing the expected channels. Pick another file.")
            st.stop()
        X, times = data
        probs = get_clf().predict_proba(X)[:, 1]
        thr = float(cfg["thr"])
        banner, m1, m2, m3 = st.empty(), None, None, None
        cols = st.columns(3); m1, m2, m3 = cols[0].empty(), cols[1].empty(), cols[2].empty()
        chart = st.empty(); alert_box = st.container()
        run, last_alarm, shown = 0, -1e9, []
        for i in range(len(probs)):
            pr, t = float(probs[i]), float(times[i])
            run = run + 1 if pr >= thr else 0
            shown.append(pr)
            m1.metric("Time in recording", f"{t:.0f} s")
            m2.metric("Seizure confidence", f"{pr:.0%}")
            m3.metric("Consecutive positives", f"{run} / {k}")
            banner.warning("Possible seizure activity ...") if pr >= thr else banner.success("Normal")
            chart.line_chart(pd.DataFrame({"confidence": shown[-120:], "threshold": thr}))
            if run >= k and t - last_alarm >= cooldown:
                msg, status = fire_alert(pr, "eeg-dashboard")
                show_alert(alert_box, msg, status)
                last_alarm, run = t, 0
            time.sleep(1.0 / speed)
        banner.info("Replay finished.")

# ------------------------------------------------------------------ Mode B
else:
    st.subheader("Mode B: Webcam shaking detector")
    st.caption("Video is processed on this laptop and never stored. Detects sustained rhythmic shaking of several "
               "limbs only. Tested on acted movement, so treat it as a demo.")
    c1, c2 = st.columns(2)
    cam_idx = c1.number_input("Camera index", 0, 5, 0)
    thr = c2.slider("Shaking score threshold", 0.2, 0.95, 0.5)
    on = st.toggle("Start camera")
    left, right = st.columns([2, 1])
    frame_ph = left.empty()
    score_ph, freq_ph, run_ph, backend_ph = (right.empty() for _ in range(4))
    chart_ph = right.empty()
    alert_box = st.container()

    if on:
        import cv2
        from camera_mode import ShakeDetector
        cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(cam_idx))
        if not cap.isOpened():
            st.error("Could not open the camera. Close other apps using it, or try another index.")
            st.stop()
        det = ShakeDetector()
        backend_ph.caption(f"Tracking backend: {det.backend}")
        run, last_alarm, last_tick, hist = 0, -1e9, 0.0, []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    st.error("Camera stopped returning frames.")
                    break
                out, info = det.process(cv2.flip(frame, 1))
                frame_ph.image(out, channels="BGR", use_container_width=True)
                now = time.time()
                if now - last_tick >= 1.0:                       # one decision per second
                    last_tick = now
                    run = run + 1 if info["score"] >= thr else 0
                    hist.append(info["score"])
                    chart_ph.line_chart(pd.DataFrame({"score": hist[-60:], "threshold": thr}))
                    if run >= k and now - last_alarm >= cooldown:
                        msg, status = fire_alert(info["score"], "camera-dashboard")
                        show_alert(alert_box, msg, status)
                        last_alarm, run = now, 0
                score_ph.metric("Shaking score", f"{info['score']:.2f}" if info["tracked"] else "no body found")
                freq_ph.metric("Dominant frequency", f"{info['freq']:.1f} Hz")
                run_ph.metric("Consecutive positives", f"{run} / {k}")
        finally:
            cap.release()

# ------------------------------------------------------------------ event log
st.divider()
st.subheader("Event log")
log = pathlib.Path("events.jsonl")
if log.exists():
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    st.dataframe(pd.DataFrame(rows).iloc[::-1], use_container_width=True)
else:
    st.caption("No alerts yet.")
