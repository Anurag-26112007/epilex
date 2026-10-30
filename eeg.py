"""EPILEX core: CHB-MIT loading, windowing, features, decision logic, event metrics."""
import re
import pathlib
import numpy as np

FS = 256                      # CHB-MIT sampling rate (Hz)
WIN_S = 2                     # window length (seconds)
WIN = WIN_S * FS              # 512 samples
STRIDE_S = 1                  # window step (seconds)
STRIDE = STRIDE_S * FS

# 18 bipolar channels present in chb01-chb05 (T8-P8 appears twice in some files; MNE names it T8-P8-0)
CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
            "FZ-CZ", "CZ-PZ"]

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]   # delta, theta, alpha, beta, gamma


# ---------------------------------------------------------------- summary files
def parse_summary(path):
    """Return {edf_name: [(start_s, end_s), ...]} from a chbXX-summary.txt file."""
    text = pathlib.Path(path).read_text(errors="ignore")
    out = {}
    for block in re.split(r"File Name:\s*", text)[1:]:
        name = block.split()[0]
        starts = [int(s) for s in re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", block)]
        ends = [int(s) for s in re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", block)]
        out[name] = list(zip(starts, ends))
    return out


# ---------------------------------------------------------------- EDF loading
def load_edf(path):
    """Load one EDF -> float32 array (n_channels, n_samples), band-passed 0.5-40 Hz. None if channels missing."""
    import mne
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    names = raw.ch_names
    if not all(c in names for c in CHANNELS):
        print(f"  skip {pathlib.Path(path).name}: channels missing")
        return None
    raw.pick(CHANNELS)
    raw.filter(0.5, 40.0, verbose="ERROR")
    return raw.get_data().astype(np.float32) * 1e6        # volts -> microvolts


# ---------------------------------------------------------------- windows + features
def window_starts(n_samples):
    return np.arange(0, n_samples - WIN + 1, STRIDE)


def features_from_signal(x, batch=256):
    """x: (C, n) -> features (n_windows, C * 7) and window centre times (s)."""
    starts = window_starts(x.shape[1])
    freqs = np.fft.rfftfreq(WIN, 1 / FS)
    hann = np.hanning(WIN).astype(np.float32)
    feats = []
    for i in range(0, len(starts), batch):
        idx = starts[i:i + batch]
        w = np.stack([x[:, s:s + WIN] for s in idx])                    # (b, C, WIN)
        power = np.abs(np.fft.rfft(w * hann, axis=-1)) ** 2             # (b, C, F)
        bp = [np.log10(power[..., (freqs >= lo) & (freqs < hi)].sum(-1) + 1e-8) for lo, hi in BANDS]
        line_len = np.log10(np.abs(np.diff(w, axis=-1)).sum(-1) + 1e-8)
        var = np.log10(w.var(-1) + 1e-8)
        f = np.stack(bp + [line_len, var], axis=-1)                     # (b, C, 7)
        feats.append(f.reshape(len(idx), -1))
    centres = (starts + WIN / 2) / FS
    return np.concatenate(feats).astype(np.float32), centres


def label_windows(centres, intervals):
    """1 if window centre lies inside any seizure interval."""
    y = np.zeros(len(centres), dtype=np.int8)
    for s, e in intervals:
        y[(centres >= s) & (centres <= e)] = 1
    return y


def build_patient(root, patient):
    """Features for every locally downloaded EDF of one patient. Cached to .npz."""
    root = pathlib.Path(root)
    cache = root / f"{patient}_features.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    summ = parse_summary(root / patient / f"{patient}-summary.txt")
    X, y, t, fid, files = [], [], [], [], []
    for name, intervals in summ.items():
        p = root / patient / name
        if not p.exists():
            continue
        print(f"{patient}: {name} ({len(intervals)} seizure(s))")
        sig = load_edf(p)
        if sig is None:
            continue
        f, c = features_from_signal(sig)
        X.append(f); y.append(label_windows(c, intervals)); t.append(c)
        fid.append(np.full(len(c), len(files))); files.append(name)
    data = dict(X=np.concatenate(X), y=np.concatenate(y), t=np.concatenate(t),
                fid=np.concatenate(fid), files=np.array(files),
                intervals=np.array([summ[f] for f in files], dtype=object))
    np.savez(cache, **data)
    return data


# ---------------------------------------------------------------- decision logic
def raise_alarms(prob, times, thr=0.7, k=3, cooldown=60.0):
    """Alarm when `k` consecutive windows exceed `thr`; then stay quiet for `cooldown` seconds."""
    alarms, run, last = [], 0, -1e9
    for p, t in zip(prob, times):
        run = run + 1 if p >= thr else 0
        if run >= k and t - last >= cooldown:
            alarms.append(float(t)); last = t; run = 0
    return alarms


def event_metrics(prob, times, fid, intervals_per_file, thr, k, cooldown=60.0, margin=60.0):
    """Event-level scoring: sensitivity, false alarms per hour, mean detection delay (s)."""
    n_seiz = n_hit = n_fa = 0
    hours = 0.0
    delays = []
    for f in np.unique(fid):
        m = fid == f
        pr, tt = prob[m], times[m]
        iv = intervals_per_file[int(f)]
        al = raise_alarms(pr, tt, thr, k, cooldown)
        seiz_time = sum(e - s + 2 * margin for s, e in iv)
        hours += max(0.0, (tt[-1] - tt[0] - seiz_time)) / 3600
        for s, e in iv:
            n_seiz += 1
            hit = [a for a in al if s - 5 <= a <= e + 5]
            if hit:
                n_hit += 1; delays.append(hit[0] - s)
        n_fa += sum(not any(s - margin <= a <= e + margin for s, e in iv) for a in al)
    return dict(seizures=n_seiz, detected=n_hit,
                sensitivity=n_hit / n_seiz if n_seiz else float("nan"),
                false_alarms=n_fa, hours=round(hours, 2),
                fa_per_hour=n_fa / hours if hours else float("nan"),
                fa_per_24h=24 * n_fa / hours if hours else float("nan"),
                mean_delay_s=float(np.mean(delays)) if delays else float("nan"))
