"""Train + evaluate the baseline seizure detector (band-power features + random forest).

Usage (cross-patient, honest):
    python train_baseline.py --root data/chbmit --train chb01 --test chb02
"""
import argparse, json, pathlib
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from eeg import build_patient, event_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/chbmit")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--neg_ratio", type=int, default=20, help="seizure-free windows kept per seizure window (training only)")
    a = ap.parse_args()

    tr = [build_patient(a.root, p) for p in a.train]
    te = [build_patient(a.root, p) for p in a.test]
    Xtr = np.concatenate([d["X"] for d in tr]); ytr = np.concatenate([d["y"] for d in tr])

    # balance TRAINING data only; the test set keeps its natural (very imbalanced) distribution
    rng = np.random.default_rng(0)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    neg = rng.choice(neg, size=min(len(neg), a.neg_ratio * max(len(pos), 1)), replace=False)
    keep = np.concatenate([pos, neg])
    print(f"train windows: {len(pos)} seizure + {len(neg)} normal")

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=0, min_samples_leaf=2)
    clf.fit(Xtr[keep], ytr[keep])

    print("\n=== TEST (patients never seen in training):", a.test, "===")
    rows = []
    for d, name in zip(te, a.test):
        prob = clf.predict_proba(d["X"])[:, 1]
        y = d["y"]
        pred = prob >= 0.5
        sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = ((~pred) & (y == 0)).sum() / max((y == 0).sum(), 1)
        print(f"{name}: window sensitivity {sens:.2f}, specificity {spec:.3f}")
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
            for k in (2, 3, 5):
                m = event_metrics(prob, d["t"], d["fid"], list(d["intervals"]), thr, k)
                rows.append((name, thr, k, m))

    print("\nEvent-level results (alarm = k consecutive windows >= threshold):")
    print(f"{'patient':8} {'thr':>4} {'k':>2} {'sens':>6} {'FA/24h':>8} {'delay_s':>8}  seizures")
    for name, thr, k, m in rows:
        print(f"{name:8} {thr:>4} {k:>2} {m['sensitivity']:>6.2f} {m['fa_per_24h']:>8.1f} {m['mean_delay_s']:>8.1f}  {m['detected']}/{m['seizures']}")

    out = pathlib.Path("models"); out.mkdir(exist_ok=True)
    joblib.dump(clf, out / "rf_baseline.joblib")
    json.dump({"thr": 0.7, "k": 3, "cooldown": 60.0}, open(out / "decision.json", "w"))
    print("\nsaved models/rf_baseline.joblib  (edit models/decision.json to change the operating point)")


if __name__ == "__main__":
    main()
