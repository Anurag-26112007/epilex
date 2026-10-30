"""Checks that everything EPILEX needs is installed. Run:  python check_setup.py"""
import importlib, sys, pathlib, shutil, subprocess, urllib.request

print("Python", sys.version.split()[0], "(want 3.10-3.12)\n")

for m in ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "mne", "torch",
          "cv2", "mediapipe", "streamlit", "fastapi", "uvicorn", "apprise", "requests", "joblib"]:
    try:
        mod = importlib.import_module(m)
        print("OK     ", m, getattr(mod, "__version__", ""))
    except Exception as e:
        print("MISSING", m, "->", str(e)[:80])

try:
    import torch
    print("\ntorch test:", float(torch.ones(3).sum()), "(want 3.0)")
except Exception as e:
    print("torch test failed:", e)

data = pathlib.Path("data/chbmit")
print("\nEEG .edf files:", len(list(data.rglob("*.edf"))),
      "| summary files:", len(list(data.rglob("*summary.txt"))), "(want 10+ edf, 2+ summary)")

print("\nGit:", "found" if shutil.which("git") else "NOT found")
if shutil.which("ollama"):
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15).stdout
    except Exception as e:
        out = f"(could not run: {e})"
    print("Ollama models:\n" + out)
else:
    print("Ollama: NOT found")

try:
    import cv2
    cap = cv2.VideoCapture(0); ok, _ = cap.read(); cap.release()
    print("Webcam:", "works" if ok else "NOT working")
except Exception as e:
    print("Webcam check failed:", e)

try:
    urllib.request.urlopen("https://ntfy.sh", timeout=10)
    print("Internet to ntfy.sh: OK")
except Exception as e:
    print("Internet to ntfy.sh: FAILED", e)