"""Download a small slice of CHB-MIT: each patient's summary file, every seizure file, and a few seizure-free files.
Run:  python get_data.py      (safe to re-run; it skips files you already have)
"""
import re, time, pathlib, urllib.request, urllib.parse

BASE = "https://physionet.org/files/chbmit/1.0.0/"
OUT = pathlib.Path("data/chbmit")
PATIENTS = ["chb01", "chb02"]          # add "chb03", "chb05" if your internet is fast
NON_SEIZURE_PER_PATIENT = 3


def get(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print("have", dest.name); return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest); print("ok  ", dest.name); return
        except Exception as e:
            print("retry", dest.name, e); time.sleep(3)
    print("FAILED", url)


for p in PATIENTS:
    summ = OUT / p / f"{p}-summary.txt"
    get(BASE + f"{p}/{p}-summary.txt", summ)
    if not summ.exists():
        continue
    blocks = re.split(r"File Name:\s*", summ.read_text(errors="ignore"))[1:]
    seiz, non = [], []
    for b in blocks:
        name = b.split()[0]
        m = re.search(r"Number of Seizures in File:\s*(\d+)", b)
        n = int(m.group(1)) if m else 0
        (seiz if n > 0 else non).append(name)
    for f in seiz + non[:NON_SEIZURE_PER_PATIENT]:
        get(BASE + f"{p}/" + urllib.parse.quote(f), OUT / p / f)
print("done")