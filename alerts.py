"""EPILEX alert engine: alarm + push + (optional) email/Telegram/SMS via Apprise.

Set these environment variables in PowerShell BEFORE running (never commit them to GitHub):
    $env:NTFY_TOPIC = "epilex-your-long-random-topic"
    $env:APPRISE_URLS = "tgram://BOT_TOKEN/CHAT_ID,mailtos://you:APP_PASSWORD@gmail.com?to=you@gmail.com"
"""
import os, json, time, pathlib
import requests

LOG = pathlib.Path("events.jsonl")


def _alarm():
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1200, 500); winsound.Beep(800, 500)
        return "ok"
    except Exception as e:                     # non-Windows or no sound device
        print("\a", end="")
        return f"fallback bell ({e.__class__.__name__})"


def _ntfy(title, msg):
    topic = os.getenv("NTFY_TOPIC")
    if not topic:
        return "skipped (NTFY_TOPIC not set)"
    r = requests.post(f"https://ntfy.sh/{topic}", data=msg.encode("utf-8"),
                      headers={"Title": title, "Priority": "urgent", "Tags": "rotating_light"}, timeout=10)
    return f"ok {r.status_code}"


def _apprise(title, msg):
    urls = os.getenv("APPRISE_URLS")
    if not urls:
        return "skipped (APPRISE_URLS not set)"
    import apprise
    ap = apprise.Apprise()
    for u in urls.split(","):
        ap.add(u.strip())
    return "ok" if ap.notify(title=title, body=msg) else "failed"


def send_alert(msg, title="EPILEX SEIZURE ALERT", source="eeg", confidence=None):
    """Fire every channel independently, so one failure never blocks the others. Returns per-channel status."""
    status = {}
    for name, fn in (("alarm", lambda: _alarm()),
                     ("push", lambda: _ntfy(title, msg)),
                     ("apprise", lambda: _apprise(title, msg))):
        try:
            status[name] = fn()
        except Exception as e:
            status[name] = f"error: {e.__class__.__name__}"
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source,
           "confidence": confidence, "message": msg, "delivery": status}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return status


def caregiver_note(confidence, when, lang="en"):
    """Short first-aid note. Uses a local Ollama model if running, otherwise a fixed safe template."""
    template_en = (f"Possible seizure detected at {when} (confidence {confidence:.0%}). Stay calm. Keep the person safe, "
                   "move hard objects away, do NOT restrain them or put anything in their mouth, turn them on their side "
                   "when shaking stops, and time the seizure. Call emergency help if it lasts over 5 minutes.")
    template_hi = (f"{when} par daura padne ki sambhavna paayi gayi (vishwas {confidence:.0%}). Shaant rahein. Vyakti ko surakshit rakhein, "
                   "kathor vastuen hataayein, use pakadein nahin aur munh mein kuch na daalein. Jhatke rukne par use karvat dilaayein. "
                   "5 minute se zyada chale to turant madad bulaayein.")
    base = template_hi if lang == "hi" else template_en
    try:   # let the open-weight LLM rephrase the vetted template; never invent new medical advice
        r = requests.post("http://localhost:11434/api/generate", timeout=20, json={
            "model": os.getenv("EPILEX_LLM", "qwen2.5:1.5b"), "stream": False,
            "prompt": "Rewrite this first-aid alert in at most 3 short, calm sentences. Do not add any new advice or facts. "
                      "Keep every instruction exactly as given:\n\n" + base})
        text = r.json().get("response", "").strip()
        return text if len(text) > 30 else base
    except Exception:
        return base
