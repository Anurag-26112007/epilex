"""EPILEX camera mode: body tracking -> 'rhythmic shaking' score between 0 and 1.

Backend 1: MediaPipe Pose (elbows, wrists, knees, ankles).
Backend 2 (automatic fallback if MediaPipe's pose API is unavailable): whole-frame motion energy.
Everything runs locally; frames are never stored.
"""
import time
import collections
import numpy as np

RESAMPLE_HZ = 20.0
WINDOW_S = 4.0
BAND = (2.0, 7.0)                    # convulsive shaking is roughly 2-7 Hz
JOINTS = [13, 14, 15, 16, 25, 26, 27, 28]   # elbows, wrists, knees, ankles (MediaPipe indices)


def _series_score(t, x, amp_min, amp_ref):
    """Score one 1-D trajectory: strong, periodic movement inside BAND -> near 1."""
    amp = float(np.std(x))
    if amp < amp_min:
        return 0.0, 0.0
    x = (x - x.mean()) * np.hanning(len(x))
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / RESAMPLE_HZ)
    total = p[(f >= 0.5) & (f <= 10)].sum() + 1e-12
    inband = (f >= BAND[0]) & (f <= BAND[1])
    if not inband.any():
        return 0.0, 0.0
    pk = int(np.argmax(np.where(inband, p, 0)))
    ratio = p[pk] / total                       # how peaky (rhythmic) the spectrum is
    rhythm = float(np.clip((ratio - 0.15) / 0.35, 0, 1))
    strength = min(1.0, amp / amp_ref)
    return strength * rhythm, float(f[pk])


def shake_score(ts, X, amp_min=0.008, amp_ref=0.02, min_joints=3):
    """ts: (T,) seconds. X: (T, J, 2) normalised coordinates, or (T, 1) for the motion fallback.
    Returns (score 0-1, dominant_freq_hz). Needs >= min_joints joints shaking (waving one hand is not a seizure)."""
    ts = np.asarray(ts, float)
    if len(ts) < 10 or ts[-1] - ts[0] < 2.5:
        return 0.0, 0.0
    grid = np.arange(ts[0], ts[-1], 1.0 / RESAMPLE_HZ)
    X = np.asarray(X, float)
    if X.ndim == 2:
        X = X[:, :, None]
    joint_scores, freqs = [], []
    for j in range(X.shape[1]):
        best, bf = 0.0, 0.0
        for c in range(X.shape[2]):
            s, fr = _series_score(grid, np.interp(grid, ts, X[:, j, c]), amp_min, amp_ref)
            if s > best:
                best, bf = s, fr
        joint_scores.append(best); freqs.append(bf)
    order = np.argsort(joint_scores)[::-1]
    need = min(min_joints, X.shape[1])
    k = order[need - 1]
    return float(joint_scores[k]), float(freqs[k])


class ShakeDetector:
    def __init__(self):
        self.ts = collections.deque()
        self.buf = collections.deque()
        self.prev = None
        self.pose = None
        self.mp = None
        try:
            import mediapipe as mp
            self.pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
            self.mp = mp
            self.backend = "MediaPipe Pose"
        except Exception:
            self.backend = "motion-energy fallback"
        self.last = (0.0, 0.0)
        self._last_eval = 0.0

    def _push(self, t, v):
        self.ts.append(t); self.buf.append(v)
        while self.ts and t - self.ts[0] > WINDOW_S:
            self.ts.popleft(); self.buf.popleft()

    def process(self, frame_bgr):
        """Returns (annotated BGR frame, info dict with score, freq, tracked)."""
        import cv2
        now = time.time()
        out, tracked = frame_bgr, False
        if self.pose is not None:
            res = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                self._push(now, np.array([[lm[j].x, lm[j].y] for j in JOINTS]))
                tracked = True
                out = frame_bgr.copy()
                self.mp.solutions.drawing_utils.draw_landmarks(
                    out, res.pose_landmarks, self.mp.solutions.pose.POSE_CONNECTIONS)
        else:
            g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame_bgr, (80, 60)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            g = g.astype(np.float32) / 255.0
            if self.prev is not None:
                self._push(now, np.array([[float(np.abs(g - self.prev).mean())]]))
                tracked = True
            self.prev = g
        if now - self._last_eval > 0.5 and len(self.ts) > 10:
            self._last_eval = now
            X = np.stack(list(self.buf))
            self.last = shake_score(list(self.ts), X, amp_min=0.008 if self.pose else 0.004,
                                    amp_ref=0.02 if self.pose else 0.02,
                                    min_joints=3 if self.pose else 1)
        return out, {"score": self.last[0], "freq": self.last[1], "tracked": tracked}
