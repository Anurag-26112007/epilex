/* EPILEX shaking score (JS port of camera_mode.py).
   shakeScore(ts, frames) -> {score 0-1, freq Hz}
   ts: seconds per frame; frames: array of frames, each an array of J joints [x, y] in normalised coordinates. */
(function (root) {
  var HZ = 20, BAND_LO = 2, BAND_HI = 7;

  function seriesScore(x, ampMin, ampRef) {
    var n = x.length, i, mean = 0;
    for (i = 0; i < n; i++) mean += x[i];
    mean /= n;
    var v = 0;
    for (i = 0; i < n; i++) v += (x[i] - mean) * (x[i] - mean);
    var amp = Math.sqrt(v / n);
    if (amp < ampMin) return [0, 0];
    var w = new Array(n);
    for (i = 0; i < n; i++) w[i] = (x[i] - mean) * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
    var half = Math.floor(n / 2), power = new Array(half + 1), k, re, im, ang;
    for (k = 0; k <= half; k++) {
      re = 0; im = 0;
      for (i = 0; i < n; i++) { ang = 2 * Math.PI * k * i / n; re += w[i] * Math.cos(ang); im -= w[i] * Math.sin(ang); }
      power[k] = re * re + im * im;
    }
    var total = 1e-12, best = -1, bestP = -1, f;
    for (k = 0; k <= half; k++) {
      f = k * HZ / n;
      if (f >= 0.5 && f <= 10) total += power[k];
      if (f >= BAND_LO && f <= BAND_HI && power[k] > bestP) { bestP = power[k]; best = k; }
    }
    if (best < 0) return [0, 0];
    var ratio = bestP / total;
    var rhythm = Math.min(1, Math.max(0, (ratio - 0.15) / 0.35));
    var strength = Math.min(1, amp / ampRef);
    return [strength * rhythm, best * HZ / n];
  }

  function shakeScore(ts, frames, opts) {
    opts = opts || {};
    var ampMin = opts.ampMin || 0.008, ampRef = opts.ampRef || 0.02, minJoints = opts.minJoints || 3;
    var T = ts.length;
    if (T < 10 || ts[T - 1] - ts[0] < 2.5) return { score: 0, freq: 0 };
    var n = Math.floor((ts[T - 1] - ts[0]) * HZ);
    if (n < 16) return { score: 0, freq: 0 };
    var J = frames[0].length, joints = [], j, c, g, p, t, s, a, best, res;
    for (j = 0; j < J; j++) {
      best = [0, 0];
      for (c = 0; c < 2; c++) {
        var series = new Array(n);
        p = 0;
        for (g = 0; g < n; g++) {
          t = ts[0] + g / HZ;
          while (p < T - 2 && ts[p + 1] < t) p++;
          a = (t - ts[p]) / Math.max(1e-9, ts[p + 1] - ts[p]);
          a = Math.min(1, Math.max(0, a));
          series[g] = frames[p][j][c] * (1 - a) + frames[p + 1][j][c] * a;
        }
        res = seriesScore(series, ampMin, ampRef);
        if (res[0] > best[0]) best = res;
      }
      joints.push(best);
    }
    joints.sort(function (u, v2) { return v2[0] - u[0]; });
    var pick = joints[Math.min(minJoints, J) - 1];
    return { score: pick[0], freq: pick[1] };
  }

  root.shakeScore = shakeScore;
  if (typeof module !== "undefined" && module.exports) module.exports = { shakeScore: shakeScore };
})(typeof window !== "undefined" ? window : globalThis);