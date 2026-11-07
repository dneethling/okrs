#!/usr/bin/env python3
"""Beat analysis for beat-synced video editing. Needs only ffmpeg + numpy.

  beats.py analyze <audio|video> [--out beats.json]   tempo, beats, downbeats, onsets, energy drops
  beats.py cuts <beats.json> <beats_per_shot,...>      cut times for a shot list measured in beats
  beats.py snap <beats.json> <t1> <t2> ...             snap times to nearest beat
  beats.py verify <video> [--thresh 0.25]              do the video's hard cuts land on its own audio beats?
"""
import sys, json, subprocess, re
import numpy as np

SR, NFFT, HOP = 22050, 1024, 256
FR = HOP / SR  # seconds per frame
LAT = 0.026    # calibrated on a synthetic 128 BPM track (kick at exact 0 -> detected 27 ms early)

def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if len(x) < SR: sys.exit("audio too short / no audio stream")
    return x

def stft_mag(x):
    w = np.hanning(NFFT); n = 1 + (len(x) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * w, axis=1))

def onset_env(mag):
    lm = np.log1p(30 * mag)
    d = np.maximum(0, np.diff(lm, axis=0, prepend=lm[:1]))
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    full = d.sum(1)
    low = d[:, freqs < 200].sum(1)          # kick / bass weight
    env = full + 1.5 * low
    env = env - np.convolve(env, np.ones(43) / 43, mode="same")   # local mean removal (~0.5 s)
    env = np.maximum(env, 0); env /= (env.std() + 1e-9)
    return env, low / (low.std() + 1e-9)

def tempo(env, lo=70, hi=180, prior=120):
    n = len(env); e = env - env.mean()
    ac = np.correlate(e, e, "full")[n - 1:]
    cands = []
    for bpm in np.arange(lo, hi + .1, .25):
        lag = 60 / bpm / FR; i = int(lag)
        if i + 1 >= n // 2: continue
        v = ac[i] * (1 - (lag - i)) + ac[i + 1] * (lag - i)
        # also reward the 2x-lag peak (bar-level periodicity) to stabilise
        j = int(2 * lag); v += .5 * (ac[j] if j < len(ac) else 0)
        v *= np.exp(-0.5 * (np.log2(bpm / prior) / 0.9) ** 2)
        cands.append((v, bpm))
    cands.sort(reverse=True)
    return cands[0][1], [round(b, 2) for _, b in cands[:5]]

def fit_grid(env, bpm0, span=3.0):
    """Comb-filter fit: search tempo (+-span BPM, 0.02 steps) and phase for the constant grid that lands on the most onset energy."""
    n = len(env); best = (-1, bpm0, 0.0)
    for bpm in np.arange(bpm0 - span, bpm0 + span + 1e-9, 0.02):
        per = 60 / bpm / FR
        for ph in np.arange(0, per, 1.0):
            pos = ph + per * np.arange(int((n - ph) / per))
            i = pos.astype(int); fr = pos - i; i1 = np.minimum(i + 1, n - 1)
            sc = (env[i] * (1 - fr) + env[i1] * fr).mean()
            if sc > best[0]: best = (sc, bpm, ph * FR)
    return best[1], best[2]

def analyze(path):
    x = decode(path); dur = len(x) / SR
    mag = stft_mag(x); env, low = onset_env(mag)
    bpm0, cands = tempo(env)
    bpm, _ = fit_grid(env, bpm0)
    # phase: hi-hats/off-beat bass can out-shout the kick, so weight the sub-200 Hz hits heavily when choosing which grid line is 'the beat'
    bpm, t0 = fit_grid(env + 2.5 * low, bpm, span=0.4); per = 60 / bpm
    t0 = (t0 + LAT) % per   # STFT frames are stamped at window start; calibrated lag so the grid sits on the audible attack
    n_beats = int((dur - t0) / per) + 1
    # steady grid, trimmed back to where the track really is
    grid = t0 + per * np.arange(n_beats)
    grid = grid[grid >= 0]
    # downbeat: bar phase whose beats carry most low-band (kick) energy; beats 1/3 vs 2/4 tie-break by overall onset
    lf = np.clip((grid / FR).astype(int), 0, len(low) - 1)
    ph = [low[lf[p::4]].mean() + .3 * env[lf[p::4]].mean() if len(lf[p::4]) else -9 for p in range(4)]
    dp = int(np.argmax(ph))
    bar = [int((i - dp) % 4 == 0) for i in range(len(grid))]
    # energy per bar -> drops / builds
    rms = np.sqrt((x[: len(x) // HOP * HOP].reshape(-1, HOP) ** 2).mean(1))
    bars = []
    for i in range(dp, len(grid) - 4, 4):
        a, b = int(grid[i] / FR), int(grid[i + 4] / FR); bars.append((grid[i], float(rms[a:b].mean())))
    drops = []
    for i in range(1, len(bars)):
        base = np.mean([e for _, e in bars[max(0, i - 2):i]]) + 1e-9
        if bars[i][1] > 1.35 * base: drops.append(round(bars[i][0], 3))
    # strong onsets (hits) for punchy accents
    thr = np.percentile(env, 92); pk = [i for i in range(1, len(env) - 1) if env[i] > thr and env[i] >= env[i - 1] and env[i] > env[i + 1]]
    return dict(file=path, duration=round(dur, 3), bpm=round(bpm, 2), bpm_candidates=cands,
                beat_period=round(per, 5), first_beat=round(float(t0), 4),
                beats=[round(float(b), 4) for b in grid], downbeats=[round(float(g), 4) for g, f in zip(grid, bar) if f],
                energy_drops=drops, onsets=[round(i * FR, 3) for i in pk],
                note="If bpm looks half/double, the true tempo is in bpm_candidates; cut on 2 beats instead of 1 to compensate.")

def cuts(j, shots):
    b = j["beats"]; i = 0; out = []
    for s in shots:
        out.append(dict(start=round(b[i], 4), end=round(b[min(i + int(s), len(b) - 1)], 4), beats=s)); i += int(s)
    return out

def scene_cuts(path, thr):
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-vf", f"select='gt(scene,{thr})',showinfo", "-an", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    return [float(m) for m in re.findall(r"pts_time:([\d.]+)", r)]

def verify(path, thr):
    j = analyze(path); cs = scene_cuts(path, thr); b = np.array(j["beats"])
    if not cs: return dict(msg="no hard cuts detected (all soft transitions?) -- lower --thresh", bpm=j["bpm"])
    off = [float(c - b[np.abs(b - c).argmin()]) for c in cs]
    on = [abs(o) <= 0.04 for o in off]
    return dict(bpm=j["bpm"], cuts_found=len(cs), on_beat_pct=round(100 * sum(on) / len(cs), 1),
                median_offset_ms=round(1000 * float(np.median(off)), 1),
                worst=[(round(c, 2), round(1000 * o)) for c, o in sorted(zip(cs, off), key=lambda z: -abs(z[1]))[:5]],
                tip="positive offset = cut late; shift shot starts earlier by median_offset_ms (or the audio later)")

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit()
    if a[0] == "analyze":
        j = analyze(a[1]); out = a[a.index("--out") + 1] if "--out" in a else None
        if out: json.dump(j, open(out, "w"))
        s = {k: v for k, v in j.items() if k not in ("beats", "onsets")}; s["n_beats"] = len(j["beats"]); s["n_onsets"] = len(j["onsets"])
        print(json.dumps(s, indent=1))
    elif a[0] == "cuts": print(json.dumps(cuts(json.load(open(a[1])), a[2].split(",")), indent=1))
    elif a[0] == "snap":
        b = np.array(json.load(open(a[1]))["beats"]); print([round(float(b[np.abs(b - float(t)).argmin()]), 4) for t in a[2:]])
    elif a[0] == "verify":
        print(json.dumps(verify(a[1], float(a[a.index("--thresh") + 1]) if "--thresh" in a else .25), indent=1))
    else: print(__doc__)
