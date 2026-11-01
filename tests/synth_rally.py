"""Synthetic doubles rally with ground truth, for testing `hud` end to end.

    python tests/synth_rally.py <cutouts dir> <out dir> [--seed N]

Renders a portrait clip filmed from behind the baseline by a shaky hand-held camera: a room
with textured walls and ceiling lights, our court plus a neighbouring court (with its own
shuttle), four players pasted from photo cut-outs (near_red, near_pink, far_red, far_blue
.png), a shuttle flown with quadratic drag between rackets, and racket clicks in the audio.
Writes synth.mp4, synth_hud.toml and synth_truth.json.
"""
import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hud import court  # noqa: E402

W, H, FPS, SR = 1080, 1920, 30.0, 44100
F_PX = 1107.0
G, VT = 9.81, 6.8
NEIGHBOUR_DX = 7.6  # m, next court to the right
PLAYERS = [  # name, cut-out, side, config colour, shirt recolour (hue deg, sat, val) or None
    ("Randy", "far_blue", "near", "blue", None),
    ("Maria", "far_red", "near", "magenta", (318, 0.70, 0.75)),
    ("Alif", "near_pink", "far", "red", (356, 0.85, 0.62)),
    ("Rizky", "near_red", "far", "pink", (335, 0.42, 0.93)),
]


def recolour(img, hsv_target):
    """Repaint the shirt (saturated torso pixels) to a target hue/saturation/value."""
    if hsv_target is None:
        return img
    h, s, v = hsv_target
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    rows = np.zeros(img.shape[:2], bool)
    rows[int(0.12 * img.shape[0]): int(0.55 * img.shape[0])] = True
    hue = hsv[..., 0] * 2
    skin = (hue > 5) & (hue < 40) & (hsv[..., 1] < 150)
    shirt = rows & (hsv[..., 1] > 70) & ~skin
    hh, ww = img.shape[:2]
    centre = np.zeros_like(shirt)
    centre[int(0.2 * hh): int(0.45 * hh), int(0.3 * ww): int(0.7 * ww)] = True
    ref = np.median(hue[shirt & centre])
    dh = np.abs(hue - ref)
    shirt &= np.minimum(dh, 360 - dh) < 22
    hsv[..., 0][shirt] = h / 2
    hsv[..., 1][shirt] = np.clip(s * 255 * hsv[..., 1][shirt] / hsv[..., 1][shirt].mean(), 0, 255)
    hsv[..., 2][shirt] = np.clip(v * 255 * hsv[..., 2][shirt] / hsv[..., 2][shirt].mean(), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------- camera

def rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    return Rz @ Rx @ Ry


def camera(t, rng_walk):
    """World->camera rotation and centre at time t (hand-held wobble)."""
    yaw, pitch, roll, dx, dz = rng_walk(t)
    # Camera axes: x right, y down, z forward. Look along +y, pitched down ~10 deg.
    base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # world (x, y, z) -> cam
    R = rot(0, np.radians(-10) + pitch, 0)
    Rw = rot(yaw, 0, 0)
    Rc = base @ R.T @ Rw.T
    Rroll = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    Rc = Rroll @ Rc
    C = np.array([3.05 + dx, -2.6, 1.55 + dz])
    return Rc, C


K = np.array([[F_PX, 0, W / 2], [0, F_PX, H / 2], [0, 0, 1]])


def proj(Rc, C, X):
    Xc = (np.atleast_2d(X) - C) @ Rc.T
    z = Xc[:, 2]
    p = Xc @ K.T
    return p[:, :2] / p[:, 2:3], z


def floor_H(Rc, C):
    P = K @ np.c_[Rc, -Rc @ C]
    Hf = P[:, [0, 1, 3]]
    return Hf / Hf[2, 2]


def wobble(seed):
    rng = np.random.default_rng(seed)
    knots = np.arange(0, 30, 0.35)
    vals = rng.normal(0, 1, (len(knots), 5)) * [np.radians(1.2), np.radians(0.9), np.radians(1.0), 0.03, 0.02]

    def at(t):
        return np.array([np.interp(t, knots, vals[:, k]) for k in range(5)])
    return at


# ---------------------------------------------------------------- scene

def room_image(Rc, C):
    """Ray-cast the room: floor (mat + lines of two courts), walls, ceiling."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.stack([(xs - W / 2) / F_PX, (ys - H / 2) / F_PX, np.ones_like(xs)], -1) @ Rc  # world dirs
    img = np.zeros((H, W, 3), np.float32)
    planes = [  # (axis, value, kind)
        (2, 0.0, "floor"), (2, 9.0, "ceiling"), (1, 18.0, "wall_far"), (1, -6.0, "wall_back"),
        (0, -5.0, "wall_left"), (0, 17.0, "wall_right")]
    sid = np.full((H, W), -1, np.int8)
    best = np.full((H, W), np.inf, np.float32)
    for k, (ax, val, kind) in enumerate(planes):
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (val - C[ax]) / d[..., ax]
        ok = (s > 0) & (s < best)
        best[ok] = s[ok]
        sid[ok] = k
    P = C + d * best[..., None]
    for k, (ax, val, kind) in enumerate(planes):
        m = sid == k
        if not m.any():
            continue
        u, v = [P[..., i][m] for i in range(3) if i != ax]
        if kind == "floor":
            x, y = u, v
            col = np.empty((m.sum(), 3), np.float32)
            col[:] = (150, 160, 40)  # teal-green mat (BGR)
            on_court = lambda x0: (x >= x0 - 1.5) & (x <= x0 + court.WIDTH + 1.5) & (y >= -2) & (y <= 15.4)
            mat = on_court(0) | on_court(NEIGHBOUR_DX)
            col[~mat] = (60, 95, 140)  # wooden surround
            for x0 in (0.0, NEIGHBOUR_DX):
                dist = _court_line_distance(x - x0, y)
                col[dist < 0.02] = (235, 238, 240)
            col += (np.sin(x * 3.1) * np.cos(y * 2.3) * 6)[:, None]
        elif kind == "ceiling":
            col = np.empty((m.sum(), 3), np.float32)
            col[:] = (70, 90, 40)
            truss = (np.abs(((u + 0.3 * v) % 3.0) - 1.5) < 0.06) | (np.abs((v % 4.0) - 2) < 0.05)
            col[truss] = (220, 220, 220)
            light = (np.abs((u % 6.0) - 3) < 0.5) & (np.abs((v % 5.0) - 2.5) < 0.35)
            col[light] = (255, 255, 255)
        else:
            col = np.empty((m.sum(), 3), np.float32)
            iu, iv = np.floor(u / 0.9).astype(int), np.floor(v / 0.7).astype(int)
            h = (iu * 73856093 ^ iv * 19349663) % 997 / 997.0
            col[:] = np.stack([60 + 120 * h, 110 + 60 * (1 - h), 70 + 140 * ((h * 7) % 1)], -1)
            col[(np.abs(v - 2.5) < 0.12)] = (40, 140, 240)  # orange band
            grain = ((np.sin(u * 11) + np.sin(v * 13)) * 12)
            col += grain[:, None]
        img[m] = col
    return np.clip(img, 0, 255).astype(np.uint8)


def _court_line_distance(x, y):
    d = np.full(x.shape, np.inf, np.float32)
    for (x0, y0), (x1, y1) in court.segments():
        if x0 == x1:
            dd = np.where((y >= y0) & (y <= y1), np.abs(x - x0), np.inf)
        else:
            dd = np.where((x >= x0) & (x <= x1), np.abs(y - y0), np.inf)
        d = np.minimum(d, dd)
    return d


def paste(img, cut, foot_px, height_px):
    h, w = cut.shape[:2]
    s = height_px / h
    if s <= 0 or height_px < 20:
        return
    c = cv2.resize(cut, (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_LINEAR)
    ch, cw = c.shape[:2]
    x0, y0 = int(foot_px[0] - cw / 2), int(foot_px[1] - ch)
    yy, xx = np.mgrid[0:ch, 0:cw]
    alpha = np.clip(1.6 - np.hypot((xx - cw / 2) / (cw / 2), (yy - ch / 2) / (ch / 2)) * 1.1, 0, 1)[..., None]
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + cw), min(H, y0 + ch)
    if X0 >= X1 or Y0 >= Y1:
        return
    a = alpha[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    img[Y0:Y1, X0:X1] = (img[Y0:Y1, X0:X1] * (1 - a) + c[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * a).astype(np.uint8)


# ---------------------------------------------------------------- shuttle physics

def fly(p0, v0, T, dt=1 / 600):
    p, v = np.array(p0, float), np.array(v0, float)
    out = [p.copy()]
    for _ in range(int(round(T / dt))):
        a = np.array([0, 0, -G]) - (G / VT ** 2) * np.linalg.norm(v) * v
        v = v + a * dt
        p = p + v * dt
        out.append(p.copy())
    return np.array(out)


def aim(p0, p1, T):
    def r(v):
        return fly(p0, v, T, 1 / 200)[-1] - p1
    guess = (np.array(p1) - p0) / T + [0, 0, G * T / 2]
    return least_squares(r, guess * 1.5).x


# ---------------------------------------------------------------- rally

def plan(rng, n_shots=11):
    """Shot list: (time, hitter, racket point). Sides alternate; first shot is a serve."""
    t = 0.8
    shots = []
    side = "near"
    names = {s: [p[0] for p in PLAYERS if p[2] == s] for s in ("near", "far")}
    for i in range(n_shots):
        hitter = names[side][rng.integers(2)] if i else names["near"][0]
        if side == "near":
            pt = (rng.uniform(1.0, 5.1), rng.uniform(1.2, 5.6) if i else 2.8)
        else:
            pt = (rng.uniform(1.0, 5.1), rng.uniform(8.0, 12.6))
        z = 1.0 if i == 0 else rng.uniform(1.6, 2.5)
        shots.append((t, hitter, side, np.array([pt[0], pt[1], z])))
        t += 1.3 if i == 0 else rng.uniform(0.55, 1.5)
        side = "far" if side == "near" else "near"
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cutouts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    cuts = {p[0]: recolour(cv2.imread(str(args.cutouts / f"{p[1]}.png")), p[4]) for p in PLAYERS}
    side_of = {p[0]: p[2] for p in PLAYERS}

    shots = plan(rng)
    # Flights between consecutive rackets; the last one lands on the floor.
    flights = []
    for (t0, h0, s0, p0), (t1, h1, s1, p1) in zip(shots, shots[1:]):
        v0 = aim(p0, p1, t1 - t0)
        flights.append((t0, t1, p0, v0))
    tl, hl, sl, pl = shots[-1]
    land = np.array([rng.uniform(0.8, 5.3), 2.0 if sl == "far" else 11.5, 0.0])
    T_last = 1.1
    flights.append((tl, tl + T_last, pl, aim(pl, land, T_last)))
    end = tl + T_last
    duration = end + 1.0
    n = int(duration * FPS)

    # Player paths: keyframes at their own hits, otherwise drifting around a base spot.
    base = {"Randy": (1.6, 2.6), "Maria": (4.4, 3.4), "Alif": (1.7, 10.5), "Rizky": (4.4, 10.0)}
    keys = {name: [(0.0, np.array(base[name]))] for name in base}
    for t, h, s, p in shots:
        keys[h].append((t, p[:2] - [0.25, 0.25 if s == "near" else -0.25]))
    for name in keys:
        for t in np.arange(0.4, duration, 1.1):
            if all(abs(t - k[0]) > 0.5 for k in keys[name]):
                keys[name].append((t, np.array(base[name]) + rng.normal(0, 0.35, 2)))
        keys[name].sort(key=lambda k: k[0])
    ppos = {}
    tt = np.arange(n) / FPS
    for name, ks in keys.items():
        kt = np.array([k[0] for k in ks])
        kp = np.array([k[1] for k in ks])
        ppos[name] = np.stack([np.interp(tt, kt, kp[:, 0]), np.interp(tt, kt, kp[:, 1])], 1)

    # Shuttle 3D per frame.
    sh = np.full((n, 3), np.nan)
    for t0, t1, p0, v0 in flights:
        path = fly(p0, v0, t1 - t0)
        for f in range(int(np.ceil(t0 * FPS)), min(n, int(t1 * FPS) + 1)):
            k = int(round((f / FPS - t0) * 600))
            sh[f] = path[min(k, len(path) - 1)]
    # Distractor shuttle on the neighbouring court.
    other = np.full((n, 3), np.nan)
    t = 0.3
    while t < duration:
        T = rng.uniform(0.7, 1.4)
        a = np.array([NEIGHBOUR_DX + rng.uniform(0.5, 5.5), rng.uniform(1, 5), 2.0])
        b = np.array([NEIGHBOUR_DX + rng.uniform(0.5, 5.5), rng.uniform(8, 12.5), 2.0])
        if rng.random() < 0.5:
            a, b = b, a
        path = fly(a, aim(a, b, T), T)
        for f in range(int(t * FPS), min(n, int((t + T) * FPS))):
            other[f] = path[min(int(round((f / FPS - t) * 600)), len(path) - 1)]
        t += T

    walk = wobble(args.seed)
    truth_h, truth_shuttle = [], []
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16",
                             "-pix_fmt", "yuv420p", str(args.out / "video_only.mp4")], stdin=subprocess.PIPE)
    prev_px = {}
    for f in range(n):
        Rc, C = camera(f / FPS, walk)
        img = room_image(Rc, C)
        Hf = floor_H(Rc, C)
        truth_h.append(Hf.tolist())
        if np.isnan(sh[f, 0]):
            truth_shuttle.append(None)
        else:
            (p,), z = proj(Rc, C, sh[f])
            truth_shuttle.append(p.tolist() if z[0] > 0 else None)
        # Players, far first.
        order = sorted(ppos, key=lambda nm: -ppos[nm][f][1])
        for nm in order:
            x, y = ppos[nm][f]
            (foot,), _ = proj(Rc, C, [x, y, 0])
            (head,), _ = proj(Rc, C, [x, y, 1.72])
            paste(img, cuts[nm], foot, foot[1] - head[1])
        for key, arr in (("ours", sh), ("other", other)):
            if np.isnan(arr[f, 0]):
                prev_px.pop(key, None)
                continue
            (p,), z = proj(Rc, C, arr[f])
            if z[0] <= 0:
                continue
            r = max(2, int(F_PX * 0.035 / z[0]))
            q = prev_px.get(key, p)
            cv2.line(img, tuple(np.int32(q)), tuple(np.int32(p)), (245, 245, 245), r, cv2.LINE_AA)
            prev_px[key] = p
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))
        proc.stdin.write(img.tobytes())
        print(f"\rframe {f + 1}/{n}", end="", flush=True)
    proc.stdin.close()
    proc.wait()
    print()

    # Audio: hall noise, racket clicks at our hits, quieter clicks from the next court.
    audio = rng.normal(0, 0.01, int(duration * SR)).astype(np.float32)
    from scipy.signal import butter, sosfilt
    sos = butter(4, [2500, 6000], btype="band", fs=SR, output="sos")

    def click(t, amp):
        i = int(t * SR)
        burst = sosfilt(sos, rng.normal(0, 1, int(0.012 * SR))) * np.exp(-np.arange(int(0.012 * SR)) / (0.003 * SR))
        audio[i:i + len(burst)] += amp * burst[: len(audio) - i]
    for t, *_ in shots:
        click(t, 0.5)
    for t in np.arange(0.3, duration, 1.05):
        click(t + rng.uniform(0, 0.3), 0.12)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(args.out / "audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(args.out / "video_only.mp4"), "-i",
                    str(args.out / "audio.wav"), "-c:v", "copy", "-c:a", "aac", "-shortest",
                    str(args.out / "synth.mp4")], check=True)

    truth = {
        "fps": FPS, "frames": n, "rally_end": end,
        "shots": [{"time": t, "hitter": h, "side": s,
                   "launch_kmh": float(np.linalg.norm(fl[3]) * 3.6)}
                  for (t, h, s, _), fl in zip(shots, flights)],
        "players": {nm: ppos[nm].tolist() for nm in ppos},
        "H": truth_h,
        "shuttle": truth_shuttle,
    }
    (args.out / "synth_truth.json").write_text(json.dumps(truth))
    cfg = "".join(f'[[players]]\nname = "{p[0]}"\ncolor = "{p[3]}"\nside = "{p[2]}"\n\n' for p in PLAYERS)
    (args.out / "synth_hud.toml").write_text(cfg + "[fixes]\nadd_shots = []\nremove_shots = []\n")
    print("shots:", [(round(s[0], 2), s[1]) for s in shots], "end", round(end, 2))


if __name__ == "__main__":
    main()
