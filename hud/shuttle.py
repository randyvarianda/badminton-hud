"""Step 7: shuttle candidates -> tracklets -> one path through the rally.

A candidate is a small bright blob that is brighter than frames t±3 and t±10, each
aligned to frame t twice: by the camera (background) homography and by the floor plane.
The path may only change direction at the racket of one of our four players, which drops
shuttles from neighbouring courts.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

OFFSETS = (-10, -3, 3, 10)
EXCESS = 22  # grey levels brighter than every aligned neighbour frame
MIN_GREY = 110


def candidates(read_frames, n, cam_abs, Hs, boxes_per_frame, progress):
    """Per frame: array of [x, y, area, peak excess]."""
    out = [None] * n
    k = max(OFFSETS)

    def process(t, frames):
        cur = frames[t]
        refs = []
        for o in OFFSETS:
            s = t + o
            if not 0 <= s < n:
                continue
            for M in (cam_abs[t] @ np.linalg.inv(cam_abs[s]), Hs[t] @ np.linalg.inv(Hs[s])):
                w = cv2.warpPerspective(frames[s], M, (cur.shape[1], cur.shape[0]),
                                        borderMode=cv2.BORDER_REPLICATE)
                refs.append(cv2.dilate(w, np.ones((5, 5), np.uint8)))
        ref = np.max(refs, axis=0)
        excess = cur.astype(np.int16) - ref
        m = ((excess > EXCESS) & (cur > MIN_GREY)).astype(np.uint8)
        for x0, y0, x1, y1 in boxes_per_frame[t]:
            # Moving bodies and clothes light up too; the racket zone above the shoulders stays.
            w = x1 - x0
            m[int(y0 + 0.2 * (y1 - y0)): int(y1) + 1, max(0, int(x0 + 0.1 * w)): int(x1 - 0.1 * w) + 1] = 0
        num, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        rows = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if 2 <= area <= 600 and max(w, h) <= 120:
                peak = excess[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].max()
                rows.append([*cents[i], area, peak])
        rows.sort(key=lambda r: -r[3] * np.sqrt(r[2]))
        out[t] = np.array(rows[:60]).reshape(-1, 4)

    frames = {}
    for t, frame in enumerate(read_frames()):
        frames[t] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if t - k >= 0:
            process(t - k, frames)
            progress(t - k + 1)
            frames.pop(t - 2 * k, None)
    for t in range(max(0, n - k), n):
        process(t, frames)
        progress(t + 1)
    return out


# ---------------------------------------------------------------- tracklets

def tracklets(cands, max_miss=2, max_step=200.0):
    """Seed-and-grow linking, robust to clutter. Each tracklet: array of (t, x, y, excess).

    Seeds are three consecutive candidates moving at constant velocity; each seed grows both
    ways with a quadratic prediction (gravity + drag bend the path) and a tight gate. Best
    seeds first; a candidate belongs to one tracklet only.
    """
    n = len(cands)
    pts = [c[:, :2] if len(c) else np.zeros((0, 2)) for c in cands]
    used = [np.zeros(len(p), bool) for p in pts]
    seeds = []
    for t in range(1, n - 1):
        A, B, C = pts[t - 1], pts[t], pts[t + 1]
        if not (len(A) and len(B) and len(C)):
            continue
        for j, b in enumerate(B):
            d_ab = np.linalg.norm(A - b, axis=1)
            for i in np.flatnonzero((d_ab > 2) & (d_ab < max_step)):
                pred = 2 * b - A[i]
                d = np.linalg.norm(C - pred, axis=1)
                k = int(np.argmin(d))
                if d[k] < 4 + 0.15 * d_ab[i]:
                    seeds.append((d[k] / (1 + d_ab[i]), t, i, j, k))
    seeds.sort()
    out = []
    for _, t, i, j, k in seeds:
        if used[t - 1][i] or used[t][j] or used[t + 1][k]:
            continue
        tr = [(t - 1, i), (t, j), (t + 1, k)]
        for direction in (1, -1):
            miss = 0
            while miss <= max_miss:
                last = tr[-1] if direction == 1 else tr[0]
                nxt = last[0] + direction * (miss + 1)
                if not 0 <= nxt < n or not len(pts[nxt]):
                    break
                seq = tr[-6:] if direction == 1 else tr[:6]
                tt = np.array([q[0] for q in seq], float)
                xy = np.array([pts[q[0]][q[1]] for q in seq])
                deg = 2 if len(seq) >= 5 else 1
                pred = np.array([np.polyval(np.polyfit(tt, xy[:, c], deg), nxt) for c in (0, 1)])
                step = np.linalg.norm(xy[-1] - xy[-2]) if direction == 1 else np.linalg.norm(xy[1] - xy[0])
                d = np.linalg.norm(pts[nxt] - pred, axis=1)
                d[used[nxt]] = np.inf
                m = int(np.argmin(d))
                if d[m] < 6 + 0.25 * step * (miss + 1):
                    if direction == 1:
                        tr.append((nxt, m))
                    else:
                        tr.insert(0, (nxt, m))
                    miss = 0
                else:
                    miss += 1
        for q in tr:
            used[q[0]][q[1]] = True
        arr = np.array([(q[0], *pts[q[0]][q[1]], cands[q[0]][q[1], 3]) for q in tr])
        for piece in _split_at_turns(arr):
            if _good(piece):
                out.append(piece)
    return sorted(out, key=lambda a: a[0, 0])


def _split_at_turns(a, angle=50):
    """Cut where the flight direction turns sharply; a turn is a hit (or noise).

    Up-then-down (image y) is left alone: gravity turns every lob over at its apex.
    """
    if len(a) < 5:
        return [a]
    v = np.diff(a[:, 1:3], axis=0) / np.diff(a[:, 0])[:, None]
    cuts = []
    for i in range(2, len(v) - 1):
        u, w = v[max(0, i - 2): i].mean(0), v[i: i + 2].mean(0)
        nu, nw = np.linalg.norm(u), np.linalg.norm(w)
        apex = u[1] < 0 < w[1] and abs(u[0] - w[0]) < 0.5 * max(nu, nw)
        if nu > 3 and nw > 3 and not apex and np.dot(u, w) / (nu * nw) < np.cos(np.radians(angle)):
            if not cuts or i - cuts[-1] > 2:
                cuts.append(i)
    pieces, s = [], 0
    for c in cuts:
        pieces.append(a[s: c + 1])
        s = c
    pieces.append(a[s:])
    return pieces


def _good(a):
    if len(a) < 4:
        return False
    if np.linalg.norm(a[-1, 1:3] - a[0, 1:3]) < 20:
        return False  # not flying
    # Smooth flight: a low-order polynomial in time fits it (cubic for long, drag-bent flights).
    t = a[:, 0] - a[0, 0]
    deg = 2 if len(a) < 10 else 3
    res = []
    for k in (1, 2):
        c = np.polyfit(t, a[:, k], deg)
        res.append(a[:, k] - np.polyval(c, t))
    speed = np.median(np.linalg.norm(np.diff(a[:, 1:3], axis=0), axis=1))
    return np.sqrt(np.mean(np.square(res))) < 5 + 0.05 * speed


def _extrap(a, t, n=5):
    seg = a[:n] if t < a[0, 0] else a[-n:]
    if len(seg) < 2:
        return seg[0, 1:3]
    deg = 2 if len(seg) >= 4 else 1
    tt = seg[:, 0] - seg[0, 0]
    return np.array([np.polyval(np.polyfit(tt, seg[:, k], deg), t - seg[0, 0]) for k in (1, 2)])


# ---------------------------------------------------------------- one path

def racket_zone(box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return (x0 - 0.7 * w, y0 - 0.55 * h, x1 + 0.7 * w, y0 + 0.75 * h)


def _in_zone(p, box):
    if box is None or np.isnan(box[0]):
        return False
    x0, y0, x1, y1 = racket_zone(box)
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def path(tracks, boxes, fps, frame_h):
    """Best chain of tracklets. boxes[name][t] = player image box (NaN if unknown).

    Links: 'flight' (continues the same flight), 'hit' (direction change inside one of our
    players' racket zones), 'apex' (leaves through the top of the frame and comes back down),
    'lost' (unseen for a while: lost at one of our rackets and found again at one).
    Clutter tracklets (dull blobs, quality < 0.2) never join the path.
    Returns (chosen tracklets, links [(type, t, (x, y), name or None, quality 0..1)]).
    """
    n = len(tracks)
    if not n:
        return [], []
    max_gap = int(0.5 * fps)
    max_lost = int(3.0 * fps)
    quality = qualities(tracks)
    score = np.array([len(a) for a in tracks], float) * quality
    score[quality < 0.2] = -np.inf
    best = score.copy()
    prev = [-1] * n
    link = [None] * n
    for j in range(n):
        for i in range(j):
            gap = tracks[j][0, 0] - tracks[i][-1, 0]
            if gap < -1 or gap > max_lost or not np.isfinite(best[i]) or not np.isfinite(score[j]):
                continue
            lk = _link(tracks[i], tracks[j], boxes, frame_h, max_gap)
            if lk is None:
                continue
            per_frame = 0.05 if lk[0] in ("lost", "apex") else 0.15
            s = best[i] + score[j] - per_frame * max(gap, 0) - (3.0 if lk[0] == "hit" else 0)
            if s > best[j]:
                best[j], prev[j], link[j] = s, i, lk
    j = int(np.argmax(best))
    chain, links = [], []
    while j >= 0:
        chain.append(tracks[j])
        if link[j] is not None:
            links.append(link[j] + (min(quality[j], quality[prev[j]]),))
        j = prev[j]
    return chain[::-1], links[::-1]


def qualities(tracks):
    """How shuttle-like each tracklet is (0..1): how much brighter than its surroundings its
    blobs are, on the clip's own scale. Clutter sits low in the range, the shuttle high."""
    if not tracks:
        return np.zeros(0)
    peaks = np.array([np.median(a[:, 3]) for a in tracks])
    lo, hi = np.percentile(peaks, 50), np.percentile(peaks, 99)
    return np.clip((peaks - lo) / max(hi - lo, 1e-6), 0.05, 1.0)


def anchored(tracks, chain, boxes, reach=6, min_quality=0.5):
    """Bright flights outside the path that start or end at one of our players' rackets.

    Some flights cannot be chained (the shuttle vanishes near a player's body and there is
    no racket sound to bridge the gap), but a flight that leaves or reaches our racket is
    still ours; a shuttle from the next court starts and ends at other people's rackets.
    Returns [(tracklet, start or None, end or None, quality)], start/end = (frame, (x, y), name).
    """
    taken = {int(t) for a in chain for t in a[:, 0]}
    out = []
    for a, q in zip(tracks, qualities(tracks)):
        if q < min_quality or len(a) < reach or sum(int(t) in taken for t in a[:, 0]) > 2:
            continue
        ends = []
        for frames in (range(int(a[0, 0]), int(a[0, 0]) - reach - 1, -1),
                       range(int(a[-1, 0]), int(a[-1, 0]) + reach + 1)):
            hit = None
            for t in frames:
                p = _extrap(a, t)
                name = _zone_at(p, t, boxes)
                if name:
                    hit = (t, tuple(p), name)
                    break
            ends.append(hit)
        if ends[0] or ends[1]:
            out.append((a, ends[0], ends[1], float(q)))
            taken |= {int(t) for t in a[:, 0]}
    return out


def _link(a, b, boxes, frame_h, max_gap):
    t_end, t_start = a[-1, 0], b[0, 0]
    gap = t_start - t_end
    if gap <= max_gap:
        p = _extrap(a, t_start)
        va, vb = a[-1, 1:3] - a[-2, 1:3], b[1, 1:3] - b[0, 1:3]
        player = _zone_of(a[-1], boxes) or _zone_of(b[0], boxes)
        if np.linalg.norm(p - b[0, 1:3]) < 25 + 8 * max(gap, 0):
            sa, sb = np.linalg.norm(va), np.linalg.norm(vb)
            if player and gap <= 2 and (sb > 1.8 * sa or sb < 0.55 * sa):
                # Same image direction, sudden speed change at a racket: the shuttle was
                # sent back along the line it came in on (a reversal in depth).
                return ("hit", int(t_end), tuple(a[-1, 1:3]), player)
            if np.dot(va, vb) > 0 or va[1] < 0 < vb[1]:  # straight on, or over the apex
                return ("flight", int(t_start), tuple(b[0, 1:3]), None)
        # Direction change: the flight in and the flight out both touch one player's racket.
        name = _zone_of(a[-1], boxes)
        if name and name == _zone_of(b[0], boxes):
            t = int(round((t_end + t_start) / 2))
            return ("hit", t, tuple((a[-1, 1:3] + b[0, 1:3]) / 2), name)
        return None
    top = 0.12 * frame_h
    if a[-1, 2] < top and a[-1, 2] < a[-2, 2] and b[0, 2] < top and b[1, 2] > b[0, 2]:
        return ("apex", int((t_end + t_start) / 2), None, None)
    if _zone_of(a[-1], boxes) and _zone_of(b[0], boxes):
        return ("lost", int((t_end + t_start) / 2), None, None)
    return None


def _zone_of(row, boxes):
    """Player whose racket zone holds this tracklet point (t, x, y, ...).

    Zones overlap on screen (a near player's big zone can cover far-court players), so pick
    the zone whose centre is closest in units of that player's height.
    """
    return _zone_at((row[1], row[2]), int(row[0]), boxes)


def _zone_at(p, t, boxes):
    x, y = p
    best = None
    for name, bx in boxes.items():
        if 0 <= t < len(bx) and _in_zone((x, y), bx[t]):
            z = racket_zone(bx[t])
            d = np.hypot(x - (z[0] + z[2]) / 2, y - (z[1] + z[3]) / 2) / (bx[t][3] - bx[t][1])
            if best is None or d < best[0]:
                best = (d, name)
    return None if best is None else best[1]


def positions(chain, n):
    """Per-frame shuttle image position (NaN when not seen); short gaps interpolated."""
    pos = np.full((n, 2), np.nan)
    for a in chain:
        for t, x, y, _ in a:
            pos[int(t)] = (x, y)
    ok = np.flatnonzero(~np.isnan(pos[:, 0]))
    for i, j in zip(ok[:-1], ok[1:]):
        if 1 < j - i <= 4:
            for k in (0, 1):
                pos[i:j + 1, k] = np.linspace(pos[i, k], pos[j, k], j - i + 1)
    return pos
