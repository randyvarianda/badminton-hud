"""Step 6: who is who - shirt hue, side of the net, and last position.

Colour decides within a team, the side of the net decides the team, and position only breaks
near-ties (position alone swaps names when partners switch sides). Track-level majority vote
then keeps one name per ByteTrack id.
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import court, people

IN_X = (-1.0, court.WIDTH + 1.0)
IN_Y = (-2.5, court.LENGTH + 2.5)
POS_WEIGHT = 0.06  # colour-cone units per metre of jump
UNASSIGNED = 0.55  # max colour distance before a box is not one of our players


def foot(box):
    return [(box[0] + box[2]) / 2, box[3]]


def _ref(player):
    h, s, v = player.hsv
    return people.cone(np.array([h]), np.array([s]), np.array([v]))[0]


def assign(dets_per_frame, Hs, players):
    """Returns per-frame {name: det} and the per-team colour centres actually used."""
    n = len(dets_per_frame)
    # 1. Court position and side for every box.
    for t, dets in enumerate(dets_per_frame):
        for d in dets:
            w = court.to_world(Hs[t], [foot(d["box"])])[0]
            _, z = court.project(Hs[t], [w])
            d["world"] = w
            # z <= 0: the feet are above the horizon (a poster on the wall), not on the floor.
            d["in_court"] = bool(z[0] > 0 and IN_X[0] < w[0] < IN_X[1] and IN_Y[0] < w[1] < IN_Y[1])
            d["side"] = "near" if w[1] < court.NET_Y else "far"

    # 2. Track-level colour: median over the track's in-court samples.
    samples = defaultdict(list)
    for dets in dets_per_frame:
        for d in dets:
            if d["in_court"] and d["color"] is not None:
                samples[d["id"]].append(d["color"])
    track_color = {k: np.median(v, 0) for k, v in samples.items() if len(v) >= 3}

    def color(d):
        return track_color.get(d["id"], d["color"])

    # 3. Re-centre each team's two reference colours on this clip's lighting (2-means).
    centres = {}
    for side in ("near", "far"):
        team = [p for p in players if p.side == side]
        C = np.array([_ref(p) for p in team])
        X = np.array([color(d) for dets in dets_per_frame for d in dets
                      if d["in_court"] and d["side"] == side and color(d) is not None])
        for _ in range(10):
            if len(X) < 10:
                break
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), 1)
            if min((lab == 0).sum(), (lab == 1).sum()) < 0.15 * len(X):
                break  # one shirt missing or a stray person: keep the named colours
            C = np.array([X[lab == k].mean(0) for k in range(2)])
        for p, c in zip(team, C):
            centres[p.name] = c

    # 4. Per-frame assignment within each side.
    last = {}
    labels = []
    for t, dets in enumerate(dets_per_frame):
        frame_lab = {}
        for side in ("near", "far"):
            team = [p.name for p in players if p.side == side]
            cand = [i for i, d in enumerate(dets) if d["in_court"] and d["side"] == side]
            if not cand:
                continue
            cost = np.zeros((len(cand), 2))
            for a, i in enumerate(cand):
                c = color(dets[i])
                for b, name in enumerate(team):
                    cd = 0.4 if c is None else np.linalg.norm(c - centres[name])
                    jump = np.linalg.norm(dets[i]["world"] - last[name]) if name in last else 0.0
                    cost[a, b] = cd + POS_WEIGHT * min(jump, 4.0)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < UNASSIGNED + POS_WEIGHT * 4:
                    frame_lab[cand[r]] = team[c]
        for i, name in frame_lab.items():
            last[name] = dets[i]["world"]
        labels.append(frame_lab)

    # 5. One name per track (majority), then fix frames where two tracks claim one name.
    votes = defaultdict(Counter)
    for t, lab in enumerate(labels):
        for i, name in lab.items():
            votes[dets_per_frame[t][i]["id"]][name] += 1
    track_name = {}
    for tid, c in votes.items():
        name, k = c.most_common(1)[0]
        if tid >= 0 and k >= 0.6 * sum(c.values()):
            track_name[tid] = (name, k)
    out = []
    side_of = {p.name: p.side for p in players}
    for t, dets in enumerate(dets_per_frame):
        lab = dict(labels[t])
        for i, d in enumerate(dets):
            if d["id"] in track_name and d["in_court"]:
                lab[i] = track_name[d["id"]][0]
        by_name = defaultdict(list)
        for i, name in lab.items():
            by_name[name].append(i)
        frame = {}
        for name, idx in by_name.items():
            idx.sort(key=lambda i: -track_name.get(dets[i]["id"], ("", 0))[1])
            frame.setdefault(name, dets[idx[0]])
            for i in idx[1:]:
                mate = next(p.name for p in players if p.side == side_of[name] and p.name != name)
                if mate not in frame and mate not in by_name:
                    frame[mate] = dets[i]
        out.append(frame)
    return out, centres


def tracks(assigned, n, fps, players):
    """Smoothed world position and image box per player per frame (NaN when unknown)."""
    res = {}
    for p in players:
        pos = np.full((n, 2), np.nan)
        box = np.full((n, 4), np.nan)
        for t in range(n):
            d = assigned[t].get(p.name)
            if d is not None:
                pos[t] = d["world"]
                box[t] = d["box"]
        pos = _fill(pos, int(fps))
        box = _fill(box, int(fps))
        res[p.name] = {"pos": _smooth(pos, max(3, int(0.25 * fps))), "box": _smooth(box, 3)}
    return res


def _fill(a, max_gap):
    """Linear interpolation over gaps up to max_gap frames."""
    a = a.copy()
    ok = ~np.isnan(a[:, 0])
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return a
    for k in range(a.shape[1]):
        a[:, k] = np.interp(np.arange(len(a)), idx, a[idx, k])
    # Re-blank long gaps and the ends beyond the first/last sighting.
    gap_start = None
    for t in range(len(a)):
        if not ok[t] and gap_start is None:
            gap_start = t
        if ok[t] and gap_start is not None:
            if t - gap_start > max_gap:
                a[gap_start:t] = np.nan
            gap_start = None
    a[: idx[0]] = np.nan
    a[idx[-1] + 1:] = np.nan
    return a


def _smooth(a, win):
    out = a.copy()
    half = win // 2
    for t in range(len(a)):
        seg = a[max(0, t - half): t + half + 1]
        seg = seg[~np.isnan(seg[:, 0])]
        if len(seg) and not np.isnan(a[t, 0]):
            out[t] = seg.mean(0)
    return out
