"""Step 8: shots, hitters and launch speeds.

Seen shots come from the shuttle path (direction changes at a racket); racket sounds fill in
where the shuttle cannot be seen. Shots always alternate sides of the net, so the final list
is the best-scoring alternating subsequence. The hitter is the team member closer to the
shuttle.

Launch speed: v0 = (e^(kD) - 1) / (kT), k = g / vt^2 with terminal velocity vt = 6.8 m/s;
D = horizontal distance from hitter to the next hitter, T = time between the two hits.
"""
from dataclasses import dataclass

import numpy as np

from . import court, shuttle

K = 9.81 / 6.8 ** 2
MIN_GAP = 0.25  # s; two shots closer than this cannot both be real


@dataclass
class Shot:
    time: float
    hitter: str
    side: str
    source: str  # seen | sound | manual
    point: tuple | None = None  # shuttle image position at the hit, if seen
    speed: float | None = None  # km/h
    distance: float | None = None  # m
    flight: float | None = None  # s
    hitter_guessed: bool = False


def launch_speed(D, T):
    return (np.exp(K * D) - 1) / (K * T) * 3.6


def _candidates(chain, links, flights, onsets, pos, boxes, side_of, fps, fixes):
    """Each candidate: (time, {side: score}, source, point, hitter)."""
    seen = []
    for _, start, end, q in flights:
        # A flight leaving a racket was hit there; one reaching a racket is about to be.
        for anchor in (start, end):
            if anchor is not None:
                t, p, name = anchor
                other = "far" if side_of[name] == "near" else "near"
                seen.append([t / fps, {side_of[name]: 0.3 + 0.5 * q, other: 0.05}, "seen", p, None])
    for typ, t, p, name, q in links:
        if typ == "hit":
            # Side from the racket zone is a strong hint, not a rule: zones overlap on screen.
            other = "far" if side_of[name] == "near" else "near"
            seen.append([t / fps, {side_of[name]: 0.4 + 0.6 * q, other: 0.1}, "seen", p, None])
    cands = []
    for tau, strength in onsets:
        if strength < 0.15:
            continue  # quiet = another court
        f = int(round(tau * fps))
        match = [c for c in seen if abs(c[0] - tau) < 0.15]
        if match:
            # Seen and heard: one shot, timed by the sound.
            c = min(match, key=lambda c: abs(c[0] - tau))
            c[0] = tau
            c[1] = {sd: v + strength for sd, v in c[1].items()}
            continue
        if any(a[0, 0] <= f - 3 and a[-1, 0] >= f + 3 for a in chain):
            strength -= 0.3  # an unbroken flight seems to pass through: less likely a hit
        sides = {"near": strength, "far": strength}
        name = _nearest_player(f, pos, boxes, fps)
        if name is not None:
            sides[side_of[name]] += 0.3  # the shuttle was last/next seen by this player
        cands.append((tau, sides, "sound", None, None))
    cands += [tuple(c) for c in seen]
    cands = [c for c in cands if not any(abs(c[0] - r) < 0.2 for r in fixes.remove_shots)]
    for tau, hitter in fixes.add_shots:
        sides = [side_of[hitter]] if hitter else ["near", "far"]
        cands.append((tau, {sd: 100.0 for sd in sides}, "manual", None, hitter))
    return sorted(cands, key=lambda c: c[0])


def _nearest_player(f, pos, boxes, fps, window=0.4):
    """Player whose racket zone is closest to the nearest shuttle sighting within the window."""
    for d in range(int(window * fps) + 1):
        for g in (f - d, f + d):
            if 0 <= g < len(pos) and not np.isnan(pos[g, 0]):
                best = None
                for name, bx in boxes.items():
                    if np.isnan(bx[g, 0]):
                        continue
                    z = shuttle.racket_zone(bx[g])
                    dist = np.hypot(pos[g, 0] - (z[0] + z[2]) / 2, pos[g, 1] - (z[1] + z[3]) / 2)
                    if best is None or dist < best[0]:
                        best = (dist, name)
                return None if best is None else best[1]
    return None


def _alternating(cands):
    """Highest-scoring subsequence whose sides alternate (DP over candidate x side)."""
    n = len(cands)
    best, back = {}, {}
    for j in range(n):
        for s, score in cands[j][1].items():
            other = "far" if s == "near" else "near"
            b, arg = 0.0, None
            for i in range(j):
                if cands[j][0] - cands[i][0] >= MIN_GAP and (i, other) in best and best[i, other] > b:
                    b, arg = best[i, other], (i, other)
            best[j, s] = score + b
            back[j, s] = arg
    if not best:
        return []
    key = max(best, key=best.get)
    out = []
    while key is not None:
        out.append((cands[key[0]], key[1]))
        key = back[key]
    return out[::-1]


def find(chain, links, flights, onsets, pos, tracks, players, Hs, fps, n, fixes):
    side_of = {p.name: p.side for p in players}
    boxes = {name: tr["box"] for name, tr in tracks.items()}
    picked = _alternating(_candidates(chain, links, flights, onsets, pos, boxes, side_of, fps, fixes))
    chain = chain + [f[0] for f in flights]  # every flight we trust, for hitter lookup
    shots = []
    for (t, _, source, p, hitter), side in picked:
        f = min(n - 1, int(round(t * fps)))
        guessed = False
        if hitter is None:
            hitter, guessed = _hitter(side, f, p, chain, boxes, players, fps)
        shots.append(Shot(t, hitter, side, source, p, hitter_guessed=guessed))

    end = fixes.rally_end
    seen = np.flatnonzero(~np.isnan(pos[:, 0]))
    if end is None:
        # The last shot's own flight: follow the path after it while it stays continuous.
        last_shot = shots[-1].time if shots else 0
        last_seen = None
        for t in seen[seen / fps > last_shot]:
            if last_seen is not None and t - last_seen > 0.5 * fps:
                break
            last_seen = t
        if last_seen is not None and last_seen / fps > last_shot + 0.2:
            end = last_seen / fps + 0.1
        else:
            end = last_shot + 1.0
        end = min(end, (n - 1) / fps)
    shots = [s for s in shots if s.time < end]

    for a, b in zip(shots, shots[1:]):
        pa, pb = _where(tracks[a.hitter]["pos"], a.time, fps), _where(tracks[b.hitter]["pos"], b.time, fps)
        a.flight = b.time - a.time
        if pa is not None and pb is not None:
            a.distance = float(np.linalg.norm(pb - pa))
            a.speed = float(launch_speed(a.distance, a.flight))
    if shots:
        last = shots[-1]
        last.flight = end - last.time
        pa = _where(tracks[last.hitter]["pos"], last.time, fps)
        after = seen[seen / fps > last.time + 0.2] if len(seen) else seen
        if pa is not None and len(after) and end - after[-1] / fps < 0.3:
            land = court.to_world(Hs[after[-1]], [pos[after[-1]]])[0]
            on_floor = -2 < land[0] < court.WIDTH + 2 and -3 < land[1] < court.LENGTH + 3
            if on_floor:  # last sighting is where it landed
                last.distance = float(np.linalg.norm(land - pa))
                last.speed = float(launch_speed(last.distance, last.flight))
    start = shots[0].time if shots else 0.0
    return shots, start, end


def _where(arr, t, fps, window=0.5):
    f = int(round(t * fps))
    for d in range(int(window * fps) + 1):
        for g in (f - d, f + d):
            if 0 <= g < len(arr) and not np.isnan(arr[g, 0]):
                return arr[g]
    return None


def _hitter(side, f, p, chain, boxes, players, fps):
    team = [pl.name for pl in players if pl.side == side]
    guessed = False
    if p is None:
        p = _shuttle_at(f, chain, fps)
    dists = []
    for name in team:
        bx = _where(boxes[name], f / fps, fps)
        if bx is None or p is None:
            dists.append(np.inf)
        else:
            z = shuttle.racket_zone(bx)
            cx, cy = (z[0] + z[2]) / 2, (z[1] + z[3]) / 2
            dists.append(np.hypot(p[0] - cx, p[1] - cy))
    if not np.isfinite(dists).any():
        guessed = True
    return team[int(np.argmin(dists))], guessed


def _shuttle_at(f, chain, fps, window=0.6):
    """Where the shuttle was at frame f: the incoming flight (or else the outgoing one)
    extrapolated to f. None if neither is within the window."""
    before = [a for a in chain if f - window * fps <= a[-1, 0] <= f]
    after = [a for a in chain if f <= a[0, 0] <= f + window * fps]
    if before:
        return shuttle._extrap(max(before, key=lambda a: a[-1, 0]), f)
    if after:
        return shuttle._extrap(min(after, key=lambda a: a[0, 0]), f)
    return None
