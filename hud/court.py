"""Steps 4-5: find the court in the first frame, then follow it through every frame.

World coordinates are metres on the floor: x across the court (0..6.10, left to right as
seen from the camera), y away from the camera (0 = near baseline, 13.40 = far baseline).
H maps world -> image.
"""
import itertools

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

WIDTH, LENGTH, NET_Y = 6.10, 13.40, 6.70
X_LINES = (0.0, 0.46, 3.05, 5.64, 6.10)
Y_BASE, Y_LONG_SERVICE = 0.0, 0.76


def segments(near_only=False):
    """Court lines as world segments ((x0, y0), (x1, y1))."""
    ys = [0.0, 0.76, 4.72] + ([] if near_only else [8.68, 12.64, 13.40])
    end = NET_Y if near_only else LENGTH
    segs = [((0, y), (WIDTH, y)) for y in ys]
    segs += [((x, 0), (x, end)) for x in (0.0, 0.46, 5.64, 6.10)]
    segs.append(((3.05, 0), (3.05, 4.72)))
    if not near_only:
        segs.append(((3.05, 8.68), (3.05, LENGTH)))
    return segs


def sample_points(segs, step=0.05):
    pts = []
    for (x0, y0), (x1, y1) in segs:
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], 1))
    return np.concatenate(pts)


NEAR_PTS = sample_points(segments(near_only=True))
ALL_PTS = sample_points(segments())
MIRROR = np.array([[-1, 0, WIDTH], [0, 1, 0], [0, 0, 1]], float)  # x -> 6.10 - x


def project(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3], p[:, 2]


def to_world(H, pts):
    return project(np.linalg.inv(H), np.asarray(pts, float).reshape(-1, 2))[0]


# ---------------------------------------------------------------- white paint

def floor_mask(bgr):
    """The court mat: the connected region coloured like the bottom of the frame."""
    small = cv2.resize(bgr, (bgr.shape[1] // 4, bgr.shape[0] // 4), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    rows = np.flatnonzero(hsv[..., 2].max(1) > 20)  # skip letterbox bars
    if not len(rows):
        return np.ones(bgr.shape[:2], np.uint8)
    top, h = rows[0], rows[-1] + 1
    bottom = hsv[int(top + 0.8 * (h - top)): h].reshape(-1, 3)
    bottom = bottom[bottom[:, 1] > 50]
    if len(bottom) < 100:
        return np.ones(bgr.shape[:2], np.uint8)  # grey floor: no colour to go on
    hue = np.median(bottom[:, 0])
    dh = np.abs(hsv[..., 0].astype(int) - hue)
    m = ((np.minimum(dh, 180 - dh) < 12) & (hsv[..., 1] > 40)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))  # fill painted lines
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if num < 2:
        return np.ones(bgr.shape[:2], np.uint8)
    rows = lab[int(top + 0.8 * (h - top)): h]
    ids, counts = np.unique(rows[rows > 0], return_counts=True)
    m = (lab == ids[np.argmax(counts)]).astype(np.uint8) if len(ids) else m
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)


def white_mask(bgr, boxes=()):
    """Court paint: brighter-and-greyer than the surrounding mat, on the floor only."""
    lo = bgr.min(axis=2)  # high only where all three channels are high = white
    tophat = cv2.morphologyEx(lo, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    m = ((tophat > 35) & (lo > 100)).astype(np.uint8) & floor_mask(bgr)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    for x0, y0, x1, y1 in boxes:  # shoes, socks and shirts are not court paint
        w = x1 - x0
        m[max(0, int(y0)): int(y1 + 0.05 * w), max(0, int(x0 - 0.1 * w)): int(x1 + 0.1 * w)] = 0
    return m


def distance_map(mask):
    return cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)


def _visible(q, z, shape, boxes):
    h, w = shape
    ok = (z > 0) & (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
    for x0, y0, x1, y1 in boxes:
        ok &= ~((q[:, 0] > x0) & (q[:, 0] < x1) & (q[:, 1] > y0) & (q[:, 1] < y1))
    return ok


def coverage(H, mask, band=7):
    """Share of the frame's white paint that lies under the drawn court lines."""
    small = mask[::2, ::2]
    total = int(small.sum())
    if total < 50:
        return 0.0
    lines = np.zeros_like(small)
    for p, q in segments():
        pts, z = project(H, sample_points([(p, q)], 0.1))
        pts = pts[(z > 0) & np.isfinite(pts).all(1)]
        if len(pts) > 1 and np.abs(pts).max() < 1e5:
            cv2.polylines(lines, [(pts / 2).astype(np.int32)], False, 1, max(1, band // 2))
    return float((small & lines).sum() / total)


def snap(H, dist, boxes=(), pts=NEAR_PTS, tau=12.0, max_move=60.0):
    """Nudge H so the (near-half) model lines land on white paint (truncated chamfer).

    H is varied through the image positions of four visible court points (far apart), so the
    parameters stay well conditioned even when the court corners are off-screen.
    """
    q, z = project(H, pts)
    ok = _visible(q, z, dist.shape, boxes)
    if ok.sum() < 30:
        return H
    pts, q = pts[ok], q[ok]
    c = q.mean(0)
    picks = [int(np.argmax(q @ v - c @ v)) for v in ([1, 1], [1, -1], [-1, 1], [-1, -1])]
    anchors = pts[picks].astype(np.float32)
    a0 = q[picks]
    if len({tuple(np.round(a, 3)) for a in anchors}) < 4 or cv2.contourArea(a0.astype(np.float32)) < 500:
        return H

    def H_of(p):
        return cv2.getPerspectiveTransform(anchors, (a0 + p.reshape(4, 2)).astype(np.float32))

    def resid(p):
        q, _ = project(H_of(p), pts)
        d = map_coordinates(dist, [q[:, 1], q[:, 0]], order=1, mode="nearest")
        return np.minimum(d, tau)

    r = least_squares(resid, np.zeros(8), loss="soft_l1", f_scale=2.0, diff_step=1e-3,
                      x_scale=4.0, max_nfev=60)
    Hn = H_of(r.x)
    _, zn = project(Hn, pts)
    if np.abs(r.x).max() > max_move or (zn <= 0).any() or np.sign(zn[0]) != np.sign(z[ok][0]):
        return H
    return Hn


# ---------------------------------------------------------------- line finding

def _hough(mask):
    segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, threshold=50, minLineLength=50, maxLineGap=12)
    return np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4).astype(float)


def _line(p, q):
    return np.cross([*p, 1.0], [*q, 1.0])


def _meet(l1, l2):
    p = np.cross(l1, l2)
    return None if abs(p[2]) < 1e-9 else p[:2] / p[2]


def _through(segs, vp, deg=1.5):
    """Which segments point at vp."""
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    mids = (segs[:, :2] + segs[:, 2:]) / 2
    dirs = (segs[:, 2:] - segs[:, :2]) / lens[:, None]
    to = vp - mids
    to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-9
    return np.abs(dirs[:, 0] * to[:, 1] - dirs[:, 1] * to[:, 0]) < np.sin(np.radians(deg))


def _vanishing_point(segs, rng):
    """RANSAC + least squares for the point the lines running away from the camera meet."""
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    lines = np.array([_line(s[:2], s[2:]) for s in segs])
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    best, best_w = None, 0
    p = lens / lens.sum()
    for _ in range(400):
        i, j = rng.choice(len(segs), 2, replace=False, p=p)
        vp = _meet(lines[i], lines[j])
        if vp is None:
            continue
        w = lens[_through(segs, vp)].sum()
        if w > best_w:
            best, best_w = vp, w
    if best is None:
        return None
    inl = _through(segs, best)
    if inl.sum() < 2:
        return None
    A, b = lines[inl, :2] * lens[inl, None], -lines[inl, 2] * lens[inl]
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _group_horizontals(segs, cx):
    """Merge near-horizontal segments into lines; returns [(line, extent px, y at centre)]."""
    out = []
    for s in sorted(segs, key=lambda s: -(s[1] + s[3])):
        a = np.arctan2(s[3] - s[1], s[2] - s[0])
        yc = s[1] + (cx - s[0]) * np.tan(a)
        for g in out:
            if abs(g["yc"] - yc) < 14 and abs(g["a"] - a) < np.radians(4):
                g["segs"].append(s)
                break
        else:
            out.append({"a": a, "yc": yc, "segs": [s]})
    lines = []
    for g in out:
        P = np.concatenate([[s[:2], s[2:]] for s in g["segs"]])
        vx, vy, x0, y0 = cv2.fitLine(P.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        line = _line((x0, y0), (x0 + vx, y0 + vy))
        xs = P[:, 0]
        lines.append((line, xs.max() - xs.min(), y0 + (cx - x0) * vy / vx))
    return lines


def _fit_1d(u, X):
    """u = (a x + b) / (c x + 1): a 1-D projective map from 3+ (world x, image u) pairs."""
    A = np.c_[X, np.ones_like(X), -X * u]
    return np.linalg.lstsq(A, u, rcond=None)[0]


def _apply_1d(m, x):
    a, b, c = m
    return (a * np.asarray(x) + b) / (c * np.asarray(x) + 1)


def _cross_ratio_match(us, tol):
    """Assign detected positions on the baseline to the court's x lines.

    Tries every monotone 3-to-3 assignment, keeps the projective map that explains the most
    detected lines (it preserves cross-ratios), then refits on all inliers.
    Returns {world x: detected index}.
    """
    X = np.array(X_LINES)
    best, best_key = None, None
    for idx in itertools.combinations(range(len(us)), 3):
        for wx in itertools.combinations(range(5), 3):
            m = _fit_1d(us[list(idx)], X[list(wx)])
            if np.any(np.abs(m[2] * np.linspace(-0.5, WIDTH + 0.5, 8) + 1) < 0.05):
                continue  # pole near the court: not a camera view
            pred = _apply_1d(m, X)
            if np.any(np.diff(pred) <= 0):
                continue
            d = np.abs(us[None, :] - pred[:, None])
            match = {}
            res = 0.0
            for k in range(5):
                j = int(np.argmin(d[k]))
                if d[k, j] < tol and j not in match.values():
                    match[k] = j
                    res += d[k, j]
            key = (len(match), -res)
            if best_key is None or key > best_key:
                best, best_key = match, key
    if best is None or len(best) < 3:
        return None
    return {X_LINES[k]: j for k, j in best.items()}


def detect(bgr, boxes=(), seed=0, pred=None):
    """Court from one frame's white lines. Returns (H, debug dict) or (None, debug dict).

    pred (a predicted H) is used on re-reads to label which painted line is which.
    """
    mask = white_mask(bgr, boxes)
    segs = _hough(mask)
    dbg = {"segments": segs}
    if len(segs) < 6:
        return None, dbg
    ang = np.degrees(np.abs(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])))
    ang = np.minimum(ang, 180 - ang)
    steep = segs[ang > 25]
    if len(steep) < 2:
        return None, dbg
    vp = _vanishing_point(steep, np.random.default_rng(seed))
    if vp is None:
        return None, dbg
    # Seen from off-centre, far running-away lines can look shallow: classify by the VP.
    through = _through(segs, vp)
    away, horiz = segs[through], segs[~through & (ang <= 35)]
    dbg["vp"] = vp
    h, w = mask.shape
    floor = horiz[np.minimum(horiz[:, 1], horiz[:, 3]) > vp[1] + 20]
    long_lines = [l for l in _group_horizontals(floor, w / 2) if l[1] > 0.35 * w]
    if len(long_lines) < 2:
        return None, dbg
    long_lines.sort(key=lambda l: -l[2])  # lowest (nearest the camera) first

    # Lines running away: one ray per painted line, as positions along a transversal
    # through the image centre (any transversal keeps the cross-ratios).
    centre = np.array([w / 2, h * 0.75])
    d = centre - vp
    trans = np.array([d[0], d[1], -d @ centre])
    us = []
    for s in away:
        p = _meet(_line(s[:2], s[2:]), trans)
        if p is not None:
            us.append(np.dot(p - centre, [-d[1], d[0]]) / np.linalg.norm(d))
    us = np.sort(np.array(us))
    merged = []
    for u in us:
        if merged and u - merged[-1][-1] < 10:
            merged[-1].append(u)
        else:
            merged.append([u])
    us = np.array([np.mean(m) for m in merged])
    if len(us) < 3:
        return None, dbg
    match = _cross_ratio_match(us, tol=max(6.0, 0.008 * w))
    if match is None:
        return None, dbg
    perp = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    rays = {x: _line(centre + us[j] * perp, vp) for x, j in match.items()}
    dbg["match"] = match

    def fit(la, ya, lb, yb):
        world, image = [], []
        for x, ray in rays.items():
            for y, l in ((ya, la), (yb, lb)):
                p = _meet(ray, l)
                if p is not None:
                    world.append((x, y))
                    image.append(p)
        if len(world) < 4:
            return None
        H, _ = cv2.findHomography(np.array(world, np.float32), np.array(image, np.float32), 0)
        return None if H is None else snap(H, dist, boxes)

    dist = distance_map(mask)
    lines = [l[0] for l in long_lines[:6]]
    if pred is not None:
        # Re-read: label each line with the court line the prediction puts nearest.
        hyps = _label_by_prediction(lines, pred, w)
    else:
        # Nearest two long lines = baseline and doubles long service line; if that covers
        # little paint (baseline cut off by the frame), try the other near-side pairings.
        hyps = [(lines[0], Y_BASE, lines[1], Y_LONG_SERVICE)]
        hyps += [(lines[i], ya, lines[j], yb)
                 for i, j in itertools.combinations(range(len(lines)), 2)
                 for ya, yb in itertools.combinations(Y_NEAR, 2)
                 if (i, ya, j, yb) != (0, Y_BASE, 1, Y_LONG_SERVICE)]
    best, best_cov = None, -1.0
    for k, (la, ya, lb, yb) in enumerate(hyps):
        H = fit(la, ya, lb, yb)
        if H is None or not _plausible(H, w, h):
            continue
        cov = coverage(H, mask)
        if cov > best_cov:
            best, best_cov = H, cov
            dbg["base"], dbg["service"], dbg["ys"] = la, lb, (ya, yb)
        if k == 0 and pred is None and cov >= 0.6:
            break  # the standard view worked
    dbg["coverage"] = best_cov
    if best is not None and pred is None:
        (a, b), _ = project(best, np.array([[0, 0], [WIDTH, 0]]))
        if a[0] > b[0]:
            best = best @ MIRROR  # x = 0 on the camera's left
    return best, dbg


Y_NEAR = (0.0, 0.76, 4.72)


def _label_by_prediction(lines, pred, w, tol=40):
    ys = [0.0, 0.76, 4.72, 8.68, 12.64, 13.40]
    labelled = []
    for l in lines:
        best = None
        for y in ys:
            (p, q), z = project(pred, np.array([[0, y], [WIDTH, y]]))
            if (z <= 0).any():
                continue
            xs = np.linspace(max(0, min(p[0], q[0])), min(w, max(p[0], q[0])), 5)
            dist = np.abs(l[0] * xs + l[1] * (p[1] + (xs - p[0]) * (q[1] - p[1]) / (q[0] - p[0] + 1e-9)) + l[2])
            dist /= np.hypot(l[0], l[1])
            if dist.max() < tol and (best is None or dist.max() < best[0]):
                best = (dist.max(), y)
        if best is not None:
            labelled.append((l, best[1]))
    return [(a, ya, b, yb) for (a, ya), (b, yb) in itertools.combinations(labelled, 2) if ya != yb][:6]


def _plausible(H, w, h):
    """The net must be in front of the camera and a fair part of the court in view."""
    _, z = project(H, np.array([[0, NET_Y], [WIDTH, NET_Y], [WIDTH / 2, 3.0]]))
    if (z <= 0).any():
        return False
    q, zq = project(H, ALL_PTS)
    inside = (zq > 0) & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    return inside.sum() > 0.25 * len(ALL_PTS)


# ---------------------------------------------------------------- every frame

def from_points(points, image_pts, frame, boxes=()):
    """Court from hand-given correspondences, snapped onto the paint."""
    H, _ = cv2.findHomography(np.float32([p[:2] for p in points]), np.float32(image_pts), 0)
    H = snap(H, distance_map(white_mask(frame, boxes)), boxes)
    return H / H[2, 2]


def follow(read_frames, H0, cam, boxes_per_frame, progress, reread=True):
    """Step 5. cam[t] maps image t-1 -> image t (cam[0] = I).

    Predict with the camera motion, snap onto white lines using the near half, and every
    2nd frame re-read the court from scratch; keep whichever covers more white paint.
    Returns (H per frame, coverage per frame, source per frame).
    """
    Hs, covs, srcs = [], [], []
    H = H0
    for t, frame in enumerate(read_frames()):
        boxes = boxes_per_frame[t]
        mask = white_mask(frame, boxes)
        dist = distance_map(mask)
        Hp = cam[t] @ H if t else H
        Hs_ = snap(Hp, dist, boxes)
        best, cov, src = Hs_, coverage(Hs_, mask), "tracked"
        if reread and t % 2 == 0 and t:
            Hd, _ = detect(frame, boxes, seed=t, pred=Hp)
            if Hd is not None:
                Hd = orient_like(Hd, Hp)
                cd = coverage(Hd, mask)
                if cd > cov and _agrees(Hd, Hp):
                    best, cov, src = Hd, cd, "re-read"
        H = best / best[2, 2]
        Hs.append(H)
        covs.append(cov)
        srcs.append(src)
        progress(t + 1)
    return np.array(Hs), np.array(covs), srcs


def orient_like(H, ref):
    """The court is mirror-symmetric; pick the labelling of x that matches ref."""
    corners = np.array([[0, 0], [WIDTH, 0], [0, 4.72], [WIDTH, 4.72]])
    a, _ = project(H, corners)
    b, _ = project(H @ MIRROR, corners)
    r, _ = project(ref, corners)
    return H if np.abs(a - r).sum() <= np.abs(b - r).sum() else H @ MIRROR


def _agrees(H, ref, px=120):
    """A re-read fit must be roughly where the camera motion says the court is."""
    corners = np.array([[0, 0], [WIDTH, 0], [0, 4.72], [WIDTH, 4.72]])
    a, _ = project(H, corners)
    r, _ = project(ref, corners)
    return np.abs(a - r).max() < px


def draw(img, H, color=(0, 255, 255), thickness=2):
    for p, q in segments():
        pts, z = project(H, sample_points([(p, q)], 0.1))
        pts = pts[z > 0]
        if len(pts) > 1:
            cv2.polylines(img, [pts.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return img
