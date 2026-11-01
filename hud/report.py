"""Step 10: error-check report - pictures to eyeball, warnings, and how to fix them."""
import html
from pathlib import Path

import cv2
import numpy as np

from . import court

SPEED_RANGE = (25, 250)  # km/h
GAP_RANGE = (0.33, 2.4)  # s between shots
THUMB = (540, 960)


def warnings(shots, start, end, cov, fps, tracks):
    w = []
    for i, s in enumerate(shots, 1):
        if s.speed is not None and not SPEED_RANGE[0] <= s.speed <= SPEED_RANGE[1]:
            w.append(("speed", s.time, f"Shot {i} ({s.time:.2f} s, {s.hitter}): {s.speed:.0f} km/h is "
                      f"outside {SPEED_RANGE[0]}-{SPEED_RANGE[1]} km/h. A shot is probably missing "
                      "or extra around it, or the next hitter is wrong."))
        if s.hitter_guessed:
            w.append(("hitter", s.time, f"Shot {i} ({s.time:.2f} s): shuttle not seen near the hit, "
                      f"hitter {s.hitter} is a guess. Check the picture."))
    for i, (a, b) in enumerate(zip(shots, shots[1:]), 1):
        g = b.time - a.time
        if g < GAP_RANGE[0]:
            w.append(("gap", a.time, f"Shots {i}-{i + 1} ({a.time:.2f}->{b.time:.2f} s) are only "
                      f"{g:.2f} s apart. One of them is probably not a shot."))
        elif g > GAP_RANGE[1]:
            w.append(("gap", a.time, f"Shots {i}-{i + 1} ({a.time:.2f}->{b.time:.2f} s) are {g:.2f} s "
                      "apart. A shot is probably missing in between (see the long-stretch pictures)."))
    if not shots:
        w.append(("shots", 0, "No shots found."))
    low = np.flatnonzero(cov < 0.35)
    if len(low):
        w.append(("court", low[0] / fps, f"Court lines cover little white paint on {len(low)} frames "
                  f"(first at {low[0] / fps:.2f} s). Check the court pictures."))
    for name, tr in tracks.items():
        miss = np.isnan(tr["pos"][:, 0])
        if miss.mean() > 0.2:
            w.append(("player", 0, f"{name} is not identified on {miss.mean():.0%} of frames."))
    return w


def _annotate(frame, t, H, tracks, colors, pos, fps, label, hit=None):
    img = frame.copy()
    court.draw(img, H, (0, 220, 255), 1)
    for name, tr in tracks.items():
        bx = tr["box"][t]
        if np.isnan(bx[0]):
            continue
        c = colors[name][::-1]
        x0, y0, x1, y1 = bx.astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
        cv2.putText(img, name, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, name, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, 2, cv2.LINE_AA)
    for k in range(max(0, t - int(0.5 * fps)), min(len(pos), t + int(0.5 * fps))):
        if not np.isnan(pos[k, 0]):
            col = (0, 255, 255) if k <= t else (255, 0, 255)
            cv2.circle(img, tuple(pos[k].astype(int)), 4, col, -1)
    if not np.isnan(pos[t, 0]):
        cv2.circle(img, tuple(pos[t].astype(int)), 22, (0, 255, 0), 2)
    if hit is not None:
        cv2.drawMarker(img, (int(hit[0]), int(hit[1])), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.rectangle(img, (0, 0), (1080, 70), (0, 0, 0), -1)
    cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA)


def _timeline(n, fps, onsets, shots, pos, colors, width=1600):
    h = 160
    img = np.full((h, width, 3), 24, np.uint8)
    sx = width / max(1, n / fps)
    seen = ~np.isnan(pos[:, 0])
    for t in np.flatnonzero(seen):
        x = int(t / fps * sx)
        cv2.line(img, (x, 120), (x, 140), (90, 200, 90), 1)
    for tau, s in onsets:
        x = int(tau * sx)
        cv2.line(img, (x, 110), (x, 110 - int(60 * s)), (160, 160, 160), 2)
    for i, s in enumerate(shots, 1):
        x = int(s.time * sx)
        cv2.line(img, (x, 10), (x, 110), colors[s.hitter][::-1], 3)
        cv2.putText(img, str(i), (x + 3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for sec in range(int(n / fps) + 1):
        x = int(sec * sx)
        cv2.line(img, (x, 145), (x, 160), (200, 200, 200), 1)
        cv2.putText(img, f"{sec}s", (x + 2, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return img


def write(out_dir: Path, read_frames, n, fps, Hs, cov, srcs, tracks, players, pos, shots, start, end,
          onsets, first_frame_img, cfg_path, clip_name):
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    colors = {p.name: p.rgb for p in players}
    warn = warnings(shots, start, end, cov, fps, tracks)

    court_frames = list(np.linspace(0, n - 1, 8).astype(int))
    shot_frames = []
    for s in shots:
        f = min(n - 1, int(round(s.time * fps)))
        shot_frames.append((f, min(n - 1, f + int(round(0.2 * fps)))))
    times = [start] + [s.time for s in shots] + [end]
    gaps = [(a, b) for a, b in zip(times, times[1:]) if b - a > GAP_RANGE[1]]
    gap_frames = {g: list(np.arange(g[0] + 0.2, g[1], 0.4)) for g in gaps}
    wanted = set(court_frames) | {f for pair in shot_frames for f in pair}
    for fr in gap_frames.values():
        wanted |= {min(n - 1, int(round(x * fps))) for x in fr}
    frames = {t: f for t, f in enumerate(read_frames()) if t in wanted}

    cv2.imwrite(str(out_dir / "court_first_frame.jpg"), first_frame_img)
    court_imgs = []
    for k, t in enumerate(court_frames):
        img = frames[t].copy()
        court.draw(img, Hs[t], (0, 255, 255), 2)
        cv2.rectangle(img, (0, 0), (1080, 70), (0, 0, 0), -1)
        cv2.putText(img, f"{t / fps:.2f} s  paint {cov[t]:.0%}  {srcs[t]}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
        name = f"court_{k}.jpg"
        cv2.imwrite(str(out_dir / name), cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA))
        court_imgs.append(name)

    shot_rows = []
    for i, (s, (fa, fb)) in enumerate(zip(shots, shot_frames), 1):
        sp = f"{s.speed:.0f} km/h" if s.speed else "-- km/h"
        a = _annotate(frames[fa], fa, Hs[fa], tracks, colors, pos, fps,
                      f"#{i} {s.time:.2f}s {s.hitter} ({s.side}) {s.source}", s.point)
        b = _annotate(frames[fb], fb, Hs[fb], tracks, colors, pos, fps, f"#{i} +0.2s  {sp}")
        name = f"shot_{i:02d}.jpg"
        cv2.imwrite(str(out_dir / name), np.hstack([a, b]))
        shot_rows.append((i, s, sp, name))

    gap_rows = []
    for k, (g, ts) in enumerate(gap_frames.items()):
        thumbs = []
        for x in ts:
            t = min(n - 1, int(round(x * fps)))
            thumbs.append(cv2.resize(_annotate(frames[t], t, Hs[t], tracks, colors, pos, fps, f"{x:.2f}s"),
                                     (270, 480)))
        while len(thumbs) % 4:
            thumbs.append(np.zeros_like(thumbs[0]))
        sheet = np.vstack([np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)])
        name = f"gap_{k}.jpg"
        cv2.imwrite(str(out_dir / name), sheet)
        gap_rows.append((g, name))

    cv2.imwrite(str(out_dir / "timeline.jpg"), _timeline(n, fps, onsets, shots, pos, colors))
    page = _html(clip_name, cfg_path, shots, start, end, warn, court_imgs, shot_rows, gap_rows)
    (out_dir / "index.html").write_text(page)
    return warn


def _html(clip_name, cfg_path, shots, start, end, warn, court_imgs, shot_rows, gap_rows):
    e = html.escape
    rows = "".join(
        f"<tr><td>{i}</td><td>{s.time:.2f}</td><td>{e(s.hitter)}</td><td>{s.side}</td><td>{s.source}</td>"
        f"<td>{sp}</td><td>{'' if s.distance is None else f'{s.distance:.1f} m'}</td>"
        f"<td>{'' if s.flight is None else f'{s.flight:.2f} s'}</td></tr>" for i, s, sp, _ in shot_rows)
    warn_html = "".join(f"<li>{e(m)}</li>" for _, _, m in warn) or "<li>None.</li>"
    court_html = "".join(f'<img src="{n}" width="270">' for n in court_imgs)
    shot_html = "".join(f'<figure><img src="{n}" width="1080"><figcaption>Shot {i}: hit (left), '
                        f'0.2 s later (right)</figcaption></figure>' for i, _, _, n in shot_rows)
    gap_html = "".join(f'<figure><img src="{n}" width="1080"><figcaption>{a:.2f}-{b:.2f} s without a '
                       f'shot</figcaption></figure>' for (a, b), n in gap_rows) or "<p>None.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HUD check: {e(clip_name)}</title>
<style>
body{{font:15px/1.5 Lato,system-ui,sans-serif;margin:24px;background:#111;color:#f6f6f6}}
h2{{margin-top:40px}} table{{border-collapse:collapse}} td,th{{padding:4px 12px;border-bottom:1px solid #303030;text-align:left}}
img{{max-width:100%;height:auto}} code,pre{{background:#262626;padding:2px 6px;border-radius:4px}}
pre{{padding:12px}} .warn li{{color:#f4c251}} figure{{margin:16px 0}}
</style></head><body>
<h1>HUD check: {e(clip_name)}</h1>
<p>Rally {start:.2f}-{end:.2f} s, {len(shots)} shots.</p>
<h2>Warnings</h2><ul class="warn">{warn_html}</ul>
<h2>How to fix</h2>
<p>Edit <code>{e(str(cfg_path))}</code> and run <code>hud {e(clip_name)}</code> again (cached steps are skipped):</p>
<pre>[fixes]
add_shots = [3.42, {{ time = 5.10, hitter = "Alif" }}]   # seconds; hitter optional
remove_shots = [7.85]                                    # nearest shot within 0.2 s
rally_end = 12.8                                         # seconds</pre>
<h2>Timeline</h2><p>Coloured = shots (hitter colour), grey = racket sounds (2-8 kHz), green = shuttle seen.</p>
<img src="timeline.jpg" width="1600">
<h2>Shots</h2><table><tr><th>#</th><th>time s</th><th>hitter</th><th>side</th><th>source</th><th>speed</th>
<th>D</th><th>T</th></tr>{rows}</table>
<h2>Court, first frame</h2><img src="court_first_frame.jpg" width="540">
<h2>Court, 8 frames</h2><div>{court_html}</div>
<h2>Every shot</h2><p>Yellow dots = shuttle before, magenta = after, red cross = hit point.</p>{shot_html}
<h2>Long stretches without a shot</h2>{gap_html}
</body></html>"""
