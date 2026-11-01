"""hud <clip>  ->  <name>_HUD.mp4 + <name>_HUD_report/index.html"""
import argparse
import dataclasses
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import camera, config, court, identity, people, prep, render, report, shots, shuttle


def log(msg):
    print(msg, flush=True)


def bar(label, total):
    t0 = time.time()

    def update(k):
        if k == total or k % 10 == 0:
            rate = k / max(1e-6, time.time() - t0)
            sys.stdout.write(f"\r   {label}: {k}/{total} frames ({rate:.1f}/s)")
            if k == total:
                sys.stdout.write("\n")
            sys.stdout.flush()
    return update


def cached(path: Path, fn):
    if path.exists():
        return pickle.loads(path.read_bytes())
    value = fn()
    path.write_bytes(pickle.dumps(value))
    return value


def first_frame_check(frame, H, dbg):
    img = frame.copy()
    for s in dbg.get("segments", []):
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (255, 120, 0), 1)
    if H is not None:
        court.draw(img, H, (0, 255, 255), 2)
        for x in court.X_LINES:
            (p,), _ = court.project(H, np.array([[x, 0.0]]))
            cv2.putText(img, f"{x:g}", (int(p[0]) - 20, int(p[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2, cv2.LINE_AA)
    for key, col in (("base", (0, 0, 255)), ("service", (0, 160, 255))):
        if key in dbg:
            a, b, c = dbg[key]
            cv2.line(img, (0, int(-c / b)), (1080, int(-(a * 1080 + c) / b)), col, 1)
    return img


def main():
    ap = argparse.ArgumentParser(prog="hud", description="Add a game-style HUD to a one-rally badminton clip.")
    ap.add_argument("clip", type=Path)
    ap.add_argument("--fresh", action="store_true", help="recompute every step (ignore the cache)")
    args = ap.parse_args()
    clip = args.clip.resolve()
    if not clip.exists():
        raise SystemExit(f"no such file: {clip}")
    folder = clip.parent / clip.stem  # everything this clip produces lives here
    folder.mkdir(exist_ok=True)
    cfg = config.load(clip, folder)
    work = folder / (clip.stem + "_hud_work")
    stamp = {"size": clip.stat().st_size, "mtime": clip.stat().st_mtime}
    if args.fresh or (work / "stamp.json").exists() and json.loads((work / "stamp.json").read_text()) != stamp:
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(exist_ok=True)
    (work / "stamp.json").write_text(json.dumps(stamp))
    out_video = folder / (clip.stem + "_HUD.mp4")
    out_report = folder / (clip.stem + "_HUD_report")
    log(f"config: {cfg.path}")

    log("1  prepare: 1080x1920, audio, racket sounds")
    video, wav, fps = prep.prepare(clip, work)
    n = cached(work / "frames.pkl", lambda: prep.frame_count(video))
    onsets = prep.racket_onsets(wav)
    log(f"   {n} frames at {fps:.2f} fps, {len(onsets)} racket-sound onsets")

    def frames():
        cap = cv2.VideoCapture(str(video))
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f

    log("3  people: YOLO11m + ByteTrack, imgsz 1280 on " + people.device())
    dets = cached(work / "people.pkl", lambda: people.track(video, bar("people", n)))
    boxes = [[d["box"] for d in fd if d["conf"] > 0.3] for fd in dets]

    frame0 = next(frames())
    fixed = cfg.court_points is not None
    if fixed:
        log("4  court from the [court] points in the config (fixed camera)")
        image_pts = prep.to_prepared(clip, [p[2:] for p in cfg.court_points])
        H0, dbg = court.from_points(cfg.court_points, image_pts, frame0, boxes[0]), {"match": {}}
    else:
        log("4  court in the first frame")
        H0, dbg = court.detect(frame0, boxes[0])
    out_report.mkdir(exist_ok=True)
    check = first_frame_check(frame0, H0, dbg)
    cv2.imwrite(str(out_report / "court_first_frame.jpg"), check)
    if H0 is None:
        raise SystemExit(f"could not find the court lines in the first frame; see "
                         f"{out_report / 'court_first_frame.jpg'}. If the camera is not behind the "
                         f"baseline, give [court] points in {cfg.path}")
    H0 = H0 / H0[2, 2]
    if not fixed:
        log(f"   matched x lines {sorted(dbg['match'])}")
    log(f"   paint coverage "
        f"{court.coverage(H0, court.white_mask(frame0, boxes[0])):.0%}")

    log("2  camera motion from ceiling and walls")
    cam, inl = cached(work / "camera.pkl", lambda: camera.motion(frames, H0, boxes, bar("camera", n)))
    log(f"   median {np.median(inl[1:]):.0f} feature inliers per frame, {int((inl[1:] < 12).sum())} frames without a fit")
    cam_abs = _chain(cam)

    log("5  court in every frame")
    Hs, cov, srcs = cached(work / "court.pkl",
                           lambda: court.follow(frames, H0, cam, boxes, bar("court", n), reread=not fixed))
    log(f"   paint coverage median {np.median(cov):.0%}, min {cov.min():.0%}; "
        f"{srcs.count('re-read')} frames took the re-read fit")

    log("6  who is who")
    assigned, centres = identity.assign(dets, Hs, cfg.players)
    tracks = identity.tracks(assigned, n, fps, cfg.players)
    for p in cfg.players:
        seen = (~np.isnan(tracks[p.name]["pos"][:, 0])).mean()
        log(f"   {p.name:<8} {p.side:<4} {p.color:<8} seen on {seen:.0%} of frames")

    log("7  shuttle")
    cands = cached(work / "shuttle.pkl", lambda: shuttle.candidates(frames, n, cam_abs, Hs, boxes, bar("shuttle", n)))
    tl = shuttle.tracklets(cands)
    player_boxes = {k: v["box"] for k, v in tracks.items()}
    chain, links = shuttle.path(tl, player_boxes, fps, 1920)
    flights = shuttle.anchored(tl, chain, player_boxes)
    pos = shuttle.positions(chain + [f[0] for f in flights], n)
    log(f"   {sum(len(c) for c in cands)} candidates, {len(tl)} tracklets, path of {len(chain)} tracklets "
        f"+ {len(flights)} flights from our rackets, "
        f"seen on {(~np.isnan(pos[:, 0])).mean():.0%} of frames")

    log("8  shots")
    shot_list, start, end = shots.find(chain, links, flights, onsets, pos, tracks, cfg.players, Hs, fps, n,
                                         cfg.fixes)
    for i, s in enumerate(shot_list, 1):
        sp = f"{s.speed:5.0f} km/h" if s.speed else "   -- km/h"
        log(f"   #{i:<2} {s.time:6.2f} s  {s.hitter:<8} {s.side:<4} {s.source:<6} {sp}")
    log(f"   rally {start:.2f}-{end:.2f} s")
    (work / "results.pkl").write_bytes(pickle.dumps({
        "fps": fps, "Hs": Hs, "coverage": cov, "shuttle": pos, "start": start, "end": end,
        "players": {k: v["pos"] for k, v in tracks.items()},
        "shots": [dataclasses.asdict(s) for s in shot_list]}))

    log("9  HUD video")
    hud = render.Hud(cfg.players, tracks, shot_list, start, end, Hs, pos, fps, n)
    render.render(hud, frames, n, fps, clip, out_video, bar("render", n))

    log("10 error report")
    warn = report.write(out_report, frames, n, fps, Hs, cov, srcs, tracks, cfg.players, pos, shot_list,
                        start, end, onsets, check, cfg.path, clip.name)
    for _, _, m in warn:
        log(f"   ! {m}")
    log(f"\nvideo:  {out_video}\nreport: {out_report / 'index.html'}\nfixes:  {cfg.path}")


def _chain(cam):
    """Absolute camera homographies: image 0 -> image t."""
    out = [np.eye(3)]
    for C in cam[1:]:
        M = C @ out[-1]
        out.append(M / M[2, 2])
    return np.array(out)


if __name__ == "__main__":
    main()
