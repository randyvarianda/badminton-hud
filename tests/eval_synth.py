"""Compare a `hud` run on a synthetic rally with its ground truth.

    python tests/eval_synth.py <synth dir>     (after `hud <synth dir>/synth.mp4`)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hud import court  # noqa: E402


def main():
    d = Path(sys.argv[1])
    truth = json.loads((d / "synth_truth.json").read_text())
    res = pickle.loads((d / "synth_hud_work" / "results.pkl").read_bytes())
    n = min(truth["frames"], len(res["Hs"]))

    # Pixel error over court-line points that are on screen in the true view.
    near_pts, far_pts = court.NEAR_PTS, court.ALL_PTS[court.ALL_PTS[:, 1] > court.NET_Y]
    near, far = [], []
    for t in range(n):
        for pts, out in ((near_pts, near), (far_pts, far)):
            a, z = court.project(np.array(truth["H"][t]), pts)
            b, _ = court.project(res["Hs"][t], pts)
            vis = (z > 0) & (a[:, 0] >= 0) & (a[:, 0] < 1080) & (a[:, 1] >= 0) & (a[:, 1] < 1920)
            out.append(np.linalg.norm(a[vis] - b[vis], axis=1).max() if vis.any() else 0)
    near, far = np.array(near), np.array(far)
    print(f"court  near-half px error: median {np.median(near):.1f}  p95 {np.percentile(near, 95):.1f}  max {near.max():.1f}")
    print(f"court  far-half  px error: median {np.median(far):.1f}  p95 {np.percentile(far, 95):.1f}  max {far.max():.1f}")
    print(f"court  first/last frame near error {near[0]:.1f} / {near[-1]:.1f} px (drift)")

    for name, tp in truth["players"].items():
        tp = np.array(tp)[:n]
        ep = res["players"][name][:n]
        ok = ~np.isnan(ep[:, 0])
        dist = np.linalg.norm(ep[ok] - tp[ok], axis=1)
        print(f"player {name:<6} identified {ok.mean():.0%} of frames, within 0.8 m on "
              f"{(dist < 0.8).mean() if len(dist) else 0:.0%} of those (median err {np.median(dist) if len(dist) else np.nan:.2f} m)")

    det = res["shots"]
    used = set()
    hit, right_hitter, dt = 0, 0, []
    for s in truth["shots"]:
        cand = [(abs(e["time"] - s["time"]), i) for i, e in enumerate(det) if i not in used]
        if cand and min(cand)[0] < 0.15:
            e_dt, i = min(cand)
            used.add(i)
            hit += 1
            dt.append(e_dt)
            right_hitter += det[i]["hitter"] == s["hitter"]
            sp = det[i]["speed"]
            print(f"  shot {s['time']:5.2f} {s['hitter']:<6} -> {det[i]['time']:5.2f} {det[i]['hitter']:<6} "
                  f"{det[i]['source']:<6} est {sp if sp is None else round(sp)} km/h, true launch {s['launch_kmh']:.0f}")
        else:
            print(f"  shot {s['time']:5.2f} {s['hitter']:<6} -> MISSED")
    for i, e in enumerate(det):
        if i not in used:
            print(f"  extra shot {e['time']:5.2f} {e['hitter']} ({e['source']})")
    print(f"shots  recall {hit}/{len(truth['shots'])}, precision {hit}/{len(det)}, hitter right {right_hitter}/{hit}, "
          f"timing error median {np.median(dt) * 1000 if dt else np.nan:.0f} ms")
    print(f"rally  end truth {truth['rally_end']:.2f} s, estimated {res['end']:.2f} s")


if __name__ == "__main__":
    main()
