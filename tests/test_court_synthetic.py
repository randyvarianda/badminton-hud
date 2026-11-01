"""Court fit on a drawn court with a known homography: python tests/test_court_synthetic.py"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hud import court  # noqa: E402


def scene(H, rng):
    img = np.full((1920, 1080, 3), (95, 140, 40), np.uint8)  # green mat
    img[:600] = (70, 60, 120)  # wall
    img = cv2.add(img, rng.integers(0, 18, img.shape, dtype=np.uint8))
    for p, q in court.segments():
        pts, z = court.project(H, court.sample_points([(p, q)], 0.05))
        pts = pts[z > 0]
        thick = max(1, int(round(6 * (pts[:, 1].mean() - 600) / 1300)))
        cv2.polylines(img, [pts.astype(np.int32)], False, (235, 235, 235), thick, cv2.LINE_AA)
    return img


def corner_error(H, Ht):
    pts = np.array([[0, 0], [6.1, 0], [0, 13.4], [6.1, 13.4], [3.05, 4.72]])
    return np.abs(court.project(H, pts)[0] - court.project(Ht, pts)[0]).max()


def main():
    rng = np.random.default_rng(1)
    world = np.float32([[0, 0], [6.1, 0], [0, 13.4], [6.1, 13.4]])
    worst = 0
    for i in range(6):
        jitter = rng.normal(0, 25, (4, 2))
        image = np.float32([[60, 1720], [1060, 1740], [400, 690], [720, 700]]) + jitter
        Ht = cv2.getPerspectiveTransform(world, image.astype(np.float32))
        img = scene(Ht, rng)
        H, dbg = court.detect(img)
        assert H is not None, f"case {i}: no court found ({len(dbg['segments'])} segments)"
        H = court.orient_like(H, Ht)
        err = corner_error(H, Ht)
        worst = max(worst, err)
        print(f"case {i}: max corner error {err:.1f} px, matched x = {sorted(dbg['match'])}")
        if i == 0:
            cv2.imwrite("/tmp/hud_synth.jpg", court.draw(img.copy(), H, (0, 0, 255), 1))
    assert worst < 6, worst
    print("ok")


if __name__ == "__main__":
    main()
