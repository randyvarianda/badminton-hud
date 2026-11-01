"""Step 2: frame-to-frame camera homography from features on the ceiling and walls."""
import cv2
import numpy as np

from . import court


def background_mask(shape, H_court, boxes):
    """Everything above the far baseline (walls, ceiling), minus people."""
    h, w = shape
    (a, b), _ = court.project(H_court, np.array([[-3, court.LENGTH], [court.WIDTH + 3, court.LENGTH]]))
    xs = np.arange(w)
    y_line = a[1] + (xs - a[0]) * (b[1] - a[1]) / (b[0] - a[0]) - 15
    mask = (np.arange(h)[:, None] < y_line[None, :]).astype(np.uint8) * 255
    for x0, y0, x1, y1 in boxes:
        pad = 0.15 * (x1 - x0)
        mask[max(0, int(y0 - pad)): int(y1 + pad), max(0, int(x0 - pad)): int(x1 + pad)] = 0
    return mask


def motion(read_frames, H0, boxes_per_frame, progress):
    """cam[t] maps image t-1 -> image t; cam[0] = I. Also returns inlier counts."""
    cams, inliers = [np.eye(3)], [0]
    prev, H_court = None, H0
    for t, frame in enumerate(read_frames()):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            mask = background_mask(gray.shape, H_court, boxes_per_frame[t - 1])
            C, n = _pair(prev, gray, mask)
            cams.append(C)
            inliers.append(n)
            H_court = C @ H_court
        prev = gray
        progress(t + 1)
    return np.array(cams), np.array(inliers)


def _pair(a, b, mask):
    p0 = cv2.goodFeaturesToTrack(a, 600, 0.01, 10, mask=mask, blockSize=7)
    if p0 is None or len(p0) < 12:
        return np.eye(3), 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21), maxLevel=4)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, winSize=(21, 21), maxLevel=4)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - p0, axis=2).ravel() < 1.0)
    if ok.sum() < 12:
        return np.eye(3), 0
    H, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
    if H is None:
        return np.eye(3), 0
    return H / H[2, 2], int(inl.sum())
