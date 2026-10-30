"""Low-poly filter for a video (Motion transfer, "Low-poly place" option). Shi-Tomasi corners tracked frame to frame (Lucas-Kanade; lost
ones re-seeded) + the frame border are Delaunay-triangulated every frame and each triangle is filled with the mean colour under it - the
layout, ledges, slopes, camera move and parallax stay, the old place's look is gone. With mattes, the subject is kept as it is on top."""
import cv2, numpy as np


def lowpoly(frames, mattes=None, n=420, grid=9, lines=False, md=None):
    """md = minimum distance between points (px); lines = draw every triangle's edges (thin, darker) over the fill"""
    H, W = frames[0].shape[:2]; out = []; md = md or max(8, W // 40)
    border = [(x, y) for x in np.linspace(0, W - 1, grid) for y in (0, H - 1)] + [(x, y) for y in np.linspace(0, H - 1, grid // 2 + 2) for x in (0, W - 1)]
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(g0, n, 0.002, md).reshape(-1, 2) if n else np.zeros((0, 2), np.float32)
    if md < W // 40:   # dense: an even grid too, so the flat ground gets small triangles as well (tracked like the corners)
        gx, gy = np.meshgrid(np.arange(md, W - md, 2 * md), np.arange(md, H - md, 2 * md))
        pts = np.vstack([pts, np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)])
    prev = g0
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if i:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts.astype(np.float32).reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
            nxt = nxt.reshape(-1, 2); ok = (st.reshape(-1) == 1) & (nxt[:, 0] >= 0) & (nxt[:, 0] < W) & (nxt[:, 1] >= 0) & (nxt[:, 1] < H)
            pts = nxt[ok]
            if len(pts) < 0.8 * n:   # re-seed where points were lost, away from the kept ones
                mask = np.full((H, W), 255, np.uint8)
                for x, y in pts:
                    cv2.circle(mask, (int(x), int(y)), md, 0, -1)
                add = cv2.goodFeaturesToTrack(g, max(1, n - len(pts)), 0.002, md, mask=mask)
                if add is not None:
                    pts = np.vstack([pts, add.reshape(-1, 2)])
        prev = g
        sub = cv2.Subdiv2D((0, 0, W, H))
        for x, y in list(map(tuple, pts)) + border:
            sub.insert((float(min(max(x, 0), W - 1)), float(min(max(y, 0), H - 1))))
        tri = sub.getTriangleList().reshape(-1, 3, 2)
        lp = np.zeros_like(f); lab = np.zeros((H, W), np.int32)
        for t, T in enumerate(tri, 1):
            if (T < -1).any() or (T[:, 0] > W).any() or (T[:, 1] > H).any():
                continue
            cv2.fillConvexPoly(lab, T.astype(np.int32), t)
        flat = f.reshape(-1, 3).astype(np.float64); idx = lab.reshape(-1)
        cnt = np.bincount(idx, minlength=len(tri) + 1).astype(np.float64); cnt[cnt == 0] = 1
        mean = np.stack([np.bincount(idx, weights=flat[:, c], minlength=len(tri) + 1) / cnt for c in range(3)], 1)
        lp = mean[idx].reshape(H, W, 3).astype(np.uint8)
        if lines:
            ed = np.zeros((H, W), np.uint8)
            cv2.polylines(ed, [T.astype(np.int32) for T in tri if not ((T < -1).any() or (T[:, 0] > W).any() or (T[:, 1] > H).any())], True, 255, 1, cv2.LINE_AA)
            lp = (lp.astype(np.float32) * (1 - 0.45 * (ed[..., None] / 255.0))).astype(np.uint8)
        if mattes is not None:   # the subject stays real
            a = cv2.GaussianBlur((cv2.resize(cv2.cvtColor(mattes[min(i, len(mattes) - 1)], cv2.COLOR_BGR2GRAY), (W, H)) > 100).astype(np.float32), (0, 0), 1.5)[..., None]
            lp = (f.astype(np.float32) * a + lp.astype(np.float32) * (1 - a)).astype(np.uint8)
        out.append(lp)
    return out
