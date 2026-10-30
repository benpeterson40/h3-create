#!/usr/bin/env python3
"""make_paste.py ROOT K OUT --pics a.png,b.png [--plate PLATE.mp4] [--scale 1.0] : the 'rough paste' start video for a masked H3 edit
(idea from Komin01/comfyui-genjutsu's classical backend). The old objects are erased (MiniMax-Remover plate), then each new object's
cut-out (white background dropped) is pasted into the old object's box in every frame, fitted to the box and centred. H3 then only has
to make the pasted object look real (light, grip, motion) instead of inventing it - for swaps where H3 keeps redrawing the old object
(K01: the phone). Pictures in shots_meta 'objs' order. OUT gets pasted.mp4 (and plate.mp4 / erase_mask.mp4 when it erases)."""
import argparse, os, subprocess, sys
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import vace_swap as V
import json

from server.config import PY_REMOVER as REMOVER_PY


def cutout(path):
    """the object from a picture on white: alpha from non-white pixels, biggest piece, holes filled, cropped"""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    rgb = im[..., :3]
    if im.shape[2] == 4 and (im[..., 3] < 128).mean() > 0.02:
        a = (im[..., 3] > 127).astype(np.uint8)
    else:  # away from the picture's own background colour (its border), so white objects keep their shading and edges
        border = np.concatenate([rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3), rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)])
        bg = np.median(border, 0)
        a = (np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(2) > 18).astype(np.uint8)
        a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(a)
    if n > 1:
        a = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ff = a.copy(); h, w = a.shape; fm = np.zeros((h + 2, w + 2), np.uint8); cv2.floodFill(ff, fm, (0, 0), 1)
    a = a | (1 - ff)  # holes filled
    ys, xs = np.nonzero(a)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    alpha = cv2.GaussianBlur(cv2.erode(a, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.0)
    return rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


def axis(mask):
    """principal axis of a binary mask: (centre x, centre y, angle in degrees, length along it, width across it)"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float64); c = pts.mean(0); q = pts - c
    w, v = np.linalg.eigh(np.cov(q.T)); d = v[:, 1]  # largest eigenvalue's vector
    ang = np.degrees(np.arctan2(d[1], d[0])); along = q @ d; across = q @ v[:, 0]
    return c[0], c[1], ang, along.max() - along.min() + 1, across.max() - across.min() + 1


def paste_axis(frame, obj, alpha, m, scale):
    """an elongated new object laid along an elongated old one (K08's racket handle, T22's cue, G07's poles): its long axis turned
    onto the old axis, its length matched"""
    A = axis(m); B = axis(alpha > 0.5)
    if A is None or B is None:
        return False
    cx, cy, a_old, l_old, w_old = A; _, _, a_obj, l_obj, w_obj = B
    rot = a_old - a_obj
    rot = (rot + 90) % 180 - 90  # the smaller turn (an axis has no direction)
    s = l_old / max(1.0, l_obj) * scale
    h0, w0 = obj.shape[:2]; diag = int(np.ceil(np.hypot(h0, w0))) + 4
    canvas = np.zeros((diag, diag, 3), np.uint8); acan = np.zeros((diag, diag), np.float32)
    oy, ox = (diag - h0) // 2, (diag - w0) // 2; canvas[oy:oy + h0, ox:ox + w0] = obj; acan[oy:oy + h0, ox:ox + w0] = alpha
    # the cut-out's own centre of mass to the canvas centre, then turn and scale about it
    B2 = axis(acan > 0.5); M = cv2.getRotationMatrix2D((B2[0], B2[1]), -rot, s)
    M[0, 2] += cx - B2[0]; M[1, 2] += cy - B2[1]
    H, W = frame.shape[:2]
    o = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    a = cv2.warpAffine(acan, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
    frame[:] = (frame.astype(np.float32) * (1 - a) + o.astype(np.float32) * a).astype(np.uint8)
    return True


def paste(frame, obj, alpha, box, scale, longest=False):
    x0, y0, x1, y1 = box; bw, bh = max(2, x1 - x0), max(2, y1 - y0)
    # longest=True: the old object's longest side x its size ratio (scale) - fitting inside the box made a wide watermelon slice in a phone's
    # narrow box tiny (K01)
    s = (max(bw, bh) / max(obj.shape[1], obj.shape[0]) if longest else min(bw / obj.shape[1], bh / obj.shape[0])) * scale
    w, h = max(2, int(obj.shape[1] * s)), max(2, int(obj.shape[0] * s))
    o = cv2.resize(obj, (w, h), interpolation=cv2.INTER_AREA); a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)[..., None]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2; px, py = cx - w // 2, cy - h // 2
    H, W = frame.shape[:2]; sx0, sy0 = max(0, -px), max(0, -py); dx0, dy0 = max(0, px), max(0, py)
    dx1, dy1 = min(W, px + w), min(H, py + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    o = o[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]; a = a[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    reg = frame[dy0:dy1, dx0:dx1].astype(np.float32)
    frame[dy0:dy1, dx0:dx1] = (reg * (1 - a) + o.astype(np.float32) * a).astype(np.uint8)


def track_masks(root, meta, k, n, H, W):
    """each picked instance on its own: {(object index, track name): bool n x H x W} (G07: two trekking poles, each its own asparagus
    along its own axis - their union is a V, not a long thing)"""
    sh = meta["shots"][str(k)]; vi = sh.get("vi", k); out = {}
    for oi_s, names in sh.get("pick", {}).items():
        oi = int(oi_s)
        if oi not in sh.get("members", []) or not names:
            continue
        for nm in names:
            p = f"{root}/obj/{vi}/{nm}.mp4"
            if not os.path.exists(p):
                continue
            M = np.zeros((n, H, W), bool)
            for i, f in enumerate(V.read_frames(p)[:n]):
                if f.shape[:2] != (H, W):
                    f = cv2.resize(f, (W, H), interpolation=cv2.INTER_NEAREST)
                M[i] = f[..., 1] > 127
            if M.any():
                out[(oi, nm)] = M
    return out


def erase(srcp, src, E, out_dir, max_side=832, chunk=121):
    """MiniMax-Remover plate of the source shot with E (bool n x H x W) erased. Big frames go through a copy scaled down to max_side
    (1344x768 x 107 frames does not fit the 16 GB card) and long shots in chunks; only the erased area is taken from the remover's
    output (scaled back up, soft edge) - everywhere else the plate is the source itself"""
    n = len(src); H, W = src[0].shape[:2]
    s = min(1.0, max_side / max(W, H)); w2, h2 = max(16, int(W * s) // 16 * 16), max(16, int(H * s) // 16 * 16)
    plate = [f.copy() for f in src]
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        if not E[c0:c1].any():
            continue
        sv, mv, pv = f"{out_dir}/erase_src_{c0}.mp4", f"{out_dir}/erase_mask_{c0}.mp4", f"{out_dir}/erase_plate_{c0}.mp4"
        V.write_video(sv, [cv2.resize(f, (w2, h2), interpolation=cv2.INTER_AREA) for f in src[c0:c1]], crf=8)
        V.write_video(mv, [cv2.resize(m.astype(np.uint8) * 255, (w2, h2), interpolation=cv2.INTER_NEAREST) for m in E[c0:c1]], crf=0, gray=True)
        V.comfy.free()  # ComfyUI may still hold H3 (14 GB): the remover runs outside it
        subprocess.run([REMOVER_PY, f"{os.path.dirname(os.path.abspath(__file__))}/remove_run.py", sv, mv, pv], check=True)
        got = V.read_frames(pv)
        for j in range(c1 - c0):
            g = cv2.resize(got[min(j, len(got) - 1)], (W, H), interpolation=cv2.INTER_CUBIC)
            a = cv2.GaussianBlur(V.grow(E[c0 + j:c0 + j + 1], 3, 0)[0].astype(np.float32), (0, 0), 2.0)[..., None]
            plate[c0 + j] = (src[c0 + j].astype(np.float32) * (1 - a) + g.astype(np.float32) * a).astype(np.uint8)
    return plate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("k", type=int); ap.add_argument("out")
    ap.add_argument("--pics", required=True); ap.add_argument("--plate", default=""); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-axis", action="store_true", help="always fit to the box (no turning onto the old object's axis)")
    ap.add_argument("--pick", default="", help="override picks, e.g. '0:obj_0+obj_3;1:obj_2'")
    ap.add_argument("--base", default="", help="the video to erase from and paste into (default the source shot) - the first take, so that only "
                    "the picked objects change and the take's other swaps stay (T22: only the red ball, not the good churro again)")
    ap.add_argument("--scales", default="", help="per object size vs the old one, e.g. '0:1.4;1:0.8' (autofix asks Qwen; K01: a watermelon "
                    "slice fitted to the phone's box came out phone-sized)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(f"{a.root}/shots_meta.json"))
    if a.pick:
        sh_ = meta["shots"][str(a.k)]
        for part in a.pick.split(";"):
            oi_s, names = part.split(":")
            sh_["pick"][oi_s] = [x for x in names.split("+") if x]
            if sh_["pick"][oi_s] and int(oi_s) not in sh_.setdefault("members", []):
                sh_["members"].append(int(oi_s))
    srcp = f"{a.root}/seg{a.k}_src.mp4"; src = V.read_frames(srcp); n = len(src); H, W = src[0].shape[:2]
    if a.base:
        bf = V.read_frames(a.base)[:n]
        src = [cv2.resize(f, (W, H)) if f.shape[:2] != (H, W) else f for f in bf] + [bf[-1]] * max(0, n - len(bf))
        srcp = f"{a.out}/base.mp4"; V.write_video(srcp, src, crf=6)
    tracks = track_masks(a.root, meta, a.k, n, H, W); members = sorted({oi for oi, _ in tracks})
    masks = {oi: np.any([m for (o, _), m in tracks.items() if o == oi], 0) for oi in members}
    if not members:
        raise SystemExit("no picked instances in this shot")
    plate_p = a.plate or f"{a.out}/plate.mp4"
    if not os.path.exists(plate_p):
        U = np.zeros((n, H, W), bool)
        for oi in members:
            U |= masks[oi]
        E = V.grow(U, W // 48, 1)
        V.write_video(plate_p, erase(srcp, src, E, a.out), crf=8)
    plate = V.read_frames(plate_p)[:n]
    pics = a.pics.split(","); cuts = {oi: cutout(pics[oi]) for oi in members}
    scl = {int(x.split(":")[0]): float(x.split(":")[1]) for x in a.scales.split(";") if ":" in x}
    long_obj = {}
    for oi in members:  # elongated new object (length / width > 2.2)?
        B = axis(cuts[oi][1] > 0.5); long_obj[oi] = bool(B and B[3] / max(1.0, B[4]) > 3.5)   # thin things only (K07: pizza boxes went askew at 2.2)
    out = []; foot = []   # foot: where the pasted new objects are (their alpha), for the redraw region (a churro is wider than the cue)
    for i in range(n):
        f = plate[i].copy(); before = f.copy()
        for (oi, nm), T in sorted(tracks.items()):
            m = T[i]
            if m.any():
                A = axis(m) if long_obj[oi] and not a.no_axis else None
                sc = a.scale * scl.get(oi, 1.0)
                if A and A[3] / max(1.0, A[4]) > 2.0 and paste_axis(f, cuts[oi][0], cuts[oi][1], m, sc):   # K07: a roller with its handle is
                    continue                                                                               # 2-3:1 - the toothbrush stood upright
                ys, xs = np.nonzero(m)
                paste(f, cuts[oi][0], cuts[oi][1], (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), sc, longest=oi in scl)
        out.append(f); foot.append(((np.abs(f.astype(np.int16) - before.astype(np.int16)).max(2) > 12) * 255).astype(np.uint8))
    V.write_video(f"{a.out}/pasted.mp4", out)
    V.write_video(f"{a.out}/pasted_mask.mp4", foot, crf=0, gray=True)
    print("pasted", n, "frames ->", f"{a.out}/pasted.mp4")


if __name__ == "__main__":
    main()
