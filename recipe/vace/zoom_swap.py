#!/usr/bin/env python3
"""zoom_swap.py ROOT K OUT [options] : the box redraw (h3edit_swap.py --video) on a ZOOMED crop around the shot's objects, pasted back (test tool).

Idea from Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder's crop-to-mask (MIT, ported, nothing installed): small objects get only a few
of H3's 32 px tokens at full frame (a phone is 1-2 tokens wide), so one fixed box around everything the shot swaps (its whole path, objects
grown, x --ctx for context, at most 2.5:1) is cut out, scaled up to the source's short side (at most --max-zoom x), redrawn there with the
shot's own prompt and sheets, scaled back and pasted inside the redrawn region only (soft edge). Shots whose box is over --max-share of the
frame need no zoom (exit code 3). OUT gets croot/ (the crop as a run folder), edit/ (the redraw), result.mp4 and panel.mp4."""
import argparse, json, os, shutil, subprocess, sys, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import vace_swap as V

from server.config import PY_APP as PY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("k", type=int); ap.add_argument("out")
    ap.add_argument("--ctx", type=float, default=1.75); ap.add_argument("--max-zoom", type=float, default=4.0)
    ap.add_argument("--max-share", type=float, default=0.6); ap.add_argument("--grow-rel", type=float, default=0.25)
    ap.add_argument("--start", default="", help="a start video at the source's size (e.g. make_paste's pasted.mp4), cropped the same way")
    ap.add_argument("--strength", type=float, default=1.0); ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--pick", default="")
    ap.add_argument("--obj", type=int, default=-1, help="zoom on this object only (its own path); the others are left out of the redraw")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    meta = json.load(open(f"{a.root}/shots_meta.json")); sh = meta["shots"][str(a.k)]; vi = sh.get("vi", a.k)
    if a.pick:
        for part in a.pick.split(";"):
            oi_s, names = part.split(":"); sh["pick"][oi_s] = [x for x in names.split("+") if x]
            if sh["pick"][oi_s] and int(oi_s) not in sh["members"]:
                sh["members"].append(int(oi_s))
    if a.obj >= 0:   # one small object (K01's phone: 15 % of the frame with its whole path, the shot's union 238 %)
        for oi_s in list(sh["pick"]):
            if int(oi_s) != a.obj:
                sh["pick"][oi_s] = []
    src = V.read_frames(f"{a.root}/seg{a.k}_src.mp4"); n = len(src); H, W = src[0].shape[:2]
    masks = V.masks_for_shot(a.root, meta, a.k, n, H, W)
    if not masks:
        raise SystemExit("no picked instances in this shot")
    R = np.zeros((n, H, W), bool)
    for oi in masks:
        R |= V.grow_rel(masks[oi], a.grow_rel, True, 1)
    ys, xs = np.nonzero(R.any(0))
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; bw, bh = (x1 - x0) * a.ctx, (y1 - y0) * a.ctx
    if bw > 2.5 * bh:
        bh = bw / 2.5
    if bh > 2.5 * bw:
        bw = bh / 2.5
    bw, bh = min(bw, W), min(bh, H)
    X0 = int(round(min(max(0, cx - bw / 2), W - bw))); Y0 = int(round(min(max(0, cy - bh / 2), H - bh))); X1, Y1 = int(X0 + bw), int(Y0 + bh)
    share = (X1 - X0) * (Y1 - Y0) / (W * H)
    log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
    if share > a.max_share:
        log(f"the box covers {share * 100:.0f}% of the frame - no zoom"); sys.exit(3)
    z = min(a.max_zoom, min(H, W) / min(X1 - X0, Y1 - Y0))
    cw, ch = max(256, int((X1 - X0) * z) // 32 * 32), max(256, int((Y1 - Y0) * z) // 32 * 32)
    log(f"shot {a.k}: box {X0},{Y0}-{X1},{Y1} ({share * 100:.0f}% of the frame), zoom {z:.2f} -> crop {cw}x{ch}")
    cr = f"{a.out}/croot"; os.makedirs(f"{cr}/obj/{vi}", exist_ok=True)
    crop = lambda f, interp=cv2.INTER_CUBIC: cv2.resize(f[Y0:Y1, X0:X1], (cw, ch), interpolation=interp)
    V.write_video(f"{cr}/seg{a.k}_src.mp4", [crop(f) for f in src], crf=6)
    for oi_s, names in sh["pick"].items():
        for nm in names or []:
            p = f"{a.root}/obj/{vi}/{nm}.mp4"
            if os.path.exists(p):
                M = V.read_frames(p)[:n]
                V.write_video(f"{cr}/obj/{vi}/{nm}.mp4", [crop(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST), cv2.INTER_NEAREST)
                                                          for m in M], crf=0)
    json.dump(meta, open(f"{cr}/shots_meta.json", "w"), indent=1)
    for f in os.listdir(a.root):
        if f == f"seg{a.k}_prompt.txt" or (f.startswith("sheet2_") and f.endswith(".png")):
            shutil.copy(f"{a.root}/{f}", f"{cr}/{f}")
    cmd = [PY, "-u", f"{HERE}/h3edit_swap.py", cr, str(a.k), f"{a.out}/edit", "--video", "--seed", str(a.seed), "--grow-rel", str(a.grow_rel)]
    if a.start:
        st = V.read_frames(a.start)[:n]; V.write_video(f"{cr}/start.mp4", [crop(f) for f in st], crf=6)
        cmd += ["--start", f"{cr}/start.mp4", "--strength", str(a.strength)]
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL)
    if r.returncode or not os.path.exists(f"{a.out}/edit/result.mp4"):
        raise SystemExit("the zoomed redraw failed")
    gen = V.read_frames(f"{a.out}/edit/result.mp4")[:n]; reg = V.read_frames(f"{a.out}/edit/region.mp4")[:n]
    base = src
    if a.start:   # pasted onto the start video (the first take: K01's cup -> cactus stays), not onto the source
        base = [cv2.resize(f, (W, H)) if f.shape[:2] != (H, W) else f for f in V.read_frames(a.start)[:n]]
        base += [base[-1]] * (n - len(base))
    res = []
    for i in range(n):
        g = cv2.resize(gen[min(i, len(gen) - 1)], (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_AREA)
        m = cv2.resize((reg[min(i, len(reg) - 1)][..., 1] > 127).astype(np.uint8), (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_NEAREST)
        m = cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(np.float32)
        al = cv2.GaussianBlur(m, (0, 0), 3)[..., None]
        f = base[i].copy(); roi = f[Y0:Y1, X0:X1].astype(np.float32)
        f[Y0:Y1, X0:X1] = (roi * (1 - al) + g.astype(np.float32) * al).astype(np.uint8)
        res.append(f)
    V.write_video(f"{a.out}/result.mp4", res)
    V.write_video(f"{a.out}/panel.mp4", [np.hstack([src[i], res[i]]) for i in range(n)], crf=16)
    json.dump(dict(root=a.root, k=a.k, box=[X0, Y0, X1, Y1], share=round(share, 3), zoom=round(z, 2), crop=[cw, ch],
                   seconds=round(time.time() - t0)), open(f"{a.out}/zoom.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f} s -> {a.out}/result.mp4")


if __name__ == "__main__":
    main()
