"""moge_ground.py OUTDIR VIDEO [VIDEO ...] : the ground of each video from MoGe-2 normals (one model load for all videos) - fast stand-in
for a SAM 3 'ground|floor|pavement' pass (~0.1 s a frame instead of ~0.4). A pixel is ground when its surface faces up (camera-space normal
y < -0.6, ~53 deg from vertical) and it is connected to the bottom of the frame; small holes are closed. Writes OUTDIR/<i>/matte.mp4 (white =
ground, the video's size and frame count). Prints 'PROGRESS <0..1> <text>'. Run with ~/sam3dbody/.venv/bin/python -I."""
import os, sys, time
import cv2, numpy as np, torch
from moge.model.v2 import MoGeModel

outdir, vids = sys.argv[1], sys.argv[2:]
P = lambda x, msg: print(f"PROGRESS {x:.3f} {msg}", flush=True)
P(0.02, "loading MoGe-2")
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval(); t0 = time.time(); total = 0
for vi, path in enumerate(vids):
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    H, W = frames[0].shape[:2]; s = min(1.0, 640 / max(H, W)); w2, h2 = int(W * s) // 2 * 2, int(H * s) // 2 * 2
    od = os.path.join(outdir, str(vi)); os.makedirs(od, exist_ok=True)
    vw = cv2.VideoWriter(os.path.join(od, "matte.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H)); prev = None
    with torch.inference_mode():
        for i, f in enumerate(frames):
            im = cv2.resize(f, (w2, h2), interpolation=cv2.INTER_AREA)
            o = model.infer(torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1))
            if "normal" in o and o["normal"] is not None:
                ny = o["normal"][..., 1].float().cpu().numpy()
            else:   # no normal head: normals from the point map
                pts = o["points"].float().cpu().numpy(); dx = np.gradient(pts, axis=1); dy = np.gradient(pts, axis=0)
                nrm = np.cross(dx, dy); nrm /= np.linalg.norm(nrm, axis=2, keepdims=True) + 1e-9; ny = nrm[..., 1]
            up = (ny < -0.6).astype(np.uint8)
            up = cv2.morphologyEx(up, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            n, lab = cv2.connectedComponents(up)
            keep = set(np.unique(lab[-max(2, h2 // 20):, :])) - {0}   # only regions that reach the bottom band of the frame
            g = np.isin(lab, list(keep)).astype(np.float32) if keep else np.zeros_like(up, np.float32)
            if prev is not None:
                g = np.maximum(g * 0.7 + prev * 0.3, 0)   # light temporal smoothing
            prev = g
            vw.write(cv2.cvtColor((cv2.resize(g, (W, H)) > 0.5).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
            total += 1
    vw.release(); P(0.05 + 0.95 * (vi + 1) / len(vids), f"video {vi + 1}/{len(vids)}")
print(f"MoGe-2 ground: {total} frames of {len(vids)} video(s) in {time.time() - t0:.0f} s", flush=True)
