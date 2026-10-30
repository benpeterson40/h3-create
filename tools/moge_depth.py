"""moge_depth.py IN OUTDIR COLORMAP [RES] [MAXFRAMES] : depth maps with MoGe-2 (Ruicheng/moge-2-vitl-normal) for a video or one image.
COLORMAP gray (near = white) | turbo | inferno | magma | viridis; RES auto (source size) | 720 | 1080 (short side); MAXFRAMES caps a video. Depth is turned into disparity (1/depth), normalised over the WHOLE clip with robust percentiles
(not per frame, so it doesn't flicker) and smoothed over time. Writes OUTDIR/depth.mp4 (video) or depth.png (image), plus depth16_first.png
(16-bit) and side_by_side.mp4/png. Prints 'PROGRESS <0..1> <text>'. Run with ~/sam3dbody/.venv/bin/python -I."""
import os, subprocess, sys, time
import cv2, numpy as np, torch
from moge.model.v2 import MoGeModel

src, outdir, style = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "gray")
RES = sys.argv[4] if len(sys.argv) > 4 else "auto"; MAXF = int(sys.argv[5]) if len(sys.argv) > 5 else 1800
CMAPS = {"turbo": cv2.COLORMAP_TURBO, "inferno": cv2.COLORMAP_INFERNO, "magma": cv2.COLORMAP_MAGMA, "viridis": cv2.COLORMAP_VIRIDIS, "color": cv2.COLORMAP_TURBO}
os.makedirs(outdir, exist_ok=True)
def P(x, msg):
    print(f"PROGRESS {x:.3f} {msg}", flush=True)

is_img = os.path.splitext(src)[1].lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
if is_img:
    frames = [cv2.imread(src, cv2.IMREAD_COLOR)]; fps = 24.0
else:
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0; frames = []
    while len(frames) < MAXF:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
if RES in ("720", "1080"):  # output size: short side 720 / 1080
    h0, w0 = frames[0].shape[:2]; k = int(RES) / min(h0, w0)
    if abs(k - 1) > 0.01:
        sz = (int(round(w0 * k / 2) * 2), int(round(h0 * k / 2) * 2)); frames = [cv2.resize(f, sz, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4) for f in frames]
N = len(frames); H, W = frames[0].shape[:2]
P(0.03, "loading MoGe-2")
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval()
s = min(1.0, 960 / max(H, W)); w2, h2 = int(W * s), int(H * s)
disp = []; t0 = time.time()
with torch.inference_mode():
    for i, f in enumerate(frames):
        im = cv2.resize(f, (w2, h2), interpolation=cv2.INTER_AREA) if s < 1 else f
        o = model.infer(torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1))
        d = o["depth"].float().cpu().numpy(); m = o.get("mask")
        m = m.cpu().numpy().astype(bool) if m is not None else np.isfinite(d)
        d = np.where(m & np.isfinite(d) & (d > 0), d, np.nan)
        disp.append(1.0 / d)
        if i % 8 == 0:
            P(0.05 + 0.8 * (i + 1) / N, f"depth {i + 1}/{N}")
print(f"MoGe-2: {N} frame(s) in {time.time() - t0:.0f} s", flush=True)
del model; torch.cuda.empty_cache()
allv = np.concatenate([x[np.isfinite(x)].ravel()[::7] for x in disp])
lo, hi = np.percentile(allv, 2), np.percentile(allv, 98.5)
out = []; prev = None
for x in disp:
    x = np.nan_to_num(x, nan=lo)  # sky / invalid = far
    x = np.clip((x - lo) / max(1e-6, hi - lo), 0, 1) ** 0.85
    if prev is not None:
        x = 0.6 * x + 0.4 * prev  # light temporal smoothing
    prev = x; out.append(cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) if s < 1 else x)
def render(x):
    g = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(g, CMAPS[style]) if style in CMAPS else cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
cv2.imwrite(os.path.join(outdir, "depth16_first.png"), (np.clip(out[0], 0, 1) * 65535).astype(np.uint16))
if is_img:
    cv2.imwrite(os.path.join(outdir, "depth.png"), render(out[0]))
    cv2.imwrite(os.path.join(outdir, "side_by_side.png"), np.concatenate([frames[0], render(out[0])], 1))
    P(1.0, "done"); sys.exit(0)
P(0.9, "writing the depth video")
def pipe(path, it, w):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{H}", "-r", f"{fps}", "-i", "-",
                          "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for fr in it:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close(); p.wait()
pipe(os.path.join(outdir, "depth.mp4"), (render(x) for x in out), W)
pipe(os.path.join(outdir, "side_by_side.mp4"), (np.concatenate([f, render(x)], 1) for f, x in zip(frames, out)), 2 * W)
P(1.0, "done")
