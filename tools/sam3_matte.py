"""sam3_matte.py IN OUTDIR TEXT MODE : SAM 3 (transformers, facebook/sam3) video tracking of TEXT ("person", "dog", "red car" ...;
several things separated by '|' are tracked one after the other and merged).
MODE matte : writes OUTDIR/matte.mp4 (white = keep), cutout.mp4 (kept part on black), transparent.webm (VP9 + alpha), green.mp4,
cutout_first.png. MODE invert : the same with the named things REMOVED (everything else kept).
MODE boxes : writes OUTDIR/boxes.json - per frame the box around every tracked object (for Reframe's subject-following crop)
Prints 'PROGRESS <0..1> <text>' lines. Run with ~/mocap2d/.venv/bin/python -I (that venv has SAM 3)."""
import json, os, subprocess, sys, time
import cv2, numpy as np, torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

src, outdir, text, mode = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "person"), (sys.argv[4] if len(sys.argv) > 4 else "matte")
os.makedirs(outdir, exist_ok=True)
def P(x, msg):
    print(f"PROGRESS {x:.3f} {msg}", flush=True)

cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0; frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
N = len(frames); H, W = frames[0].shape[:2]
# SAM 3 works on a copy no bigger than 1024 px (masks are scaled back up)
s = min(1.0, 1024 / max(H, W)); w2, h2 = int(round(W * s / 2) * 2), int(round(H * s / 2) * 2)
small = [cv2.cvtColor(cv2.resize(f, (w2, h2), interpolation=cv2.INTER_AREA) if s < 1 else f, cv2.COLOR_BGR2RGB) for f in frames]
P(0.05, f"loading SAM 3 ({N} frames)")
model = Sam3VideoModel.from_pretrained("facebook/sam3").to("cuda", dtype=torch.bfloat16).eval()
model.max_num_objects = int(os.environ.get("MAX_OBJ", "8"))
proc = Sam3VideoProcessor.from_pretrained("facebook/sam3")
masks = [np.zeros((h2, w2), np.float32) for _ in range(N)]; boxes = [[] for _ in range(N)]; t0 = time.time()
concepts = [c.strip() for c in text.split("|") if c.strip()] or ["person"]
for ci, concept in enumerate(concepts):
    sess = proc.init_video_session(video=small, inference_device="cuda", processing_device="cpu", video_storage_device="cpu", dtype=torch.bfloat16)
    sess = proc.add_text_prompt(inference_session=sess, text=concept)
    with torch.inference_mode():
        for mo in model.propagate_in_video_iterator(inference_session=sess, max_frame_num_to_track=N):
            r = proc.postprocess_outputs(sess, mo); f = int(mo.frame_idx)
            for k in range(len(r["object_ids"])):
                m = r["masks"][k]
                m = (m.float().cpu().numpy() if m.dtype != torch.bool else m.cpu().numpy().astype(np.float32))
                if m.shape != (h2, w2):
                    m = cv2.resize(m, (w2, h2), interpolation=cv2.INTER_LINEAR)
                masks[f] = np.maximum(masks[f], (m > 0.5).astype(np.float32))
                b = r["boxes"][k].float().cpu().numpy().tolist()
                boxes[f].append([v / s for v in b] if max(b) > 1.5 else [b[0] * W, b[1] * H, b[2] * W, b[3] * H])
            if f % 10 == 0:
                P(0.1 + 0.75 * (ci + f / max(1, N - 1)) / len(concepts), f"tracking '{concept}' {f + 1}/{N}")
    del sess
found = sum(1 for m in masks if m.any())
print(f"tracked in {time.time() - t0:.0f} s; '{text}' found on {found}/{N} frames", flush=True)
del model; torch.cuda.empty_cache()

if mode == "boxes":
    json.dump({"W": W, "H": H, "N": N, "fps": fps, "boxes": boxes, "found": found}, open(os.path.join(outdir, "boxes.json"), "w"))
    P(1.0, "done"); sys.exit(0)
if found == 0:
    print(f"ERROR nothing matching '{text}' was found in the video", flush=True); sys.exit(3)

# soft, steady edges: 3-frame temporal average, then a small blur, at full size
P(0.88, "refining edges")
A = []
for i in range(N):
    m = (masks[max(0, i - 1)] + 2 * masks[i] + masks[min(N - 1, i + 1)]) / 4.0
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) if s < 1 else m
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    a = np.clip((m - 0.15) / 0.7, 0, 1)
    A.append(1.0 - a if mode == "invert" else a)

def pipe(args, frames_iter, pix):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-"] + args, stdin=subprocess.PIPE)
    for fr in frames_iter:
        p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait()
    if p.returncode:
        raise RuntimeError("ffmpeg failed: " + " ".join(args[-1:]))

P(0.92, "writing videos")
pipe(["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", os.path.join(outdir, "matte.mp4")], ((a * 255).astype(np.uint8) for a in A), "gray")
pipe(["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", os.path.join(outdir, "cutout.mp4")],
     ((f.astype(np.float32) * a[..., None]).astype(np.uint8) for f, a in zip(frames, A)), "bgr24")
green = np.zeros((H, W, 3), np.float32); green[:] = (0, 177, 64)
pipe(["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", os.path.join(outdir, "green.mp4")],
     ((f.astype(np.float32) * a[..., None] + green * (1 - a[..., None])).astype(np.uint8) for f, a in zip(frames, A)), "bgr24")
pipe(["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24", "-row-mt", "1", "-auto-alt-ref", "0", os.path.join(outdir, "transparent.webm")],
     (np.dstack([f, (a * 255).astype(np.uint8)]) for f, a in zip(frames, A)), "bgra")
cv2.imwrite(os.path.join(outdir, "cutout_first.png"), np.dstack([frames[0], (A[0] * 255).astype(np.uint8)]))
P(1.0, "done")
