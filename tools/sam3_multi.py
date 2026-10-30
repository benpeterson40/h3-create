"""sam3_multi.py OUTDIR STRIDE TEXT VIDEO [VIDEO ...] : SAM 3 subject mattes for several videos with ONE model load, tracking every
STRIDE-th frame (the frames between get the union of their two tracked neighbours, grown a little for the motion). TEXT as in
sam3_matte.py ('person|skateboard': each concept is one tracking pass, so list each thing once). Writes OUTDIR/<i>/matte.mp4 (white = the
things, the video's size and frame count). Prints 'PROGRESS <0..1> <text>'. Run with ~/mocap2d/.venv/bin/python -I."""
import json, os, sys, time
import cv2, numpy as np, torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

outdir, stride, text, vids = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
P = lambda x, msg: print(f"PROGRESS {x:.3f} {msg}", flush=True)
P(0.02, "loading SAM 3")
model = Sam3VideoModel.from_pretrained("facebook/sam3").to("cuda", dtype=torch.bfloat16).eval()
model.max_num_objects = int(os.environ.get("MAX_OBJ", "8")); proc = Sam3VideoProcessor.from_pretrained("facebook/sam3")
# no mask NMS: without the 'kernels' package transformers skips the NMS anyway - but only AFTER building the pairwise mask-IoU matrix of every
# detection, which ran out of memory on a market full of oranges (2026-10-28, M03: 5.9 GB for one frame). The same detections, no matrix
model.det_nms_thresh = 0.0
concepts = [c.strip() for c in text.split("|") if c.strip()] or ["person"]; t0 = time.time(); tot = 0
for vi, path in enumerate(vids):
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    N = len(frames); H, W = frames[0].shape[:2]; s = min(1.0, 1024 / max(H, W)); w2, h2 = int(round(W * s / 2) * 2), int(round(H * s / 2) * 2)
    idx = list(range(0, N, stride)) + ([N - 1] if (N - 1) % stride else [])
    small = [cv2.cvtColor(cv2.resize(frames[i], (w2, h2), interpolation=cv2.INTER_AREA) if s < 1 else frames[i], cv2.COLOR_BGR2RGB) for i in idx]
    got = [np.zeros((h2, w2), np.float32) for _ in idx]
    per = {}   # PER_OBJ=1: every tracked object on its own (concept, id) -> per tracked frame mask
    cap0 = model.max_num_objects

    def track(concept):
        """one concept's pass -> ({frame: union mask}, {(concept, id): {frame: mask}})"""
        g_, p_ = {}, {}
        sess = proc.init_video_session(video=small, inference_device="cuda", processing_device="cpu", video_storage_device="cpu", dtype=torch.bfloat16)
        sess = proc.add_text_prompt(inference_session=sess, text=concept)
        try:
            with torch.inference_mode():
                for mo in model.propagate_in_video_iterator(inference_session=sess, max_frame_num_to_track=len(small)):
                    r = proc.postprocess_outputs(sess, mo); f = int(mo.frame_idx)
                    for k in range(len(r["object_ids"])):
                        m = r["masks"][k]; m = (m.float().cpu().numpy() if m.dtype != torch.bool else m.cpu().numpy().astype(np.float32))
                        if m.shape != (h2, w2):
                            m = cv2.resize(m, (w2, h2), interpolation=cv2.INTER_LINEAR)
                        g_[f] = np.maximum(g_.get(f, 0), (m > 0.5).astype(np.float32))
                        if os.environ.get("PER_OBJ") == "1":
                            p_.setdefault((concept, int(r["object_ids"][k])), {})[f] = (m > 0.5).astype(np.uint8)
        finally:
            del sess
        return g_, p_
    for concept in concepts:
        model.max_num_objects = cap0
        while True:   # out of memory (2026-10-28, M03: a market full of oranges, 32 tracked at once = 14.9 GB): again with half as many objects
            try:
                g_, p_ = track(concept); break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if model.max_num_objects <= 4:
                    raise
                model.max_num_objects //= 2
                print(f"PROGRESS 0.05 out of GPU memory on '{concept}' - again with at most {model.max_num_objects} objects", flush=True)
        for f, m in g_.items():
            got[f] = np.maximum(got[f], m)
        for key, fm in p_.items():
            dst = per.setdefault(key, [np.zeros((h2, w2), np.uint8) for _ in idx])
            for f, m in fm.items():
                dst[f] = m
        torch.cuda.empty_cache()
    tot += len(small) * len(concepts)
    grey = [cv2.cvtColor(cv2.resize(f, (w2, h2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames]
    gx, gy = np.meshgrid(np.arange(w2, dtype=np.float32), np.arange(h2, dtype=np.float32)); full = []

    def pull(m, i, j):   # the mask of frame j moved to frame i along the optical flow (frame i -> frame j)
        fl = cv2.calcOpticalFlowFarneback(grey[i], grey[j], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        return cv2.remap(m, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    for i in range(N):   # tracked frames as they are; the others = both neighbours moved along the flow (no ghost of a fast leg)
        j = np.searchsorted(idx, i)
        if j < len(idx) and idx[j] == i:
            full.append(got[j])
        else:
            a = pull(got[max(0, j - 1)], i, idx[max(0, j - 1)]); b = pull(got[min(len(idx) - 1, j)], i, idx[min(len(idx) - 1, j)])
            full.append(cv2.dilate(np.maximum(a, b), np.ones((3, 3), np.uint8)))
    od = os.path.join(outdir, str(vi)); os.makedirs(od, exist_ok=True)
    vw = cv2.VideoWriter(os.path.join(od, "matte.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
    for i in range(N):   # soft steady edges like sam3_matte.py: a 3-frame average, a small blur
        m = np.mean([full[max(0, i - 1)], full[i], full[min(N - 1, i + 1)]], axis=0)
        m = cv2.GaussianBlur(cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 1.2)
        vw.write(cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    vw.release(); P(0.05 + 0.95 * (vi + 1) / len(vids), f"video {vi + 1}/{len(vids)}")
    if per:   # per object: a matte video at the tracked frames' size (stride 1 expected) + where each one is
        info = []
        for (concept, oid), ms in per.items():
            name = f"obj_{len(info)}"; vo = cv2.VideoWriter(os.path.join(od, f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H)); seen = []
            for j, m in enumerate(ms):
                vo.write(cv2.cvtColor(cv2.resize(m * 255, (W, H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR))
                if m.any():
                    ys, xs = np.nonzero(m); seen.append([idx[j], int(xs.min() / s), int(ys.min() / s), int(xs.max() / s), int(ys.max() / s), int(m.sum())])
            vo.release(); info.append({"name": name, "concept": concept, "id": oid, "frames": seen})
        json.dump(info, open(os.path.join(od, "objects.json"), "w"))
print(f"SAM 3: {len(vids)} video(s), {tot} frame-passes in {time.time() - t0:.0f} s", flush=True)
