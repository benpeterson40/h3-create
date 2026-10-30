"""tools.py : the Edit Video tools that are not H3 generations - Remove Background
(SAM 3), Depth Map (MoGe-2), SDR to HDR (inverse tone mapping -> HDR10) and Reframe (subject-following crop). Each GPU helper runs as its
own process in the Python environment that has its model; 'PROGRESS x text' lines become the job's progress. H3 tools (Edit, Extend,
Reframe 'expand with AI') live in pipeline.py."""
import json, math, os, subprocess, time
import cv2, numpy as np
from . import comfy, llm

from . import config
TOOLS_DIR = os.path.join(config.ROOT, "tools")
PY = {"sam3": config.PY_SAM3, "moge": config.PY_MOGE, "app": config.PY_APP}


def stop_orphan_llm():
    """llm.down() only stops a llama-server THIS process started; one left by a script that ended (its parent is now pid 1) would hold ~14 GB
    and starve SAM 3 / H3 (2026-10-27: a test script's server made SAM 3 run out of memory). The app's own server keeps its parent."""
    out = subprocess.run(["ps", "-eo", "pid,ppid,cmd"], capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1] == "1" and "llama-server" in parts[2]:
            try:
                os.kill(int(parts[0]), 15)
            except OSError:
                pass


def gpu_take(job):
    """the GPU helpers need the card: no LLM, ComfyUI idle and its models unloaded"""
    llm.down(); stop_orphan_llm()
    t0 = time.time()
    while time.time() - t0 < 3 * 3600:
        try:
            if comfy.busy() == 0:
                break
        except Exception:
            break
        job.set(stage="Waiting for the GPU (another render is running)")
        time.sleep(5)
    comfy.free(); time.sleep(2)


def run_proc(job, args, lo, hi, label, timeout=3 * 3600, env=None):
    """run a helper; map its PROGRESS lines into [lo, hi]; raise with its last lines on failure"""
    job.log("run: " + " ".join(os.path.basename(a) if i < 2 else a for i, a in enumerate(args))[:400])
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, **(env or {})})
    tail = []; t0 = time.time()
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith("PROGRESS "):
            parts = line.split(" ", 2)
            try:
                job.set(stage=f"{label} - {parts[2]}" if len(parts) > 2 else label, progress=lo + (hi - lo) * float(parts[1]))
            except ValueError:
                pass
        elif line:
            tail = (tail + [line])[-25:]
            if not any(s in line for s in ("Warning", "warn(", "FutureWarning", "UserWarning")):
                job.log(line[:300])
        if job.cancelled():
            p.kill(); raise comfy.Cancelled()
        if time.time() - t0 > timeout:
            p.kill(); raise RuntimeError(f"{label} timed out")
    p.wait()
    if p.returncode:
        err = next((l[6:] for l in reversed(tail) if l.startswith("ERROR ")), None)
        raise RuntimeError(err or f"{label} failed ({p.returncode}): " + " | ".join(tail[-4:])[:500])


def ffprobe(path):
    d = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration", "-of", "json", path],
                                  capture_output=True, text=True).stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video"); n, dd = v["r_frame_rate"].split("/")
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": float(n) / max(1.0, float(dd)), "duration": float(d["format"].get("duration") or 0),
            "audio": any(s["codec_type"] == "audio" for s in d["streams"])}


def trim(src, dst, seconds):
    """the first `seconds` of src (re-encoded, audio kept) - tools have length limits"""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-t", f"{seconds}", "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-crf", "12",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", dst], check=True)
    return dst


def poster(video, dst):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-frames:v", "1", "-q:v", "3", dst])
    return dst if os.path.exists(dst) else None


# ------------------------------------------------------------------ Remove Background
RMBG_SYSTEM = """You turn a request about cutting things out of a video into input for an image segmenter that finds things by short English
noun phrases. Reply with JSON only: {"mode": "keep" or "remove", "items": ["short noun phrase", ...]}. "keep" = cut these out and drop the
rest (the default when the request names things to save or says nothing about removing); "remove" = erase these and keep everything else.
Use 1 to 4 simple, visible things ("woman", "bottle", "dog", "red car"); never sentences."""


def rmbg_plan(text):
    """the user's words -> (mode, [noun phrases]) via the LLM, with a plain fallback"""
    text = (text or "").strip()
    if not text:
        return "keep", ["person"]
    import re
    if llm.available():
        try:
            r = llm.ask(f"Request: {text}", system=RMBG_SYSTEM, max_tokens=120)
            items = [str(x).strip()[:40] for x in r.get("items", []) if str(x).strip()][:4]
            if items:
                return ("remove" if r.get("mode") == "remove" else "keep"), items
        except Exception:
            pass
    mode = "remove" if re.match(r"(?i)\s*(remove|erase|delete|get rid of|take out)\b", text) else "keep"
    t = re.sub(r"(?i)^\s*(remove|erase|delete|get rid of|take out|keep|save|cut out|isolate)\s+(only\s+)?(the\s+)?", "", text)
    items = [re.sub(r"(?i)^(the|a|an)\s+", "", x.strip()).strip(" .") for x in re.split(r"(?i),|\band\b|\+", t)]
    return mode, [x[:40] for x in items if x][:4] or ["person"]


def remove_bg(job, d, jdir, src):
    P = d["params"]; mode, items = rmbg_plan(d.get("prompt") or P.get("subject") or "")
    job.log(f"remove background: {mode} {items}")
    gpu_take(job)
    od = os.path.join(jdir, "rmbg")
    run_proc(job, [PY["sam3"], "-I", f"{TOOLS_DIR}/sam3_matte.py", src, od, "|".join(items), "invert" if mode == "remove" else "matte"], 0.03, 0.97,
             f"{'Removing' if mode == 'remove' else 'Cutting out'} {', '.join(items)}")
    main = os.path.join(od, "matte.mp4" if P.get("output") == "mask" else "cutout.mp4")
    out = os.path.join(jdir, "result.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", main, "-i", src, "-map", "0:v", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest", out], check=True)
    ex = [{"label": "Transparent (WebM)", "path": os.path.join(od, "transparent.webm")}, {"label": "Green screen", "path": os.path.join(od, "green.mp4")}]
    ex.append({"label": "Cutout video", "path": os.path.join(od, "cutout.mp4")} if P.get("output") == "mask" else {"label": "Alpha mask", "path": os.path.join(od, "matte.mp4")})
    return {"result": out, "extras": ex}


# ------------------------------------------------------------------ Depth Map
def depth(job, d, jdir, src):
    P = d["params"]; cmap = P.get("colormap", P.get("style", "gray")); res = str(P.get("resolution", "auto")); maxf = str(int(P.get("maxframes", 1800)))
    sbs = str(P.get("sbs", "0")) in ("1", "true", "True")
    gpu_take(job)
    od = os.path.join(jdir, "depth")
    run_proc(job, [PY["moge"], "-I", f"{TOOLS_DIR}/moge_depth.py", src, od, cmap, res, maxf], 0.03, 0.97, "Estimating depth (MoGe-2)")
    vid = os.path.exists(os.path.join(od, "depth.mp4")); ext = "mp4" if vid else "png"
    a, b = (os.path.join(od, f"side_by_side.{ext}"), os.path.join(od, f"depth.{ext}")) if sbs else (os.path.join(od, f"depth.{ext}"), os.path.join(od, f"side_by_side.{ext}"))
    return {"result": a, "extras": [{"label": "Depth only" if sbs else "Side by side", "path": b},
                                    {"label": "16-bit PNG (first frame)" if vid else "16-bit PNG", "path": os.path.join(od, "depth16_first.png")}]}


# ------------------------------------------------------------------ SDR to HDR
def sdr_to_hdr(job, d, jdir, src):
    P = d["params"]; fmt = P.get("format", "hdr10"); peak = float(P.get("peak", 1000)); strength = P.get("strength", "standard")
    out = os.path.join(jdir, f"result_{fmt}.mp4"); name = "HLG" if fmt == "hlg" else "HDR10"
    run_proc(job, [PY["app"], f"{TOOLS_DIR}/sdr2hdr.py", src, out, str(peak), strength, fmt], 0.02, 0.98, f"Converting to {name}")
    return {"result": out, "extras": [], "note": f"{name}, H.265 10-bit, BT.2020. It looks right on an HDR screen; a normal screen may show it flat or dim."}


# ------------------------------------------------------------------ Reframe (subject-following crop)
ASPECTS = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0, "4:5": 4 / 5, "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9}
SHORT = {"480p": 480, "720p": 720, "1080p": 1080}


def _smooth(x, win):
    """zero-lag smoothing (forward-backward EMA)"""
    a = 2.0 / (win + 1); y = np.array(x, np.float64)
    for i in range(1, len(y)):
        y[i] = a * y[i] + (1 - a) * y[i - 1]
    for i in range(len(y) - 2, -1, -1):
        y[i] = a * y[i] + (1 - a) * y[i + 1]
    return y


def reframe(job, d, jdir, src):
    m = ffprobe(src); W, H = m["w"], m["h"]
    ar = ASPECTS.get(d["params"].get("aspect", "9:16"), 9 / 16); short = SHORT.get(d["params"].get("quality", "720p"), 720)
    subj = (d["params"].get("subject") or "person").strip() or "person"
    # crop window: the biggest window of the new aspect that fits inside the source
    cw, ch = (W, W / ar) if W / H < ar else (H * ar, H)
    cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
    if ar < 1:
        ow, oh = short, int(round(short / ar / 2) * 2)
    else:
        ow, oh = int(round(short * ar / 2) * 2), short
    cx_path = cy_path = None
    if cw < W - 2 or ch < H - 2:
        gpu_take(job)
        od = os.path.join(jdir, "track")
        run_proc(job, [PY["sam3"], "-I", f"{TOOLS_DIR}/sam3_matte.py", src, od, subj, "boxes"], 0.03, 0.6, f"Finding the {subj}")
        B = json.load(open(os.path.join(od, "boxes.json")))
        cx, cy, last = [], [], (W / 2, H / 2)
        for bs in B["boxes"]:
            if bs:
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bs]
                main = bs[int(np.argmax(areas))]; big = max(areas)
                group = [b for b, a in zip(bs, areas) if a >= 0.7 * big]   # people about as big as the main one
                x0 = min(b[0] for b in group); x1 = max(b[2] for b in group)
                if x1 - x0 > 0.9 * cw:   # they don't all fit side by side: follow the biggest one
                    group = [main]; x0, x1 = main[0], main[2]
                y0 = min(b[1] for b in group); y1 = max(b[3] for b in group)
                last = ((x0 + x1) / 2, y0 + 0.42 * (y1 - y0))   # a little above the middle: heads stay in frame
            cx.append(last[0]); cy.append(last[1])
        win = max(6, int(m["fps"] * 0.8))
        cx_path, cy_path = _smooth(cx, win), _smooth(cy, win)
        job.log(f"subject '{subj}' found on {B['found']}/{B['N']} frames")
    job.set(stage=f"Reframing to {d['params'].get('aspect', '9:16')}", progress=0.65)
    out = os.path.join(jdir, "result.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", f"{m['fps']}", "-i", "-",
                            "-i", src, "-map", "0:v", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out],
                           stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(src); i = 0; n = max(1, int(m["duration"] * m["fps"]))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if cx_path is not None:
            k = min(i, len(cx_path) - 1)
            x0 = int(np.clip(cx_path[k] - cw / 2, 0, W - cw)); y0 = int(np.clip(cy_path[k] - ch / 2, 0, H - ch))
        else:
            x0, y0 = (W - cw) // 2, (H - ch) // 2
        crop = f[y0:y0 + ch, x0:x0 + cw]
        enc.stdin.write(cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_AREA if cw > ow else cv2.INTER_LANCZOS4).tobytes())
        i += 1
        if i % 24 == 0:
            job.set(progress=0.65 + 0.33 * min(1.0, i / n))
    enc.stdin.close(); enc.wait()
    if enc.returncode:
        raise RuntimeError("encoding the reframed video failed")
    job.log(f"{W}x{H} -> crop {cw}x{ch} following '{subj}' -> {ow}x{oh}")
    return {"result": out, "extras": []}


RUNNERS = {"rmbg": remove_bg, "depth": depth, "hdr": sdr_to_hdr, "reframe": reframe}
LIMITS = {"rmbg": 75, "depth": 75, "hdr": 60, "reframe": 60, "edit": 30, "extend": 30}
