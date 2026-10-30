"""main.py : the H3 Create web app (FastAPI). Run: ./run.sh  ->  http://127.0.0.1:7880
Uploads, the job queue (one GPU job at a time, in a worker thread), history, the motion library (H3C_LIBRARY), styles."""
import glob, importlib, json, os, queue, shutil, subprocess, threading, time, traceback, uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import comfy, durable, llm, pipeline
from . import tools as tools_mod
from .store import DATA, Store
from .config import ROOT, LIBRARY
UP = f"{DATA}/uploads"; LIB = f"{DATA}/library"
for d in (UP, LIB):
    os.makedirs(d, exist_ok=True)
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_IMAGES = 8

app = FastAPI(title="H3 Create")
store = Store()
work = queue.Queue()
current = {"id": None}


def url(path):
    if not path:
        return None
    path = os.path.abspath(path)
    return "/data/" + os.path.relpath(path, DATA) if path.startswith(DATA) else None


# ------------------------------------------------------------------ uploads
def _thumb_video(src, dst):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.3", "-i", src, "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "3", dst])
    if not os.path.exists(dst):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "3", dst])


def _record(uid):
    p = f"{UP}/{uid}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def _public(rec):
    return {k: v for k, v in rec.items() if k != "path"} | {"url": url(rec["path"]), "thumb": url(rec.get("thumb_path"))}


def _ingest(path, name):
    """file already saved at path -> upload record"""
    uid = os.path.splitext(os.path.basename(path))[0]; ext = os.path.splitext(path)[1].lower()
    rec = {"id": uid, "name": name, "path": path, "created": time.time()}
    if ext in VIDEO_EXT:
        meta = pipeline.probe(path); rec.update(kind="video", meta=meta); th = f"{UP}/{uid}_thumb.jpg"; _thumb_video(path, th); rec["thumb_path"] = th
    else:
        import cv2
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise HTTPException(400, "not a readable image")
        h, w = im.shape[:2]; s = min(1.0, 512 / max(h, w)); th = f"{UP}/{uid}_thumb.jpg"
        cv2.imwrite(th, cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
        rec.update(kind="image", meta={"w": w, "h": h}, thumb_path=th)
        if ext not in (".png", ".jpg", ".jpeg"):  # keep a PNG copy for ComfyUI's LoadImage
            png = f"{UP}/{uid}.png"; cv2.imwrite(png, im); rec["path"] = png
    durable.write_json(f"{UP}/{uid}.json", rec)
    return rec


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_EXT | IMAGE_EXT:
        raise HTTPException(400, f"unsupported file type {ext or '?'}")
    uid = uuid.uuid4().hex[:12]; path = f"{UP}/{uid}{ext}"
    with open(path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    try:
        rec = _ingest(path, file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"could not read the file: {e}")
    return _public(rec)


@app.get("/api/uploads")
def uploads(kind: str = "video", limit: int = 30):
    recs = [r for r in (json.load(open(p)) for p in glob.glob(f"{UP}/*.json")) if r.get("kind") == kind and not r.get("library")]
    return [_public(r) for r in sorted(recs, key=lambda r: -r["created"])[:limit]]


# ------------------------------------------------------------------ motion library (the clips in H3C_LIBRARY, default examples/library)
def _library():
    """every video in the library folder (sub-folders too); a library.json there may give titles: {"file.mp4": "Title"}"""
    titles = {}
    try:
        titles = json.load(open(os.path.join(LIBRARY, "library.json")))
    except Exception:
        pass
    items = []
    for p in sorted(glob.glob(os.path.join(LIBRARY, "**", "*"), recursive=True)):
        if os.path.splitext(p)[1].lower() in VIDEO_EXT and os.path.isfile(p):
            rel = os.path.relpath(p, LIBRARY); lid = "".join(c if c.isalnum() else "_" for c in os.path.splitext(rel)[0]).lower()
            items.append({"id": lid, "title": titles.get(rel) or os.path.splitext(os.path.basename(p))[0].replace("_", " "), "path": p})
    return items


@app.get("/api/library")
def library():
    out = []
    for it in _library():
        th = f"{LIB}/{it['id']}.jpg"
        if not os.path.exists(th):
            _thumb_video(it["path"], th)
        meta_p = f"{LIB}/{it['id']}.json"
        if not os.path.exists(meta_p):
            json.dump(pipeline.probe(it["path"]), open(meta_p, "w"))
        out.append({"id": it["id"], "title": it["title"], "thumb": url(th), "video": f"/api/library/{it['id']}/video", "meta": json.load(open(meta_p))})
    return out


@app.get("/api/library/{lid}/video")
def library_video(lid: str):
    it = next((x for x in _library() if x["id"] == lid), None)
    if not it:
        raise HTTPException(404)
    return FileResponse(it["path"], media_type="video/mp4")


@app.post("/api/library/{lid}/use")
def library_use(lid: str):
    it = next((x for x in _library() if x["id"] == lid), None)
    if not it:
        raise HTTPException(404)
    uid = uuid.uuid4().hex[:12]; path = f"{UP}/{uid}.mp4"; shutil.copy(it["path"], path)
    rec = _ingest(path, it["title"]); rec["library"] = lid; durable.write_json(f"{UP}/{uid}.json", rec)
    return _public(rec)


@app.get("/api/styles")
def styles():
    return [{"id": k, "name": v[0], "text": v[1]} for k, v in pipeline.STYLES.items()]


# ------------------------------------------------------------------ jobs
class NewJob(BaseModel):
    mode: str = "motion"            # motion | swap
    video: str                      # upload id
    images: list[str] = []          # upload ids, in @image order
    prompt_on: bool = True
    prompt: str = ""
    style: str = "none"
    duration: str = "5s"            # 5s | full
    quality: str = "standard"       # standard (768p) | fast (480p)
    mt_style: str = "standard"      # Motion transfer: standard | lowpoly (the old place turned low-poly + grey before H3 paints the new one)
    audit: bool = True
    keep_audio: bool = True
    project: dict | None = None     # test-set label for the Ratings view ({set, code, title, order, note}); the web UI never sends it


def _next_version(code):
    """1 + the project's delivered or live versions (a cancelled re-run never became a version: held re-runs made labels jump)"""
    return 1 + sum(1 for x in store.all() if (x.get("project") or {}).get("code") == code and x.get("status") != "cancelled")


def job_view(j, full=False):
    v = {k: j.get(k) for k in ("id", "created", "mode", "mt_style", "status", "stage", "progress", "duration", "quality", "style", "audit", "prompt_on", "prompt",
                               "error", "seconds", "frames", "canvas", "video_name")}
    v["result"] = url(j.get("result")); v["poster"] = url(j.get("poster")); v["source_thumb"] = url(j.get("video_thumb"))
    v["tool"] = j.get("tool", "create"); v["tool_name"] = TOOL_NAMES.get(v["tool"], v["tool"]); v["params"] = j.get("params", {})
    v["extras"] = [{"label": x["label"], "url": url(x["path"])} for x in j.get("extras", []) if os.path.exists(x.get("path", ""))]
    v["scene_used"] = j.get("scene_used"); v["from_job"] = j.get("from_job"); v["note"] = j.get("note"); v["result_kind"] = "image" if (j.get("result") or "").lower().endswith((".png", ".jpg")) else "video"
    v["image_thumbs"] = [url(p) for p in j.get("image_thumbs", [])]
    v["source"] = url(j.get("video")); v["rating"] = j.get("rating"); v["project"] = j.get("project")
    v["seg_scores"] = [next(((a.get("audit") or {}).get("score") for a in s.get("attempts", []) if a.get("attempt") == s.get("chosen")), None)
                       for s in j.get("segments", [])] if j.get("status") == "done" else []
    v["queue_pos"] = None
    if j.get("status") == "queued":
        ids = [x for x in list(work.queue)]
        v["queue_pos"] = (ids.index(j["id"]) + 1 + (1 if current["id"] else 0)) if j["id"] in ids else None
    audits = [a.get("audit") for s in j.get("segments", []) for a in s.get("attempts", []) if a.get("attempt") == s.get("chosen") and a.get("audit")]
    v["audit_score"] = round(sum(a.get("score", 0) for a in audits) / len(audits), 1) if audits and all("score" in a for a in audits) else None
    v["audit_pass"] = all(a.get("pass") for a in audits) if audits else None
    if full:
        v["log"] = j.get("log", []); v["spec_source"] = (j.get("spec") or {}).get("_source"); v["first_frame"] = url(j.get("first_frame"))
        v["segments"] = [{"k": s["k"], "start": s["start"], "length": s["length"], "status": s.get("status"), "chosen": s.get("chosen"),
                          "attempts": [{"attempt": a["attempt"], "seed": a["seed"], "prompt": a["prompt"], "video": url(a.get("video")), "audit": a.get("audit"), "repair": a.get("repair")}
                                       for a in s.get("attempts", [])]} for s in j.get("segments", [])]
    return v


@app.post("/api/jobs")
def create_job(b: NewJob):
    vrec = _record(b.video)
    if not vrec or vrec.get("kind") != "video":
        raise HTTPException(400, "upload a video first")
    imgs = [_record(i) for i in b.images[:MAX_IMAGES]]
    if any(r is None or r.get("kind") != "image" for r in imgs):
        raise HTTPException(400, "a reference picture is missing")
    if b.mode not in ("motion", "swap"):
        raise HTTPException(400, "unknown mode")
    if b.mode == "swap" and not (b.prompt_on and b.prompt.strip()) and not imgs:
        raise HTTPException(400, "describe what to change, or add a picture")
    if b.mode == "motion" and not imgs:
        raise HTTPException(400, "add at least one picture of the new character, object or place")
    if b.style not in pipeline.STYLES:
        raise HTTPException(400, "unknown style")
    jid = store.create({"mode": b.mode, "video": vrec["path"], "video_name": vrec.get("name"), "video_thumb": vrec.get("thumb_path"),
                        "images": [r["path"] for r in imgs], "image_thumbs": [r.get("thumb_path") for r in imgs], "prompt_on": b.prompt_on,
                        "prompt": b.prompt, "style": b.style, "duration": b.duration if b.duration in ("5s", "full") else "5s",
                        "quality": b.quality if b.quality in ("standard", "fast") else "standard", "audit": b.audit, "keep_audio": b.keep_audio,
                        "mt_style": b.mt_style if b.mode == "motion" and b.mt_style in ("standard", "lowpoly") else "standard",
                        **({"project": {**b.project, "version": _next_version(b.project.get("code"))}} if b.project and b.project.get("code") else {})})
    work.put(jid)
    return job_view(store.get(jid))


TOOL_NAMES = {"create": "Create Video", "improve": "Improve", "control": "Motion Control", "edit": "Edit Video", "extend": "Extend Video", "reframe": "Reframe", "rmbg": "Remove Background",
              "hdr": "SDR to HDR", "depth": "Depth Map"}


class NewToolJob(BaseModel):
    tool: str
    video: str                      # upload id (an image is allowed for Depth Map)
    images: list[str] = []
    prompt: str = ""
    params: dict = {}
    audit: bool = True
    keep_audio: bool = True
    mask: str | None = None         # Edit Video, Draw mode: upload id of the painted mask (white = change)


@app.post("/api/tools")
def create_tool_job(b: NewToolJob):
    if b.tool not in TOOL_NAMES or b.tool == "create":
        raise HTTPException(400, "unknown tool")
    rec = _record(b.video)
    if not rec or (rec.get("kind") != "video" and not (b.tool == "depth" and rec.get("kind") == "image")):
        raise HTTPException(400, "add a video first")
    imgs = [_record(i) for i in b.images[:MAX_IMAGES]]
    if any(r is None or r.get("kind") != "image" for r in imgs):
        raise HTTPException(400, "a reference picture is missing")
    dur = rec.get("meta", {}).get("duration", 0)
    if b.tool == "extend" and rec.get("kind") == "video" and dur < 1.0:
        raise HTTPException(400, "the video must be at least 1 second long")
    common = {"tool": b.tool, "video": rec["path"], "video_name": rec.get("name"), "video_thumb": rec.get("thumb_path"), "images": [r["path"] for r in imgs],
              "image_thumbs": [r.get("thumb_path") for r in imgs], "prompt": b.prompt, "prompt_on": bool(b.prompt.strip()), "params": b.params,
              "audit": b.audit, "keep_audio": b.keep_audio, "style": "none"}
    if b.tool == "control":
        if not imgs:
            raise HTTPException(400, "add a picture of your character")
        if rec.get("kind") != "video":
            raise HTTPException(400, "add a motion video")
        P = b.params; seed = P.get("seed")
        try:
            seed = int(seed) if str(seed or "").strip() else None
        except ValueError:
            raise HTTPException(400, "the seed must be a whole number")
        common.update(mode="motion", duration=P.get("length", "full") if P.get("length") in ("full", "5s") else "full",
                      quality=P.get("quality", "standard") if P.get("quality") in ("standard", "fast") else "standard",
                      scene=P.get("scene", "auto") if P.get("scene") in ("auto", "video", "image") else "auto", anchor=bool(P.get("anchor", True)),
                      images=common["images"][:1], image_thumbs=common["image_thumbs"][:1], char_image=common["images"][0], seed=seed)
    if b.tool == "edit":
        if not b.prompt.strip():
            raise HTTPException(400, "describe what to change")
        mask = None
        if b.mask:
            m = _record(b.mask)
            if not m or m.get("kind") != "image":
                raise HTTPException(400, "the painted mask is missing")
            mask = m["path"]
        common.update(mode="swap", duration="full", quality=b.params.get("quality", "standard"), mask=mask)
    jid = store.create(common); work.put(jid)
    return job_view(store.get(jid))


@app.get("/api/jobs")
def jobs(limit: int = 60):
    return [job_view(j) for j in store.all()[:limit]]


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = store.get(jid)
    if not j:
        raise HTTPException(404)
    return job_view(j, full=True)


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    j = store.get(jid)
    if not j:
        raise HTTPException(404)
    if j["status"] in ("queued", "running"):
        store.update(jid, cancel=True)
        if j["status"] == "queued":
            store.update(jid, status="cancelled", stage="Cancelled")
    return job_view(store.get(jid))


@app.post("/api/jobs/{jid}/retry")
def retry(jid: str):
    j = store.get(jid)
    if not j:
        raise HTTPException(404)
    keep = {k: j[k] for k in ("tool", "mode", "video", "video_name", "video_thumb", "images", "image_thumbs", "prompt_on", "prompt", "style", "duration", "quality",
                              "audit", "keep_audio", "params", "mask", "scene", "anchor", "char_image") if k in j}
    if j.get("tool") == "control" and j.get("char_image"):  # start again from the character picture alone (no plate, no old first frame)
        keep["images"] = [j["char_image"]]; keep["image_thumbs"] = j.get("image_thumbs", [])[:1]
    if j.get("project", {}).get("code"):   # a test video made again: same label, next version, shown next to the earlier ones in Ratings
        keep["project"] = {**{k: v for k, v in j["project"].items() if k not in ("note", "rerun_of")}, "version": _next_version(j["project"]["code"]), "rerun_of": jid}
    nid = store.create(keep); work.put(nid)
    return job_view(store.get(nid))


@app.post("/api/jobs/{jid}/improve")
def improve(jid: str):
    j = store.get(jid)
    if not j:
        raise HTTPException(404)
    if j.get("status") != "done" or j.get("tool", "create") not in ("create", "edit", "control", "improve") or j.get("mask") or not j.get("segments"):
        raise HTTPException(400, "only finished Create Video, Edit Video (prompt) and Motion Control videos can be improved")
    keep = {k: j[k] for k in ("mode", "video", "video_name", "video_thumb", "images", "image_thumbs", "prompt_on", "prompt", "style", "duration", "quality",
                              "keep_audio", "params", "scene_used", "char_image") if k in j}
    nid = store.create({**keep, "tool": "improve", "from_job": jid, "audit": True}); work.put(nid)
    return job_view(store.get(nid))


class Rating(BaseModel):
    score: int | None = None        # 1-10, or None to clear
    comment: str = ""


@app.post("/api/jobs/{jid}/rating")
def rate(jid: str, b: Rating):
    """the user's own 1-10 score and comment for a finished video (kept in its job.json)"""
    j = store.get(jid)
    if not j:
        raise HTTPException(404)
    if j.get("status") != "done":
        raise HTTPException(400, "only finished videos can be rated")
    if b.score is not None and not 1 <= b.score <= 10:
        raise HTTPException(400, "the score must be a whole number from 1 to 10")
    store.update(jid, rating={"score": b.score, "comment": (b.comment or "")[:5000], "updated": time.time()})
    return job_view(store.get(jid))


@app.get("/api/ratings")
def ratings():
    """every finished video (newest first) with its source, pictures, check scores, test-set label and rating - the Ratings view"""
    return [job_view(j) for j in store.all() if j.get("status") == "done" and j.get("result")
            and not str(j.get("result")).lower().endswith((".png", ".jpg", ".jpeg"))]


@app.delete("/api/jobs/{jid}")
def remove(jid: str):
    j = store.get(jid)
    if not j:
        raise HTTPException(404)
    if j["status"] == "running":
        raise HTTPException(409, "cancel it first")
    store.remove(jid)
    return {"ok": True}


@app.get("/api/status")
def status():
    try:
        q = comfy.busy(); comfy_ok = True
    except Exception:
        q, comfy_ok = None, False
    return {"llm": {"name": llm.NAME, "available": llm.available()}, "comfy": {"ok": comfy_ok, "queue": q},
            "worker": {"current": current["id"], "waiting": work.qsize()}}


# ------------------------------------------------------------------ worker (one GPU job at a time)
def _label_ai(jid):
    """tag the finished video as AI-generated in its metadata (lossless remux) - MiniMax H3's use policy asks for a machine-generated label"""
    out = (store.get(jid) or {}).get("result") or ""
    ext = os.path.splitext(out)[1].lower()
    if ext not in (".mp4", ".mov", ".mkv", ".webm") or not os.path.exists(out):
        return
    tmp = out[:-len(ext)] + ".label" + ext
    args = ["ffmpeg", "-v", "error", "-y", "-i", out, "-map", "0", "-c", "copy", "-metadata", "comment=AI-generated with MiniMax H3 (H3 Create)",
            "-metadata", "description=AI-generated video"] + (["-movflags", "+faststart"] if ext in (".mp4", ".mov") else []) + [tmp]
    try:
        if subprocess.run(args, timeout=600).returncode == 0 and os.path.getsize(tmp) > 0:
            os.replace(tmp, out)
    except Exception as e:
        store.log(jid, f"could not add the AI-generated label: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def worker():
    while True:
        jid = work.get(); j = store.get(jid)
        if not j or j.get("status") != "queued" or j.get("cancel"):
            continue
        current["id"] = jid
        try:
            try:  # pick up prompt / pipeline changes without restarting the app (and losing the queue position of a running job)
                importlib.reload(tools_mod); importlib.reload(pipeline)
            except Exception as e:
                store.log(jid, f"pipeline reload failed, using the loaded code: {e}")
            pipeline.Job(store, jid).run()
            _label_ai(jid)
        except comfy.Cancelled:
            store.update(jid, status="cancelled", stage="Cancelled")
        except Exception as e:
            store.update(jid, status="failed", stage="Failed", error=str(e)[:500]); store.log(jid, traceback.format_exc()[-1500:])
        finally:
            try:
                llm.down()
            except Exception:
                pass
            current["id"] = None


@app.on_event("startup")
def startup():
    for j in store.all():
        if j["status"] == "running":
            store.update(j["id"], status="failed", stage="Failed", error="stopped by an app restart - press Retry")
    for j in sorted(store.all(), key=lambda j: j["created"]):
        if j["status"] == "queued":
            work.put(j["id"])
    threading.Thread(target=worker, daemon=True).start()


app.mount("/data", StaticFiles(directory=DATA), name="data")
app.mount("/static", StaticFiles(directory=f"{ROOT}/static"), name="static")


@app.get("/")
def index():
    return FileResponse(f"{ROOT}/static/index.html", headers={"Cache-Control": "no-cache"})
