"""abtest.py TEST... : offline A/B renders of the no-install ideas on finished test jobs' segments (GPU - run only while the app is idle).
  dp:CODE:K    new place: view plate (place_view + plate_window) vs depth plate (depth_place_plate); both rendered by H3 from the plate
               with a composite pin of the plate's first frame (on_plate)
  mask:CODE:K  object swap: plain render vs masked swap (old object grey + H3 inpaint inside its grown outline + follow_alpha paste-back)
  hair:CODE:K  hair of the replaced people: TELEA fill (strip_hair) vs flat grey (grey_regions)
CODE may be a job id. Arms -> data/abtest/<CODE>_<K>_<test>/<arm>.mp4 ; each arm is judged with physics_check + audit_dense ->
data/abtest/results.json (read-only on the jobs)"""
import copy, json, os, re, shutil, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); args = sys.argv[1:]; sys.argv = ["x"]
import cv2, numpy as np
from server import pipeline as P, llm, h3, comfy, tools
from server.config import DATA as D
OUT = os.environ.get("R3_OUT") or f"{D}/abtest"; os.makedirs(OUT, exist_ok=True)
RES = f"{OUT}/results.json"; res = json.load(open(RES)) if os.path.exists(RES) else {}
BASE = {"M9": "1023-201757-40b9"}
FAST = os.environ.get("AB_FULL") != "1"
ARMS = (os.environ.get("AB_ARMS") or "view,depth").split(",")   # dp test: which arms to (re)render    # M9 v2 has the new spec (travel / spot / closeup); the others use the rated job


class Store:
    def __init__(s, root, job):
        s.root, s.job = root, job

    def dir(s, jid):
        return s.root

    def get(s, jid):
        return s.job

    def update(s, jid, **kw):
        s.job.update({k: v for k, v in kw.items() if k not in ("stage", "progress")})

    def log(s, jid, msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)


class FakeJob:
    def __init__(s, root, job):
        s.store, s.id = Store(root, job), job["id"]

    @property
    def d(s):
        return s.store.job

    def set(s, **kw):
        s.store.update(s.id, **kw)

    def log(s, msg):
        s.store.log(s.id, msg)

    def cancelled(s):
        return False


def job_for(code):
    if code in BASE:
        jid = BASE[code]
    elif "-" in code:
        jid = code
    else:
        items = {(x.get("project") or {}).get("code"): x for x in json.load(urllib.request.urlopen("http://127.0.0.1:7880/api/ratings"))
                 if x.get("project") and not (x.get("project") or {}).get("rerun_of")}
        jid = items[code]["id"]
    return jid, json.load(open(f"{D}/jobs/{jid}/job.json"))


def render(fj, job, spec, k, window, guides, L, seed, dst, inpaint=None):
    W, H = job["canvas"]
    prompt, pics = P.segment_prompt(spec, k, P.closeup_refs(fj, job, spec, k, job["images"]), job.get("style"))
    llm.down()
    wf = h3.build(prompt, window, pics, W, H, L, seed, guides)
    if inpaint:
        h3.add_inpaint(wf, *inpaint)
    t0 = time.time(); files = comfy.run(wf, None, lambda: False); mp4 = next((f for f in files if f.endswith(".mp4")), None)
    if not mp4:
        raise RuntimeError("no video")
    shutil.copy(mp4, dst); fj.log(f"rendered {os.path.basename(dst)} in {time.time() - t0:.0f} s")
    open(dst[:-4] + "_prompt.txt", "w").write(prompt)
    return P.read_frames(dst)


def judge(job, spec, k, win_real, frames, n):
    seg = next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), None)
    try:
        a = P.audit_dense(job, spec, win_real, frames[:n], seg=seg)
    except Exception as e:
        a = {"error": str(e)[:300], "score": 0}
    return {"score": a.get("score"), "issues": [{x2: x.get(x2) for x2 in ("severity", "time", "what", "check")} for x in a.get("issues", []) if isinstance(x, dict)]}


def guides_for(fj, job, spec, k, sg, frame0, tag, on_plate):
    """the production pins: a continuation follows the previous segment's chosen result; a new shot gets a composite of its first frame"""
    W, H = job["canvas"]
    if sg.get("cont"):
        prev = next((s for s in job["segments"] if s["k"] == k - 1), None)
        if prev:
            pv = next(a["video"] for a in prev["attempts"] if a["attempt"] == prev.get("chosen", 0))
            fr = P.read_frames(pv); n_prev = min(prev["length"], prev.get("end", 10 ** 9) - prev["start"])
            gp = os.path.join(fj.store.root, f"{tag}_guide.mp4"); P.write_video(fr[:n_prev][-5:], gp, crf=10)
            return [{"video": gp, "frame": 0}]
    if "end" in sg:
        seg = next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), {})
        comp, _ = P.compose_frame(fj, job, spec, P.cover_resize(frame0, W, H), W, H, tag, kept=seg.get("kept"), shots=seg.get("shots") or "", on_plate=on_plate)
        return [{"image": comp, "frame": 0}] if comp else []
    return []


def run(test, code, k):
    jid, job0 = job_for(code); k = int(k)
    root = f"{OUT}/{code}_{k}_{test}"; os.makedirs(root, exist_ok=True)
    job = copy.deepcopy(job0); job["id"] = f"ab-{code}-{k}-{test}"; fj = FakeJob(root, job); spec = job["spec"]
    spec.pop("_corrections", None)   # the job's own retake notes are not part of a first render (M9 v2's asked for 'trees and a low building')
    sg = next(s for s in job["segments"] if s["k"] == k); L = sg["length"]
    if FAST:   # 480p previews (~3x quicker renders); the winner is confirmed at full size in the real re-run
        job["canvas"] = list(P.canvas(*job["canvas"], "fast")[0])
    W, H = job["canvas"]
    n = min(L, sg.get("end", 10 ** 9) - sg["start"]); jd = f"{D}/jobs/{jid}"
    win = f"{root}/src.mp4"; shutil.copy(f"{jd}/seg{k}_src.mp4", win)
    win_real = f"{root}/src_real.mp4"; shutil.copy(f"{jd}/seg{k}_src_real.mp4", win_real)
    seed = int(job.get("seed") or 1234) + 1000 * k + 991
    job["_context_frames"] = []
    out = {"job": jid, "code": code, "k": k, "test": test, "arms": {}}
    key = f"{code}:{k}:{test}"; out["arms"].update({a_: v_ for a_, v_ in (res.get(key) or {}).get("arms", {}).items() if a_ not in ARMS}); res[key] = out
    chosen = next(a for a in sg["attempts"] if a["attempt"] == sg.get("chosen", 0))   # the version the user rated (its in-app score; not re-judged)
    out["arms"]["rated"] = {"video": chosen["video"], "score": (chosen.get("audit") or {}).get("score"), "issues": []}
    json.dump(res, open(RES, "w"), indent=1)
    if test == "dp":
        arms = {}
        view = None
        if "view" in ARMS:
            try:
                view = P.place_view(fj, job, spec, k, P.cover_resize(P.read_frames(win, [0])[0], W, H), W, H, f"seg{k}")
            except Exception as e:
                print("place view failed", e)
        if "view" in ARMS:
            arms["view"] = P.plate_window(fj, job, spec, win, L, W, H, "view", view=view)
        if "depth" in ARMS:
            arms["depth"] = P.depth_place_plate(fj, job, spec, k, win, L, W, H, "depth")
        for arm, plate in arms.items():
            g = guides_for(fj, job, spec, k, sg, P.read_frames(plate, [0])[0], f"{arm}_pin", True)
            fr = render(fj, job, spec, k, plate, g, L, seed, f"{root}/{arm}.mp4")
            out["arms"][arm] = {"video": f"{root}/{arm}.mp4", "plate": plate, "pinned": bool(g), **judge(job, spec, k, win_real, fr, n)}
            print(f"== {key} {arm}: {out['arms'][arm]['score']}", flush=True); json.dump(res, open(RES, "w"), indent=1)
    elif test == "rp":   # the new place re-painted from the shot's own first frame (repaint_plate), rendered from the plate with an on-plate pin
        plate = P.repaint_plate(fj, job, spec, k, win, L, W, H, "rp")
        g = guides_for(fj, job, spec, k, sg, P.read_frames(plate, [0])[0], "rp_pin", True)
        fr = render(fj, job, spec, k, plate, g, L, seed, f"{root}/repaint.mp4")
        out["arms"]["repaint"] = {"video": f"{root}/repaint.mp4", "plate": plate, "pinned": bool(g), **judge(job, spec, k, win_real, fr, n)}
        print(f"== {key} repaint: {out['arms']['repaint']['score']} (pinned {bool(g)})", flush=True); json.dump(res, open(RES, "w"), indent=1)
    elif test == "mask":
        sub = next(s for s in spec["subjects"] if s.get("kind") == "object")
        words = re.sub(r"\s*\b(in|of|from|held by|on)\b.*$", "", re.sub(r"<[^>]+>", "", sub.get("replaces") or "object")).replace("the ", "").strip()[:40] or "object"
        g = guides_for(fj, job, spec, k, sg, P.read_frames(win, [0])[0], "plain_pin", False)
        fr = render(fj, job, spec, k, win, g, L, seed, f"{root}/plain.mp4")
        out["arms"]["plain"] = {"video": f"{root}/plain.mp4", "pinned": bool(g), **judge(job, spec, k, win_real, fr, n)}
        print(f"== {key} plain: {out['arms']['plain']['score']}", flush=True); json.dump(res, open(RES, "w"), indent=1)
        grey, masks = P.grey_regions(fj, win, words, "obj", grow_frac=0.012)
        big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.04 * max(W, H)) + 1,) * 2)
        mvid = [cv2.cvtColor(cv2.dilate(m * 255, big), cv2.COLOR_GRAY2BGR) for m in masks]
        mclip = f"{root}/obj_mask.mp4"; P.write_video([cv2.resize(m, (W, H)) for m in mvid], mclip, crf=0)
        greyc = f"{root}/obj_grey_canvas.mp4"; P.write_video([P.cover_resize(f, W, H) for f in P.read_frames(grey)], greyc, crf=12)
        g2 = guides_for(fj, job, spec, k, sg, P.read_frames(grey, [0])[0], "mask_pin", False)
        gen = render(fj, job, spec, k, greyc, g2, L, seed, f"{root}/masked_raw.mp4", inpaint=(greyc, mclip))
        S = [P.cover_resize(f, W, H) for f in P.read_frames(win)]
        comp = []
        for i in range(min(len(gen), len(S))):
            gi = cv2.resize(gen[i], (W, H)); a = P.follow_alpha(S[i], gi, cv2.resize(mvid[min(i, len(mvid) - 1)][..., 0], (W, H)).astype(np.float32) / 255.0)
            comp.append((S[i].astype(np.float32) * (1 - a) + gi.astype(np.float32) * a).astype(np.uint8))
        P.write_video(comp, f"{root}/masked.mp4", crf=12)
        sel = P.selectivity(S[:len(comp)], comp, [cv2.resize(m[..., 0], (W, H)) for m in mvid][:len(comp)])
        out["arms"]["masked"] = {"video": f"{root}/masked.mp4", "selectivity": round(float(sel), 2), "words": words, **judge(job, spec, k, win_real, comp, n)}
        print(f"== {key} masked: {out['arms']['masked']['score']} selectivity {sel:.1f}", flush=True); json.dump(res, open(RES, "w"), indent=1)
    elif test == "hair":
        for arm in ("telea", "grey"):
            w2 = P.strip_hair(fj, win, "hair") if arm == "telea" else P.grey_regions(fj, win, "hair", "hairgrey", grow_frac=0.01)[0]
            g = guides_for(fj, job, spec, k, sg, P.read_frames(w2, [0])[0], f"{arm}_pin", False)
            fr = render(fj, job, spec, k, w2, g, L, seed, f"{root}/{arm}.mp4")
            out["arms"][arm] = {"video": f"{root}/{arm}.mp4", "window": w2, "pinned": bool(g), **judge(job, spec, k, win_real, fr, n)}
            print(f"== {key} {arm}: {out['arms'][arm]['score']}", flush=True); json.dump(res, open(RES, "w"), indent=1)
    json.dump(res, open(RES, "w"), indent=1)


if __name__ == "__main__":
    for a in args:
        t, code, k = a.split(":")
        try:
            run(t, code, k)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"== {a} FAILED: {e}", flush=True)
    llm.down()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import shotcheck; shotcheck.stop_orphan_llm()
