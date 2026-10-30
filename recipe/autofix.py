"""autofix.py JOBID ROOT [OUT.mp4] : 'Check & auto-fix' for object swaps (2026-10-27, user: "The goal is to have this run locally" - the tool must
fix its videos itself, not Claude by hand). Runs after the first take of redoutline3.py (ROOT = that run's folder), all on this machine:
  1. check: SAM 3 looks for the OLD objects in every shot of the result; each one found gets a close-up that Qwen3.8 compares with the new
     object's picture (new / old / neither). An old object on a place the shot's plan wanted replaced = a missed swap:
       'dropped' - the pick step had dropped that instance (G09 5 s test: the close-up check called the wheelbarrow 'a shovel' - it stayed green)
       'kept'    - it was outlined, but the video kept the old object (N06's corgi, F22's rope)
  2. remake each flagged shot, quick local ways first (each checked and judged on the spot):
     'dropped' -> the box redraw with the missed instance added, then erase + paste + refine of only the missed (and kept) instances on the
                  first take, then redoutline3 again for that shot with the pick forced (R3_FORCE), new seed, same tracks and sheets;
     'kept'    -> erase + paste + refine of only the kept instances on the first take / the box redraw (crowds first), the zoomed redraw, then
                  a new seed, a pinned edited frame, and the shot made from its edited first frame (Qwen-Image + H3)
  3. the remake is checked the same way; the judge (judge_pair, both A/B orders) compares it with the first take; per shot the better one is kept
     (a remake that fixed the miss wins unless both orders prefer the first take; otherwise it needs both orders). A judge vote counts only
     when its own error lists name a difference (founded); 'extra object' complaints are left to the stray check (judge_overrule)
  4. strays (new objects off their places): the source pasted back; else erased by MiniMax-Remover in the shot (source empty there); a new
     object drawn on another replaced object's place: that area redrawn with only that object in the prompt; else remakes (split / seed)
  5. the shots are joined, the source sound put back -> OUT.mp4 (default ROOT/result_fixed.mp4) + ROOT/autofix.json
A shot with nothing flagged is not touched. Progress lines start with '== autofix'. (2026-10-28 evening: steps marked above as 'erase',
'mixed area redrawn', 'founded', the 'every' rule for instances the pick step left out, meant() before any 'kept', on a wide view.)"""
import json, os, re, shutil, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT_ = os.path.dirname(HERE); sys.path.insert(0, ROOT_); sys.path.insert(0, HERE)
import cv2, numpy as np
from server import pipeline as P, llm, tools, comfy, h3

from server import config as CFG
DATA = os.environ.get("R3_DATA") or CFG.DATA; PY = CFG.PY_APP
jid, root = sys.argv[1], os.path.abspath(sys.argv[2]); OUTV = sys.argv[3] if len(sys.argv) > 3 else f"{root}/result_fixed.mp4"
job = json.load(open(f"{DATA}/jobs/{jid}/job.json")); meta = json.load(open(f"{root}/shots_meta.json"))
objs = meta["objs"]; segs = job["segments"]; t_all = time.time()
code = (job.get("project") or {}).get("code") or jid; base = os.path.basename(root)
TAG0 = base[len(f"redline3_{code}"):] if base.startswith(f"redline3_{code}") else ""
N_OUT = [min(sg["length"], sg.get("end", 10 ** 9) - sg["start"]) for sg in segs]
SHOT = {str(sg["k"]): i for i, sg in enumerate(segs)}                  # shot k -> its index (vi) in the run
VS = ('Picture 1 shows an object: {full}. Picture 2 is a close-up from one frame of an edited video, with a yellow box. The edit should have '
      'replaced the {old} with the object of Picture 1. Is the {old} itself still there inside the yellow box - its own body, shape, colours '
      'or material (for an animal or a person: its head, face, fur, legs) - even partly, even with the new object attached to it or in front '
      'of it? Parts that the object of Picture 1 has itself (its own handles, wheels or straps) do not count. "old" = some of the {old} is '
      'still there; "new" = only the object of Picture 1 is there (from any angle, partly hidden or blurred); "neither" = something else '
      '(a hand, a wall, the floor, another object). Reply with JSON only: {{"inside": "old" or "new" or "neither", "what": "2-6 words"}}')
# (2026-10-27: 'new / old / neither' called N06's corgi with a robot vacuum strapped to its side 'new' - a mix is 'old': the dog is still there)


def say(m):
    print(f"== autofix: {m}", flush=True)


def picture(oi):
    return job["images"][int(objs[oi]["picture"]) - 1]


def to_check(k):
    """the objects shot k should show replaced: what its plan lists or what was outlined, minus 'not meant' (stored spares) and people"""
    s = meta["shots"][str(k)]
    return [oi for oi in sorted(set(s["planned"]) | set(s["outlined"])) if oi not in s["not_meant"] and objs[oi]["word"] != "person"]


def shot_frames(fold, k):
    """shot k's own frames in a run folder: after the extra-copy guard if it pasted anything, else the render (or the source shot)"""
    i = SHOT[str(k)]
    for nm in (f"seg{k}_guard.mp4", f"seg{k}.mp4"):
        if os.path.exists(f"{fold}/{nm}"):
            return P.read_frames(f"{fold}/{nm}")[:N_OUT[i]]
    raise FileNotFoundError(f"{fold}/seg{k}.mp4")


def to_timeline(per_shot):
    """the shots in order -> the video; a long shot's next chunk (cont) overlaps the previous by 5 frames: cross-faded (as redoutline3)"""
    res = []
    for i, sg in enumerate(segs):
        fr = list(per_shot[i][:N_OUT[i]])
        if sg.get("cont") and len(res) >= 5 and len(fr) > 5:
            for j in range(5):
                a = (j + 1) / 6.0; res[-5 + j] = cv2.addWeighted(res[-5 + j], 1 - a, fr[j], a, 0)
            fr = fr[5:]
        res += fr
    return res


def masks(path, n, W, H):
    return [(cv2.cvtColor(cv2.resize(m, (W, H)), cv2.COLOR_BGR2GRAY) > 127) for m in P.read_frames(path)][:n]


def detect(fold, ks):
    """SAM 3 on shots ks of a run folder, searching the old objects -> ({k: [tracks]}, {k: frames}); a track = {oi, name, frames, mask}"""
    tmpd = tempfile.mkdtemp(prefix="afx_", dir=fold); vids = []; idx = []; R = {}
    concepts = []
    for k in ks:
        R[k] = shot_frames(fold, k); P.write_video(R[k], f"{tmpd}/s{k}.mp4", crf=14); vids.append(f"{tmpd}/s{k}.mp4"); idx.append(k)
        concepts += [objs[oi]["samw"] for oi in to_check(k)]
    concepts = list(dict.fromkeys(concepts))
    if not concepts:
        return {}, R
    llm.down(); tools.stop_orphan_llm(); comfy.free()
    r = subprocess.run([tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_multi.py", f"{tmpd}/det", "1", "|".join(concepts)] + vids,
                       capture_output=True, text=True, env={**os.environ, "PER_OBJ": "1", "MAX_OBJ": "16"})
    if r.returncode:
        raise RuntimeError("SAM 3 failed: " + (r.stdout + r.stderr)[-300:])
    out = {}
    for j, k in enumerate(idx):
        info = json.load(open(f"{tmpd}/det/{j}/objects.json")) if os.path.exists(f"{tmpd}/det/{j}/objects.json") else []
        tr = []
        for o in info:
            ois = [oi for oi in to_check(k) if o["concept"] in (objs[oi]["samw"], objs[oi]["word"])]
            if ois and len(o["frames"]) >= 3:
                tr.append({**o, "oi": ois[0], "mask": f"{tmpd}/det/{j}/{o['name']}.mp4"})
        out[k] = tr
    return out, R


def classify(R, k, t, tag):
    """Qwen: is the detected thing at its biggest frame the new object, still the old one, or neither?"""
    f_ = max(t["frames"], key=lambda x: x[5]); return classify_box(R[k][min(f_[0], len(R[k]) - 1)], f_[1:5], t["oi"], f"{tag}_s{k}_{t['name']}")


def second_look(frs, track_frames, oi, name):
    """an 'old' needs a second look - the track's middle frame, a crop twice as wide (G09: at the edge of the frame only the bathtub's own
    handles showed, and they were called 'wheelbarrow handles'; with the tub in view they are the new object)"""
    fr_ = sorted(track_frames, key=lambda x: x[0]); f2 = fr_[len(fr_) // 2]
    what, desc = classify_box(frs[min(f2[0], len(frs) - 1)], f2[1:5], oi, name + "_2", wide=True)
    return what == "old", desc


PIC_Q = ('Picture 1 is a close-up from one frame of a video, with a yellow box around a {old}. Is that {old} only an image of one - part of a '
         'poster, a sign, a mural, a painting or a screen (flat artwork on a wall, a board or a display), or a reflection in a mirror or a window? '
         'Things that are carried, held, worn, stored in a bag or a pocket, or that stand on a table or the floor are real objects, even when blurry '
         'or seen from the side. Reply with JSON only: {{"image": true or false, "what": "short reason"}}')


FLAT_CTX = (r"\b(poster|mural|screen|display|displayed|monitor|tv|television|sign|billboard|banner|painting|painted|photo|photograph|mirror|window|"
            r"reflection|reflected|animated|animation|video game|graphic|graphics|illustration|drawing|drawn|cartoon|sticker|logo|printed|print)\b")
WORN_CTX = r"\b(fabric|cloth|clothing|backpack|bag|pocket|shirt|t-shirt|jacket|sleeve|label|packaging|box)\b"


def is_picture(views, oi, name):
    """a leftover that is only an image does not count (K03 shot 1: her umbrella became the lollipop and uncovered the umbrella PRINTED on
    the poster behind her - 'kept' sent the shot through three remakes, the first take was right all along; G04: a pin on the bowling mural).
    views = [(frame, box), ...], all asked. An image needs one view answering 'image' and naming flat artwork (poster, banner, screen ...),
    and no 'image' answer naming something worn or carried as its surface (G07: a real bottle in a backpack pocket was once called 'printed
    image on the backpack fabric'). The views may disagree: K03's poster umbrella and G04's screen pin were each called real in one view"""
    ans = []
    for v, (img, box) in enumerate(views):
        x0, y0, x1, y1 = [int(x) for x in box]; m = max(x1 - x0, y1 - y0) + 48
        cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(img.shape[1], x1 + m), min(img.shape[0], y1 + m)
        c = img[cy0:cy1, cx0:cx1].copy(); cv2.rectangle(c, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), (0, 255, 255), 2)
        sc = 512 / max(c.shape[:2]); c = cv2.resize(c, (int(c.shape[1] * sc), int(c.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
        os.makedirs(f"{root}/autofix", exist_ok=True); p = f"{root}/autofix/{name}_pic{v}.png"; cv2.imwrite(p, c)
        try:
            r = llm.ask([llm.img_part(p, 512)], system=PIC_Q.format(old=objs[oi]["samw"]), max_tokens=60, temperature=0.0)
        except Exception as e:
            return False, str(e)[:50]
        ans.append((r.get("image") is True, str(r.get("what") or "")[:300]))   # matched in full: K03's 'banner' came after 60 characters
    why = " / ".join(f"{'image' if a else 'real'}: {w[:60]}" for a, w in ans)
    flat = any(a and re.search(FLAT_CTX, w, re.I) for a, w in ans)
    worn = any(a and re.search(WORN_CTX, w, re.I) for a, w in ans)
    return flat and not worn, why[:110]


def sees_new_only(d3, oi):
    """the image check's own descriptions of the thing name the NEW object and never the old one (K05 shot 2: 'apple body still visible',
    then both views said 'a real donut with pink frosting sitting in a glass bowl') - the 'old' is not sure: the remakes could only lose"""
    w_ = str(d3 or "").lower(); new = str(objs[oi].get("detect") or objs[oi]["full"]).lower().split()[-1].rstrip("s"); old = str(objs[oi]["samw"]).lower().split()[-1].rstrip("s")
    return bool(new) and re.search(rf"\b{re.escape(new)}", w_) is not None and re.search(rf"\b{re.escape(old)}", w_) is None


def pic_views(frs, track_frames):
    """two moments of a track (a third and two thirds in) for the image check"""
    fr_ = sorted(track_frames, key=lambda x: x[0]); out = []
    for q in (1 / 3, 2 / 3):
        f_ = fr_[min(len(fr_) - 1, int(q * len(fr_)))]
        out.append((frs[min(f_[0], len(frs) - 1)], f_[1:5]))
    return out


def classify_box(img, box, oi, name, wide=False):
    x0, y0, x1, y1 = [int(v) for v in box]; m = (max(x1 - x0, y1 - y0) + 48) if wide else (max(x1 - x0, y1 - y0) // 2 + 24)
    cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(img.shape[1], x1 + m), min(img.shape[0], y1 + m)
    c = img[cy0:cy1, cx0:cx1].copy(); cv2.rectangle(c, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), (0, 255, 255), 2)
    sc = 448 / max(c.shape[:2]); c = cv2.resize(c, (int(c.shape[1] * sc), int(c.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
    os.makedirs(f"{root}/autofix", exist_ok=True); p = f"{root}/autofix/{name}.png"; cv2.imwrite(p, c)
    try:
        r = llm.ask([llm.img_part(picture(oi), 384), llm.img_part(p, 448)], system=VS.format(full=objs[oi]["full"], old=objs[oi]["samw"]),
                    max_tokens=80, temperature=0.0)
        return str(r.get("inside") or "?").lower().strip(), str(r.get("what") or "")[:50]
    except Exception as e:
        return "?", str(e)[:50]


def source_match(k, t, W, H):
    """the source instance (first take's SAM 3 track of the same kind) that the detected thing sits on -> (name, mean IoU)"""
    i = SHOT[str(k)]; n = N_OUT[i]; oi = t["oi"]
    info = json.load(open(f"{root}/obj/{i}/objects.json")) if os.path.exists(f"{root}/obj/{i}/objects.json") else []
    D = masks(t["mask"], n, W, H); best = (None, 0.0)
    for c in info:
        if c["concept"] not in (objs[oi]["samw"], objs[oi]["word"]):
            continue
        S = masks(f"{root}/obj/{i}/{c['name']}.mp4", n, W, H); ious = []
        for j in range(0, min(len(D), len(S)), 3):
            u = int((D[j] | S[j]).sum())
            if u and D[j].any():
                ious.append(int((D[j] & S[j]).sum()) / u)
        if ious and np.mean(ious) > best[1]:
            best = (c["name"], float(np.mean(ious)))
    return best


ALIKE_Q = ('Picture 1 shows an object: {full}. Picture 2 is a close-up from a video with a {old} in the yellow box. In a video, could a viewer '
           'clearly tell the object of Picture 1 apart from the {old} - by its shape, colour, size or details? Reply with JSON only: '
           '{{"distinct": true or false, "why": "short"}}')
ALIKE = {}


def alike(oi, k, R):
    """the new object looks like the old one (G06: wooden chopsticks for wooden knitting needles) - a 'kept' there cannot be seen, and a remake
    would not change it: such objects are not judged (asked once per object, on its biggest source instance)"""
    if oi in ALIKE:
        return ALIKE[oi]
    i = SHOT[str(k)]; info = json.load(open(f"{root}/obj/{i}/objects.json")) if os.path.exists(f"{root}/obj/{i}/objects.json") else []
    cs = [c for c in info if c["concept"] in (objs[oi]["samw"], objs[oi]["word"]) and len(c["frames"]) >= 3]
    if not cs:
        return False
    c = max(cs, key=lambda c_: max(f[5] for f in c_["frames"])); f_ = max(c["frames"], key=lambda x: x[5])
    S = P.read_frames(f"{root}/seg{k}_src.mp4", [f_[0]])
    if not S:
        return False
    ok, why = ask_box(S[0], f_[1:5], ALIKE_Q.format(full=objs[oi]["full"], old=objs[oi]["samw"]), picture(oi), f"alike_{oi}", key="distinct", wkey="why")
    ALIKE[oi] = ok is False
    if ALIKE[oi]:
        say(f"{objs[oi]['full']} looks like the {objs[oi]['samw']} it replaces ({why}) - a kept one is not judged")
    return ALIKE[oi]


MEANT_Q = ('You look at one frame of a video edit. The edit replaces: "{what}". Is the {old} in the yellow box the one the edit means - the one '
           'the person holds, carries, uses or owns, or one that is part of what the people do in the shot? Spare ones stored in the background (on '
           'a rack, a shelf, a hook), ones that clearly belong to other people, and pictures of it (on a poster, a sign, a screen, a painting) or '
           'reflections are NOT meant. Being only partly visible, small, blurred or cut off at the edge of the frame does not matter. Reply with JSON '
           'only: {{"yes": true or false, "what": "short reason"}}')
NOT_MEANT_CTX = (r"\b(spare|spares|stored|storage|rack|racks|shelf|shelves|hook|hooks|hanging|on display|displayed|for sale|someone else|belongs?|"
                 r"other (person|people|man|woman|customer|player|diner|guest)|another (person|man|woman|customer|player)|poster|picture|"
                 r"print|printed|painting|reflection|mirror|screen|background|(?:on|at) (?:a|the|another|other|a different|their) table|"
                 r"guests?'?s?|customers?'?s?|diners?)\b")   # (not 'pile' / 'not held': G09's wheelbarrow 'with a pile of sand'; but F04's
# glass 'sitting on the table in the background' (a guest's) is a reason that counts)
SPEC_OBJS = [s_ for s_ in job["spec"]["subjects"] if s_.get("kind") != "location" and str(s_.get("picture") or "").strip().isdigit()]


MEANT = {}


def meant(k, oi, cname, strict=False):
    """before an unpicked old one counts as 'dropped': is it the one the edit means? (G10: spare ribbons on the gym's rack). Since
    2026-10-28 evening also before a picked one counts as 'kept' (F04's fresh run: one of the two picks for 'the two glasses on the tray' was
    a guest's glass at the frame edge - the take rightly left it, and the fixer redrew it). Asked once per instance. strict: only an explicit
    yes counts (an instance the pick step itself left out - K05's 'spare' apples on the counter for 'every red apple')"""
    if (k, oi, cname) not in MEANT:
        MEANT[(k, oi, cname)] = meant_(k, oi, cname)
    gated, raw = MEANT[(k, oi, cname)]
    return raw is True if strict else gated


def every_(oi):
    """the edit swaps EVERY one of the object, with no place or owner that narrows it (K05: 'every red apple, including ...' - yes; L01:
    'every glass of orange juice on the waiter's silver tray' - no: the guests' glasses stay, and the rule had turned two of them into
    cupcakes; M03: 'every orange in the vendor's crate' - no)"""
    what = str(objs[oi].get("replaces") or (str(SPEC_OBJS[oi].get("replaces") or "") if oi < len(SPEC_OBJS) else ""))
    head = re.split(r"[,;.(]|\bincluding\b", what, 1)[0]
    return bool(re.search(r"\b(every|all|each)\b", head, re.I)) and not re.search(
        r"\b(on|in|at|inside|into|from|under|near|by|with|that|which|who|held|carried|belonging|his|her|their|its)\b|'s\b", head, re.I)


def meant_(k, oi, cname):
    """-> (gated answer, Qwen's own yes/no)"""
    i = SHOT[str(k)]; info = json.load(open(f"{root}/obj/{i}/objects.json")) if os.path.exists(f"{root}/obj/{i}/objects.json") else []
    c = next((c_ for c_ in info if c_["name"] == cname), None)
    if c is None:
        return True, None
    f_ = max(c["frames"], key=lambda x: x[5]); S = P.read_frames(f"{root}/seg{k}_src.mp4", [f_[0]])
    if not S:
        return True, None
    what = objs[oi].get("replaces") or (str(SPEC_OBJS[oi].get("replaces") or "") if oi < len(SPEC_OBJS) else "") or objs[oi]["word"]
    up = re.sub(r"@image\d+", "the picture", str(job.get("prompt") or "")).strip()
    if up:   # the user's own words, with what stays as it is (L01: "The glasses the guests hold stay as they are.")
        what = f"{re.sub(r'<[^>]+>', '', what)}\" (the user asked: \"{up[:400]}\")"
    # a wide view: whose it is shows only around it (F04: a guest's glass stem at the frame edge, cropped tight, was 'meant' - with the tables
    # and the waiter's tray in view it is a guest's)
    ok, why = ask_box(S[0], f_[1:5], MEANT_Q.format(what=re.sub(r"<[^>]+>", "", what), old=objs[oi]["samw"]), None, f"meant_s{k}_{cname}",
                      full=True, wide=True)
    if ok is False and not re.search(NOT_MEANT_CTX, why, re.I):   # a 'no' must give a reason the edit allows (G09 5 s: the wheelbarrow being
        say(f"shot {k}: the {objs[oi]['samw']} {cname} - 'not meant' without a reason that counts ({why[:60]}) - meant")   # used was 'not meant'
        return True, ok                                                                       # because it was 'only partially visible')
    if ok is False:
        say(f"shot {k}: the {objs[oi]['samw']} {cname} is not the one the edit means ({why[:140]}) - not a miss")
    return ok is not False, ok


WHOLE_Q = ('Picture 1 shows an object: {full}. Picture 2 is a frame of an edited video. The edit replaced every {old} with the object of Picture 1. '
           'Is any {old} still visible in Picture 2 - one that was not replaced? Objects that look like Picture 1 do not count. Reply with JSON only: '
           '{{"yes": true or false, "where": "where in the frame, or empty"}}')


def check(fold, ks, tag):
    """-> {k: {"dropped": {oi: [names]}, "kept": [oi], "found": [...]}} for the shots ks of fold"""
    det, R = detect(fold, ks); res = {}
    for k in ks:
        s = meta["shots"][str(k)]; i = SHOT[str(k)]; r = {"dropped": {}, "kept": [], "found": []}; H, W = R[k][0].shape[:2]
        big = lambda fr_: max(((f[3] - f[1]) * (f[4] - f[2]) for f in fr_), default=0)
        for t in sorted(det.get(k, []), key=lambda t_: -big(t_["frames"]))[:int(os.environ.get("AFX_MAX_DET", "8"))]:   # the biggest few (G04: 10 pins
            if len(t["frames"]) < min_frames(k, t["oi"], N_OUT[i]) or not big_enough(t["frames"], W, H, crowded(k, t["oi"])):                     # a shot took 10 min)
                continue
            what, desc = classify(R, k, t, tag); oi = t["oi"]; nm, iou = source_match(k, t, W, H) if what == "old" else (None, 0.0)
            picked = s["pick"].get(str(oi)) or []
            kind = ""
            if what == "old":
                conf, d2 = second_look(R[k], t["frames"], oi, f"{tag}_s{k}_{t['name']}")
                if not conf:
                    what, desc = "old?", f"{desc} / 2nd look: {d2}"
            if what == "old":
                pic, d3 = is_picture(pic_views(R[k], t["frames"]), oi, f"{tag}_s{k}_{t['name']}")
                if pic:
                    what, desc = "picture", f"{desc} / {d3}"
                elif sees_new_only(d3, oi):
                    what, desc = "old?", f"{desc} [{d3}] / the image check saw the new object"
                else:
                    desc = f"{desc} [{d3}]"
            if what == "old":
                if nm and iou >= 0.08 and nm in picked and meant(k, oi, nm):   # its best match is the picked one (K03: a
                    kind = "kept"   # half-swapped umbrella behind the lollipop overlaps its old place only 18 % - the lollipop fills the rest); when the
                    # pick itself is doubtful (many alike, more picks than the edit names) it must be the one the edit means (F04: 'both wine
                    # glasses' had picked 14 in the restaurant - every glass on a guest's table counted as kept, every remake was refused)
                elif oi in s["planned"] and (not picked or (nm and nm in s["rejected"]) or oi in s["false_only"]) and (not nm or meant(k, oi, nm)):
                    kind = "dropped"
                elif oi in s["planned"] and nm and iou >= 0.08 and nm not in picked and every_(oi) and meant(k, oi, nm, strict=True):
                    kind = "dropped"   # 'every apple': one the pick step left out as a 'spare on the counter', and Qwen says it is meant (K05)
            if kind and alike(oi, k, R):
                kind = "alike"
            r["found"].append({"oi": oi, "track": t["name"], "frames": len(t["frames"]), "inside": what, "what": desc, "source": nm,
                               "iou": round(iou, 2), "verdict": kind or "ok", "at": int(max(t["frames"], key=lambda x: x[5])[0])})
            if kind == "kept" and oi not in r["kept"]:
                r["kept"].append(oi)
            elif kind == "dropped":
                r["dropped"].setdefault(oi, [])
                if nm and nm not in r["dropped"][oi]:
                    r["dropped"][oi].append(nm)
        # the old objects' own places in the source looked at in the result: SAM 3 misses thin or small things in the result (G01's baton in the
        # wide shot) - each instance that should be replaced (picked, or dropped by the checks while the plan shows the object) gets a close-up at
        # its biggest source frame; instances SAM 3 already matched above are skipped
        seen_src = {x["source"] for x in r["found"] if x.get("source") and x.get("iou", 0) >= 0.08}   # (a weak match - M02's 'cup' at IoU 0.05 -
        # does not count as having looked at that instance)
        info = json.load(open(f"{root}/obj/{i}/objects.json")) if os.path.exists(f"{root}/obj/{i}/objects.json") else []
        for oi in to_check(k):
            picked = s["pick"].get(str(oi)) or []; asked = 0
            for c in sorted(info, key=lambda c_: -big(c_["frames"])):
                if c["concept"] not in (objs[oi]["samw"], objs[oi]["word"]) or len(c["frames"]) < min_frames(k, oi, N_OUT[i]) or c["name"] in seen_src \
                        or not big_enough(c["frames"], W, H, crowded(k, oi)):
                    continue
                if asked >= int(os.environ.get("AFX_MAX_SRC", "6")):
                    break
                left_out = oi in s["planned"] and picked and c["name"] not in picked and c["name"] not in s["rejected"] and every_(oi)
                want = c["name"] in picked or (oi in s["planned"] and (not picked or c["name"] in s["rejected"] or oi in s["false_only"])) or left_out
                if not want:
                    continue
                f_ = max(c["frames"], key=lambda x: x[5]); j = min(f_[0], len(R[k]) - 1); asked += 1
                what, desc = classify_box(R[k][j], f_[1:5], oi, f"{tag}_s{k}_src_{c['name']}")
                if what == "old":
                    conf, d2 = second_look(R[k], c["frames"], oi, f"{tag}_s{k}_src_{c['name']}")
                    if not conf:
                        what, desc = "old?", f"{desc} / 2nd look: {d2}"
                if what == "old":
                    pic, d3 = is_picture([(R[k][j], f_[1:5])] + pic_views(R[k], c["frames"])[1:], oi, f"{tag}_s{k}_src_{c['name']}")
                    if pic:
                        what, desc = "picture", f"{desc} / {d3}"
                    elif sees_new_only(d3, oi):
                        what, desc = "old?", f"{desc} [{d3}] / the image check saw the new object"
                    else:
                        desc = f"{desc} [{d3}]"
                kind = (("kept" if meant(k, oi, c["name"]) else "") if c["name"] in picked else
                        ("dropped" if meant(k, oi, c["name"], strict=left_out) else "")) if what == "old" else ""
                if kind and alike(oi, k, R):
                    kind = "alike"
                r["found"].append({"oi": oi, "track": f"src:{c['name']}", "frames": len(c["frames"]), "inside": what, "what": desc, "source": c["name"],
                                   "iou": 1.0, "verdict": kind or "ok", "at": int(j)})
                if kind == "kept" and oi not in r["kept"]:
                    r["kept"].append(oi)
                elif kind == "dropped":
                    r["dropped"].setdefault(oi, [])
                    if c["name"] not in r["dropped"][oi]:
                        r["dropped"][oi].append(c["name"])
        # an object the plan shows here that SAM 3 found neither in the source nor in the result (K02: the juggling balls in a fast toss, swapped
        # from the text alone - red balls stayed in his hand): whole frames of the result asked directly; two of three 'still there' = kept
        for oi in to_check(k):
            if any(x["oi"] == oi for x in r["found"]) or crowded(k, oi) is None:
                continue
            if any(c_["concept"] in (objs[oi]["samw"], objs[oi]["word"]) for c_ in info):
                continue
            yes = 0; notes = []; yes_q = 0.5; os.makedirs(f"{root}/autofix", exist_ok=True)
            for q in (0.15, 0.35, 0.55, 0.75, 0.92):   # five moments (K02: the red balls showed only in the shot's last third - 1 of 3)
                fimg = R[k][min(len(R[k]) - 1, int(q * (len(R[k]) - 1)))]; pth = f"{root}/autofix/{tag}_s{k}_whole_{oi}_{int(q * 100)}.png"
                cv2.imwrite(pth, cv2.resize(fimg, (int(fimg.shape[1] * 768 / max(fimg.shape[:2])), int(fimg.shape[0] * 768 / max(fimg.shape[:2])))))
                try:
                    a_ = llm.ask([llm.img_part(picture(oi), 384), llm.img_part(pth, 768)], system=WHOLE_Q.format(full=objs[oi]["full"], old=objs[oi]["samw"]),
                                 max_tokens=80, temperature=0.0)
                    yes += bool(a_.get("yes")); notes.append(str(a_.get("where") or "")[:30])
                    if a_.get("yes"):
                        yes_q = q
                except Exception:
                    pass
            if yes == 1:   # one moment only (K02: the red balls show in the shot's last sixth): a second look 5 frames away confirms it
                j2 = int(yes_q * (len(R[k]) - 1)); j2 = j2 - 5 if j2 + 5 >= len(R[k]) else j2 + 5
                fimg = R[k][max(0, j2)]; pth = f"{root}/autofix/{tag}_s{k}_whole_{oi}_2nd.png"
                cv2.imwrite(pth, cv2.resize(fimg, (int(fimg.shape[1] * 768 / max(fimg.shape[:2])), int(fimg.shape[0] * 768 / max(fimg.shape[:2])))))
                try:
                    a_ = llm.ask([llm.img_part(picture(oi), 384), llm.img_part(pth, 768)], system=WHOLE_Q.format(full=objs[oi]["full"], old=objs[oi]["samw"]),
                                 max_tokens=80, temperature=0.0)
                    yes += bool(a_.get("yes")); notes.append("2nd: " + str(a_.get("where") or "")[:24])
                except Exception:
                    pass
            v_ = "kept" if yes >= 2 and not alike(oi, k, R) else ""
            r["found"].append({"oi": oi, "track": "whole", "frames": 0, "inside": f"{yes}/5 still", "what": "; ".join(notes), "source": None, "iou": 0,
                               "verdict": v_ or "ok", "at": int(yes_q * (len(R[k]) - 1))})
            if v_ and oi not in r["kept"]:
                r["kept"].append(oi)
        res[k] = r
    return res


def timeline(fold, frames_by_k):
    """fold/result_silent.mp4: the first take with the given shots replaced (the judge reads whole timelines) + fold/seg<k>.mp4 of those shots"""
    os.makedirs(fold, exist_ok=True); per = []
    for sg in segs:
        k = sg["k"]
        if k in frames_by_k:
            fr = frames_by_k[k]; fr = [fr[min(j, len(fr) - 1)] for j in range(N_OUT[SHOT[str(k)]])]
            P.write_video(fr, f"{fold}/seg{k}.mp4", crf=12); per.append(fr)
        else:
            per.append(shot_frames(root, k))
    P.write_video(to_timeline(per), f"{fold}/result_silent.mp4", crf=14)


def remake(ks, extra, suffix):
    """redoutline3 again for shots ks (R3_ONLY_SHOTS) with extra settings, the first take's tracks and sheets reused -> ROOT<suffix>"""
    fix = f"{root}{suffix}"; os.makedirs(fix, exist_ok=True)
    llm.down(); tools.stop_orphan_llm(); comfy.free()   # this process's Qwen server must not hold the card while the remake renders (OOM 2026-10-27)
    for f in os.listdir(root):
        if f.startswith("sheet") and f.endswith(".png") and not os.path.exists(f"{fix}/{f}"):
            shutil.copy(f"{root}/{f}", f"{fix}/{f}")
    env = {**os.environ, "R3_DATA": DATA, "R3_OUT": os.path.dirname(root), "R3_TAG": TAG0 + suffix, "R3_ONLY_SHOTS": ",".join(str(k) for k in ks),
           "R3_OBJ_FROM": f"{root}/obj", "R3_SEED_OFF": "7919", **extra}
    env.pop("R3_PICKS_ONLY", None); env.pop("R3_DRY", None)
    with open(f"{root}/autofix_remake{suffix}.log", "w") as lf:
        r = subprocess.run([PY, "-u", f"{HERE}/redoutline3.py", jid], stdout=lf, stderr=subprocess.STDOUT, env=env,
                           cwd=ROOT_)
    if r.returncode != 0 or not os.path.exists(f"{fix}/result_silent.mp4"):
        raise RuntimeError("the remake failed: " + open(f"{root}/autofix_remake{suffix}.log").read()[-300:])
    return fix


def remake_forced(ks, forces):
    """the dropped picks forced, a new seed"""
    return remake(ks, {"R3_FORCE": ",".join(forces)}, "_fix")


def remake_from_frame(k, ois, size):
    """the shot made again from its first frame with the new objects edited in (Qwen-Image), H3 plain generation from that frame"""
    from abtest import FakeJob
    i = SHOT[str(k)]; sg = segs[i]; fj = FakeJob(root, {"id": f"autofix-{k}"}); llm.down(); tools.stop_orphan_llm()
    W, H = (job["canvas"] if os.environ.get("R3_QUALITY") == "standard" else P.canvas(*job["canvas"], "fast")[0])
    f0 = P.read_frames(f"{root}/seg{k}_src.mp4", [0])[0]; cur = f"{root}/autofix/s{k}_frame0.png"; os.makedirs(f"{root}/autofix", exist_ok=True)
    cv2.imwrite(cur, cv2.resize(f0, (W, H)))
    for oi in ois:
        pr = (f"Picture 1 is a frame of a video. Replace the {objs[oi]['word']} in it with the {objs[oi]['full']} shown in Picture 2 - exactly that "
              f"object, its shape, colours and details - at the same place and size, held the same way. Remove every part of the "
              f"{objs[oi]['word']}. Keep everything else exactly the same: the people, their faces, hands and clothes, the background, the light "
              f"and the framing.")
        cur = P.qwen_edit(fj, [cur, picture(oi)], pr, int(job.get("seed") or 1234) + 400 + 7 * int(k) + oi, f"autofix_s{k}_edit{oi}")
    members = meta["shots"][str(k)]["members"]; pics = [picture(oi) for oi in members]
    plan = next((g for g in job["spec"].get("segments", []) if str(g.get("index")) == str(k)), {}); text = str(plan.get("shots") or "")
    for oi in range(len(objs)):   # the shot's replaced objects by their pictures; any other replaced object by its old name
        if objs[oi].get("tag"):
            text = text.replace(objs[oi]["tag"], f"the {objs[oi]['full']} from <Picture {members.index(oi) + 1}>" if oi in members else f"the {objs[oi]['word']}")
    full = (text + " Photorealistic live-action footage, natural light and motion blur, each object looking exactly like its picture; "
            "the first frame is exactly the given frame.")
    llm.down(); tools.stop_orphan_llm(); comfy.free()
    files = comfy.run(h3.build_gen(full, pics, W, H, sg["length"], int(job.get("seed") or 1234) + 777 * int(k), [{"image": cur, "frame": 0}]),
                      None, lambda: False)
    mp4 = next(f for f in files if f.endswith(".mp4")); fr = P.read_frames(mp4)
    fr = [cv2.resize(f, size) for f in fr]
    fold = f"{root}_regen{k}"; timeline(fold, {k: fr}); shutil.copy(mp4, f"{fold}/seg{k}_regen_raw.mp4")
    write_edit(fold, k)
    return fold


def write_edit(fold, k):
    """fold/seg<k>_prompt.txt for the judge (it reads the shot's edit from there; without one it is told 'nothing is replaced'): the first
    take's own prompt, or a plain one naming the shot's objects when the first take never rendered the shot (L05: every pick dropped)"""
    if os.path.exists(f"{fold}/seg{k}_prompt.txt"):
        return
    if os.path.exists(f"{root}/seg{k}_prompt.txt"):
        shutil.copy(f"{root}/seg{k}_prompt.txt", f"{fold}/seg{k}_prompt.txt"); return
    s_ = meta["shots"][str(k)]; ois = sorted(set(s_.get("members") or []) | set(s_.get("planned") or [])) or list(range(len(objs)))
    subs, defs = [], []
    for j, oi in enumerate(ois, 1):
        old = re.sub(r"<[^>]+>", "", str(objs[oi].get("replaces") or ("the " + objs[oi]["word"]))).strip()
        subs.append(f"{old} in <Video 1> with the object from <Picture {j}>")
        defs.append(f"<Subject {j}> is the object whose appearance comes from <Picture {j}> and whose performance comes from {old} in <Video 1>: "
                    f"{objs[oi]['full']}")
    open(f"{fold}/seg{k}_prompt.txt", "w").write("Replace " + " and ".join(subs) + ", each at exactly the place, size and angle of its original."
                                                 "\n\nsubject_definitions:\n" + "\n".join(defs) + "\n")


VACE_DIR = f"{HERE}/vace"


def box_picks(k, extra=None):
    """the shot's picks plus extra {oi: [instance names]} -> ('oi:a+b;...' for the vace tools, {oi: [names]})"""
    pk = {str(oi): list(v or []) for oi, v in meta["shots"][str(k)]["pick"].items()}
    for oi, nms in (extra or {}).items():
        pk[str(oi)] = list(dict.fromkeys(pk.get(str(oi), []) + [n_ for n_ in nms if n_]))
    return ";".join(f"{oi}:{'+'.join(v)}" for oi, v in pk.items() if v), pk


SIZE_Q = ('Typical real-world sizes: compare a {new} with a {old}. How long is the {new} (its longest side) compared with the {old}? Reply '
          'with JSON only: {{"ratio": the {new}\'s length divided by the {old}\'s length, as a number}}')
SIZES = {}


def size_ratio(oi):
    """the new object's real size against the old one's (Qwen, text only; 0.5-2.5) - the pasted cut-out is scaled by it"""
    if oi not in SIZES:
        try:
            r = llm.ask([{"type": "text", "text": "Answer the question."}], system=SIZE_Q.format(new=objs[oi]["full"], old=objs[oi]["samw"]),
                        max_tokens=40, temperature=0.0)
            SIZES[oi] = min(2.5, max(0.5, float(r.get("ratio"))))
        except Exception:
            SIZES[oi] = 1.0
    return SIZES[oi]


def remake_boxes(k, suffix, extra=None, paste=False, strength=0.75, only=None, only_inst=None):
    """H3 redraws ONLY boxes around the shot's old objects (vace/h3edit_swap.py: the source shot is the start latent with a per-token noise
    mask; <Video 1> = the plain source, the shot's own prompt and sheets) - nothing outside the boxes can change, so no strays or leaks there
    (2026-10-27, from D3voz/openjutsu's object mode; K05's bowl of apples became donuts in one go). paste=True: erase + paste + refine
    (vace/make_paste.py): the old objects erased (MiniMax-Remover), the new objects' cut-outs pasted into their boxes, and H3 starts from that
    at strength 0.75 - K01's phone survived every full remake but the pinned one, this removed it in 70 s. -> ROOT<suffix>"""
    fold = f"{root}{suffix}"; work = f"{fold}/work_s{k}"; os.makedirs(work, exist_ok=True)
    ratios = {}
    if paste:   # a small old object's paste is scaled to the new object's real size (K01: the watermelon came out phone-sized); a big one keeps
        i_ = SHOT[str(k)]; info_ = json.load(open(f"{root}/obj/{i_}/objects.json")) if os.path.exists(f"{root}/obj/{i_}/objects.json") else []
        for oi in meta["shots"][str(k)]["members"]:   # its box fit (K07: a pizza box x1.5 filled a third of the frame)
            nm_ = set(meta["shots"][str(k)]["pick"].get(str(oi)) or [])
            ar = [(f_[3] - f_[1]) * (f_[4] - f_[2]) for c in info_ if c["name"] in nm_ for f_ in c["frames"]]
            if ar and np.median(ar) / (SIZE[0] * SIZE[1]) < 0.02:
                ratios[oi] = min(1.5, max(0.67, size_ratio(oi)))
    llm.down(); tools.stop_orphan_llm(); comfy.free()
    parg, pk = box_picks(k, extra)
    if not parg:
        raise RuntimeError("no picked instances to draw boxes around")
    take = None
    if only is not None:   # only these objects (the kept ones), on top of the first take: its other swaps stay as they are (T22: the red ball
        # among the macarons - erase + paste + refine of the whole shot drew the churro again, thick and pasted-looking, and was refused)
        # only_inst {oi: [names]}: just the instances seen kept (K05: the two loose apples by the herbs - not the whole bowl of 18, whose erasing
        # took the glass bowl with it)
        parg = ";".join(f"{oi}:{'+'.join(([n_ for n_ in v if n_ in only_inst[int(oi)]] if only_inst and only_inst.get(int(oi)) else v) if int(oi) in only else [])}"
                        for oi, v in pk.items())
        take = next((f"{root}/{nm}" for nm in (f"seg{k}_guard.mp4", f"seg{k}.mp4") if os.path.exists(f"{root}/{nm}")), None)
    pics = ",".join(picture(oi) for oi in range(len(objs)))
    cmd = [PY, "-u", f"{VACE_DIR}/h3edit_swap.py", root, str(k), f"{work}/edit", "--video", "--pick", parg, "--fallback-refs", pics,
           "--seed", str(int(job.get("seed") or 1234) + 101 * int(k))]
    if paste:
        say(f"shot {k}: new-object sizes against the old ones: " + (", ".join(f"{objs[oi]['full']} x{v:.2f}" for oi, v in ratios.items()) or
                                                                     "none scaled (no small old object)"))
        with open(f"{root}/autofix_paste_s{k}.log", "w") as lf:
            r = subprocess.run([PY, "-u", f"{VACE_DIR}/make_paste.py", root, str(k), f"{work}/paste", "--pics", pics, "--pick", parg,
                                "--scales", ";".join(f"{oi}:{v:.2f}" for oi, v in ratios.items() if only is None or oi in only)]
                               + (["--base", take] if take else []),
                               stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT_)
        if r.returncode or not os.path.exists(f"{work}/paste/pasted.mp4"):
            raise RuntimeError("erase + paste failed: " + open(f"{root}/autofix_paste_s{k}.log").read()[-300:])
        cmd += ["--start", f"{work}/paste/pasted.mp4", "--strength", str(strength),   # room for a bigger new object
                "--grow-rels", ";".join(f"{oi}:{max(0.25, (v - 1) / 2 + 0.2):.2f}" for oi, v in ratios.items()),
                "--extra-region", f"{work}/paste/pasted_mask.mp4"]
    with open(f"{root}/autofix_box{suffix}.log", "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT_)
    if r.returncode or not os.path.exists(f"{work}/edit/result.mp4"):
        raise RuntimeError("the box redraw failed: " + open(f"{root}/autofix_box{suffix}.log").read()[-300:])
    fr = [cv2.resize(f, SIZE) if (f.shape[1], f.shape[0]) != SIZE else f for f in P.read_frames(f"{work}/edit/result.mp4")]
    if os.path.exists(f"{work}/edit/prompt.txt"):   # the judge reads the shot's edit from it (without it: 'nothing is replaced in this shot')
        shutil.copy(f"{work}/edit/prompt.txt", f"{fold}/seg{k}_prompt.txt")
    write_edit(fold, k)
    m2 = json.loads(json.dumps(meta)); sh2 = m2["shots"][str(k)]; sh2["pick"] = pk
    for oi_s, v in pk.items():
        if v and int(oi_s) not in sh2["members"]:
            sh2["members"].append(int(oi_s))
    timeline(fold, {k: fr}); json.dump(m2, open(f"{fold}/shots_meta.json", "w"), indent=1)
    return fold


def remake_zoom(k, oi, suffix):
    """the one small object redrawn on a ZOOMED crop of its own path (vace/zoom_swap.py: up to 4x more H3 tokens for it; ported from
    Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder's crop-to-mask) on top of the first take - the rest of the take stays (K01's phone,
    1-2 tokens wide at full frame, became the watermelon slice it is meant to be in both shots, 60 s each). Raises if the object's path
    needs most of the frame (exit 3: no zoom)."""
    fold = f"{root}{suffix}"; work = f"{fold}/work_s{k}"; os.makedirs(work, exist_ok=True)
    llm.down(); tools.stop_orphan_llm(); comfy.free()
    base = next((f"{root}/{nm}" for nm in (f"seg{k}_guard.mp4", f"seg{k}.mp4") if os.path.exists(f"{root}/{nm}")), None)
    with open(f"{root}/autofix_zoom{suffix}.log", "w") as lf:
        r = subprocess.run([PY, "-u", f"{VACE_DIR}/zoom_swap.py", root, str(k), f"{work}/zoom", "--obj", str(oi), "--start", base,
                            "--seed", str(int(job.get("seed") or 1234) + 131 * int(k))], stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT_)
    if r.returncode == 3:
        raise RuntimeError("its path covers most of the frame - no zoom")
    if r.returncode or not os.path.exists(f"{work}/zoom/result.mp4"):
        raise RuntimeError("the zoomed redraw failed: " + open(f"{root}/autofix_zoom{suffix}.log").read()[-300:])
    fr = [cv2.resize(f, SIZE) if (f.shape[1], f.shape[0]) != SIZE else f for f in P.read_frames(f"{work}/zoom/result.mp4")]
    timeline(fold, {k: fr})
    write_edit(fold, k)   # the judge reads the shot's edit from it
    return fold


SEVERE_ERR = (r"\b(face|faces|person|people|man's|woman's|hands?|fingers?|arms?|body|hair|identity|distort\w*|broken|melt\w*|duplicat\w*|"
              r"blend\w*|wrong (?:object|new object|way)|backwards|background|place changed|missing)\b")
EXTRA_ERR = r"\b(extra|floating|not present in the source|added|stray|additional|appears? where)\b"


PASTE_ERR = r"\b(float\w*|sticker\w*|paste\w*|flat|cut-?outs?|huge|giant|oversized|enormous|too (?:big|large))\b"


def judge_overrule(k, fold, errs_b, pasted=False):
    """the judge's only complaints about a remake that fixed the miss are extra / floating objects: the stray check decides - it knows where
    the old objects were (K05: the judge called two donuts at the frame edge 'extra objects not present in the source' - they replace two
    small apples there, and the remake that swapped them was refused A/A). Complaints about people, hands, distortion or a wrong object stand."""
    errs = [e for e in (errs_b or []) if not re.search(r"\b(which is correct|is correct|acceptable|no errors?)\b", str(e), re.I)]
    if not errs or any(re.search(SEVERE_ERR, str(e), re.I) for e in errs) or not all(re.search(EXTRA_ERR, str(e), re.I) for e in errs):
        return False
    if pasted and any(re.search(PASTE_ERR, str(e), re.I) for e in errs):   # a pasted cut-out that did not blend in (K05: erasing a bowl of apples
        return False                                                     # took the bowl - flat donuts floated where it was, one huge)
    try:
        st = stray_check({k: shot_frames(fold, k)}, [k], {k: meta_of(fold)}).get(k, [])
    except Exception as e:
        say(f"shot {k}: stray check for the judge's complaint failed: {str(e)[:120]}"); return False
    bad = [x for x in st if x["verdict"] in ("stray", "mixed")]
    say(f"shot {k}: the judge's complaint is only extra objects; the stray check finds {len(bad)} - " +
        ("the remake is taken" if not bad else "the judge stands"))
    return not bad


def judge_marks(fa, fb, k):
    """j -> the boxes (shares of the frame) of the old instances that either take replaces in shot k at frame j (the picks in both folders'
    plans; a box remake adds the missed ones) - drawn on the judge's SOURCE frames. Not for objects whose picks are in doubt (more than the
    edit names: F04's 14 glasses for 'both' - the guests' would be marked as to be replaced)"""
    if os.environ.get("AFX_JUDGE_MARKS", "0") != "1":   # OFF: on my 14 labelled pairs the boxed judge was right 7 / wrong 4, the plain one
        return None                                    # 9 / 2 - the boxes drew new complaints ('distorted donut', 'a bottle, not a maraca')
    i = SHOT[str(k)]; info = json.load(open(f"{root}/obj/{i}/objects.json")) if os.path.exists(f"{root}/obj/{i}/objects.json") else []
    names = set()
    for f in (fa, fb):
        s_ = meta_of(f)["shots"].get(str(k)) or {}
        for oi in to_check(k):
            if not doubt_pick(k, oi):
                names |= set((s_.get("pick") or {}).get(str(oi)) or [])
    tracks = [{f_[0]: f_[1:5] for f_ in c["frames"]} for c in info if c["name"] in names]
    if not tracks:
        return None
    W, H = SIZE

    def mk(j):
        out = []
        for t in tracks:
            near = min(t, key=lambda x: abs(x - j))
            if abs(near - j) <= 3:
                x0, y0, x1, y1 = t[near]; out.append((x0 / W, y0 / H, x1 / W, y1 / H))
        return out
    return mk


def judge(fa, fb, k):
    import judge_pair as JP
    os.makedirs(f"{root}/autofix", exist_ok=True)
    i = SHOT[str(k)]; sg = segs[i]
    subs = [s for s in job["spec"]["subjects"] if s.get("kind") != "location"]
    edit = "; ".join(f"{re.sub(r'<[^>]+>', '', str(s.get('replaces') or ''))[:60]} becomes {re.sub(r'<[^>]+>', '', str(s.get('description') or ''))[:70]}" for s in subs)
    JP.D = DATA
    try:
        mk = judge_marks(fa, fb, k)
    except Exception as e:
        mk = None; say(f"shot {k}: no marks for the judge ({str(e)[:80]})")
    v = JP.judge_shot(job, fa, fb, sg, edit, f"{root}/autofix/judge_s{k}.png", marks=mk) or {}
    w = JP.judge_shot(job, fa, fb, sg, edit, f"{root}/autofix/judge_s{k}.png", swap=True, marks=mk) or {}
    b1, b2 = v.get("better"), w.get("better")
    # a vote counts only when its own error lists back it: an error of the other take that it does not list for the chosen one (K04: both takes
    # 'the cup is still a cup' + 'a skateboard', word for word, and still SAME/A - the redraw that removed the second bouquet was refused)
    if b1 in ("A", "B") and not founded(v, b1):
        say(f"shot {k}: the judge's {b1} (first order) names no difference - same"); b1 = "SAME"
    if b2 in ("A", "B") and not founded(w, b2):
        say(f"shot {k}: the judge's {b2} (second order) names no difference - same"); b2 = "SAME"
    JLOG.append({"shot": k, "a": os.path.relpath(fa, os.path.dirname(root)), "b": os.path.relpath(fb, os.path.dirname(root)),
                 "votes": [v.get("better"), w.get("better")], "counted": [b1, b2], "errors_a": [v.get("errors_a"), w.get("errors_a")],
                 "errors_b": [v.get("errors_b"), w.get("errors_b")]})   # both orders, for reading the judge's reasons afterwards
    return b1, b2, v.get("errors_a"), v.get("errors_b")


JLOG = []


def norm_err(e):
    return " ".join(re.findall(r"[a-z0-9]+", str(e).lower()))


def founded(r, better):
    """the judge's answer r names an error of the take it voted against that it does not name for the take it chose ('... which is correct'
    entries are not errors)"""
    ok_ = lambda e: norm_err(e) and not re.search(r"\b(which is correct|is correct|acceptable|no errors?|none)\b", str(e), re.I)
    ea = {norm_err(e) for e in (r.get("errors_a") or []) if ok_(e)}; eb = {norm_err(e) for e in (r.get("errors_b") or []) if ok_(e)}
    return bool((eb - ea) if better == "A" else (ea - eb))


CHK_NEW = ('Picture 1 shows an object: {new}. Picture 2 is a close-up from one frame of a video, with a yellow box. Is the object of Picture 1 '
           '(or something that looks like it, from any angle) inside the yellow box? Reply with JSON only: {{"yes": true or false, "what": "what is '
           'in the box, 2-6 words"}}')
CHK_OLD = ('You look at a close-up from one frame of a video, with a yellow box. Is there a real {old} inside the yellow box - possibly small, '
           'blurred by motion, partly hidden or seen from an unusual angle? A picture, painting or print of one on a wall, poster or screen does not '
           'count. Reply with JSON only: {{"yes": true or false, "what": "what is in the box, 2-6 words"}}')
TINY = float(os.environ.get("AFX_TINY", "0.004"))   # instances whose box stays under this share of the frame are not judged (G04: pins far down
                                                    # the other lanes stayed white; a remake rarely changes things that small)


def big_enough(frames, W, H, crowd=True):
    """crowd = one of many alike things (pins far down the lanes): the TINY floor; a single meant thing (T12's flying tennis ball, N10's kicked
    ball) is judged down to a tenth of that"""
    return max(((f[3] - f[1]) * (f[4] - f[2]) for f in frames), default=0) >= (TINY if crowd else TINY / 10) * W * H


def min_frames(k, oi, n):
    """how long a leftover must show to count: a fifth of the shot for one of many, a twelfth for a single meant thing (N10: the kicked ball
    stayed a soccer ball 'for a moment')"""
    return max(6, int(0.2 * n)) if crowded(k, oi) else max(5, int(0.08 * n))


def doubt_pick(k, oi):
    """the shot's picks of object oi may hold some the edit does not mean: more than the number the edit names ('both' with 14 picks - F04).
    (Not every crowd: 'every apple' in a bowl is meant, and asking each one would risk dropping real leftovers)"""
    n_ = len(meta["shots"][str(k)]["pick"].get(str(oi)) or [])
    m = re.search(r"\b(both|two|pair|three|four|five|six|\d)\b", str(objs[oi].get("replaces") or ""), re.I)
    named = (int(m.group(1)) if m.group(1).isdigit() else {"both": 2, "two": 2, "pair": 2, "three": 3, "four": 4, "five": 5, "six": 6}[m.group(1).lower()]) if m else None
    return bool(named and n_ > named)


def crowded(k, oi):
    return len((meta["shots"][str(k)]["pick"].get(str(oi)) or [])) > 3
DRY = os.environ.get("AFX_DRY") == "1"   # check only: report, no remakes, nothing pasted (to measure the checks on finished videos)


NOT_REAL = r"\b(reflection|reflected|picture|painting|painted|print|printed|poster|photo|image|drawing|drawn|shadow|mural|screen|toy|sticker|logo|cartoon|icon|statue)\b"


def names_it(what, word):
    """an answer that says no but names the thing itself ('sledgehammer handle and head') - not a reflection, a picture and the like"""
    w_ = str(what or "").lower(); head = str(word).lower().split()[-1].rstrip("s") if str(word).strip() else ""
    if head and re.search(rf"\b(no|not|without|never|isn't)\s+(?:an?\s+|the\s+|any\s+|visible\s+)?(?:\w+\s+){{0,2}}{re.escape(head)}s?\b", w_):
        return False   # its own words deny it ('no case' - L07)
    return bool(head) and re.search(rf"\b{re.escape(head)}s?\b", w_) is not None and not re.search(NOT_REAL, w_)


def ask_box(img, box, system, pic=None, tag="x", key="yes", wkey="what", full=False, wide=False):
    x0, y0, x1, y1 = [int(v) for v in box]; m = (max(x1 - x0, y1 - y0) * 2 + 96) if wide else (max(x1 - x0, y1 - y0) // 2 + 24)
    cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(img.shape[1], x1 + m), min(img.shape[0], y1 + m)
    c = img[cy0:cy1, cx0:cx1].copy(); cv2.rectangle(c, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), (0, 255, 255), 2)
    sc = 448 / max(c.shape[:2]); c = cv2.resize(c, (int(c.shape[1] * sc), int(c.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
    os.makedirs(f"{root}/autofix", exist_ok=True); p = f"{root}/autofix/{tag}.png"; cv2.imwrite(p, c)
    try:
        r = llm.ask(([llm.img_part(pic, 384)] if pic else []) + [llm.img_part(p, 448)], system=system, max_tokens=80, temperature=0.0)
        return bool(r.get(key)), str(r.get(wkey) or "")[:400 if full else 50]
    except Exception as e:
        return None, str(e)[:50]


def meta_of(fold):
    try:
        return json.load(open(f"{fold}/shots_meta.json"))
    except Exception:
        return meta


def new_concepts(k, m):
    """SAM 3 name -> object: the new objects outlined in shot k (judged by their places) and those not meant in shot k at all - any copy of
    those is wrong (K08: in the man's shot his racket became a pan too); objects swapped there from the text alone have no place to judge by"""
    s = m["shots"][str(k)]; out = {}
    for oi in range(len(objs)):
        if objs[oi]["word"] != "person" and (oi in s["outlined"] or (oi not in s["planned"] and oi not in s["members"])):
            out.setdefault(objs[oi].get("detect") or objs[oi]["full"], oi)
    return out


def owned_elsewhere(k, m, oi):
    """an object the edit gives an owner ('the woman's tennis racket') picked in a shot whose plan does not show it - the same rule as the pick
    step's since 2026-10-27: someone else's thing (K08: the man's racket in his shot became a pan); new objects there are over-swaps"""
    s = m["shots"][str(k)]
    return oi not in s["planned"] and bool(re.search(r"\b\w+'s\b|\b(her|his|their)\b", str(objs[oi].get("replaces") or ""), re.I))


def places(k, m, n, W, H):
    """per outlined object: (its source instances, the same +-3 frames and grown W/14) - every instance of its old kind in the shot, picked or
    not (H3 swapping the other one is a swap, not a copy), minus the ones the close-up check rejected"""
    i = SHOT[str(k)]; s = m["shots"][str(k)]; ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (W // 14) + 1, 2 * (W // 14) + 1))
    info = json.load(open(f"{root}/obj/{i}/objects.json")) if os.path.exists(f"{root}/obj/{i}/objects.json") else []; out = {}
    for oi in s["outlined"]:
        if owned_elsewhere(k, m, oi):
            continue
        nms = [c["name"] for c in info if c["concept"] in (objs[oi]["samw"], objs[oi]["word"]) and c["name"] not in s["rejected"] and len(c["frames"]) >= 3]
        nms += [x for x in (s["pick"].get(str(oi)) or []) if x not in nms]
        u = None
        if named_count(oi) and s["pick"].get(str(oi)):   # the edit names how many (M01 'the two eggs he holds'; the carton's eggs stay): only the
            pk_ = list(s["pick"].get(str(oi)) or [])     # picked ones are its places - a new one on another of the kind is an over-swap
            fr_of = {c["name"]: {f_[0] for f_ in c["frames"]} for c in info}
            pf_ = set().union(*[fr_of.get(x, set()) for x in pk_])
            # ... but only another one that is there at the same time as the picked ones: one that is gone when they come is the same thing
            # tracked in two pieces (M01: the whole egg before he cracks it, then its two picked halves - the lemon in his hand was pasted back
            # into an egg for a second)
            un_ = [x for x in nms if x not in pk_ and len(fr_of.get(x, set()) & pf_) >= 0.5 * max(1, len(fr_of.get(x, set())))]
            nms = pk_ + [x for x in nms if x not in pk_ and x not in un_]; u = [np.zeros((H, W), bool) for _ in range(n)]
            for nm in un_:
                for j, mm in enumerate(masks(f"{root}/obj/{i}/{nm}.mp4", n, W, H)):
                    u[j] |= mm
        t = [np.zeros((H, W), bool) for _ in range(n)]
        for nm in nms:
            for j, mm in enumerate(masks(f"{root}/obj/{i}/{nm}.mp4", n, W, H)):
                t[j] |= mm
        # +-6 frames: the edit runs a little ahead of or behind the source (K01: the swapped phone left the frame 7 frames later than the real one
        # and was called a stray as it went)
        g = [cv2.dilate(np.logical_or.reduce([t[min(n - 1, max(0, j + d))] for d in range(-12, 13)]).astype(np.uint8), ker) > 0 for j in range(n)]
        # (+-12 since K05: the apple hidden by his hands while he bit it - its donut was called a stray and the 'fix' lost his thumbs-up)
        out[oi] = (t, g, nms, u)
    return out


def named_count(oi):
    """the number the edit names for the object ('both', 'two', '3' ...) or None"""
    m = re.search(r"\b(both|two|pair|three|four|five|six|\d)\b", str(objs[oi].get("replaces") or ""), re.I)
    return (int(m.group(1)) if m.group(1).isdigit() else {"both": 2, "two": 2, "pair": 2, "three": 3, "four": 4, "five": 5, "six": 6}[m.group(1).lower()]) if m else None


def stray_check(per, ks, metas):
    """new objects away from every place of their kind -> {k: [{oi, track, frames, mask, verdict ...}]}; Qwen confirms each one: the new object
    is in the result there AND the source shows no old one there (else SAM 3 lost the real one in motion blur). 'mixed' = it sits on another
    replaced object's place (H22: the bottle drawn as the paddle) - a paste would bring the other old object back, so it is only reported"""
    tmpd = tempfile.mkdtemp(prefix="afs_", dir=root); vids, idx, conc = [], [], []
    for k in ks:
        nc = new_concepts(k, metas[k])
        if nc:
            P.write_video(per[k], f"{tmpd}/s{k}.mp4", crf=14); vids.append(f"{tmpd}/s{k}.mp4"); idx.append(k); conc += list(nc)
    if not vids:
        return {}
    llm.down(); tools.stop_orphan_llm(); comfy.free()
    r = subprocess.run([tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_multi.py", f"{tmpd}/det", "1", "|".join(dict.fromkeys(conc))] + vids,
                       capture_output=True, text=True, env={**os.environ, "PER_OBJ": "1", "MAX_OBJ": "16"})
    if r.returncode:
        raise RuntimeError("SAM 3 failed: " + (r.stdout + r.stderr)[-300:])
    res = {}
    for j_, k in enumerate(idx):
        m = metas[k]; nc = new_concepts(k, m); fr = per[k]; n = len(fr); H, W = fr[0].shape[:2]
        S = [cv2.resize(x, (W, H)) for x in P.read_frames(f"{root}/seg{k}_src.mp4")][:n]; pl = places(k, m, n, W, H); out = []
        info = json.load(open(f"{tmpd}/det/{j_}/objects.json")) if os.path.exists(f"{tmpd}/det/{j_}/objects.json") else []; asked = 0
        for o in info:
            oi = nc.get(o["concept"])
            if oi is None or len(o["frames"]) < 3:
                continue
            unplaced = oi not in pl   # not meant in this shot: every frame of it is off its places
            M = masks(f"{tmpd}/det/{j_}/{o['name']}.mp4", n, W, H); off = []; oth_n = {}
            for j in range(min(n, len(M), len(S))):
                if not M[j].any():
                    continue
                a = int(M[j].sum()); on_own = 0.0 if unplaced else int((M[j] & pl[oi][1][j]).sum()) / a
                ov_ = {o2: int((M[j] & pl[o2][0][j]).sum()) / a for o2 in pl if o2 != oi}
                on_oth = max(list(ov_.values()) + [0.0])
                if on_oth > 0:
                    oth_n[max(ov_, key=ov_.get)] = oth_n.get(max(ov_, key=ov_.get), 0) + 1
                cov_oth = max([int((M[j] & pl[o2][0][j]).sum()) / max(1, int(pl[o2][0][j].sum())) for o2 in pl if o2 != oi] + [0.0])
                # right next to another replaced object (touching its place grown by W/16): the two objects' prompts mixed (L02: a carrot
                # sprouting from the pineapple where the mug's handle was - the pen's carrot drawn into the mug's place)
                near = any(int((M[j] & (cv2.dilate(pl[o2][0][j].astype(np.uint8), np.ones((W // 16 | 1, W // 16 | 1), np.uint8)) > 0)).sum()) > 0.05 * a
                           for o2 in pl if o2 != oi)
                if on_own < 0.15:
                    on_unp = int((M[j] & pl[oi][3][j]).sum()) / a if not unplaced and pl[oi][3] is not None else 0.0
                    off.append((j, on_oth, cov_oth, near, on_unp))
            if len(off) < max(6, int(0.1 * n)) or not big_enough(o["frames"], W, H, crowded(k, oi)):
                continue
            if asked >= int(os.environ.get("AFX_MAX_STRAY", "6")):
                break
            asked += 1
            # off its (generous: +-12 frames, grown W/14) places in under half of its frames - is it the swapped object itself, or another one
            # that only passes near it? the share of its frames in which it covers a third of one source instance of its kind (K08: the giant
            # lemon at the man's feet was 'on its places' in 21 of 39 frames because the bouncing ball passed near - it covered the ball in 0)
            present_ = [j for j in range(min(n, len(M))) if M[j].any()]; tight_own = None; tight_fr = []
            if not unplaced:   # the frames in which it covers a third of one of its own (picked) instances: there it IS the swapped object
                inst = [masks(f"{root}/obj/{SHOT[str(k)]}/{nm}.mp4", n, W, H) for nm in pl[oi][2] if os.path.exists(f"{root}/obj/{SHOT[str(k)]}/{nm}.mp4")]
                hit_, den_ = 0, 0
                for j in present_:
                    cv_ = [int((M[j] & I[j]).sum()) / int(I[j].sum()) for I in inst if j < len(I) and I[j].any()]
                    if cv_:
                        den_ += 1
                        if max(cv_) >= 0.3:
                            hit_ += 1; tight_fr.append(j)
                if len(off) < 0.5 * len(present_):
                    tight_own = round(hit_ / den_, 2) if den_ >= 3 else None
            jb = max((j for j, *_ in off), key=lambda j: int(M[j].sum())); ys, xs = np.nonzero(M[jb]); box = (xs.min(), ys.min(), xs.max(), ys.max())
            # mixed = drawn AS another replaced object (H22's bottle as the paddle): mostly on its place AND covering much of it - a peach next to
            # the pan's handle only touches the racket's place (T12) and is a stray
            mixed = float(np.mean([x[1] for x in off])) >= 0.3 and float(np.mean([x[2] for x in off])) >= 0.3
            on_other = float(np.mean([x[1] for x in off])); near_other = float(np.mean([x[3] for x in off])); on_unpicked = float(np.mean([x[4] for x in off]))
            in_res, w_res = ask_box(fr[jb], box, CHK_NEW.format(new=objs[oi]["full"]), picture(oi), f"stray_s{k}_{o['name']}_res")
            if in_res is False and names_it(w_res, objs[oi]["full"]):
                in_res = True
            # was an old one there? the plain name (G04: the plan's 'blue' ball was white), and three moments must all say no - a fast small one
            # is a blur in one frame and plain in another (T12: the tossed tennis ball in the serve)
            jo = sorted(j for j, *_ in off); looks = [jb] + [x for x in (jo[len(jo) // 4], jo[(3 * len(jo)) // 4]) if abs(x - jb) >= 3]
            in_src, w_src = None, ""
            for jj in looks[:3]:
                ys_, xs_ = np.nonzero(M[jj]) if M[jj].any() else (ys, xs)
                b_ = (xs_.min(), ys_.min(), xs_.max(), ys_.max())
                y_, w_ = ask_box(S[jj], b_, CHK_OLD.format(old=objs[oi]["samw"]), None, f"stray_s{k}_{o['name']}_src{jj}")
                if y_ is False and names_it(w_, objs[oi]["samw"]):   # N30: no - 'sledgehammer handle and head'
                    y_ = True
                in_src, w_src = y_, w_
                if y_ is not False:
                    break
            verdict = "mixed" if mixed and in_res else ("stray" if in_res and (in_src is False or unplaced) else "ok")
            if verdict == "stray" and unplaced:
                verdict = "stray"; w_src = f"not meant in this shot ({w_src})"
            if verdict == "ok" and in_res and on_unpicked >= 0.3:   # on one the edit leaves as it is (M01: a carton egg drawn as a lemon) - the
                verdict = "stray"; w_src = f"on one the edit leaves as it is ({w_src})"   # source pasted back brings that one back
            out.append({"oi": oi, "track": o["name"], "frames": [j for j, *_ in off], "n_off": len(off), "frame": int(jb), "box": [int(v) for v in box],
                        "on_other": round(on_other, 2), "near_other": round(near_other, 2), "on_unpicked": round(on_unpicked, 2),
                        "other": max(oth_n, key=oth_n.get) if oth_n else None,
                        "tight_own": tight_own, "tight_frames": tight_fr,
                        "result": [in_res, w_res], "source": [in_src, w_src], "verdict": verdict, "mask": f"{tmpd}/det/{j_}/{o['name']}.mp4"})
        res[k] = out
    return res


OUTLINE_BGR = [(0, 0, 255), (255, 90, 0), (0, 200, 0), (0, 220, 255)]   # redoutline3's COLORS, object oi -> COLORS[oi % 4]


def leak_fix(k, fr, m):
    """thin strokes in an outline's own colour that the source does not have, near where that outline was drawn (the object's old edge) or
    along the frame border (K04: a blue line at the left edge; N04: a red line along the border) - painted out from the pixels around them.
    -> (frames, number of frames fixed)"""
    s_ = m["shots"][str(k)]; n = len(fr); H, W = fr[0].shape[:2]; i = SHOT[str(k)]
    if not s_["outlined"]:
        return fr, 0
    S = [cv2.resize(x, (W, H)) for x in P.read_frames(f"{root}/seg{k}_src.mp4")][:n]
    b_ = max(6, W // 60); kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b_ + 1, 2 * b_ + 1))
    kt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, W // 90) | 1, max(5, W // 90) | 1)); kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (W // 40) + 1,) * 2)
    border = np.zeros((H, W), bool); e_ = max(10, W // 12); border[:e_] = border[-e_:] = True; border[:, :e_] = border[:, -e_:] = True   # K04's
    # blue line ran 20-70 px in from the left edge
    bands = {}
    for oi in s_["outlined"]:
        t = [np.zeros((H, W), bool) for _ in range(n)]
        for nm in (s_["pick"].get(str(oi)) or []):
            if os.path.exists(f"{root}/obj/{i}/{nm}.mp4"):
                for j, mm in enumerate(masks(f"{root}/obj/{i}/{nm}.mp4", n, W, H)):
                    t[j] |= mm
        bands[oi] = t
    out = list(fr); fixed = 0; found_ = {}
    hsv = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    for j in range(min(n, len(S))):
        hr, hs = hsv(fr[j]), hsv(S[j]); leak = np.zeros((H, W), bool)
        for oi, t in bands.items():
            cb, cg, cr = OUTLINE_BGR[oi % 4]; im = fr[j].astype(np.int16)
            near = border   # only along the frame border: near the object's own edge the new object's stripes have outline colours too (K03's
            # rainbow lollipop lost a red stripe, K07's blue bristles were smudged); the leaks seen so far all ran along the border (K04, N04)
            col = (np.abs(im[..., 0] - cb) + np.abs(im[..., 1] - cg) + np.abs(im[..., 2] - cr) < 130) & (hr[..., 1] > 140) & (hr[..., 2] > 110)
            im_s = S[j].astype(np.int16)
            col_s = (np.abs(im_s[..., 0] - cb) + np.abs(im_s[..., 1] - cg) + np.abs(im_s[..., 2] - cr) < 160) & (hs[..., 1] > 110)
            c8 = (col & near & ~(cv2.dilate(col_s.astype(np.uint8), kd) > 0)).astype(np.uint8)
            thin = c8 & ~cv2.morphologyEx(c8, cv2.MORPH_OPEN, kt)   # strokes, not filled shapes (a red bottle, a blue sweater)
            ncc, lab, st_, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), 8); hit = np.zeros((H, W), bool)
            for c in range(1, ncc):
                x, y, w, h, a = st_[c]
                along = (h >= 3 * w and (x < e_ or x + w > W - e_)) or (w >= 3 * h and (y < e_ or y + h > H - e_))   # running along the border
                if max(w, h) >= W // 25 and a >= 25 and a <= 0.35 * w * h and along:
                    hit |= lab == c
            if hit.any():   # the whole stroke it belongs to, its thicker corners too (K04: the line's foot stayed blue)
                nc2, lab2 = cv2.connectedComponents(c8, 8)
                for c in np.unique(lab2[hit]):
                    if c:
                        leak |= lab2 == c
        if leak.sum() >= 40:
            found_[j] = leak
    # a leak persists (K04: 9 frames); a lone frame is a stripe of the new object passing the border (K03's lollipop) - and painting one
    # frame alone would flicker: only frames with 2+ other leak frames within +-3 are painted
    core_ = [j for j in found_ if sum(1 for d in (-3, -2, -1, 1, 2, 3) if j + d in found_) >= 2]
    if not core_:
        return out, 0
    # around a confirmed run (+-4 frames): every outline-coloured pixel inside the run's area - its faint start, a lone flash after a gap
    # (K04: 37-41 strong, 36 faint, 44 back for one frame)
    area = cv2.dilate(np.logical_or.reduce([found_[c] for c in core_]).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    cols = [OUTLINE_BGR[oi % 4] for oi in bands]
    for j in range(max(0, min(core_) - 4), min(n, max(core_) + 5)):
        im = fr[j].astype(np.int16); hj = hsv(fr[j]); anyc = np.zeros((H, W), bool)
        for cb, cg, cr in cols:
            anyc |= (np.abs(im[..., 0] - cb) + np.abs(im[..., 1] - cg) + np.abs(im[..., 2] - cr) < 130) & (hj[..., 1] > 120)
        mk = (anyc & area) | (found_.get(j, np.zeros((H, W), bool)))
        if mk.sum() >= 30:
            out[j] = cv2.inpaint(fr[j], cv2.dilate(mk.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))), 5, cv2.INPAINT_TELEA)
            fixed += 1
    return out, fixed


def not_itself(st):
    """a stray that is on its object's places in many frames only by passing near them: it covers a third of a source instance of its kind
    in under a tenth of its frames (K08's giant lemon, 0 of 39) - the whole track is the stray, not a swapped object that strays a moment"""
    return st.get("tight_own") is not None and st["tight_own"] < 0.1


def paste_back(fr, k, st):
    """the source back where a stray sits (its mask +-1 frame, grown 9 px, +-2 frames, soft edge) - only where the surroundings still match the
    source (H3 redraws everything; where the person moved, a paste would show a ghost) -> new frames or None"""
    n = len(fr); H, W = fr[0].shape[:2]; S = [cv2.resize(x, (W, H)) for x in P.read_frames(f"{root}/seg{k}_src.mp4")][:n]
    M = masks(st["mask"], n, W, H); fs = set()
    present = [j for j in range(min(n, len(M))) if M[j].any()]
    if not present or (len(st["frames"]) < 0.5 * len(present) and not not_itself(st)):   # all or nothing: a part-painted stray flickers in and out
        st["ring_diff"] = None; st["why_not"] = f"off its places in only {len(st['frames'])} of {len(present)} frames"; return None
    keep_ = set(st.get("tight_frames") or [])   # where SAM 3's track IS a swapped instance (it jumped onto it - M01: the carton lemon's track
    for j in present:   # took the cracked lemon in his hands for a while, and the paste showed half an eggshell there): left alone
        if j not in keep_:   # every other frame it shows in, not only the frames it was off its places
            fs |= {min(n - 1, max(0, j + d)) for d in (-2, -1, 0, 1, 2)} - keep_
    g9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)); g25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)); plan = {}; diffs = []
    for j in sorted(fs):
        if j >= len(S) or j >= len(M):
            continue
        base = np.logical_or.reduce([M[min(len(M) - 1, max(0, j + d))] for d in (-1, 0, 1)])
        if not base.any():
            continue
        b = cv2.dilate(base.astype(np.uint8), g9) > 0; ring = (cv2.dilate(b.astype(np.uint8), g25) > 0) & ~b
        gr = cv2.GaussianBlur(cv2.cvtColor(fr[j], cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32)
        gs = cv2.GaussianBlur(cv2.cvtColor(S[j], cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32)
        if ring.any():
            diffs.append(float(np.abs(gr - gs)[ring].mean()))
        plan[j] = b
    st["ring_diff"] = round(float(np.median(diffs)), 1) if diffs else None
    if not plan or (st["ring_diff"] is not None and st["ring_diff"] > float(os.environ.get("AFX_RING", "22"))):
        return None
    out = list(fr)
    for j, b in plan.items():
        a = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 4)[..., None]
        out[j] = (fr[j].astype(np.float32) * (1 - a) + S[j].astype(np.float32) * a).clip(0, 255).astype(np.uint8)
    return out


def erase_strays(k, fr, sts):
    """strays the source cannot be pasted over (the person moved there, or it sits next to another replaced object) erased from the shot
    itself by MiniMax-Remover (vace/erase_strays.py), which fills in what is behind them (L02: the left woman's mug handle drawn as a carrot
    in 74 frames beside the pineapple - the ring differed from the source, the split remake lost her pen). All or nothing per stray, as the
    paste: one that is off its places in under half of the frames it shows in is left alone. -> (new frames or None, [tracks erased])"""
    n = len(fr); H, W = fr[0].shape[:2]; E = np.zeros((n, H, W), bool); used = []
    for st in sts:
        M = masks(st["mask"], n, W, H); present = [j for j in range(min(n, len(M))) if M[j].any()]
        if not present or (len(st["frames"]) < 0.5 * len(present) and not not_itself(st)):
            continue
        keep_ = set(st.get("tight_frames") or [])
        for j in present:
            if j in keep_:
                continue
            for d in (-1, 0, 1):
                if 0 <= j + d < n and j + d not in keep_:
                    E[j + d] |= M[j]
        used.append(st["track"])
    if not used:
        return None, []
    g = max(5, W // 48); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    wd = f"{root}_erase{k}/work"; os.makedirs(wd, exist_ok=True)
    P.write_video(fr, f"{wd}/cur.mp4", crf=8)
    P.write_video([cv2.cvtColor(((cv2.dilate(e.astype(np.uint8), ker) > 0) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for e in E],
                  f"{wd}/mask.mp4", crf=4)
    llm.down(); tools.stop_orphan_llm(); comfy.free()
    with open(f"{root}/autofix_erase_s{k}.log", "w") as lf:
        r = subprocess.run([PY, "-u", f"{VACE_DIR}/erase_strays.py", f"{wd}/cur.mp4", f"{wd}/mask.mp4", f"{wd}/erased.mp4"], stdout=lf,
                           stderr=subprocess.STDOUT, cwd=ROOT_)
    if r.returncode or not os.path.exists(f"{wd}/erased.mp4"):
        raise RuntimeError("the eraser failed: " + open(f"{root}/autofix_erase_s{k}.log").read()[-300:])
    new = [cv2.resize(f, (W, H)) if f.shape[:2] != (H, W) else f for f in P.read_frames(f"{wd}/erased.mp4")][:n]
    new += [new[-1]] * (n - len(new))
    return new, used


def mixed_fix(k, fr, sts, o2, ref="plain"):
    """a new object drawn on ANOTHER replaced object's place (K04: a second bouquet on the teddy's cart where the dog was; H22: the bottle
    drawn as the paddle): H3 redraws only that area on the shot (vace/h3edit_swap.py --region-video, the shot as the start), with a plain
    prompt that names only the other object o2 - nothing asks for the stray's own object there (the one-render-per-object remake did the
    same for the whole shot and could lose other swaps: L02's pen). -> new frames"""
    n = len(fr); H, W = fr[0].shape[:2]; E = np.zeros((n, H, W), bool)
    for st in sts:
        M = masks(st["mask"], n, W, H)
        for j in range(min(n, len(M))):
            if M[j].any():
                for d in (-1, 0, 1):
                    if 0 <= j + d < n:
                        E[j + d] |= M[j]
    g = max(9, W // 32); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    wd = f"{root}_mixed{k}/work" + ("" if ref == "plain" else f"_{ref}"); os.makedirs(wd, exist_ok=True)
    P.write_video(fr, f"{wd}/cur.mp4", crf=8)
    P.write_video([cv2.cvtColor(((cv2.dilate(e.astype(np.uint8), ker) > 0) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for e in E],
                  f"{wd}/region.mp4", crf=4)
    nms = list(meta_of(choice[k])["shots"][str(k)]["pick"].get(str(o2)) or meta["shots"][str(k)]["pick"].get(str(o2)) or [])
    if not nms:
        raise RuntimeError(f"no picked instances of the {objs[o2]['word']} in shot {k}")
    llm.down(); tools.stop_orphan_llm(); comfy.free()
    pics = ",".join(picture(oi) for oi in range(len(objs)))
    with open(f"{root}/autofix_mixed_s{k}.log", "w") as lf:
        r = subprocess.run([PY, "-u", f"{VACE_DIR}/h3edit_swap.py", root, str(k), f"{wd}/edit", "--video", "--pick", f"{o2}:{'+'.join(nms)}",
                            "--only-picked", "--region-video", f"{wd}/region.mp4", "--start", f"{wd}/cur.mp4", "--strength", "1.0", "--ref-video", ref,
                            "--fallback-refs", pics, "--seed", str(int(job.get("seed") or 1234) + 151 * int(k))],
                           stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT_)
    if r.returncode or not os.path.exists(f"{wd}/edit/result.mp4"):
        raise RuntimeError("the redraw failed: " + open(f"{root}/autofix_mixed_s{k}.log").read()[-300:])
    new = [cv2.resize(f, (W, H)) if f.shape[:2] != (H, W) else f for f in P.read_frames(f"{wd}/edit/result.mp4")][:n]
    return new + [new[-1]] * (n - len(new))


BUDGET = float(os.environ.get("AFX_BUDGET", "2100"))   # no new remake starts after this many seconds (a job must not grow without end; 1500
# until 2026-10-27 evening - the two quick box remakes tried first took F04's last slow remake out of the budget)


def time_left(what):
    if time.time() - t_all > BUDGET:
        say(f"no time left for {what} (budget {BUDGET:.0f} s)"); return False
    return True


report = {"shots": {}, "remade": [], "seconds": 0}
ks = [sg["k"] for sg in segs if to_check(sg["k"])]
say(f"checking {len(ks)} shot(s)")
first = check(root, ks, "a") if ks else {}
f0_ = shot_frames(root, segs[0]["k"])[0]; SIZE = (f0_.shape[1], f0_.shape[0])
for k, r in first.items():
    report["shots"][str(k)] = {"first": r}
    say(f"shot {k}: " + ("; ".join(f"{objs[x['oi']]['word']} -> {x['inside']} ({x['what']}), {x['verdict']}" for x in r["found"]) or "no old object seen"))
flag_d = {k: r["dropped"] for k, r in first.items() if r["dropped"] and not DRY}
flag_k = {k: r["kept"] for k, r in first.items() if r["kept"] and k not in flag_d and not DRY}
choice = {sg["k"]: root for sg in segs}
pre = {}   # shot -> (folder, judge result) of a box remake already judged on the spot
BOXES = os.environ.get("AFX_BOXES", "1") == "1"


def try_box(k, how, suffix, **kw):
    """a box remake of shot k, checked and judged on the spot; -> its folder when it fixed the miss and the judge does not prefer the first
    take in both orders, else None (the slower remakes follow)"""
    if not time_left(how):
        return None
    say(f"shot {k}: {how}")
    try:
        fb = remake_zoom(k, kw["zoom_obj"], suffix) if "zoom_obj" in kw else remake_boxes(k, suffix, **kw)
        r_ = check(fb, [k], suffix.strip("_")).get(k, {"dropped": {}, "kept": []})
        ent = report["shots"][str(k)].setdefault("remake_boxes", []); ent.append({"how": how, "folder": fb, "check": r_})
        if r_["dropped"] or r_["kept"]:
            say(f"shot {k}: still missed after {how.split(' -')[0]}"); return None
        # the first look still sees the old object on the very instance being fixed and only the second look (the track's middle moment) says
        # otherwise: a flash of it (K08: the box redraw showed the tennis ball for a few frames and a racket in a swing - it was taken)
        fixing = {(x["oi"], x["source"]) for x in first.get(k, {}).get("found", []) if x.get("source") and (x["verdict"] in ("kept", "dropped")
                                                                                                         or x["inside"] == "new")}   # (or was right
        # in the first take and is not now: the same redraw's racket in a swing, swapped well in the first take)
        flash = [x for x in r_.get("found", []) if x["inside"] == "old?" and "2nd look" in str(x.get("what")) and (x["oi"], x.get("source")) in fixing]
        if flash:
            say(f"shot {k}: {how.split(' -')[0]} still shows the old {objs[flash[0]['oi']]['word']} at moments ({flash[0]['what'][:60]}) - not taken")
            return None
        v = judge(root, fb, k); ent[-1]["judge"] = list(v[:2]); ent[-1]["errors_first"] = v[2]; ent[-1]["errors_remake"] = v[3]
        if (v[0] == "A" and v[1] != "B") or (v[1] == "A" and v[0] != "B"):   # leans to the first take (K07: A/SAME - a pizza box filling a
            if not judge_overrule(k, fb, v[3], pasted=bool(kw.get("paste"))):                                # third of the frame)
                say(f"shot {k}: it fixed the miss, but the judge leans to the first take ({v[0]}/{v[1]})"); return None
            ent[-1]["overruled"] = True
        pre[k] = (fb, v); say(f"shot {k}: fixed by {how.split(' -')[0]} (judge {v[0]}/{v[1]})")
        report["shots"][str(k)]["remake"] = {"how": how.split(" -")[0], "folder": fb}
        return fb
    except Exception as e:
        say(f"shot {k}: {how.split(' -')[0]} failed: {str(e)[:200]}"); return None


boxed = {}
if flag_d and BOXES:   # a missed instance of an object the shot already swaps: only the boxes redrawn, the instance added (about a minute)
    for k, d in sorted(flag_d.items()):
        if all(nms for oi, nms in d.items()):   # the missed instances are known (a member or not: the redraw adds them to the prompt)
            fb = try_box(k, "the box redraw - only boxes around the objects redrawn, the missed one added", f"_box{k}",
                         extra={int(oi): nms for oi, nms in d.items()})
            if not fb:   # erase + paste + refine on the first take, only the missed instances (and any kept ones in the shot): the rest of the
                # take stays (K05: the box redraw rebuilt every apple of the shot and still left one; the take's other swaps were right)
                inst = {int(oi): list(nms) for oi, nms in d.items()}
                for x in first[k]["found"]:
                    if x["verdict"] == "kept" and x.get("source") and x["source"] not in inst.setdefault(x["oi"], []):
                        inst[x["oi"]].append(x["source"])
                if all(len(v) <= 6 for v in inst.values()):
                    fb = try_box(k, "erase + paste + refine - the missed instances erased in the first take, the new ones pasted", f"_epr{k}",
                                 extra={int(oi): nms for oi, nms in d.items()}, paste=True, only=sorted(inst), only_inst=inst)
            if fb:
                boxed[k] = fb
flag_d2 = {k: v for k, v in flag_d.items() if k not in boxed}
if flag_d2 and time_left("the dropped-pick remake"):
    forces = [f"{k}:{oi}:{'|'.join(nms)}" for k, d in flag_d2.items() for oi, nms in d.items()]
    say(f"remaking shot(s) {sorted(flag_d2)} with the dropped picks forced: {forces}")
    try:
        fx = remake_forced(sorted(flag_d2), forces)
        for k in flag_d2:
            report["shots"][str(k)]["remake"] = {"how": "forced pick", "folder": fx}
    except Exception as e:
        say(f"remake failed: {str(e)[:200]}"); fx = None
else:
    fx = None
regen = {**boxed}
if flag_k and time_left("the kept-object remake"):   # the old object kept on its outlined place: first erase + paste + refine (only its boxes
    still = dict(flag_k)                               # redrawn, about a minute), then the same recipe with a new seed (keeps the source's
    for k in (sorted(flag_k) if BOXES else []):        # motion - G03's sword swing), a pinned edited frame, and the shot made from its
        s_ = meta["shots"][str(k)]                     # edited first frame (N06's corgi survived every retake)
        if all(s_["pick"].get(str(oi)) for oi in flag_k[k]):   # the kept object has picked instances to erase (K02's balls had none)
            # both quick ways before the slow remakes; a crowd (K05's bowl of apples: erasing took the bowl with it and the pasted donuts
            # floated as stickers) tries the plain box redraw first, anything else erase + paste + refine first (T22: the red ball among the
            # macarons stayed through the box redraw and every slow remake - erase + paste + refine had removed it)
            tries = [("box", "the box redraw - only boxes around the objects redrawn", f"_box{k}", False),
                     ("epr", "erase + paste + refine - the kept objects erased in the first take, the new ones pasted, H3 redraws only their boxes", f"_epr{k}", True)]
            kept_inst = {}   # the instances seen kept: erase + paste + refine touches only those
            for x in first[k]["found"]:
                if x["verdict"] == "kept" and x.get("source") and x["source"] in (s_["pick"].get(str(x["oi"])) or []):
                    kept_inst.setdefault(x["oi"], [])
                    if x["source"] not in kept_inst[x["oi"]]:
                        kept_inst[x["oi"]].append(x["source"])
            few = all(0 < len(kept_inst.get(oi) or []) <= 3 for oi in flag_k[k])   # a few known instances: erasing them on the first take
            if not any(crowded(k, oi) for oi in s_["members"]) or few:           # changes nothing else - first (the box redraw rebuilds every box)
                tries.reverse()
            if any(len(kept_inst.get(oi) or s_["pick"].get(str(oi)) or []) > 6 for oi in flag_k[k]):   # a big crowd kept (K05: 15-19 apples in a
                tries = [t for t in tries if t[0] != "epr"]   # glass bowl): erasing them took the bowl too - only the box redraw (T22's 4 balls: fine)
            tries = [(t[0], t[1], t[2], t[3]) for t in tries]
            if len(flag_k[k]) == 1 and not crowded(k, flag_k[k][0]):   # one small kept object: its zoomed redraw on the first take, LAST of
                # the quick tries - it keeps the rest of the first take, mistakes included (K01: the phone became the watermelon slice, but the
                # first take's floating watermelon stayed - two of them); the two others rebuild the shot from the source
                tries.append(("zoom", "the zoomed redraw - the small object redrawn on a zoomed crop of its path", f"_zoom{k}", None))
            for kind_, how_, suf_, pst_ in tries:
                fe = try_box(k, how_, suf_, **({"zoom_obj": flag_k[k][0]} if kind_ == "zoom" else
                                                ({"paste": True, "only": list(flag_k[k]), "only_inst": kept_inst} if kind_ == "epr" else {"paste": pst_})))
                if fe:
                    regen[k] = fe; still.pop(k, None); break
    ks_ = sorted(still)
    if ks_:
        say(f"remaking shot(s) {ks_} with a new seed - the old object stayed")
    try:
        if not ks_:
            raise StopIteration
        fs_ = remake(ks_, {"R3_SEED_OFF": "23209"}, "_kseed")
        for k in ks_:
            r_ = check(fs_, [k], "kseed").get(k, {"dropped": {}, "kept": []})
            report["shots"][str(k)]["remake_seed"] = {"folder": fs_, "check": r_}
            if not (r_["dropped"] or r_["kept"]):
                regen[k] = fs_; still.pop(k, None); say(f"shot {k}: the new seed swapped it")
                report["shots"][str(k)]["remake"] = {"how": "a new seed", "folder": fs_}
            else:
                say(f"shot {k}: the new seed kept it too")
    except StopIteration:
        pass
    except Exception as e:
        say(f"seed remake failed: {str(e)[:200]}")
    if still and time_left("the pinned-first-frame remake"):   # next: the same recipe with its first frame edited and pinned (the source still
        ks2 = sorted(still)                                    # leads the motion - the full rebuild below loses it and the judge often prefers
        say(f"remaking shot(s) {ks2} with a frame edited and pinned")   # the first take: K01's phone, K03's umbrella, K02's balls)
        at_ = {}; ob_ = {}
        for k in ks2:   # the frame the old object was seen in (K02's balls show only late in the shot - frame 0 does not have them)
            ent = [x for x in first[k]["found"] if x["verdict"] == "kept"]
            at_[k] = ent[0].get("at", 0) if ent else 0; ob_[k] = sorted({x["oi"] for x in ent})
        say(f"pinning frames {at_} with {ob_} edited")
        try:
            fp_ = remake(ks2, {"R3_SEED_OFF": "31337", "R3_PIN": "1", "R3_PIN_AT": ",".join(f"{k}:{at_[k]}" for k in ks2),
                               "R3_PIN_OBJS": ",".join(f"{k}:{oi}" for k in ks2 for oi in ob_[k])}, "_kpin")
            for k in ks2:
                r_ = check(fp_, [k], "kpin").get(k, {"dropped": {}, "kept": []})
                report["shots"][str(k)]["remake_pin"] = {"folder": fp_, "check": r_}
                if not (r_["dropped"] or r_["kept"]):
                    regen[k] = fp_; still.pop(k, None); say(f"shot {k}: the pinned frame swapped it")
                    report["shots"][str(k)]["remake"] = {"how": "a frame pinned", "folder": fp_}
                else:
                    say(f"shot {k}: the pinned frame kept it too")
        except Exception as e:
            say(f"pinned remake failed: {str(e)[:200]}")
    for k, ois in (still.items() if time_left("remakes from an edited first frame") else []):
        say(f"remaking shot {k} from its first frame with {', '.join(objs[oi]['word'] + ' -> ' + objs[oi]['full'] for oi in ois)} edited")
        try:
            regen[k] = remake_from_frame(k, ois, SIZE); report["shots"][str(k)]["remake"] = {"how": "from the edited first frame", "folder": regen[k]}
        except Exception as e:
            say(f"remake of shot {k} failed: {str(e)[:200]}")
cands = {**({k: fx for k in flag_d2} if fx else {}), **regen}
for k, fold in cands.items():
    say(f"comparing shot {k}")
    try:
        if k in pre and pre[k][0] == fold:   # a box remake: checked and judged when it was made
            r2 = report["shots"][str(k)]["remake_boxes"][-1]["check"]; v1, v2, ea, eb = pre[k][1]
        else:
            r2 = check(fold, [k], "b").get(k, {"dropped": {}, "kept": [], "found": []})
            v1, v2, ea, eb = judge(root, fold, k)
        miss1 = bool(first[k]["dropped"] or first[k]["kept"]); miss2 = bool(r2["dropped"] or r2["kept"])
        if miss1 and not miss2:
            take = not (v1 == "A" and v2 == "A") or judge_overrule(k, fold, eb, pasted="_epr" in os.path.basename(fold))
        else:
            take = v1 == "B" and v2 == "B"
        report["shots"][str(k)].update({"remake_check": r2, "judge": [v1, v2], "errors_first": ea, "errors_remake": eb, "chosen": "remake" if take else "first"})
        say(f"shot {k}: first take {'missed' if miss1 else 'ok'}, remake {'missed' if miss2 else 'ok'}, judge {v1}/{v2} -> "
            f"{'the remake' if take else 'the first take'}")
        if take:
            choice[k] = fold; report["remade"].append(k)
    except Exception as e:
        say(f"comparing shot {k} failed: {str(e)[:200]} - the first take stays")
per = {sg["k"]: shot_frames(choice[sg["k"]], sg["k"]) for sg in segs}; metas = {k: meta_of(choice[k]) for k in per}
report["strays"] = {}; report["pasted"] = []; found = {}
if os.environ.get("AFX_STRAYS", "1") == "1":
    say("looking for stray new objects")
    found = {}
    try:
        found = stray_check(per, [sg["k"] for sg in segs], metas)
    except Exception as e:
        found = {}; say(f"stray check failed: {str(e)[:200]}")
    for k, lst in found.items():
        for st in lst:
            say(f"shot {k}: {objs[st['oi']]['full']} off its places in {st['n_off']} frames (frame {st['frame']}): result {st['result']}, source "
                f"{st['source']} -> {st['verdict']}")
            if st["verdict"] == "stray" and not DRY:
                new = paste_back(per[k], k, st) if st.get("on_other", 0) < 0.15 else None
                if new is not None:
                    per[k] = new; report["pasted"].append([k, st["track"]]); say(f"shot {k}: the source pasted back over it (ring diff {st['ring_diff']})")
                else:
                    say(f"shot {k}: not pasted - " + (st.get("why_not") or f"the surroundings differ from the source (ring diff {st.get('ring_diff')})"))
            report["strays"].setdefault(str(k), []).append({x: v for x, v in st.items() if x != "mask"})
# strays that could not be pasted over: erased by MiniMax-Remover in the shot itself (a minute; nothing else in the shot changes), kept when the
# stray check finds fewer problems after it and the judge does not lean to the shot before it (2026-10-28 evening, L02's mug-handle carrot)
report["erased"] = []
for k, lst in (sorted(found.items()) if os.environ.get("AFX_ERASE", "1") == "1" and not DRY else []):
    # only where the source had none of the old kind (an over-swapped thing that is not meant here - K08's racket in the man's hand - must get
    # its old self back, not an empty hand)
    # and not on another replaced object's place (K04's 2nd bouquet on the teddy's cart, T12's peach on the pan: erasing would cut a hole in
    # that object - the one-render-per-object remake below is for those)
    bad_ = [st for st in lst if st["verdict"] == "stray" and [k, st["track"]] not in report["pasted"] and (st.get("source") or [None])[0] is False
            and st.get("on_other", 0) < 0.3]
    if not bad_ or not time_left("erasing strays"):
        continue
    try:
        new, used = erase_strays(k, per[k], bad_)
        if new is None:
            say(f"shot {k}: no stray to erase (each is off its places in under half of its frames)"); continue
        fe, fc = f"{root}_erase{k}", f"{root}_erase{k}/before"
        timeline(fe, {k: new}); timeline(fc, {k: per[k]}); write_edit(fe, k); write_edit(fc, k)
        n_before = len([st for st in lst if (st["verdict"] == "stray" and [k, st["track"]] not in report["pasted"]) or st["verdict"] == "mixed"])
        s2 = stray_check({k: new}, [k], {k: metas[k]}).get(k, [])
        n_after = len([st for st in s2 if st["verdict"] in ("stray", "mixed")])
        v1, v2, ea, eb = judge(fc, fe, k)
        lean_a = (v1 == "A" and v2 != "B") or (v2 == "A" and v1 != "B")
        take = n_after < n_before and not lean_a
        say(f"shot {k}: erased {used} - {n_before} stray problem(s) before, {n_after} after, judge {v1}/{v2} -> "
            f"{'the erased shot' if take else 'the shot before'}")
        report["shots"].setdefault(str(k), {})["erase"] = {"tracks": used, "problems_before": n_before, "problems_after": n_after,
                                                          "judge": [v1, v2], "errors_before": ea, "errors_erased": eb, "chosen": take}
        if take:
            per[k] = new; report["erased"].append([k, used]); found[k] = [{**st, "track": "e:" + st["track"]} for st in s2]   # (new SAM names: not
            # to be taken for the strays pasted before)
    except Exception as e:
        say(f"shot {k}: erasing strays failed: {str(e)[:200]}")
# a new object drawn on another replaced object's place (a mix, or a stray mostly on its place): only that area redrawn on the shot with the
# other object alone in the prompt (about a minute), kept when the stray check finds fewer problems, no old object came back, and the judge
# does not lean to the shot before it (2026-10-28 evening)
report["mixed_fixed"] = []
for k, lst in (sorted(found.items()) if os.environ.get("AFX_MIXFIX", "1") == "1" and not DRY else []):
    mx_ = [st for st in lst if (st["verdict"] == "mixed" or (st["verdict"] == "stray" and st.get("on_other", 0) >= 0.3)) and st.get("other") is not None
           and [k, st["track"]] not in report["pasted"]]
    if not mx_ or not time_left("redrawing mixed objects"):
        continue
    o2 = max({st["other"] for st in mx_}, key=lambda o_: sum(1 for st in mx_ if st["other"] == o_))
    mx_ = [st for st in mx_ if st["other"] == o2]
    try:
        say(f"shot {k}: {len(mx_)} new object(s) drawn on the {objs[o2]['word']}'s place - that area redrawn with only the {objs[o2]['full']}")
        fm, fc = f"{root}_mixed{k}", f"{root}_mixed{k}/before"
        n_before = len([st for st in lst if (st["verdict"] == "stray" and [k, st["track"]] not in report["pasted"]) or st["verdict"] == "mixed"])
        cur_r = (report["shots"].get(str(k), {}).get("remake_check") if choice[k] != root else first.get(k)) or {}
        miss_cur = bool(cur_r.get("dropped") or cur_r.get("kept"))
        for ref_ in ("plain", "blur"):   # the cited source shows the old object right there: when H3 copies it back (K08: the racket in the
            new = mixed_fix(k, per[k], mx_, o2, ref_)   # lemon's place), once more with the old objects blurred in <Video 1>
            timeline(fm, {k: new}); timeline(fc, {k: per[k]}); write_edit(fm, k); write_edit(fc, k)
            r2 = check(fm, [k], "mx").get(k, {"dropped": {}, "kept": []}); miss2 = bool(r2["dropped"] or r2["kept"])
            if not (miss2 and not miss_cur):
                break
            say(f"shot {k}: the redraw brought the old {objs[o2]['word']} back" + (" - again with it blurred in the cited source" if ref_ == "plain" else ""))
        s2 = stray_check({k: new}, [k], {k: metas[k]}).get(k, [])
        n_after = len([st for st in s2 if st["verdict"] in ("stray", "mixed")])
        v1, v2, ea, eb = judge(fc, fm, k)
        lean_a = (v1 == "A" and v2 != "B") or (v2 == "A" and v1 != "B")
        take = n_after < n_before and (not miss2 or miss_cur) and not lean_a
        say(f"shot {k}: redrawn - {n_before} stray problem(s) before, {n_after} after, {'an old object is back, ' if miss2 and not miss_cur else ''}"
            f"judge {v1}/{v2} -> {'the redrawn shot' if take else 'the shot before'}")
        report["shots"].setdefault(str(k), {})["mixed_fix"] = {"other": o2, "problems_before": n_before, "problems_after": n_after, "missed": miss2,
                                                              "judge": [v1, v2], "errors_before": ea, "errors_fixed": eb, "chosen": take}
        if take:
            per[k] = new; report["mixed_fixed"].append(k); found[k] = [{**st, "track": "m:" + st["track"]} for st in s2]
    except Exception as e:
        say(f"shot {k}: redrawing the mixed area failed: {str(e)[:200]}")
# shots whose strays could not be painted out (the person moved there), or where two new objects mixed: made again - a new seed, one render
# per object for a mix (R3_SPLIT fixed H22 by hand) - then checked the same way; the remake is kept when it has fewer problems and the judge
# does not prefer the current take in both orders, or when both judge orders prefer it
redo = {}
for k, lst in (found.items() if os.environ.get("AFX_STRAYS", "1") == "1" else []):
    bad = [st for st in lst if (st["verdict"] == "stray" and [k, st["track"]] not in report["pasted"]) or st["verdict"] == "mixed"]
    if bad and not DRY and os.environ.get("AFX_STRAY_REMAKE", "1") == "1":
        # on another replaced object's place (a mix, or K04's second bouquet sitting on the teddy's cart where the dog was) = the two objects'
        # prompts interfere: one render per object; elsewhere a new seed
        redo[k] = ("split" if any(st["verdict"] == "mixed" or st.get("on_other", 0) >= 0.15 or st.get("near_other", 0) >= 0.5 for st in bad)
                   else "seed", len(bad))   # (near_other: L02's carrot growing out of the pineapple - one render per object)
forced_env = ",".join(f"{k}:{oi}:{'|'.join(nms)}" for k, d in flag_d.items() for oi, nms in d.items() if choice.get(k) != root)
split_lost = []   # a split remake that was refused (L02: one render per object dropped the stray carrot but kept her pen): a new seed next
for how in ("seed", "split", "seed2"):
    ks_ = sorted(k for k, (h, _) in redo.items() if h == how) if how != "seed2" else sorted(split_lost)
    if not ks_ or not time_left(f"the {how} remake for strays"):
        continue
    say(f"remaking shot(s) {ks_} ({'one render per object' if how == 'split' else 'a new seed'}) for strays that could not be painted out")
    try:
        fold = remake(ks_, {"R3_SEED_OFF": "15401" if how != "seed2" else "19937", **({"R3_SPLIT": "1"} if how == "split" else {}),
                            **({"R3_FORCE": forced_env} if forced_env else {})}, f"_{how}")
    except Exception as e:
        say(f"remake failed: {str(e)[:200]}"); continue
    for k in ks_:
        try:
            m2 = meta_of(fold); fr2 = shot_frames(fold, k)
            miss2 = check(fold, [k], f"c{how}").get(k, {"dropped": {}, "kept": []}); miss2 = bool(miss2["dropped"] or miss2["kept"])
            s2 = stray_check({k: fr2}, [k], {k: m2}).get(k, []); fixed2 = []
            for st in s2:
                if st["verdict"] == "stray" and st.get("on_other", 0) < 0.15:
                    new = paste_back(fr2, k, st)
                    if new is not None:
                        fr2 = new; fixed2.append(st["track"])
            bad2 = len([st for st in s2 if (st["verdict"] == "stray" and st["track"] not in fixed2) or st["verdict"] == "mixed"])
            v1, v2, ea, eb = judge(choice[k], fold, k)
            # a stray is a small flaw; a remake may change the action (K05: it lost the chef's thumbs-up) - the judge must not prefer the current
            # take in EITHER order
            cur_r = (report["shots"].get(str(k), {}).get("remake_check") if choice[k] != root else first.get(k)) or {}
            miss_cur = bool(cur_r.get("dropped") or cur_r.get("kept"))   # a leftover the current take has too is no reason to refuse (K08 shot 2)
            lean_a = (v1 == "A" and v2 != "B") or (v2 == "A" and v1 != "B")   # K05 A/SAME = leans to the current take (it kept his thumbs-up);
            take = (not miss2 or miss_cur) and bad2 < redo[k][1] and not lean_a  # K04 B/A = a tie - the remake without the 2nd bouquet wins
            say(f"shot {k}: {redo[k][1]} unfixed problem(s) before, {bad2} in the remake, judge {v1}/{v2} -> {'the remake' if take else 'the current take'}")
            report["shots"].setdefault(str(k), {})[f"remake_{how}"] = {"folder": fold, "problems_before": redo[k][1], "problems_after": bad2,
                                                                        "missed": miss2, "judge": [v1, v2], "chosen": take}
            if take:
                per[k] = fr2; choice[k] = fold; report["remade"].append(k)
            elif how == "split":
                split_lost.append(k)
            if not take and miss2 and not miss_cur:
                say(f"shot {k}: the remake lost a swap (an old object stayed) - not taken")
        except Exception as e:
            say(f"comparing shot {k} failed: {str(e)[:200]} - the current take stays")
            if how == "split":
                split_lost.append(k)
report["leaks"] = {}
if os.environ.get("AFX_LEAKS", "1") == "1" and not DRY:
    for sg in segs:
        k = sg["k"]
        try:
            per[k], nf = leak_fix(k, per[k], meta_of(choice[k]))
            if nf:
                report["leaks"][str(k)] = nf; say(f"shot {k}: outline-coloured strokes painted out in {nf} frame(s)")
        except Exception as e:
            say(f"leak check of shot {k} failed: {str(e)[:120]}")
llm.down(); tools.stop_orphan_llm()
P.write_video(to_timeline([per[sg["k"]] for sg in segs]), f"{root}/result_fixed_silent.mp4", crf=14)
r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{root}/result_fixed_silent.mp4", "-i", job["video"], "-map", "0:v", "-map", "1:a:0?", "-c:v",
                    "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", OUTV])
if r.returncode != 0:
    shutil.copy(f"{root}/result_fixed_silent.mp4", OUTV)
report["seconds"] = round(time.time() - t_all); report["choice"] = {str(k): v for k, v in choice.items()}; report["judge_log"] = JLOG
json.dump(report, open(f"{root}/autofix.json", "w"), indent=1)
say(f"done in {report['seconds']} s - {len(report['remade'])} shot(s) remade: {report['remade']}, {len(report['pasted'])} stray(s) painted out, "
    f"{sum(len(x[1]) for x in report.get('erased') or [])} erased, {len(report.get('mixed_fixed') or [])} mixed area(s) redrawn -> {OUTV}")
