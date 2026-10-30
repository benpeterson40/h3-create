"""judge_pair.py CALIB.json : pairwise judge of two takes of an object swap, shot by shot (2026-10-26) - Qwen3.8 sees, for 4 moments of a shot,
the source | take A | take B side by side and says which take follows the edit better. CALIB.json = [{"code", "jid" (v1 job), "a": folder,
"b": folder, "shots": {"k": "A"|"B"|"same"} (my visual verdicts, optional)}, ...]; folders are data/abtest names or paths. Prints the verdicts
and, where my verdicts are given, the agreement. -> data/runs/r3/judge_<calib name>.json"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from server import pipeline as P, llm

from server.config import DATA as D
JUDGE = ('You compare two edited versions (A and B) of the same shot of a video. The edit in THIS shot: {edit} Nothing else is replaced in this '
         'shot - every other thing must look as in the source. Each row shows one moment: the SOURCE video on the left, version A in the middle, '
         'version B on the right. First look at the SOURCE to see which of the named objects are really in this shot. Judge which version does the '
         'edit better. Count as errors: an old object still there where it should be replaced; a new object added where the source has no matching old object (an extra copy, a floating or '
         'stray object); a replaced object drawn as the wrong new object or two new objects blended; a new object turned or held the wrong way for '
         'what the person does; distorted, broken or duplicated objects; people, hands or the place changed. Ignore small differences in light or '
         'detail. Reply with JSON only: {{"errors_a": ["..."], "errors_b": ["..."], "better": "A" or "B" or "same"}}')


MARKS = (' In the SOURCE frames thin yellow boxes mark the old objects that this edit replaces (some are small or at the edge of the frame): '
         'a new object at the place of a yellow box replaces that old object and is NOT an extra or floating object; an old object still '
         'there at a yellow box\'s place in A or B is an error. The boxes are drawn on the SOURCE only.')


def frame_at(path, t):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * 24))); ok, f = c.read(); c.release()
    return f if ok else None


def folder(x):
    return x if os.path.isabs(x) else f"{D}/abtest/{x}"


def shot_edit(folder_, k, fallback):
    """the shot's own edit: the first line of its rendered prompt ('Replace the X outlined in red in <Video 1> with ...'), pictures named"""
    try:
        line = open(f"{folder_}/seg{k}_prompt.txt").read().split("\n")[0]
        subj = re.findall(r"<Subject (\d+)> is the \w+ whose appearance comes from <Picture (\d+)>[^:]*: ([^\n]+)", open(f"{folder_}/seg{k}_prompt.txt").read())
        for n_, pic, desc in subj:
            line = line.replace(f"<Picture {pic}>", f"({desc[:80]})")
        return re.sub(r"\s*in <Video 1>", "", line).replace("outlined in red ", "").replace("outlined in blue ", "").replace("outlined in green ", "")
    except Exception:
        return fallback


def judge_shot(job, a, b, sg, edit, out_png, swap=False, marks=None):
    """marks(j) -> [(x0, y0, x1, y1) as shares of the frame]: the old objects the edit replaces at shot frame j, drawn as yellow boxes on the
    SOURCE (2026-10-28 evening: the judge called K05's two donuts at the frame edge 'extra objects not present in the source' - they
    replace two small apples there that it could not see at 240 px; T22's box redraw was taken once and refused once)"""
    if swap:   # the same question with the takes' places exchanged; the answer is mapped back
        v = judge_shot(job, b, a, sg, edit, out_png.replace(".png", "_sw.png"), marks=marks)
        if v:
            v = {**v, "better": {"A": "B", "B": "A"}.get(v.get("better"), v.get("better")), "errors_a": v.get("errors_b"), "errors_b": v.get("errors_a")}
        return v
    n_out = min(sg["length"], sg.get("end", 10 ** 9) - sg["start"]); t0 = sg["start"] / 24; dur = n_out / 24
    rows = []
    for q in (0.15, 0.4, 0.65, 0.9):
        t = t0 + q * dur
        fs = [frame_at(f"{D}/jobs/{job['id']}/src24.mp4", t), frame_at(f"{folder(a)}/result_silent.mp4", t), frame_at(f"{folder(b)}/result_silent.mp4", t)]
        if any(f is None for f in fs):
            continue
        h = 240; fs = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in fs]
        if marks:
            for x0, y0, x1, y1 in marks(int(round(t * 24)) - sg["start"]) or []:
                w_, h_ = fs[0].shape[1], fs[0].shape[0]; px = 3
                cv2.rectangle(fs[0], (max(0, int(x0 * w_) - px), max(0, int(y0 * h_) - px)), (min(w_ - 1, int(x1 * w_) + px), min(h_ - 1, int(y1 * h_) + px)),
                              (0, 255, 255), 1, cv2.LINE_AA)
        for f, lab in zip(fs, ("SOURCE", "A", "B")):
            cv2.rectangle(f, (0, 0), (90, 24), (0, 0, 0), -1); cv2.putText(f, lab, (4, 18), 0, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.hstack([fs[0], np.full((h, 6, 3), 255, np.uint8), fs[1], np.full((h, 6, 3), 255, np.uint8), fs[2]]))
    if not rows:
        return None
    w = max(r.shape[1] for r in rows); grid = np.vstack([np.hstack([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)]) for r in rows])
    cv2.imwrite(out_png, grid)
    try:
        # 'nothing is replaced' only when NEITHER take has the shot's prompt (2026-10-28: autofix's box / erase+paste / zoom remakes wrote no
        # prompt - the judge was told nothing should change and preferred the take that kept the old phone, L05)
        if os.path.exists(f"{folder(b)}/seg{sg['k']}.mp4") and not any(os.path.exists(f"{folder(x)}/seg{sg['k']}_prompt.txt") for x in (a, b)):
            e2 = "nothing is replaced in this shot (the source has none of the objects)."
        else:
            e2 = shot_edit(folder(b), sg["k"], None) or shot_edit(folder(a), sg["k"], edit)
        sysm = JUDGE.format(edit=e2) + (MARKS if marks and not str(e2 or '').startswith('nothing is replaced') else '')
        r = llm.ask([llm.img_part(out_png, 1536)], system=sysm, max_tokens=400, temperature=0.0)
    except Exception as e:
        return {"better": "?", "why": str(e)[:80]}
    b_ = str(r.get("better") or "?").strip().upper()
    return {"better": "SAME" if b_.startswith("S") else b_[:1], "errors_a": r.get("errors_a"), "errors_b": r.get("errors_b")}


if __name__ == "__main__":
    calib = json.load(open(sys.argv[1])); out = []; agree = total = 0
    od = f"{D}/runs/r3/judge"; os.makedirs(od, exist_ok=True)
    for c in calib:
        job = json.load(open(f"{D}/jobs/{c['jid']}/job.json"))
        subs = [s for s in job["spec"]["subjects"] if s.get("kind") != "location"]
        edit = "; ".join(f"{re.sub(r'<[^>]+>', '', str(s.get('replaces') or ''))[:60]} becomes {re.sub(r'<[^>]+>', '', str(s.get('description') or ''))[:70]}" for s in subs)
        for sg in job["segments"]:
            k = str(sg["k"])
            if c.get("shots") and k not in c["shots"]:
                continue
            v1 = judge_shot(job, c["a"], c["b"], sg, edit, f"{od}/{c['code']}_s{k}.png")
            v2 = judge_shot(job, c["a"], c["b"], sg, edit, f"{od}/{c['code']}_s{k}.png", swap=True) if os.environ.get("JP_BOTH") == "1" else None
            if v1 is None:
                continue
            v = dict(v1)
            if v2 is not None:   # both orders must agree, else 'unsure'
                v["better"] = v1["better"] if v1.get("better") == v2.get("better") else "UNSURE"
            mine = (c.get("shots") or {}).get(k)
            ok = None
            if mine:
                total += 1; ok = (v["better"] == mine.upper()) or (mine.lower() == "same"); agree += bool(ok)
            out.append({"code": c["code"], "shot": k, "judge": v["better"], "mine": mine, "agree": ok, "errors_a": v.get("errors_a"), "errors_b": v.get("errors_b")})
            print(f"{c['code']} s{k}: judge {v['better']}  mine {mine}  {'OK' if ok else ('MISS' if ok is False else '')}  A: {str(v.get('errors_a'))[:90]} | B: {str(v.get('errors_b'))[:90]}", flush=True)
    llm.down()
    name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    json.dump(out, open(f"{D}/runs/r3/judge_{name}.json", "w"), indent=1)
    if total:
        print(f"agreement with my verdicts: {agree}/{total}", flush=True)
