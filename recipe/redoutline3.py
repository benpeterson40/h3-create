"""redoutline3.py JOBID : object swap = redoutline2.py (one outline colour per replaced object, Qwen3.8 picks the meant instances, 480p, no
pins) + fixes from the user's ratings of 2026-10-25 ("Most of the issues are with object swapping" - new objects added where the source has
none: H16 pumpkins in the jump-rope shot, H24 cheese + credit card next to the bags, T14 ice cream in the record shots, N04 a pizza held by a
man with no record):
 1. each shot's prompt names ONLY the objects of that shot - redoutline2's summary / retention / replace lines were clip-wide and named every
    swap of the video, and unmatched <Subject n> / <Picture n> numbers collided after the per-shot renumbering (H16 shot 4: both outlines got
    Picture 1); now tags and pictures are renumbered 1..n per shot from the shot's own objects
 2. an outlined object that the plan does not list for the shot needs a close-up yes from Qwen3.8 (H16 shot 4: a rope handle taken for a
    kettlebell); an object the plan lists but SAM 3 did not find is replaced without an outline; a shot with nothing to replace IS the
    source shot (no render)
 3. after the renders SAM 3 looks for the new objects; one that does not sit on a replaced object is an extra copy -> the source shot is
    pasted back there, feathered (H3 keeps the rest of the frame aligned with the source, so the patch blends)
Env: R3_TAG (output folder suffix), R3_NOGUARD=1 (skip 3), SRC_OVERRIDE as in redoutline2, R3_USE=1 (a how-it-is-held sentence per object),
R3_FILL=1 (flat grey inside the outlines), R3_SHEET=1 (user 2026-10-25: "add a composite image for the object made up of several views,
for example the cheerleader's traffic cone used as a megaphone" - each new object's picture becomes a 4-view sheet made by Qwen-Image edit from
the object picture: two plain views + two as it is held / placed in this video, the views planned by Qwen3.8 from the frame where the original
is biggest), R3_SPLIT=1 (a shot with 2+ objects is rendered once per object - its own picture and outline only -
and the renders are joined by where each object is: the swaps cannot mix, H22 bottle -> paddle, T24 tortilla on the ball, T12 peach in the pan).
-> data/abtest/redline3_<code><tag>/result.mp4 (SAM 3 instances of the source reused from redline2_<code>/obj when they are there)"""
import copy, json, os, re, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]; sys.argv = ["x"]
import cv2, numpy as np
from server import pipeline as P, llm, h3, comfy, tools
from abtest import FakeJob, OUT

COLORS = [((0, 0, 255), "red"), ((255, 90, 0), "blue"), ((0, 200, 0), "green"), ((0, 220, 255), "yellow")]   # BGR
PICK = ('You look at one frame of a video edit. The edit replaces: "{what}". The numbered boxes are every {word} that was found. Which numbers '
        'are the thing the edit means - the one the person holds, carries, uses or owns in the edit? Other people\'s things of the same kind (cups '
        'standing on a table in front of other people, bags on other chairs) are NOT meant - but when the edit names the thing in the plural or '
        'says both / all / every (both rackets, the pom-poms, the billiard balls, the records), every one that is part of what the people do in the '
        'shot is meant - held, used, in play on the table or the decks, rolling or flying - but NOT spare ones stored in the background (dumbbells '
        'on a rack, bottles on a shelf) and not ones that clearly belong to other people. '
        'Reply with JSON only: {{"ids": [numbers], "why": "short"}}')
PICK_PERSON = ('You look at one frame of a video edit. The edit replaces: "{what}". The numbered boxes are the people that were found. Which '
               'number is that person (by the clothes and place the description gives)? Usually one number; an empty list if that person is not in '
               'this frame. Reply with JSON only: {{"ids": [numbers], "why": "short"}}')
PICK_ALL = ('You look at one frame of a video edit. The edit replaces "{what}" - EVERY one of them that is part of what the people do in the shot '
            '(held, used, in play on the table or the decks, rolling or flying). The numbered boxes are every {word} that was found. Which numbers must '
            'NOT be replaced: spare ones stored in the background (dumbbells on a rack, bottles on a shelf, cues in a wall rack), ones that clearly belong '
            'to other people, or boxes that do not show a {word} at all? Usually none. Reply with JSON only: {{"ids": [numbers], "why": "short"}}')
PICK_N = ('You look at one frame of a video edit. The edit replaces: "{what}" - exactly {n} of them. The numbered boxes are every {word} that '
          'was found. Which numbers are those {n}: the ones the person the edit is about holds, carries or uses, or that stand on the thing they '
          'carry? Ones in front of other people, on other tables, and spare ones in the background are NOT meant. Give at most {n} numbers. Reply '
          'with JSON only: {{"ids": [numbers], "why": "short"}}')
COUNT_W = {"both": 2, "two": 2, "pair": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def count_of(t):
    """the number of things an edit names: 'both wine glasses' -> 2 (F04: 'both' made every one of 14 glasses in the restaurant a pick)"""
    m = re.search(r"\b(both|two|pair|three|four|five|six|\d)\b", str(t or ""), re.I)
    if not m:
        return None
    w = m.group(1).lower()
    return int(w) if w.isdigit() else COUNT_W[w]


CHECK = ('You look at a close-up from one frame of a video. Is the thing inside the yellow box a {word} or a part of one - possibly seen end-on, '
         'from an unusual angle, blurred by motion or partly hidden? Answer no only when it is clearly something else: a wall, a shadow, a hand, a '
         'piece of clothing, a different kind of object (F14: a sword pointed at the camera looked like a barrel and was wrongly dropped). '
         'Reply with JSON only: {{"yes": true or false, "what": "what the thing in the box is, 2-6 words"}}')
CHECK2 = ('You look at a close-up from one frame of a video. Is there a {word} - or a visible part of one (an edge, rim, handle, wheel, blade, tip) - '
          'inside the yellow box? The box may also hold other things (hands, people, other objects, the floor): that does not matter, only whether '
          'some of a real {word} is inside it, possibly seen end-on, from an unusual angle, blurred by motion or partly hidden. A picture, painting '
          'or print of a {word} on a wall, poster or screen does not count. Answer no when there is no real {word} in the box at all - just a wall, '
          'a shadow, a hand, a piece of clothing, the background or a different kind of object. '
          'Reply with JSON only: {{"yes": true or false, "what": "the {word} or what is there instead, 2-6 words"}}')
CHECK3 = ('You look at a close-up from one frame of a video. Is the thing outlined in yellow (shown brighter than the rest) a {word} or a part '
          'of one - possibly seen end-on, from an unusual angle, blurred by motion or partly hidden? Answer no only when the outlined thing is '
          'clearly something else: a wall, a shadow, a hand, a piece of clothing, the background or a different kind of object. A picture, '
          'painting or print of a {word} on a wall, poster or screen does not count. '
          'Reply with JSON only: {{"yes": true or false, "what": "what the outlined thing is, 2-6 words"}}')
CHECK_MODE = os.environ.get("R3_CHECK", "4")   # (default 4 since 2026-10-27: 0 wrong yes on the 3-way A/B, the wrong no's are caught by autofix)
# 1: the box, 'is THE thing in it a X'; 2: the box, 'is there an X in it'; 3: the thing's own SAM 3 outline; 4: 2 and 3
# (2026-10-27: with a box, 'is THE thing in the box a wheelbarrow' named the shovel in front of a wheelbarrow's rim; 'is there one in it' passed big
#  boxes on a fence or a window that only happened to hold the racket / the punching bag)
NAMES = ('For each numbered description of an object give (a) "name": what it is in 1-4 words, as the description has it (e.g. "string of '
         'sausages", "pepperoni pizza", "red gift box") and (b) "detect": the singular everyday noun an object detector would search for, 1-3 words '
         '(e.g. "sausage", "pizza", "gift box", "ice cream cone"). Reply with JSON only: {"items": [{"name": "...", "detect": "..."}, ...]}')
USE = ('You see one frame of a video shot (the {old} has a coloured outline) and a picture of a new object. In the edited video the outlined '
       '{old} becomes the {new} from the picture, doing the same job in the same hands or place. Write ONE short sentence for a video model saying '
       'how the {new} is held or placed and which way it faces so that the person uses it the natural way for what they do in this frame: which '
       'part of the {new} is at their hand, mouth or body, and which part points where. Name ONLY the {new} and the person\'s body parts - no other '
       'object, no comparison ("like a ..."), not the {old}. Reply with JSON only: {{"sentence": "..."}}')
jid = args[0]; t_all = time.time(); TAG = os.environ.get("R3_TAG", "")
from server.config import DATA as _CFG_DATA
DATA_ = os.environ.get("R3_DATA") or _CFG_DATA   # the app's data dir (a test instance runs on a copy)
if os.environ.get("R3_OUT"):
    OUT = os.environ["R3_OUT"]; os.makedirs(OUT, exist_ok=True)
job0 = json.load(open(f"{DATA_}/jobs/{jid}/job.json")); code = (job0.get("project") or {}).get("code") or jid
root = f"{OUT}/redline3_{code}{TAG}"; os.makedirs(root, exist_ok=True)
job = copy.deepcopy(job0); job["id"] = f"red3-{code}"; fj = FakeJob(root, job); spec = job["spec"]; spec.pop("_corrections", None)
for s_ in spec.get("subjects", []):   # stored jobs carry the picture numbers as text ('1')
    if isinstance(s_.get("picture"), str) and s_["picture"].strip().isdigit():
        s_["picture"] = int(s_["picture"])
if os.environ.get("R3_QUALITY") != "standard":   # batches render at 480p; the app's 'standard' keeps its own canvas
    job["canvas"] = list(P.canvas(*job["canvas"], "fast")[0])
W, H = job["canvas"]; jd = f"{DATA_}/jobs/{jid}"
objs = [s for s in spec["subjects"] if s.get("kind") != "location" and isinstance(s.get("picture"), int)]   # everything that is replaced
PERSON = r"\b(man|woman|person|boy|girl|child|kid|guy|lady|dancer|rider|people|couple|skater|player)\b"
GENERIC = (r"(?:ball|piece|pieces|slice|slices|stack|stacks|pile|piles|bunch|pair|bag|bags|cup|cups|glass|glasses|bottle|bottles|box|boxes|plate|bowl|"
           r"loaf|sheet|roll|jar|can|cans|carton|tray|lump|chunk|clump|bundle|string|strand|coil|handful|scoop|bar|block|wedge|stick|sticks)")
def noun(t):   # the leading noun phrase: "the red sport motorcycle the rider is on ..." -> "red sport motorcycle"
    """2026-10-26: brackets dropped (F28 'every chess piece (both the black and the light wooden ones) on the board' went to SAM 3 whole and
    found one box round the board), and a generic head keeps its 'of ...' (F08 'the round ball of pizza dough' -> 'round ball' found an olive)"""
    t = re.sub(r"\([^)]*\)", " ", re.sub(r"<[^>]+>", "", t or "object"))
    t = re.sub(r"\s+", " ", re.sub(r",\s*", " ", t)).strip()
    # 'black and white soccer ball' is ONE describer (N10 searched for 'black'; F14 for 'brass' of 'long brass and dark wood telescope'): two
    # colours / materials around 'and' are joined before the stop at 'and' ('pink gymnastic ribbon and its stick' still stops)
    t = re.sub(rf"\b({ADJ})\s+(and|or)\s+(?=(?:{ADJ})\b)", lambda m_: f"{m_.group(1)}-{m_.group(2)}-", t, flags=re.I)
    stop = (r"(?:the|a|an|that|which|who|in|from|on|with|held|he|she|they|her|his|at|into|onto|under|over|near|by|for|to|being|is|are|as|"
            r"while|and|or|plus|its|their|including|except|excluding|like|such|apart|besides|inside|outside|within|behind|beside|beneath|"
            r"below|above|around|across|through|along|against|among|between|toward|towards|upon)")   # M03: 'the oranges inside the vendor's crate'   # K05: 'every red apple, including the one on the ...'   # 'pink gymnastic ribbon and its stick' -> 'pink gymnastic ribbon' (G10 searched for 'its stick')
    m = re.match(r"^(?:(?:the|a|an|her|his|their|both|two|three|four|all|every|each|its|some|several|many)\s+)*([\w\s'-]+?)(?=\s+(?:of|" + stop[3:] +
                 r"\b|[,.;:]|$)", t, re.I)
    head = (m.group(1).strip() if m else t) or "object"
    rest = t[m.end():] if m else ""
    if re.search(r"\b" + GENERIC + r"$", head, re.I) and re.match(r"\s+of\s+", rest, re.I):
        m2 = re.match(r"\s+of\s+(?:(?:the|a|an|her|his|their)\s+)?([\w\s'-]+?)(?=\s+" + stop + r"\b|[,.;:]|$)", rest, re.I)
        if m2:
            head = f"{head} of {m2.group(1).strip()}"
    return head
ADJ = (r"white|black|silver|grey|gray|red|blue|green|yellow|orange|pink|purple|brown|golden|gold|chrome|clear|dark|light|bright|small|"
       r"large|big|tiny|giant|long|short|round|square|flat|thin|thick|old|new|shiny|glossy|matte|wooden|metal|metallic|steel|stainless|plastic|"
       r"glass|leather|paper|cardboard|ceramic|rubber|straw|woven|plush|vintage|retro|classic|single|empty|full|brass|wood|copper|iron|bronze|"
       r"marble|stone|cotton|wool|silk|denim|velvet|crystal|porcelain|tin|aluminium|aluminum|bamboo|felt|canvas|beige|cream|navy|turquoise|teal|"
       r"maroon|violet|lilac|tan|khaki|olive|rainbow|striped|spotted|checkered|colorful|colourful")
def core(word):
    """the thing's plain name for a close-up check: colours, materials and hyphenated describers dropped (F20: 'white-headed snare drum' made
    Qwen reject the real, silver-sparkle snare)"""
    toks = str(word).split(); head, rest = (toks, []) if "of" not in toks else (toks[:toks.index("of")], toks[toks.index("of"):])
    head = [t for t in head if not re.fullmatch(r"its|his|her|their|my|your|our|and|or", t, re.I)] or head
    keep = [t for t in head[:-1] if not re.fullmatch(ADJ, t, re.I) and "-" not in t] + head[-1:]   # never the head noun itself (2026-10-27:
                                                                                                    # 'martini glass' -> 'martini', 'toilet paper' -> 'toilet')
    if rest and all(re.fullmatch(r"pair|pairs|set|sets|stack|stacks|pile|piles|bunch|bundle|handful|group|couple|lot", t, re.I) for t in keep):
        return core(" ".join(rest[1:]))   # 'pair of white opera gloves' -> 'opera gloves'; a SHAPE stays ('ball of yarn' - G06: 'yarn' found the knitted fabric)
    return " ".join(keep[-2:] + rest)


words = ["person" if s.get("kind") != "object" and re.search(PERSON, str(s.get("replaces") or ""), re.I) else noun(s.get("replaces"))
         for s in objs]
SAMW = [w if w == "person" else core(w) for w in words]   # what SAM 3 searches for: the thing's plain name, not its colour (G04)
for i_ in range(len(SAMW)):
    if SAMW.count(SAMW[i_]) > 1:   # two objects with the same plain name ('red ball', 'blue ball'): keep their full names
        SAMW = [words[k_] if SAMW[k_] == SAMW[i_] else SAMW[k_] for k_ in range(len(SAMW))]
seed0 = int(job.get("seed") or 1234); shots = []
SRC = os.environ.get("SRC_OVERRIDE")
SF = P.read_frames(SRC) if SRC else None
for sg in job["segments"]:
    if SF is not None:
        real = SF[sg["start"]:min(len(SF), sg.get("end", sg["start"] + sg["length"]))]
        S = [P.cover_resize(f, W, H) for f in (real + [real[-1]] * max(0, sg["length"] - len(real)))][:sg["length"]]
    else:
        S = [P.cover_resize(f, W, H) for f in P.read_frames(f"{jd}/seg{sg['k']}_src.mp4")][:sg["length"]]
    w = f"{root}/seg{sg['k']}_src.mp4"; P.write_video(S, w, crf=12); shots.append(dict(k=sg["k"], sg=sg, S=S, win=w))
def plan_of(k):
    return next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), None) or {}
t0 = time.time(); od = f"{root}/obj"; old = f"{OUT}/redline2_{code}/obj"; tools.gpu_take(fj)   # no LLM, ComfyUI's models unloaded
if os.environ.get("R3_OBJ_FROM"):   # a second take reuses the first take's SAM 3 instances (best-of-2)
    old = os.environ["R3_OBJ_FROM"] if os.path.isabs(os.environ["R3_OBJ_FROM"]) else f"{OUT}/{os.environ['R3_OBJ_FROM']}/obj"
if not all(os.path.exists(f"{od}/{i}/objects.json") for i in range(len(shots))):
    if not SRC and all(os.path.exists(f"{old}/{i}/objects.json") for i in range(len(shots))) and \
            {o["concept"] for i in range(len(shots)) for o in json.load(open(f"{old}/{i}/objects.json"))} <= set(words) | set(SAMW):
        shutil.rmtree(od, ignore_errors=True); shutil.copytree(old, od); print(f"== instances reused from {old}", flush=True)
    else:
        many = any(re.search(r"\b(both|all|every|each)\b", str(s_.get("replaces") or ""), re.I) or (w_.endswith("s") and not w_.endswith("ss"))
                   for s_, w_ in zip(objs, words))
        tools.run_proc(fj, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_multi.py", od, "1", "|".join(dict.fromkeys(SAMW))] + [sh["win"] for sh in shots],
                       0.5, 0.5, "instances", env={"PER_OBJ": "1", "MAX_OBJ": "32" if many else "16"})
        print(f"== instances of {words} tracked in {time.time() - t0:.0f} s", flush=True)
# Qwen3.8: the new objects' short names (for the shot lines and the extra-copy search), the meant instances per shot, close-up checks
t0 = time.time()
try:
    r = llm.ask([{"type": "text", "text": "\n".join(f"{i}. {re.sub(r'<[^>]+>', '', str(s.get('description') or ''))[:300]}" for i, s in enumerate(objs, 1))}],
                system=NAMES, max_tokens=200, temperature=0.0)
    items = [x for x in (r.get("items") or []) if isinstance(x, dict)]
except Exception as e:
    items = []; print(f"   names failed: {str(e)[:80]}", flush=True)
fallback = lambda i: noun(objs[i].get("description"))
full = [str(items[i].get("name") or "").strip().lower()[:50] if i < len(items) else "" for i in range(len(objs))]
names = [str(items[i].get("detect") or "").strip().lower()[:40] if i < len(items) else "" for i in range(len(objs))]   # SAM 3 concept
full = [full[i] or names[i] or fallback(i) for i in range(len(objs))]; names = [names[i] or full[i] for i in range(len(objs))]
print(f"   new objects: {[(w, f, n) for w, f, n in zip(words, full, names)]}", flush=True)
def cand_mask(vi, name, f, shape):
    """a SAM 3 instance's mask in shot vi at frame f, at the size of that shot's frames (None if missing)"""
    M = P.read_frames(f"{od}/{vi}/{name}.mp4", [f]) if os.path.exists(f"{od}/{vi}/{name}.mp4") else []
    return (cv2.resize(cv2.cvtColor(M[0], cv2.COLOR_BGR2GRAY), (shape[1], shape[0])) > 127) if M else None


NOT_REAL = r"\b(reflection|reflected|picture|painting|painted|print|printed|poster|photo|image|drawing|drawn|shadow|mural|screen|toy|sticker|logo|cartoon|icon|statue)\b"


def says_yes(r_, word):
    """a 'no' that names the thing itself ('part of a drumstick', 'drum head' for a snare drum) is a yes - unless it is a reflection, a
    picture on a wall and the like (the 3-way A/B, 2026-10-27)"""
    w_ = str(r_.get("what") or "").lower(); head = core(word).lower().split()[-1].rstrip("s")
    if head and re.search(rf"\b(no|not|without|never|isn't)\s+(?:an?\s+|the\s+|any\s+|visible\s+)?(?:\w+\s+){{0,2}}{re.escape(head)}s?\b", w_):
        return False   # its own words deny it (L07: 'yes - man holding guitar, no case' made the guitar the guitar case's pick)
    if r_.get("yes"):
        return True
    return bool(head) and re.search(rf"\b{re.escape(head)}s?\b", w_) is not None and not re.search(NOT_REAL, w_)


def crop_check(img, box, word, mask=None):
    x0, y0, x1, y1 = box; bw, bh = x1 - x0, y1 - y0; m = max(bw, bh) // 2 + 24
    cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(img.shape[1], x1 + m), min(img.shape[0], y1 + m)
    def shot(kind):
        c = img[cy0:cy1, cx0:cx1].copy()
        if kind == "mask":   # the thing itself: its outline in yellow, everything else dimmed
            mk = (np.asarray(mask)[cy0:cy1, cx0:cx1] > 0).astype(np.uint8)
            c = np.where(mk[..., None] > 0, c, (c.astype(np.float32) * 0.45).astype(np.uint8))
            cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(c, cs, -1, (0, 255, 255), 2)
        else:
            cv2.rectangle(c, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), (0, 255, 255), 2)
        sc = 448 / max(c.shape[:2]); c = cv2.resize(c, (int(c.shape[1] * sc), int(c.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
        os.makedirs(f"{root}/checks", exist_ok=True); p_ = f"{root}/checks/{time.time_ns()}{'_m' if kind == 'mask' else ''}.png"; cv2.imwrite(p_, c)
        return p_
    use_mask = CHECK_MODE in ("3", "4") and mask is not None and bool(np.asarray(mask)[cy0:cy1, cx0:cx1].any())
    ask = lambda sysp, path: llm.ask([llm.img_part(path, 448)], system=sysp.format(word=core(word)), max_tokens=80, temperature=0.0)
    yn = lambda r_: "yes" if r_.get("yes") else "NO"
    try:
        if CHECK_MODE == "4":   # both must say yes: the box shows the surroundings (a ball PAINTED on a mural, a reflection), the outline shows
            pb = shot("box"); r2 = ask(CHECK2, pb)   # exactly which thing it is (a window frame in a big box) - a wrong yes puts the new
            r3 = ask(CHECK3, shot("mask")) if use_mask and (r2.get("yes") or os.environ.get("R3_CHECK_AB") == "1") else {"yes": True}
            if os.environ.get("R3_CHECK_AB") == "1":   # object in the wrong place and nothing later catches it; a wrong no is caught by autofix
                r1 = ask(CHECK, pb)
                print(f"   CHECK-AB3 {core(word)} | {pb} | old {yn(r1)} ({str(r1.get('what'))[:40]}) | box {yn(r2)} ({str(r2.get('what'))[:40]}) | "
                      f"outline {yn(r3) if use_mask else '-'} ({str(r3.get('what'))[:40]})", flush=True)
            ok = says_yes(r2, word) and says_yes(r3, word)
            return ok, str((r2 if not r2.get("yes") else r3).get("what") or r2.get("what") or "")[:60]
        sysp = CHECK3 if use_mask else (CHECK2 if CHECK_MODE == "2" else CHECK)
        p = shot("mask" if use_mask else "box")
        r = ask(sysp, p)
        if os.environ.get("R3_CHECK_AB") == "1" and sysp is not CHECK:   # an A/B: the old box question logged beside the chosen one
            po = shot("box"); ro = ask(CHECK, po)
            print(f"   CHECK-AB {core(word)} | {p} | old {yn(ro)} ({str(ro.get('what'))[:40]}) | new {yn(r)} ({str(r.get('what'))[:40]})", flush=True)
        return bool(r.get("yes")), str(r.get("what") or "")[:60]
    except Exception as e:
        return True, f"check failed: {str(e)[:40]}"
def as_source(text):
    """the shot plan's text as the SOURCE shows it: tags and new-object names become the old objects' names (G01: 'the lit sparkler in his
    right hand' made the pick reject the real baton); placeholders first, so a short new name ('gloves') never matches inside an old one"""
    t = str(text or "")
    for oi_, s_ in enumerate(objs):
        if s_.get("tag"):
            t = t.replace(s_["tag"], f"\x00{oi_}\x00")
    for oi_ in range(len(objs)):
        for nm in sorted({full[oi_], names[oi_]}, key=len, reverse=True):
            if nm and len(nm) > 2:
                t = re.sub(r"(?i)\b(?:the |a |an )?" + re.escape(nm) + r"s?\b", f"\x00{oi_}\x00", t)
    t = re.sub(r"(\x00(\d+)\x00)(?:\s*\(?\s*(?:the |a |an )?\x00\2\x00\s*\)?)+", r"\1", t)   # '<Subject 1> (the lit sparkler)' -> once
    t = re.sub(r"\x00(\d+)\x00", lambda m: f"the {words[int(m.group(1))]}", t)
    return P.rn_plain(t)


UNPICKED = {}   # (shot, object) -> other real instances of the old object in the shot (they stay; otherwise the 'stays' line only names it again)
for vi, sh in enumerate(shots):
    info = json.load(open(f"{od}/{vi}/objects.json")) if os.path.exists(f"{od}/{vi}/objects.json") else []
    plan = plan_of(sh["k"]); present = plan.get("present")
    sh["pick"] = {}; sh["rejected"] = set(); sh["not_meant"] = set(); sh["false_only"] = set()
    for oi, (s_, word) in enumerate(zip(objs, words)):
        cands = [o for o in info if o["concept"] in (SAMW[oi], word) and len(o["frames"]) >= 3]
        if not cands:
            continue
        counts = {}
        for o in cands:
            for fr in o["frames"]:
                counts[fr[0]] = counts.get(fr[0], 0) + 1
        fi = max(counts, key=lambda f_: (counts[f_], -f_))
        img = sh["S"][fi].copy(); boxes = []
        for n, o in enumerate(cands, 1):
            b = min(o["frames"], key=lambda fr: abs(fr[0] - fi))
            x0, y0, x1, y1 = [int(v * W / sh["S"][0].shape[1]) for v in b[1:5]]; boxes.append((x0, y0, x1, y1))
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2); cv2.putText(img, str(n), (x0 + 2, max(14, y0 - 4)), 0, 0.6, (0, 255, 255), 2)
        bp = f"{root}/seg{sh['k']}_{oi}_boxes.png"; cv2.imwrite(bp, img)
        n_named = count_of(s_.get("replaces")) if word != "person" else None
        counted = bool(n_named and len(cands) > n_named)   # 'both wine glasses' with 14 found: pick those two, not every one
        plural = not counted and word != "person" and len(cands) > 1 and (re.search(r"\b(both|all|every|each)\b", str(s_.get("replaces") or ""), re.I) or
                                                                          (word.endswith("s") and not word.endswith("ss")))
        try:   # a plural edit: every instance unless Qwen names it as stored / someone else's (T22 got 6 of 16 balls when asked which ARE meant)
            sysp = PICK_N if counted else (PICK_ALL if plural else (PICK_PERSON if word == "person" else PICK))
            sysp = sysp.format(what=s_.get("replaces"), word=word, n=n_named or 0)
            for att_ in (0, 1):   # a reply that is not JSON (M03 retry: 'Expecting value' - the shot got no picks at all): once more, more room
                try:
                    r = llm.ask([{"type": "text", "text": f"This shot (as the source video shows it): {as_source(plan.get('shots', ''))[:300]}"},
                                 llm.img_part(bp, 768)], system=sysp + ("" if not att_ else " Reply with the JSON object only, nothing before or after it."),
                                max_tokens=150 if not att_ else 400, temperature=0.0 if not att_ else 0.2)
                    break
                except Exception:
                    if att_:
                        raise
            ids = [int(x) for x in (r.get("ids") or []) if str(x).isdigit() and 1 <= int(x) <= len(cands)]
            if counted:
                ids = ids[:n_named]
            if plural:
                ids = [i for i in range(1, len(cands) + 1) if i not in ids]
                if not ids:   # every one found is a stored spare / someone else's: not part of this shot's edit (N34: the whole rack became chickens)
                    sh["not_meant"].add(oi)
            why = ("all but " + str(sorted(set(range(1, len(cands) + 1)) - set(ids))) + ": " if plural else "") + str(r.get("why") or "")[:120]
        except Exception as e:
            ids, why = [], str(e)[:80]
        listed = present is None or s_.get("tag") in (present or [])
        forced = next((x for x in (os.environ.get("R3_FORCE") or "").split(",") if x.count(":") >= 1 and x.split(":")[0].strip() == str(sh["k"])
                       and x.split(":")[1].strip() == str(oi)), None)
        if forced:   # autofix.py: the first take kept the old object here (the close-up check had dropped the pick) - take the named instances
            nm_ = [x for x in (forced.split(":")[2] if forced.count(":") >= 2 else "").split("|") if x]
            ids = [i for i, c in enumerate(cands, 1) if c["name"] in nm_] or ids or [max(range(1, len(cands) + 1), key=lambda i: len(cands[i - 1]["frames"]))]
            why = "forced by autofix - " + why; sh["not_meant"].discard(oi)
        held = re.search(r"\b(hold|holds|holding|held|grip|grips|gripping|catch|catches|catching|swing|swings|swinging|carry|carries|carrying|"
                         r"in (his|her|their) hands?|in play|being hit|hits?|kick|kicks|kicked|bounc\w*|roll|rolls|rolling)\b", why, re.I)
        # ('in play' since 2026-10-27: the app test's 5 s cut of K08 - the plan of the man's 1.2 s shot did not list the ball, and the ball he
        #  returned stayed a tennis ball)
        owned = re.search(r"\b\w+'s\b|\b(her|his|their)\b", str(s_.get("replaces") or ""), re.I)   # K08: 'the woman's tennis racket' - in the
        # man's shot (the plan did not list it) the racket HE held was picked as 'held by the person' and became a pan too
        if ids and not forced and not listed and word != "person" and not plural and (not held or owned) and os.environ.get("R3_TRUST_PLAN", "1") == "1":
            # a single thing the plan does not show in this shot: the picks are other copies of it (H20: five glasses on the counter became
            # light bulbs; T12: a second tennis ball made a peach in the pan) - trust the plan
            print(f"   seg{sh['k']} {word}: not in the shot plan, named in the singular and not held - {len(ids)} pick(s) ignored", flush=True)
            sh["false_only"].add(oi); ids = []
        if ids and not forced and not listed and word != "person":   # the plan does not show this object in the shot: each box needs a close-up yes
            keep = []
            for i in ids:
                ok, what = crop_check(sh["S"][fi], boxes[i - 1], word,
                                      mask=cand_mask(vi, cands[i - 1]["name"], fi, sh["S"][fi].shape) if CHECK_MODE in ("3", "4") else None)
                print(f"   seg{sh['k']} {word} #{i} not in the shot plan - close-up: {'yes' if ok else 'no'} ({what})", flush=True)
                if ok:
                    keep.append(i)
                else:
                    sh["rejected"].add(cands[i - 1]["name"])
            ids = keep
        elif ids and not forced and word != "person" and len(ids) <= 6:   # every pick gets a close-up at its biggest frame; a second look before dropping it
            keep = []                                       # (T14 shot 4: the 'record' track sat on the wall and the wall got a pizza)
            for i in ids:
                fr_ = sorted(cands[i - 1]["frames"], key=lambda f_: -f_[5])
                looks = [fr_[0]] + ([fr_[len(fr_) // 2]] if len(fr_) > 2 else [])
                ok = False
                for f_ in looks:
                    bx = tuple(int(v * W / sh["S"][0].shape[1]) for v in f_[1:5])
                    fF = min(f_[0], len(sh["S"]) - 1)
                    ok, what = crop_check(sh["S"][fF], bx, word,
                                          mask=cand_mask(vi, cands[i - 1]["name"], f_[0], sh["S"][fF].shape) if CHECK_MODE in ("3", "4") else None)
                    if ok:
                        break
                if ok:
                    keep.append(i)
                else:
                    sh["rejected"].add(cands[i - 1]["name"]); print(f"   seg{sh['k']} {word} #{i} dropped - close-ups say: {what}", flush=True)
            if not keep:   # SAM 3 found only things that are not it: the object is not in this shot (not even 'listed' by the plan)
                sh["false_only"].add(oi)
            ids = keep
        sh["pick"][oi] = [cands[i - 1]["name"] for i in ids]
        UNPICKED[(sh["k"], oi)] = len([c for c in cands if c["name"] not in sh["pick"][oi] and c["name"] not in sh["rejected"]])
        print(f"   seg{sh['k']} {word}: {len(cands)} found, picked {ids} ({why})", flush=True)
    for pair in (os.environ.get("R3_NOOUTLINE") or "").split(","):   # "k:oi" - still swapped in shot k, but from the text alone, no outline
        if pair.strip() and pair.split(":")[0].strip() == str(sh["k"]):   # (G04: every pick in the cheer shot was a ball painted on the wall)
            sh["pick"][int(pair.split(":")[1])] = []
    for pair in (os.environ.get("R3_SKIP") or "").split(","):   # R3_SKIP right after the picks: no outline, no sheet, no shot-text fix for it
        if pair.strip() and pair.split(":")[0].strip() == str(sh["k"]):   # (T08_rt: FIXTXT had written the teddy into the close-up's text)
            oi_ = int(pair.split(":")[1]); sh["pick"][oi_] = []; sh["not_meant"].add(oi_)
if os.environ.get("R3_USE") == "1":   # how each outlined new object is held, per shot, from the shot's outlined frame
    for vi, sh in enumerate(shots):
        sh["use"] = {}
        for oi, nm in sh["pick"].items():
            if not nm or words[oi] == "person":
                continue
            Ms = [P.read_frames(f"{od}/{vi}/{x}.mp4") for x in nm]; best, fa = -1, 0
            for j in range(0, len(sh["S"]), 4):
                a = sum(int((cv2.cvtColor(M[min(j, len(M) - 1)], cv2.COLOR_BGR2GRAY) > 127).sum()) for M in Ms)
                if a > best:
                    best, fa = a, j
            img = sh["S"][fa].copy(); m = np.zeros(img.shape[:2], np.uint8)
            for M in Ms:
                m |= (cv2.resize(cv2.cvtColor(M[min(fa, len(M) - 1)], cv2.COLOR_BGR2GRAY), (W, H)) > 127).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cs, -1, (0, 0, 255), 3, cv2.LINE_AA)
            fp = f"{root}/seg{sh['k']}_{oi}_use.png"; cv2.imwrite(fp, img)
            try:
                r = llm.ask([{"type": "text", "text": "The frame:"}, llm.img_part(fp, 768), {"type": "text", "text": f"The new object ({full[oi]}):"},
                             llm.img_part(job["images"][objs[oi]["picture"] - 1], 384)],
                            system=USE.format(old=words[oi], new=full[oi]), max_tokens=120, temperature=0.0)
                sh["use"][oi] = re.sub(r"\s+", " ", str(r.get("sentence") or "")).strip()[:300]
            except Exception as e:
                print(f"   seg{sh['k']} use sentence failed: {str(e)[:60]}", flush=True)
            print(f"   seg{sh['k']} {full[oi]}: {sh['use'].get(oi)}", flush=True)
SHEET = os.environ.get("R3_SHEET") == "1"
VIEWS = ('You see one frame of a video (the {old} has a red outline) and a picture of a new object. In the edited video the {new} takes the '
         'place of the outlined {old} and does the same job in the same hands or place. FIRST match the parts: which part of the {new} goes where '
         'each part of the {old} is in this frame - the end at the person\'s mouth, face or hand, the end that points away from them, the side that '
         'faces the camera (H14: a traffic cone used as a megaphone has its NARROW TIP at the lips, where the mouthpiece is, and its WIDE BASE pointing '
         'away, where the bell is) - and the hand holds the {new} at the part that matches where it holds the {old} (the megaphone\'s handle sits '
         'under the mouthpiece, so the hand holds the cone near its narrow tip, not by its base). Say which side of the {old} faces the camera in this '
         'frame (its front, its back, a side, its top or underside) and so which side of the {new} faces the camera (N20: the camera sees the back of '
         'the laptop lid, so it sees the BACK of the typewriter - the paper roller - and the keys face the person, away from the camera). THEN plan a reference sheet of the {new} for a video model: exactly four views of the same single {new} on plain '
         'white. Views 1 and 2: the {new} alone, from two different sides. Views 3 and 4: the {new} exactly as your match places it in this frame - '
         'held by a hand (only a hand and forearm, no body, no face) if a person holds it, or standing or lying as it is placed here - from the side '
         'and from this frame\'s camera angle, where view 4 shows exactly the side your camera answer names (the back of the object when its back '
         'faces the camera); say which part points where in words like "the narrow tip toward the mouth side". ALSO write "hold": '
         'one sentence for the video model saying how the {new} is held and which of its parts is where in this video. In the views and in "hold" '
         'name only the {new}, the hand, body parts and directions - never the {old} or any other object. Reply with JSON only: '
         '{{"match": "...", "camera_side": "...", "hold": "...", "views": ["...", "...", "...", "..."]}}')
sheet_views = {}; sheet_hold = {}
if SHEET:   # where each new object is biggest in the source -> the four views of its sheet
    for oi in range(len(objs)):
        if words[oi] == "person":
            continue
        best = None
        for vi, sh in enumerate(shots):
            for nm in sh["pick"].get(oi, []):
                M = P.read_frames(f"{od}/{vi}/{nm}.mp4")
                for j in range(0, len(M), 4):
                    a = int((cv2.cvtColor(M[j], cv2.COLOR_BGR2GRAY) > 127).sum())
                    if best is None or a > best[0]:
                        best = (a, vi, j)
        if not best:
            continue
        _, vi, j = best; sh = shots[vi]; img = sh["S"][min(j, len(sh["S"]) - 1)].copy(); m = np.zeros((H, W), np.uint8)
        for nm in sh["pick"][oi]:
            M = P.read_frames(f"{od}/{vi}/{nm}.mp4"); m |= (cv2.resize(cv2.cvtColor(M[min(j, len(M) - 1)], cv2.COLOR_BGR2GRAY), (W, H)) > 127).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cs, -1, (0, 0, 255), 3, cv2.LINE_AA)
        fp = f"{root}/sheet_{oi}_frame.png"; cv2.imwrite(fp, img)
        try:
            r = llm.ask([{"type": "text", "text": "The frame:"}, llm.img_part(fp, 768), {"type": "text", "text": f"The new object ({full[oi]}):"},
                         llm.img_part(job["images"][objs[oi]["picture"] - 1], 384)], system=VIEWS.format(old=words[oi], new=full[oi]), max_tokens=400,
                        temperature=0.0)
            v = [re.sub(r"\s+", " ", str(x)).strip()[:420] for x in (r.get("views") or []) if str(x).strip()][:4]
            if len(v) >= 3:
                sheet_views[oi] = v
                hold = re.sub(r"\s+", " ", str(r.get("hold") or "")).strip()[:300]
                if hold and not re.search(rf"\b{re.escape(words[oi].split()[-1])}\b", hold, re.I):   # never the old object's name
                    sheet_hold[oi] = hold
                print(f"   match for {full[oi]}: {str(r.get('match'))[:300]} | camera side: {str(r.get('camera_side'))[:160]}", flush=True)
        except Exception as e:
            print(f"   sheet views for {full[oi]} failed: {str(e)[:60]}", flush=True)
        print(f"   sheet views for {full[oi]} (shot {sh['k']} frame {j}): {sheet_views.get(oi)}", flush=True)
SIDE = ('You see a crop of one video frame showing a {old}, and a picture of a new object. An image editor will replace the {old} in this crop '
        'with the {new}, doing the same job in the same hands or place, and the camera must see the {new} from the same side as it sees the {old} '
        'now (H14: the megaphone\'s wide bell faces the camera, so the cone\'s WIDE OPEN END faces the camera and its narrow tip is at her lips; '
        'N20: the camera sees the back of the laptop lid, so it sees the typewriter\'s plain BACK panel and the paper sticking up, its keys hidden on '
        'the far side facing him). Write "edit": two sentences for the editor - which side of the {new} faces the camera, said plainly in capitals '
        '(its FRONT / BACK / LEFT SIDE / TOP / UNDERSIDE / OPEN END faces the camera), which of its parts the camera sees and which are hidden on '
        'the far side, which way its parts point, and - only if a hand touches the {old} in the crop - where the hand holds it. Write "hold": the '
        'same in one sentence for a video model, naming '
        'only the {new}, body parts and directions - never the {old}. Also say "hand": true only if a hand or fingers touch the {old} in this crop, '
        'and "contents": what is inside, on or coming out of the {old} in this crop that is NOT part of it (a rabbit coming out of a hat, food in a pan, '
        'liquid pouring from it), or "" if nothing. Reply with JSON only: {{"edit": "...", "hold": "...", "hand": true or false, "contents": "..."}}')
shot_hold = {}
if os.environ.get("R3_SHEET") == "2":
    for vi, sh in enumerate(shots):
        sh["panel"] = {}
        for oi, nm in sh["pick"].items():
            if not nm or words[oi] == "person":
                continue
            Ms = [P.read_frames(f"{od}/{vi}/{x}.mp4") for x in nm]; cand = []
            for j in range(min(len(M) for M in Ms)):
                m = np.zeros((H, W), bool)
                for M in Ms:
                    m |= cv2.resize(cv2.cvtColor(M[j], cv2.COLOR_BGR2GRAY), (W, H)) > 127
                if m.sum() >= 200:
                    cand.append((int(m.sum()), j, m))
            cand.sort(key=lambda c: -c[0]); best = None
            for c in cand[:1] + [x for x in cand[1:] if all(abs(x[1] - y[1]) >= 12 for y in cand[:1])][:2]:   # the biggest, then two others
                ys, xs = np.nonzero(c[2]); box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                ok, what = crop_check(sh["S"][min(c[1], len(sh["S"]) - 1)], box, words[oi], mask=c[2])   # a drifted track must not become the panel (T14)
                print(f"   seg{sh['k']} panel frame {c[1]} for {words[oi]}: {'yes' if ok else 'no'} ({what})", flush=True)
                if ok:
                    best = c; break
            if not best:
                continue
            _, j, m = best; f = sh["S"][min(j, len(sh["S"]) - 1)]
            ys, xs = np.nonzero(m); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); bw, bh = x1 - x0, y1 - y0
            pad = int(0.45 * max(bw, bh)) + 20; side = max(bw, bh) + 2 * pad; cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            X0, Y0 = max(0, min(W - side, cx - side // 2)), max(0, min(H - side, cy - side // 2)); X1, Y1 = min(W, X0 + side), min(H, Y0 + side)
            crop = f[Y0:Y1, X0:X1]; sc = 768 / max(crop.shape[:2])
            crop = cv2.resize(crop, (int(crop.shape[1] * sc), int(crop.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
            cp = f"{root}/panel_s{sh['k']}_{oi}_crop.png"; cv2.imwrite(cp, crop)
            try:
                r = llm.ask([{"type": "text", "text": "The crop:"}, llm.img_part(cp, 640), {"type": "text", "text": f"The new object ({full[oi]}):"},
                             llm.img_part(job["images"][objs[oi]["picture"] - 1], 384)], system=SIDE.format(old=words[oi], new=full[oi]),
                            max_tokens=300, temperature=0.0)
                ed = re.sub(r"\s+", " ", str(r.get("edit") or "")).strip()[:500]; hd = re.sub(r"\s+", " ", str(r.get("hold") or "")).strip()[:300]
                hand_ = bool(r.get("hand")); cont_ = re.sub(r"\s+", " ", str(r.get("contents") or "")).strip()[:160]
            except Exception as e:
                ed, hd = "", ""; print(f"   seg{sh['k']} side words failed: {str(e)[:60]}", flush=True)
            if ed:
                sh["panel"][oi] = (cp, ed, hand_, cont_)
                if hd and not re.search(rf"\b{re.escape(words[oi].split()[-1])}\b", hd, re.I):
                    shot_hold[(sh["k"], oi)] = hd
            print(f"   seg{sh['k']} {full[oi]} seen as: {ed[:220]}", flush=True)
FIXTXT = ('Here is the description of one shot of an edited video, and how a new object must be held in it: "{hold}". Change ONLY the words that '
          'say how the {new} ({tag}) is held, turned, or which of its parts is where, so that they agree with that; if the description says nothing '
          'about it, leave it as it is. Keep every other word exactly. Reply with JSON only: {{"text": "..."}}')
for sh in shots:   # R3_SHEET: the plan's own words about the grip must not contradict the sheet
    txt = str(plan_of(sh["k"]).get("shots") or "")
    for oi in sh["pick"]:
        hold_ = shot_hold.get((sh["k"], oi)) or sheet_hold.get(oi)
        if not sh["pick"][oi] or not hold_ or not objs[oi].get("tag") or objs[oi]["tag"] not in txt:
            continue
        try:
            r = llm.ask([{"type": "text", "text": txt}], system=FIXTXT.format(hold=hold_, new=full[oi], tag=objs[oi]["tag"]), max_tokens=900,
                        temperature=0.0)
            t2 = str(r.get("text") or "").strip()
            if t2 and objs[oi]["tag"] in t2:
                if t2 != txt:
                    print(f"   seg{sh['k']} shot text corrected for {full[oi]}: {t2[:200]}", flush=True)
                txt = t2; sh["fixed_text"] = txt
        except Exception as e:
            print(f"   seg{sh['k']} shot text fix failed: {str(e)[:60]}", flush=True)
json.dump({str(sh["k"]): {str(oi): v for oi, v in sh["pick"].items()} for sh in shots}, open(f"{root}/picks.json", "w"), indent=1)
print(f"== picks in {time.time() - t0:.0f} s", flush=True)


for vi, sh in enumerate(shots):   # the objects of each shot: outlined, or listed by the plan
    plan = plan_of(sh["k"]); present = plan.get("present"); outlined = [oi for oi, nm in sh["pick"].items() if nm]
    if present is None:   # no list in the plan: the objects its shot text names, else the outlined ones
        listed = [oi for oi in range(len(objs)) if objs[oi].get("tag") and objs[oi]["tag"] in str(plan.get("shots") or "")] or outlined
    else:
        listed = [oi for oi in range(len(objs)) if objs[oi].get("tag") in present]
    sh["planned"] = sorted(set(listed))   # what the plan shows in this shot (autofix.py checks these even when the picks dropped them)
    listed = [oi for oi in listed if oi not in sh.get("not_meant", set()) and oi not in sh.get("false_only", set())]
    for pair in (os.environ.get("R3_SKIP") or "").split(","):   # "k:oi" - this object is not swapped in shot k (T08: only the backpack's straps show
        if pair.strip() and pair.split(":")[0].strip() == str(sh["k"]):   # in the close-up, and a whole teddy bear was drawn on his chest)
            oi_ = int(pair.split(":")[1]); listed = [oi for oi in listed if oi != oi_]; outlined = [oi for oi in outlined if oi != oi_]
            sh["pick"][oi_] = []
    sh["members"] = sorted(set(outlined) | set(listed)); sh["outlined"] = outlined
json.dump({"objs": [{"word": words[oi], "full": full[oi], "samw": SAMW[oi], "detect": names[oi], "tag": objs[oi].get("tag"), "picture": objs[oi].get("picture"),
                     "replaces": str(objs[oi].get("replaces") or "")}
                    for oi in range(len(objs))],
           "shots": {str(sh["k"]): {"vi": vi, "members": sh["members"], "outlined": sh["outlined"], "planned": sh.get("planned", []),
                                    "pick": {str(oi): v for oi, v in sh["pick"].items()}, "rejected": sorted(sh["rejected"]),
                                    "not_meant": sorted(sh["not_meant"]), "false_only": sorted(sh["false_only"])} for vi, sh in enumerate(shots)}},
          open(f"{root}/shots_meta.json", "w"), indent=1)   # each shot's pick decisions, for autofix.py
if os.environ.get("R3_PICKS_ONLY") == "1":   # a test of the pick step alone (and shots_meta.json for autofix.py): no sheets, no renders
    print(f"== picks only, in {time.time() - t0:.0f} s", flush=True); sys.exit(0)
SPLIT = os.environ.get("R3_SPLIT") == "1"
ALONE = ('You get the description of one shot of an edited video. Several objects are replaced in the edit, but this version of the shot keeps '
         'only ONE replacement: {one}. Rewrite the description so that it names only that new object; each other replaced object is described '
         'as the original thing it really is in the source video ({others}), without any <...> tag. Keep everything else - people, actions, '
         'camera, timing - word for word. Reply with JSON only: {{"text": "..."}}')
if SPLIT:
    for sh in shots:
        sh["alone"] = {}
        if len(sh["members"]) < 2 or any(words[oi] == "person" for oi in sh["members"]):
            continue
        txt = str(plan_of(sh["k"]).get("shots") or "")
        for oi in sh["members"]:
            one = f"the {words[oi]} becomes the {full[oi]} ({objs[oi].get('tag')})"
            oth = "; ".join(f"{objs[o].get('tag')} / the {full[o]} is really the {words[o]}" for o in sh["members"] if o != oi)
            try:
                r = llm.ask([{"type": "text", "text": txt}], system=ALONE.format(one=one, others=oth), max_tokens=900, temperature=0.0)
                sh["alone"][oi] = str(r.get("text") or "").strip() or None
            except Exception as e:
                sh["alone"][oi] = None; print(f"   seg{sh['k']} alone text failed: {str(e)[:60]}", flush=True)
            print(f"   seg{sh['k']} alone {full[oi]}: {str(sh['alone'][oi])[:150]}", flush=True)


def remap(t, tagmap, picmap):
    t = str(t or "")
    if tagmap:   # all tags in one pass (sequential replaces turned a renumbered '<Subject 1>' into the next mapping's target)
        pat = re.compile("|".join(re.escape(a) for a in sorted(tagmap, key=len, reverse=True)))
        t = pat.sub(lambda m: tagmap[m.group(0)], t)
    return re.sub(r"<Picture (\d+)>", lambda m: f"<Picture {picmap.get(int(m.group(1)), int(m.group(1)))}>", t)


def shot_spec(k, members, outlined, use=None, shots_text=None):
    """the spec for shot k naming ONLY its own objects (members), as <Subject 1..n> with <Picture 1..n>; outlined = the members with an outline"""
    plan = plan_of(k); sp = copy.deepcopy(spec)
    tagmap = {objs[oi]["tag"]: f"<Subject {i}>" for i, oi in enumerate(members, 1) if objs[oi].get("tag")}
    for oi in range(len(objs)):   # an object the shot does not swap keeps its real name in the shot text (R3_SKIP, false_only)
        if oi not in members and objs[oi].get("tag") and objs[oi]["tag"] not in tagmap:
            tagmap[objs[oi]["tag"]] = f"the {words[oi]}"
    picmap = {objs[oi]["picture"]: i for i, oi in enumerate(members, 1)}
    subs = []
    for i, oi in enumerate(members, 1):
        s = copy.deepcopy(objs[oi]); s["tag"] = f"<Subject {i}>"; s["picture"] = i
        s["description"] = remap(s.get("description"), tagmap, picmap); s["replaces"] = remap(s.get("replaces"), tagmap, picmap); subs.append(s)
    sp["subjects"] = subs
    seg = copy.deepcopy(plan) or {"index": k}; seg["present"] = [f"<Subject {i}>" for i in range(1, len(members) + 1)]
    seg["shots"] = remap(shots_text or seg.get("shots"), tagmap, picmap); seg["kept"] = remap(seg.get("kept"), tagmap, picmap) if seg.get("kept") else seg.get("kept")
    for oi in range(len(objs)):   # an object this shot does not swap: the shot text's words for its NEW look go back to the old thing
        if oi in members:         # (T08_rt2: 'the teddy bear is visible on his back' drew a teddy on his chest)
            continue
        for nm in sorted({full[oi], names[oi]}, key=len, reverse=True):
            if nm and len(nm) > 2:
                for key in ("shots", "kept"):
                    if seg.get(key):
                        seg[key] = re.sub(r"(?i)\b(?:the |a |an )?" + re.escape(nm) + r"s?\b", f"the {words[oi]}", seg[key])
        for key in ("shots", "kept"):
            if seg.get(key):
                seg[key] = re.sub(r"(?i)(the " + re.escape(words[oi]) + r")(?:\s*\(?the " + re.escape(words[oi]) + r"\)?)+", r"\1", seg[key])
    sp["segments"] = [seg if str(g.get("index")) == str(k) else g for g in sp.get("segments", [])]
    if not any(str(g.get("index")) == str(k) for g in sp["segments"]):
        sp["segments"].append(seg)
    others = [x for oi in range(len(objs)) if oi not in members for x in (names[oi], full[oi])]
    pairs = [(i, oi) for i, oi in enumerate(members, 1)]
    sp["summary"] = ("The target video is <Video 1> with " + " and ".join(f"the {words[oi]} replaced by the {full[oi]} (<Subject {i}>)" for i, oi in pairs) +
                     ", keeping every movement, camera move, cut and the timing.")
    ret = []
    for i, oi in pairs:   # the plan's own line for this object when it names nothing else, else a plain one
        line = next((r for r in spec.get("retention", []) if objs[oi].get("tag") and str(r).startswith(objs[oi]["tag"])), "")
        body = re.sub(r"^<[^>]+>\s*(\([^)]*\))?\s*:?\s*", "", str(line))
        if not body or re.search(r"<(Subject|Picture) \d+>|<[A-Z][^>]*>", body) or any(o and o in body.lower() for o in others):
            body = f"fully_preserved - the {full[oi]} keeps the look of <Picture {i}> in every frame it appears."
        ret.append(f"<Subject {i}> (throughout): {body}")
    sp["retention"] = ret
    sp["replace"] = ", ".join(f"{words[oi]} to {full[oi]}" for i, oi in pairs)
    col = {oi: COLORS[oi % len(COLORS)][1] for oi in members}
    sp["instruction"] = ("Replace " + " and ".join((f"the {words[oi]} outlined in {col[oi]} in <Video 1>" if oi in outlined else f"the {words[oi]} in <Video 1>") +
                                                   f" with the {objs[oi].get('kind', 'object')} from <Picture {i}>" for i, oi in pairs) +
                         ", each at exactly the place, size and angle of its original, held and used exactly the same way.")
    corr = ""
    if outlined:
        corr = (("Inside each coloured outline <Video 1> shows only flat grey: that is where the new object is, drawn whole with its own natural "
                 "shape, colours and edges, and no grey is left. " if os.environ.get("R3_FILL") == "1" else
                 "The coloured outlines in <Video 1> only mark which objects to replace: draw each new object's own natural edges there."))
        stay = [words[oi] for oi in outlined if UNPICKED.get((k, oi), 0) > 0]   # N06: 'every corgi without an outline stays' kept the corgi
        if stay:
            corr += " Every " + " and every ".join(stay) + " without an outline stays exactly as it is in <Video 1>."
    if any(words[oi] == "person" for oi in members):
        sp["instruction"] = sp["instruction"].replace("held and used exactly the same way", "moving exactly as the original moves")
    for i, oi in pairs:   # an animal replacing a person: the whole figure (C3)
        if objs[oi].get("kind") == "animal" and words[oi] == "person":
            corr += (f" The whole figure outlined in {col[oi]} - head, face, hands and body - becomes the animal from <Picture {i}>: from every side, also "
                     f"from behind and while spinning, its head is the animal's furry head with its round ears.")
    for i, oi in pairs:   # an animal replacing a vehicle (C6)
        if objs[oi].get("kind") == "animal" and re.search(r"\b(motorcycle|motorbike|bike|bicycle|scooter|car|quad|atv|jet ?ski|snowmobile|vehicle)\b",
                                                           str(objs[oi].get("replaces") or ""), re.I):
            corr += (" The rider sits astride the animal in the saddle, legs down both its sides, and his gloved hands hold the reins just above its "
                     "shoulders; its neck and head are well ahead of his hands, at their natural distance. It gallops the whole time: with every stride "
                     "its neck and head rise and fall, its mane flies and the reins swing, and the rider's body moves with it, rising and sinking in the saddle.")
            sp["instruction"] = sp["instruction"].replace("held and used exactly the same way", "ridden by the same rider in the same place")
    for i, oi in pairs:   # R3_USE: how the new object is held / faces (names only it and body parts)
        if (use or {}).get(oi):
            corr += " " + re.sub(r"(?i)\bthe " + re.escape(full[oi]) + r"\b", f"the {full[oi]} (<Subject {i}>)", use[oi], count=1)
    for i, oi in pairs:   # R3_SHEET: the picture is a sheet of several views of the ONE object
        if oi in sheets or str(oi) in (GIVEN.get(str(k)) or {}):
            corr += (f" <Picture {i}> shows the one {full[oi]} several times: alone from two sides, and held or placed the way it is used in this "
                     f"video. The video shows a single {full[oi]} at each place of its original, turned and held like in those views.")
            if oi in outlined and (shot_hold.get((k, oi)) or sheet_hold.get(oi)):
                corr += " " + (shot_hold.get((k, oi)) or sheet_hold[oi])
    if os.environ.get("R3_PROMPT2", "1") == "1":   # 2026-10-27, wording from Higgsfield's own Genjutsu object-swap presets and BytePlus's object-swap
        # (default ON since 2026-10-28: the sixth batch re-rendered with it, everything else pinned - leftovers 6 -> 2 in 32 shots, the judge 3:2
        # for it, strays 4 -> 4; R3_PROMPT2=0 turns it off)
        # guide: how many there are, a positive 'fully replaced in every frame' line, when each one is in view (objects entering mid-shot),
        # a firm grip for held ones and their own real size (all positive: 'no ...' lines made H3 bring the thing back - NariP, D3voz)
        vi_ = next((i_ for i_, s_ in enumerate(shots) if s_["k"] == k), None); sh_ = shots[vi_] if vi_ is not None else {}
        info_ = json.load(open(f"{od}/{vi_}/objects.json")) if vi_ is not None and os.path.exists(f"{od}/{vi_}/objects.json") else []
        nfr = len(sh_.get("S") or []) or 1
        for i, oi in pairs:
            nm_ = (sh_.get("pick") or {}).get(oi) or []; n_ = len(nm_); thing = full[oi]
            fr_ = sorted({f_[0] for c in info_ if c["name"] in nm_ for f_ in c["frames"]})
            if 1 <= n_ <= 3:
                corr += (f" The shot shows exactly {('one', 'two', 'three')[n_ - 1]} {thing}{'s' if n_ > 1 else ''} (<Subject {i}>), one at each "
                         f"place of a {words[oi]} in <Video 1>.")
            if UNPICKED.get((k, oi), 0) > 0 and oi in outlined:   # others of its kind stay (K01: the other customers' cups) - not 'every'
                corr += f" The outlined {words[oi]} is fully replaced by the {thing} in every frame it appears; only the {thing} is seen there."
            else:
                corr += f" Every {words[oi]} is fully replaced by the {thing} in every frame it appears; only the {thing} is seen there."
            if fr_ and (fr_[0] > 3 or fr_[-1] < nfr - 4):
                corr += f" The {thing} is in view from {fr_[0] / 24:.1f} s to {(fr_[-1] + 1) / 24:.1f} s of the shot."
            if oi in outlined and (shot_hold.get((k, oi)) or sheet_hold.get(oi)):
                corr += f" The {thing} stays firmly in the same hand the whole time, gripped by the fingers."
            corr += f" The {thing} has its own real-world size and shape."
    for item in (os.environ.get("R3_EXTRA_CORR") or "").split("||"):   # "k:sentence||k:sentence" - a hand-written correction for one shot (N18 straws)
        if ":" in item and item.split(":", 1)[0].strip() == str(k):
            corr += " " + item.split(":", 1)[1].strip()
    sp["_corrections"] = {str(k): corr.strip()} if corr.strip() else {}
    shot_sheet = {int(a): b for a, b in (GIVEN.get(str(k)) or {}).items()}   # a sheet for this shot's own camera angle wins
    return sp, [shot_sheet.get(oi) or sheets.get(oi) or job["images"][objs[oi]["picture"] - 1] for oi in members], others


def gapfill(M, most=12):
    """a tracked mask lost for a few frames (motion blur: N10's kicked ball stayed a soccer ball at 8.5 s) -> the nearest tracked mask moved along
    the line between the tracked positions before and after the gap"""
    have = [j for j, m in enumerate(M) if m.any()]
    if len(have) < 2:
        return M
    out = list(M); cen = {j: np.argwhere(M[j]).mean(0) for j in have}
    for a_, b_ in zip(have, have[1:]):
        if 1 < b_ - a_ <= most:
            for j in range(a_ + 1, b_):
                w_ = (j - a_) / (b_ - a_); src = a_ if w_ <= 0.5 else b_
                dy, dx = (cen[a_] + (cen[b_] - cen[a_]) * w_) - cen[src]
                out[j] = cv2.warpAffine(M[src], np.float32([[1, 0, dx], [0, 1, dy]]), (M[src].shape[1], M[src].shape[0]), flags=cv2.INTER_NEAREST)
    return out


def outline_video(vi, sh, which, path):
    """the shot's source with the outlines of the objects in `which` (FILL: flat grey inside), nothing on the frame border"""
    S = sh["S"]; th = max(3, W // 160); frames = [f.copy() for f in S]; FILL = os.environ.get("R3_FILL") == "1"
    for oi in which:
        Ms = [gapfill([(cv2.resize(cv2.cvtColor(f_, cv2.COLOR_BGR2GRAY), (W, H)) > 127).astype(np.uint8) for f_ in P.read_frames(f"{od}/{vi}/{nm}.mp4")])
              for nm in sh["pick"][oi]]
        c_ = COLORS[oi % len(COLORS)][0]
        for j in range(len(frames)):
            m = np.zeros((H, W), np.uint8)
            for M in Ms:
                m |= M[min(j, len(M) - 1)]
            if os.environ.get("R3_GROUP") == "1" and len(Ms) > int(os.environ.get("R3_GROUP_MIN") or 8):   # opt-in: many small pieces (F28: 32 chess pieces): ONE outline round the group -
                gk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (W // 24) + 1, 2 * (W // 24) + 1))   # v1's single board-wide box swapped
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, gk)                                               # them, per-piece outlines did not
                cs0, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); m = np.zeros_like(m); cv2.drawContours(m, cs0, -1, 1, -1)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if FILL:   # the old object's own pixels go: flat grey inside its outline (the hands in front of it are not in its mask and stay)
                frames[j][cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0] = 127
            cv2.drawContours(frames[j], [c for c in cs if cv2.contourArea(c) > 30], -1, c_, th, cv2.LINE_AA)
    e_ = max(th + 2, W // 40)   # an object cut by the frame edge: its outline ran along the border and H3 kept it as a red line (N04 at 11 s; 7 px was not enough)
    for j in range(len(frames)):
        for sl in ((slice(0, e_), slice(None)), (slice(H - e_, H), slice(None)), (slice(None), slice(0, e_)), (slice(None), slice(W - e_, W))):
            frames[j][sl] = S[j][sl]
    P.write_video(frames, path, crf=12)


def inpaint_mask(vi, sh, which, path):
    """R3_INPAINT: white = where H3 may draw (the replaced objects' places, grown W/20 and +-3 frames, outline gaps bridged), black = the source
    (H3 Fun ControlNet inpaint) - nothing can appear anywhere else, so no extra copies by construction"""
    n = len(sh["S"]); U = [np.zeros((H, W), np.uint8) for _ in range(n)]
    for oi in which:
        for nm in sh["pick"][oi]:
            M = gapfill([(cv2.resize(cv2.cvtColor(f_, cv2.COLOR_BGR2GRAY), (W, H)) > 127).astype(np.uint8) for f_ in P.read_frames(f"{od}/{vi}/{nm}.mp4")])
            for j in range(n):
                U[j] |= M[min(j, len(M) - 1)]
    pad = int(os.environ.get("R3_INPAINT_PAD") or max(16, W // 28))   # W/20 let T14's pizzas grow past the turntables
    g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    out = [cv2.dilate(np.logical_or.reduce([U[min(n - 1, max(0, j + d))] for d in range(-3, 4)]).astype(np.uint8), g) * 255 for j in range(n)]
    P.write_video([cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in out], path, crf=0)
    return out


def render(k, sg, ov, spec2, imgs2, others, name, inpaint=None, guides=()):
    """one H3 render of shot k (the earlier one is reused with R3_REUSE when its prompt is the same) -> root/<name>.mp4;
    inpaint = (source window, mask clip, masks): only the white part is drawn, the rest is the source (pasted back exactly, feathered)"""
    prompt, pics = P.segment_prompt(spec2, k, imgs2, job.get("style"))
    leak = [o for o in others if o and re.search(rf"\b{re.escape(o)}", prompt, re.I)]
    if leak:
        print(f"   {name} WARNING the prompt still names {leak}", flush=True)
    pf = f"{root}/{name}_prompt.txt"
    if os.environ.get("R3_DRY") == "1":
        open(pf, "w").write(prompt); print(f"   {name} DRY prompt written, pictures {[os.path.basename(x) for x in pics]}", flush=True); return None
    if os.environ.get("R3_REUSE") == "1" and os.path.exists(f"{root}/{name}.mp4") and os.path.exists(pf) and open(pf).read() == prompt:
        print(f"   {name} same prompt - earlier render reused", flush=True)
    else:
        wf = h3.build(prompt, ov, pics, W, H, sg["length"], seed0 + 1000 * k + 991 + int(os.environ.get("R3_SEED_OFF") or 0), list(guides))
        if inpaint:
            h3.add_inpaint(wf, inpaint[0], inpaint[1])
        mp4 = next(f for f in comfy.run(wf, None, lambda: False) if f.endswith(".mp4"))
        shutil.copy(mp4, f"{root}/{name}.mp4"); open(pf, "w").write(prompt)
        if inpaint:   # outside the drawn area: the source, exactly
            R = P.read_frames(f"{root}/{name}.mp4"); S0 = P.read_frames(inpaint[0]); Ms = inpaint[2]; G = []; offs = []
            e9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
            for j in range(len(R)):   # the drawn area's tone drift (N06: a lighter halo round the dog) measured on its inner rim
                m = Ms[min(j, len(Ms) - 1)] > 127; ring = m & ~(cv2.erode(m.astype(np.uint8), e9) > 0)
                src = cv2.resize(S0[min(j, len(S0) - 1)], (W, H))
                offs.append((R[j][ring].astype(np.float32).mean(0) - src[ring].astype(np.float32).mean(0)) if ring.sum() > 50 else np.zeros(3, np.float32))
            offs = [np.mean(offs[max(0, j - 4):j + 5], axis=0) for j in range(len(offs))]   # steady over +-4 frames
            for j in range(len(R)):
                a = cv2.GaussianBlur(Ms[min(j, len(Ms) - 1)].astype(np.float32) / 255.0, (0, 0), 3)[..., None]
                r_ = R[j].astype(np.float32) - offs[j][None, None, :]
                G.append((r_ * a + cv2.resize(S0[min(j, len(S0) - 1)], (W, H)).astype(np.float32) * (1 - a)).clip(0, 255).astype(np.uint8))
            P.write_video(G, f"{root}/{name}.mp4", crf=12)
    return f"{root}/{name}.mp4"


llm.down(); sheets = {}
GIVEN = json.load(open(os.environ["R3_SHEETS_JSON"])) if os.environ.get("R3_SHEETS_JSON") else {}   # {"*" or shot k: {oi: sheet.png}} made by make_sheet.py
if os.environ.get("R3_SHEET") == "2" and any(sh.get("panel") for sh in shots):
    t0 = time.time(); plain = {}; canvas2 = f"{root}/sheet_canvas2.png"; cv2.imwrite(canvas2, np.full((768, 1536, 3), 255, np.uint8))
    for sh in shots:
        for oi, (cp, ed, hand_, cont_) in (sh.get("panel") or {}).items():
            pic_ = job["images"][objs[oi]["picture"] - 1]
            if oi not in plain:
                pp = f"{root}/sheet2_{oi}_plain.png"
                if not os.path.exists(pp):
                    shutil.copy(P.qwen_edit(fj, [canvas2, pic_], "Picture 1 is an empty white sheet. Show the object from Picture 2 on it twice, side by side, "
                                "each whole on plain white: on the left from the front, on the right from the side. Exactly the same object - the same shape, "
                                "proportions, colours, materials and details. No text, no labels, no other objects.", seed0 + 177 + oi, f"sheet2_{oi}_plain_raw"), pp)
                plain[oi] = pp
            dst = f"{root}/sheet2_s{sh['k']}_{oi}.png"
            if not os.path.exists(dst):
                pr = (f"Picture 1 is a photo of a {words[oi]}. Replace the {words[oi]} with the {full[oi]} from Picture 2 - exactly the same object as in "
                      f"Picture 2: the same shape, proportions, colours, materials and details - at exactly the same place and size as the {words[oi]}: {ed} "
                      + ("Keep the hands and fingers exactly as they are, holding it the same way; remove the person's head, face and body - only the "
                         "hands and forearms stay (H20: a panel kept the whole bartender). " if hand_ else
                         "There is no hand and no person in this picture - do not add any. ")
                      + (f"Keep {cont_} exactly as it is, in the same place. " if cont_ and cont_.lower() not in ("none", "nothing") else "")
                      + "Make everything else plain white: no background, no other objects.")
                panel = cv2.imread(P.qwen_edit(fj, [cp, pic_], pr, seed0 + 300 + 7 * int(sh["k"]) + oi, f"sheet2_s{sh['k']}_{oi}_panel"))
                row = []
                for im in (cv2.imread(plain[oi]), panel):
                    im = cv2.resize(im, (int(im.shape[1] * 600 / im.shape[0]), 600), interpolation=cv2.INTER_AREA); row += [im, np.full((600, 24, 3), 255, np.uint8)]
                cv2.imwrite(dst, np.hstack(row[:-1])); open(f"{root}/sheet2_s{sh['k']}_{oi}_prompt.txt", "w").write(pr)
            GIVEN.setdefault(str(sh["k"]), {})[str(oi)] = dst
    print(f"== per-shot sheets in {time.time() - t0:.0f} s: {sorted((k_, o_) for k_, v_ in GIVEN.items() if k_ != '*' for o_ in v_)}", flush=True)
for oi_, pth in (GIVEN.get("*") or {}).items():
    sheets[int(oi_)] = pth
if sheet_views:   # Qwen-Image edit: a blank white sheet (Picture 1, its shape is kept) + the object picture -> the 4-view sheet
    canvas = f"{root}/sheet_canvas.png"; cv2.imwrite(canvas, np.full((928, 1664, 3), 255, np.uint8)); t0 = time.time()
    for oi, views in sheet_views.items():
        if oi in sheets:   # a sheet made from the source frames was given
            continue
        dst = f"{root}/sheet_{oi}.png"
        if not os.path.exists(dst):
            sp_ = ("Picture 1 is an empty white sheet. Turn it into a clean reference sheet of the object shown in Picture 2 - exactly the same object: "
                   "the same shape, proportions, colours, materials and details. Four views side by side, evenly spaced, each showing the whole object "
                   "on plain white: " + " ".join(f"({i}) {v}" for i, v in enumerate(views, 1)) +
                   " No text, no numbers, no labels, no frames, no other objects; no person, only the hands that the views name.")
            png = P.qwen_edit(fj, [canvas, job["images"][objs[oi]["picture"] - 1]], sp_, seed0 + 77 + oi, f"sheet_{oi}_raw")
            shutil.copy(png, dst); open(f"{root}/sheet_{oi}_prompt.txt", "w").write(sp_)
        sheets[oi] = dst
    print(f"== {len(sheets)} object sheet(s) in {time.time() - t0:.0f} s", flush=True)
out = []; t0 = time.time(); rendered = {}; passes = {}
for vi, sh in enumerate(shots):
    k, sg, S = sh["k"], sh["sg"], sh["S"]; members, outlined = sh["members"], sh["outlined"]
    n_out = min(sg["length"], sg.get("end", 10 ** 9) - sg["start"])
    if not members:   # nothing to replace: the source shot itself
        P.write_video(S, f"{root}/seg{k}.mp4", crf=12); out += S[:n_out]
        print(f"   seg{k}: nothing to replace - the source shot", flush=True); continue
    ONLY = [x.strip() for x in (os.environ.get("R3_ONLY_SHOTS") or "").split(",") if x.strip()]
    if ONLY and str(k) not in ONLY:   # a retake of some shots only: the others are placeholders (the source) - best_of.py keeps the first take there
        P.write_video(S, f"{root}/seg{k}.mp4", crf=12); out += S[:n_out]
        print(f"   seg{k}: not retaken (R3_ONLY_SHOTS)", flush=True); continue
    if SPLIT and sh.get("alone"):   # one render per object, joined after the SAM 3 pass below
        passes[vi] = {}
        for oi in members:
            ov = f"{root}/seg{k}_p{oi}_outlined.mp4"; outline_video(vi, sh, [oi] if oi in outlined else [], ov)
            alone = sh["alone"].get(oi) or (f"[Shot 1] The shot exactly as in <Video 1>, with the {words[oi]} replaced by {objs[oi].get('tag')}."
                                            if objs[oi].get("tag") else None)   # never the plan's text: it names the other new objects
            spec2, imgs2, others = shot_spec(k, [oi], [oi] if oi in outlined else [], sh.get("use"), shots_text=alone)
            inp = None   # R3_SPLIT + R3_INPAINT: each object drawn alone inside its own place, the rest the source
            if os.environ.get("R3_INPAINT") == "1" and oi in outlined:
                mc = f"{root}/seg{k}_p{oi}_inmask.mp4"; inp = (sh["win"], mc, inpaint_mask(vi, sh, [oi], mc))
            passes[vi][oi] = render(k, sg, ov, spec2, imgs2, others, f"seg{k}_p{oi}", inpaint=inp)
        print(f"   seg{k} rendered once per object ({time.time() - t0:.0f} s so far): {[full[oi] for oi in members]}", flush=True)
        rendered[vi] = f"{root}/seg{k}.mp4"; continue
    ov = f"{root}/seg{k}_outlined.mp4"; outline_video(vi, sh, outlined, ov)
    spec2, imgs2, others = shot_spec(k, members, outlined, sh.get("use"), shots_text=sh.get("fixed_text"))
    inp = None
    if os.environ.get("R3_INPAINT") == "1" and outlined and all(oi in outlined for oi in members):
        mc = f"{root}/seg{k}_inmask.mp4"; inp = (sh["win"], mc, inpaint_mask(vi, sh, outlined, mc))
    gd = []
    pin_at = next((int(x.split(":")[1]) for x in (os.environ.get("R3_PIN_AT") or "").split(",") if x.count(":") == 1 and x.split(":")[0].strip() == str(k)), 0)
    pin_objs = [int(x.split(":")[1]) for x in (os.environ.get("R3_PIN_OBJS") or "").split(",") if x.count(":") == 1 and x.split(":")[0].strip() == str(k)]
    pin_at = max(0, min(pin_at, len(S) - 1))
    if os.environ.get("R3_PIN") == "1" and (outlined or pin_objs):   # a frame with the new objects already in place, pinned (N06: the corgi
        # close-up kept the dog whatever the prompt said). R3_PIN_AT=k:j pins frame j instead of 0, R3_PIN_OBJS=k:oi edits those objects there
        # even without an outline (autofix, 2026-10-27: K02's red balls showed only late in the shot and were swapped from the text alone)
        f0p = f"{root}/seg{k}_pin_src.png"; cv2.imwrite(f0p, S[pin_at]); cur = f0p
        for oi in (pin_objs or outlined):
            names_ = sh["pick"].get(oi) or []
            if oi in outlined and not pin_objs and not any((cv2.cvtColor(P.read_frames(f"{od}/{vi}/{nm}.mp4", [pin_at])[0], cv2.COLOR_BGR2GRAY) > 127).any()
                                                         for nm in names_):
                continue   # not in the pinned frame
            pic_ = (GIVEN.get(str(k)) or {}).get(str(oi)) or job["images"][objs[oi]["picture"] - 1]
            pr = (f"Picture 1 is a frame of a video. Replace the {words[oi]} in it with the {full[oi]} shown in Picture 2 - exactly that object, its "
                  f"shape, colours and details - at the same place and size, turned the way the {words[oi]} is. Remove every part of the "
                  f"{words[oi]}. Keep everything else exactly the same: the people, their faces and hands, the background, the light and the framing.")
            try:
                cur = P.qwen_edit(fj, [cur, pic_], pr, seed0 + 400 + 7 * int(k) + oi, f"seg{k}_pin_{oi}")
            except Exception as e:
                print(f"   seg{k} pin edit failed: {str(e)[:80]}", flush=True)
        if cur != f0p:
            gd = [{"image": cur, "frame": pin_at}]; print(f"   seg{k} frame {pin_at} pinned: {os.path.basename(cur)}", flush=True)
    if render(k, sg, ov, spec2, imgs2, others, f"seg{k}", inpaint=inp, guides=gd) is None:
        continue
    rendered[vi] = f"{root}/seg{k}.mp4"
    out += P.read_frames(f"{root}/seg{k}.mp4")[:n_out]
    print(f"   seg{k} rendered ({time.time() - t0:.0f} s so far), outlined: {[words[oi] for oi in outlined]}, "
          f"without outline: {[words[oi] for oi in members if oi not in outlined]}", flush=True)
if os.environ.get("R3_DRY") == "1":
    sys.exit(0)
def src_place(vi, oi, n):
    """every source instance of object oi's kind in shot vi (picked or not, minus close-up rejects), +-1 frame"""
    sh = shots[vi]; t = [np.zeros((H, W), bool) for _ in range(n)]
    info = json.load(open(f"{od}/{vi}/objects.json")) if os.path.exists(f"{od}/{vi}/objects.json") else []
    for o_ in info:
        if o_["concept"] in (SAMW[oi], words[oi]) and len(o_["frames"]) >= 3 and o_["name"] not in sh.get("rejected", set()):
            for j, m in enumerate(P.read_frames(f"{od}/{vi}/{o_['name']}.mp4")[:n]):
                t[j] |= cv2.cvtColor(cv2.resize(m, (W, H)), cv2.COLOR_BGR2GRAY) > 127
    return [t[j] | t[max(0, j - 1)] | t[min(n - 1, j + 1)] for j in range(n)]
if passes:   # R3_SPLIT: each object comes from its own render, pasted where that object is; the base = the object with the biggest place
    t0 = time.time(); pd = f"{root}/passobj"; plist = [(vi, oi, p) for vi in sorted(passes) for oi, p in passes[vi].items()]
    tools.gpu_take(fj)
    tools.run_proc(fj, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_multi.py", pd, "1", "|".join(dict.fromkeys(names))] + [p for _, _, p in plist],
                   0.5, 0.5, "objects in the per-object renders", env={"PER_OBJ": "1", "MAX_OBJ": "16"})
    wide = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (W // 16) + 1, 2 * (W // 16) + 1)); g24 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (W // 24) + 1, 2 * (W // 24) + 1))
    g5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for vi in sorted(passes):
        sh = shots[vi]; k = sh["k"]; R = {oi: P.read_frames(p) for oi, p in passes[vi].items()}; n = min(len(v) for v in R.values())
        place = {oi: src_place(vi, oi, n) for oi in R}; inst = {}
        for oi in R:   # the new object in its own render: SAM 3 instances of its name near its place (anywhere when it has no outline)
            gi = [i for i, (v_, o_, _) in enumerate(plist) if v_ == vi and o_ == oi][0]; m_ = [np.zeros((H, W), bool) for _ in range(n)]
            info = json.load(open(f"{pd}/{gi}/objects.json")) if os.path.exists(f"{pd}/{gi}/objects.json") else []
            near = [cv2.dilate(place[oi][j].astype(np.uint8), wide) > 0 for j in range(n)]
            for o_ in info:
                if o_["concept"] != names[oi]:
                    continue
                M = [cv2.cvtColor(cv2.resize(m, (W, H)), cv2.COLOR_BGR2GRAY) > 127 for m in P.read_frames(f"{pd}/{gi}/{o_['name']}.mp4")[:n]]
                for j in range(n):
                    if M[j].any() and (not place[oi][j].any() or (M[j] & near[j]).sum() / M[j].sum() >= 0.12):
                        m_[j] |= M[j]
            inst[oi] = m_
        order = sorted(R, key=lambda oi: -sum(int(p.sum()) for p in place[oi]))
        comp = [f.astype(np.float32) for f in R[order[0]][:n]]
        for ob in order[1:]:
            al = []
            for j in range(n):
                reg = (cv2.dilate(place[ob][j].astype(np.uint8), g24) > 0) | (cv2.dilate(inst[ob][j].astype(np.uint8), g5) > 0)
                prot = np.zeros((H, W), bool)
                for o in R:
                    if o != ob:
                        prot |= cv2.dilate((place[o][j] | inst[o][j]).astype(np.uint8), g5) > 0
                al.append((reg & ~prot).astype(np.float32))
            for j in range(n):
                a = np.maximum.reduce([al[max(0, j - 1)], al[j], al[min(n - 1, j + 1)]]); a = cv2.GaussianBlur(a, (0, 0), 3)[..., None]
                comp[j] = comp[j] * (1 - a) + R[ob][j].astype(np.float32) * a
        comp = [c.clip(0, 255).astype(np.uint8) for c in comp]
        P.write_video(comp, f"{root}/seg{k}.mp4", crf=12)
        print(f"   seg{k} joined: base {full[order[0]]}, pasted {[full[o] for o in order[1:]]}", flush=True)
    out = []
    for vi, sh in enumerate(shots):   # the whole video again in shot order
        n_out = min(sh["sg"]["length"], sh["sg"].get("end", 10 ** 9) - sh["sg"]["start"])
        out += P.read_frames(f"{root}/seg{sh['k']}.mp4")[:n_out]
    print(f"== per-object renders joined in {time.time() - t0:.0f} s", flush=True)
def to_timeline(frames):
    """the shots' frames one after another -> the video: a long shot's next chunk (cont) overlaps the previous one by 5 frames - cross-faded
    as the app's join_segment does, not shown twice (before 2026-10-27 each join repeated 5 frames and the sound drifted)"""
    res = []; p_ = 0
    for sh_ in shots:
        sg_ = sh_["sg"]; n_ = min(sg_["length"], sg_.get("end", 10 ** 9) - sg_["start"]); fr_ = frames[p_:p_ + n_]; p_ += n_
        if sg_.get("cont") and len(res) >= 5 and len(fr_) > 5:
            for i_ in range(5):
                a_ = (i_ + 1) / 6.0; res[-5 + i_] = cv2.addWeighted(res[-5 + i_], 1 - a_, fr_[i_], a_, 0)
            fr_ = fr_[5:]
        res += fr_
    return res


P.write_video(to_timeline(out), f"{root}/result_noguard.mp4", crf=14)
# 3. extra copies: a new object that does not sit on a replaced object gets the source back
guard = [(vi, oi) for vi in rendered for oi in shots[vi]["outlined"] if words[oi] != "person"]
if guard and os.environ.get("R3_NOGUARD") != "1":
    t0 = time.time(); nd = f"{root}/newobj"; vis = sorted(rendered); concepts = list(dict.fromkeys(names[oi] for oi in range(len(objs)) if words[oi] != "person"))
    tools.gpu_take(fj)
    tools.run_proc(fj, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_multi.py", nd, "1", "|".join(concepts)] + [rendered[vi] for vi in vis],
                   0.5, 0.5, "new objects", env={"PER_OBJ": "1", "MAX_OBJ": "16"})
    r_ = max(9, W // 40); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1)); grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    def masks(path, n):
        M = P.read_frames(path); return [(cv2.cvtColor(cv2.resize(m, (W, H)), cv2.COLOR_BGR2GRAY) > 127) for m in M][:n]
    out = []; log = []
    for vi, sh in enumerate(shots):
        k = sh["k"]; n_out = min(sh["sg"]["length"], sh["sg"].get("end", 10 ** 9) - sh["sg"]["start"])
        R = P.read_frames(f"{root}/seg{k}.mp4")
        if vi not in rendered or not os.path.exists(f"{nd}/{vis.index(vi)}/objects.json"):
            out += R[:n_out]; continue
        n = len(R); S = sh["S"]; tgt = {}
        for oi in sh["outlined"]:
            t = [np.zeros((H, W), bool) for _ in range(n)]
            for nm in sh["pick"][oi]:
                for j, m in enumerate(masks(f"{od}/{vi}/{nm}.mp4", n)):
                    t[j] |= m
            tgt[oi] = t
        for oi, t in tgt.items():   # SAM 3 drop-outs in the source: an empty frame between found ones gets its neighbours' masks
            have = [j for j in range(n) if t[j].any()]
            for j in range(n):
                if not t[j].any() and have:
                    a = max([x for x in have if x < j], default=None); b = min([x for x in have if x > j], default=None)
                    if (a is not None and j - a <= 12) or (b is not None and b - j <= 12):
                        t[j] = (t[a] if a is not None and j - a <= 12 else False) | (t[b] if b is not None and b - j <= 12 else False)
        # every source instance of a replaced kind, picked or not: H3 sometimes swaps the other one (H08: the hat he wears instead of the one
        # he holds) - a new object there is a swap, not a copy, and pasting the source back would only bring the old object back
        info = json.load(open(f"{od}/{vi}/objects.json")) if os.path.exists(f"{od}/{vi}/objects.json") else []
        for oi in sh["outlined"]:
            for o_ in info:
                if o_["concept"] in (SAMW[oi], words[oi]) and len(o_["frames"]) >= 3 and o_["name"] not in sh["pick"][oi] and o_["name"] not in sh["rejected"]:
                    for j, m in enumerate(masks(f"{od}/{vi}/{o_['name']}.mp4", n)):
                        tgt[oi][j] = tgt[oi][j] | m
        # where the new object may be: W/16 around its original (+-3 frames), more where the original moves fast (H16's swinging rope) -
        # a still original keeps the tight radius so a copy next to it is still a copy (N04's pizza by the decks)
        kers = {}
        def kern(r):
            r = int(r)
            if r not in kers:
                kers[r] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            return kers[r]
        near = {}
        for oi, t in tgt.items():
            cen = [(np.argwhere(m).mean(0) if m.any() else None) for m in t]; ar = [int(m.sum()) for m in t]; near[oi] = []
            for j in range(n):   # speed only between frames of a steady mask size (a flickering or drifting mask is not motion - T14)
                sp = max([float(np.hypot(*(cen[min(n - 1, max(0, j + d))] - cen[j]))) for d in (-3, -2, -1, 1, 2, 3)
                          if cen[j] is not None and cen[min(n - 1, max(0, j + d))] is not None
                          and 0.6 <= ar[min(n - 1, max(0, j + d))] / max(1, ar[j]) <= 1.6] + [0.0])
                rad = min(W // 8, W // 16 + int(round(sp / 16)) * 16)
                near[oi].append(cv2.dilate(np.logical_or.reduce([t[min(n - 1, max(0, j + d))] for d in (-3, -2, -1, 0, 1, 2, 3)]).astype(np.uint8),
                                           kern(rad)) > 0)
        tight = [np.zeros((H, W), bool) for _ in range(n)]   # the replaced objects' own places (+-2 frames): pasting the source there brings them back
        for t in tgt.values():
            for j in range(n):
                tight[j] |= np.logical_or.reduce([t[min(n - 1, max(0, j + d))] for d in (-2, -1, 0, 1, 2)])
        tight = [cv2.dilate(z.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1))) > 0 for z in tight]
        keepm = [np.zeros((H, W), bool) for _ in range(n)]; extram = [np.zeros((H, W), bool) for _ in range(n)]; gi = vis.index(vi); inst = []
        for o in json.load(open(f"{nd}/{gi}/objects.json")):
            ois = [oi for oi in range(len(objs)) if names[oi] == o["concept"]]
            M = masks(f"{nd}/{gi}/{o['name']}.mp4", n); seen = [j for j in range(n) if M[j].any()]
            if not seen:
                continue
            if any(oi in sh["members"] and oi not in sh["outlined"] for oi in ois):   # replaced without an outline here: no place to check
                for j in seen:
                    keepm[j] |= M[j]
                continue
            # per frame (one track can hold the real new object and, later, a copy somewhere else - T14's pizza): a frame is a copy when the
            # instance is not near a place of ITS OWN kind (a pizza belongs on a record, not next to the microphone) and does not mostly sit on
            # another replaced object's place (SAM 3 calling the other new object by this name); runs under 3 frames do not count (no flicker)
            own = [oi for oi in ois if oi in tgt]; oth = [oi for oi in tgt if oi not in own]; ex = {}
            for j in seen:
                a = int(M[j].sum())
                on_own = max([int((M[j] & near[oi][j]).sum()) / a for oi in own] + [0.0])
                on_oth = max([int((M[j] & tgt[oi][j]).sum()) / a for oi in oth] + [0.0])
                cov_oth = max([int((M[j] & tgt[oi][j]).sum()) / max(1, int(tgt[oi][j].sum())) for oi in oth] + [0.0])
                # on another replaced object's place = the swaps mixed (H22: the bottle drawn as the paddle) - a paste cannot fix that and
                # cut the wrong object into a blob, so it stays
                ex[j] = on_own < 0.12 and on_oth < 0.2 and cov_oth < 0.3
                if own and not any(tgt[oi][j].any() for oi in own):   # its own kind's original is not tracked in this frame (T10: the ball lost in
                    ex[j] = False                                      # the throw) - no place to judge by; a paste would bring the ball back
            cm = {j: np.argwhere(M[j]).mean(0) for j in seen}
            allt = [np.logical_or.reduce([t[j] for t in tgt.values()]) if tgt else np.zeros((H, W), bool) for j in range(n)]
            for j in seen:   # a fast-moving piece near the action is the real thing whose source mask was lost to motion blur (H26's flying clubs,
                if not ex[j]:  # H16's swinging rope) - pasting the source there brought the white clubs back; the real copies stood still
                    continue
                prv, nxt = [x for x in seen if x < j][-1:], [x for x in seen if x > j][:1]
                if prv and nxt:
                    sp = float(np.hypot(*(cm[nxt[0]] - cm[prv[0]]))) / max(1, nxt[0] - prv[0])
                    if sp > W / 80 and allt[j].any():
                        d = cv2.distanceTransform((~allt[j]).astype(np.uint8), cv2.DIST_L2, 3)
                        if float(d[M[j]].min()) < W / 4:
                            ex[j] = False
            for j in seen:
                run = [x for x in range(j - 2, j + 3) if x in ex]
                if ex[j] and sum(ex[x] for x in run) < 3:
                    ex[j] = False
            inst.append((o, M, seen, ex))
        touch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        for _ in range(6):   # a piece touching a kept piece of the same kind is part of it (H16: a string of sausages tracked as many pieces)
            changed = False
            for a_, (o, M, seen, ex) in enumerate(inst):
                for j in seen:
                    if not ex[j]:
                        continue
                    grown = cv2.dilate(M[j].astype(np.uint8), touch) > 0
                    if any(b_ != a_ and o2["concept"] == o["concept"] and j in ex2 and not ex2[j] and (grown & M2[j]).any()
                           for b_, (o2, M2, seen2, ex2) in enumerate(inst)):
                        ex[j] = False; changed = True
            if not changed:
                break
        for o, M, seen, ex in inst:
            bad = [j for j in seen if ex[j]]
            msg = f"   seg{k} {o['concept']} {o['name']}: frames {seen[0]}-{seen[-1]} ({len(seen)}), a copy in {len(bad)}" + (f" ({bad[0]}-{bad[-1]}) -> source pasted back there" if bad else " -> keep")
            print(msg, flush=True); log.append(msg)
            for j in seen:
                (extram if ex[j] else keepm)[j] |= M[j]
        five = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        paste = [((cv2.dilate(extram[j].astype(np.uint8), grow) > 0) & ~(cv2.dilate(keepm[j].astype(np.uint8), five) > 0) & ~tight[j]).astype(np.float32)
                 for j in range(n)]
        if not any(p.any() for p in paste) and os.path.exists(f"{root}/seg{k}_guard.mp4"):
            os.remove(f"{root}/seg{k}_guard.mp4")   # an earlier run's patch (this run pastes nothing here)
        if any(p.any() for p in paste):
            G = []
            for j in range(n):
                a = np.maximum.reduce([paste[max(0, j - 1)], paste[j], paste[min(n - 1, j + 1)]]); a = cv2.GaussianBlur(a, (0, 0), 4)[..., None]
                G.append((R[j].astype(np.float32) * (1 - a) + S[j].astype(np.float32) * a).clip(0, 255).astype(np.uint8))
            P.write_video(G, f"{root}/seg{k}_guard.mp4", crf=12); R = G
        out += R[:n_out]
    open(f"{root}/guard.log", "w").write("\n".join(log) + "\n")
    print(f"== extra-copy check in {time.time() - t0:.0f} s", flush=True)
P.write_video(to_timeline(out), f"{root}/result_silent.mp4", crf=14)
r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{root}/result_silent.mp4", "-i", SRC or job0["video"], "-map", "0:v", "-map", "1:a:0?", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", f"{root}/result.mp4"])
if r.returncode != 0:
    shutil.copy(f"{root}/result_silent.mp4", f"{root}/result.mp4")
print(f"== VIDEO READY: {root}/result.mp4 ({len(out)} frames) in {(time.time() - t_all) / 60:.1f} min", flush=True)
import shotcheck; shotcheck.stop_orphan_llm()
