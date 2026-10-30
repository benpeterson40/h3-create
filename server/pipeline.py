"""pipeline.py : one Create Video job, start to finish.
prep (24 fps motion clip, canvas size, segment plan) -> edit spec (Qwen3.8 looks at the source frames + pictures + the user's text; a
template when the LLM is not installed) -> per segment: H3 render (segment k >= 1 continues from the previous segment's last 5 frames)
-> Qwen audit against the source frames and the pictures -> one automatic retake with the audit's fix when it fails -> cross-faded
joins, the source's audio muxed back in, poster. Segments are <= 5.2 s (124 frames on H3's 17k+5 grid); segment k starts at 119k."""
import json, os, random, re, shutil, subprocess, time, copy
import cv2, numpy as np
from . import comfy, config, durable, h3, llm, tools

FPS = 24
MAX_SECONDS = 30
RETAKES = 1
PRES = "Preserve the reference character's facial features, hairstyle, skin tone, body proportions, clothing, colors, and distinctive visual style."
KEEP = "The reference image supplies appearance only, not the character's pose, expression, background, or framing."
TAIL = """Maintain each reference character's identity through every expression and viewing angle. Hair and clothing move naturally with the transferred performance. Preserve interactions with source-scene objects and the ordering of foreground and background occlusions.

Keep the source video's crop and shot scale. Show only the portions of the replacement characters that belong within that framing; do not widen the shot to reproduce the full-body reference. Do not import the reference images' backgrounds, static poses, or handheld props. Preserve the source camera movement and action timing through the final frame.

overall_soundscape:
Retain the source clip's existing sound without adding new dialogue or sound effects.

non_diegetic_music:
Retain existing source music, if present; add no new music."""
FOLLOW = ("Follow <Video 1> exactly: every shot, cut, camera move, action and its timing. Match the original performance moment by moment: "
          "posture, steps, gestures, head turns, gaze, expressions and timing.")

STYLES = {
    "none": ("Original look", ""),
    "photoreal": ("Photoreal", "Photorealistic live-action footage with natural light, real skin texture and real materials."),
    "anime": ("Anime", "Japanese 2D anime style: clean line art, cel shading, flat vivid colours, expressive anime faces."),
    "3d": ("3D animation", "Polished 3D animated feature-film style: soft global illumination, stylized proportions, smooth shaded materials."),
    "clay": ("Claymation", "Stop-motion claymation style: hand-sculpted plasticine characters and sets, visible fingerprints, miniature lighting."),
    "comic": ("Comic book", "Bold comic-book style: thick ink outlines, halftone shading, saturated flat colours."),
    "watercolor": ("Watercolor", "Soft watercolor painting style: paper texture, bleeding pigment edges, light washes of colour."),
    "film": ("Vintage film", "1970s 35 mm film look: warm faded colours, heavy grain, soft halation, gentle gate weave."),
}

SPEC_SYSTEM = """You are the prompt writer for a local AI video editor built on MiniMax H3 reference-to-video with a Character Swap LoRA.
The model re-renders a SOURCE video, called <Video 1>, which supplies motion, poses, camera moves, cuts and timing, with the people, animals, objects and places shown in REFERENCE PICTURES, called <Picture 1>, <Picture 2>, ...
You get the mode, the user's instruction, the reference pictures, and frames of the source video with timestamps, grouped by SEGMENT (each segment is rendered separately and is at most 5.2 s long).
Reply with JSON only, with exactly these keys:
{
 "instruction": "One or two sentences starting with 'Replace' that say what in <Video 1> is replaced by what in which <Picture n>.",
 "subjects": [{"tag": "<Subject 1>", "picture": 1, "kind": "person|animal|object|location", "replaces": "what in <Video 1> it replaces, described so it can be found (or 'new')", "description": "exact visual description taken from the picture"}],
 "kept": "what stays exactly as in <Video 1>: other people, props, captions or text overlays, the place when it is not replaced",
 "summary": "One sentence: the target video is <Video 1> with ... replaced, keeping every movement, camera move, cut and the timing.",
 "retention": ["<Subject 1> (throughout): fully_preserved - ...", "..."],
 "replace": "a short phrase: what is replaced and what stays",
 "treatment": "One sentence about the overall look (photoreal or stylized, colour grade); match the source unless a style is requested. Never name places, objects or the light of particular shots here - those belong in each segment's shots text.",
 "hair_off": "true only if EVERY person seen in <Video 1> is replaced by an animal, robot, creature or helmeted figure without long hair AND one of those people has long loose hair, braids or a ponytail; otherwise false",
 "segments": [{"index": 0, "shots": "[Shot 1] (0.0 to 2.1 seconds) camera, framing, and what each subject does where, with subject tags. [Shot 2] (2.1 to 5.2 seconds) ...", "present": ["<Subject 1>"], "kept": "what stays exactly as in <Video 1> in THIS segment: its own place, people and props", "closeup": "only when the whole segment is a close-up that shows just part of a person or animal: the parts that fill the frame, e.g. 'feet and lower legs' or 'hands'; otherwise empty", "camera": "the camera in this segment: its height (ground level, knee, waist, eye level, high, aerial), its angle to the subject (side view, from behind, from the front, from above) and how it moves", "travel": "which way the main subject moves across the frame: left-to-right, right-to-left, toward-camera, away-from-camera, up, down or none", "spot": "only when a location picture replaces the place: where in the new place this segment happens, chosen so its movement makes sense there - e.g. 'on the stone walkway on top of the wall, seen from the side, the walkway running left to right'; otherwise empty"}]
}
Rules:
- One subject per reference picture that is used. Describe only what is visible in that picture: for people the face, age, skin tone, hair colour/length/style, build, every garment with its colour and material, shoes and accessories - look at EVERY view of a reference sheet: a backpack, a hood or a print may show only in the back or side view, and it belongs in the description; facial hair exactly as the close-up shows it (stubble is not a beard); for places every key element; for objects shape, colours, materials and markings. Never invent clothes or details.
- Every person keeps the face of their own picture; never merge two pictures into one person.
- When an animal, robot, creature or helmeted figure replaces a person with long hair, braids or a ponytail, end its description with one POSITIVE sentence about its head, for example "Its head has only short grey fur and two round ears, also when it spins and jumps." Never name the hair it must not have.
- The same when a person with short hair, a buzz cut or a shaved head replaces a person with long hair, braids, a ponytail or a hood or scarf over long hair: end the replacement's description with one POSITIVE sentence about its head, for example "His head has only a very short buzz cut close to the scalp, also when he spins and runs." Never name the long hair.
- If the same person or thing appears more than once in a frame (a reflection, a mirror, a double exposure, a screen showing them), EVERY instance is replaced - say so in the shot text.
- A location picture replaces the whole surrounding place in every shot of every segment, and after a cut the place never changes back to the source place; write that in its retention line.
- A close-up that shows only part of a subject (feet, hands, a wheel): say which part fills the frame for the WHOLE shot and where the camera stays (for example "Only her feet and ankles in pointe shoes fill the frame from start to end; the camera stays at floor level"), so the shot never widens to show the face.
- In close-ups after a change of place, describe only the part of the new place the camera would see at that distance (its floor surface, a wall, an out-of-focus backdrop in its colours) - never the whole view of the place picture, and never a second copy of the subject in the background.
- When the place changes, restage every movement onto solid, walkable ground of the new place (its floor, path, deck or ground) and onto real surfaces in it: say where each action happens; nobody walks, crawls, sits or lies in water, in the air or inside walls.
- When the place changes, stage each shot where its motion makes sense in the new place: a subject that travels across the frame travels ALONG the path, road, walkway, slope or water of the new place (for example "skates along the top of the wall, parallel to the parapet"), never toward a wall, cliff or obstacle the source does not have; wheels, feet, hooves and hulls stay on the surface.
- When the place changes, every object of the old place that a subject uses or touches (a ramp, a ledge, a rail, a bench, a table, a book used as a ramp) becomes a matching object that belongs in the new place (a rock slab, a dirt jump, a stone ledge, a fallen log): name what each one becomes in the shot text. Nothing from the old place appears in the new place.
- Hands and held objects: when an object that is held, ridden or played is replaced or removed (handlebars, a guitar, a cup, a bag), say exactly what each hand holds instead and where (for example "his gloved fists grip the black reins just above the horse's neck"). The replacement sits exactly where the original was and is held, worn or played the same way; it never hangs somewhere else, and no part of the original stays.
- Give one "segments" entry per segment shown, with times RELATIVE to that segment's first frame (0.0). Write one [Shot n] for EVERY shot - even one that lasts half a second - with its start and end time, its place, framing, camera move and what each subject does, so the timing matches the source frames. Use the possible cut times given with each segment only where the frames confirm a change of shot; fold a flash, a dissolve frame or anything shorter than 0.3 s into the shot around it. "present" lists only the subjects visible in that segment.
- Look-alikes: when the thing being replaced has look-alikes in a segment that are NOT replaced (other people's cups on a table, spare balls, other bags), that segment's "kept" names them and says they stay as they are (for example "the other people's paper coffee cups on the table stay paper coffee cups"), and its shot text says which one is replaced and where it is held or placed in every moment.
- A segment marked ONE continuous shot gets exactly one [Shot 1] covering all of it. A segment's "shots" and "kept" describe only what its own frames show: never mention places, people or things that appear only in other segments.
- Keep on-screen text, captions and logos from <Video 1> unless the user asks to change them.
- Costume changes: only when a person of <Video 1> changes into a THEMED costume in a later shot - traditional dress (a kimono, a hanbok), a uniform, armour, a stage or fancy-dress costume - their replacement keeps the face, hair, skin and build of their picture but wears THAT costume - say so in that shot's text. Ordinary clothes that differ between scenes (another t-shirt, top, cardigan or jacket) are NOT a costume change: the replacement wears its picture's outfit in every such shot, and the shot text never describes a replaced person by the source person's own clothes.
- Keep each reference character in its own art style: a cartoon, anime, illustrated or 3D-rendered character stays in that style inside the source footage (never call it photoreal) unless the user asks for a style change; "treatment" then describes the footage's look AND the character's own style.
- Objects swap mode: change ONLY what the user's instruction asks; everything else stays exactly as in <Video 1>.
- Motion transfer mode: the pictures define the new people, objects and place; <Video 1> supplies only motion, camera and timing.
- Objects swap mode with a location picture: the new place replaces the place it stands in for (where the people of <Video 1> are, also in shots of that place without them); a shot set somewhere clearly different - outer space, a planet seen from space, a dream, explosion or special-effect world - keeps its own look unless the user's instruction names it. Name only the replaced place in the location's "replaces", and give "spot" only to the segments where the new place is seen.
- Never add people, objects or text that are in neither the source nor the pictures.
- Props: name only objects you can clearly see being held, used or standing in the scene; parts of clothing (sleeves, scarves, hoods) are never props. Describe each hand action exactly (opens the lid, pours, drinks, wipes the mouth with the back of the hand)."""

AUDIT_SYSTEM = """You are a strict QA reviewer for AI video edits. You compare GENERATED frames with the SOURCE frames at the same moments and with the REFERENCE pictures, against the edit spec. Report only problems you can actually see in the frames.
Reply with JSON only:
{"score": 0-10, "pass": true or false,
 "checks": {"identity": "ok or the problem", "location": "ok or the problem", "motion": "ok or the problem", "physics": "ok or the problem", "extras": "ok or the problem", "quality": "ok or the problem"},
 "issues": [{"severity": "critical|major|minor", "time": seconds, "what": "..."}],
 "fix": "one or two imperative sentences to add to the prompt that would fix the critical and major issues, or an empty string"}
The SOURCE frames are the ground truth for objects and actions, even where the spec says otherwise: anything held, used or added in a generated frame
that the source does not show at that moment is an invented object (major), and a different hand action (holding a cloth instead of opening a lid)
is a different action (critical).
Judge every check against the SOURCE frame at the same moment: if the source person is in the air (jumping, dancing, flipping), off the ground or
in water at that moment, the same in the generated frame is correct, not a problem. A cartoon, anime or illustrated reference character that keeps
its own art style is correct (not a quality problem) unless the user asked for a different style.
critical = a wrong or merged identity, the place changing back to the source place, a missing or duplicated main subject, a different action, garbled faces in close shots,
a person who is not supported by solid ground the way the source person is (walking, crawling or lying in water or in mid-air, sinking into a floor, passing through objects or walls).
major = clothing or colour clearly different from the picture, source objects left that should be replaced, timing off by more than about half a second, invented objects or text.
pass = score >= 7 and no critical issue.
When pass is false, "fix" must not be empty: say exactly what to remove, keep or change (for example "No balloons or objects hang from the top of the frame; above the hills there is only open blue sky.")."""


# ---------------------------------------------------------------------------- media helpers
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate,nb_frames:stream_side_data=rotation:format=duration",
                          "-of", "json", path], capture_output=True, text=True, check=True)
    d = json.loads(out.stdout); v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        raise ValueError("no video stream")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) in (90, 270):
        w, h = h, w
    num, den = (v.get("r_frame_rate") or "24/1").split("/"); fps = float(num) / max(1.0, float(den))
    dur = float(d.get("format", {}).get("duration") or 0)
    return {"w": w, "h": h, "fps": round(fps, 3), "duration": round(dur, 3), "audio": any(s.get("codec_type") == "audio" for s in d.get("streams", []))}


def canvas(w, h, quality):
    r = w / h
    if r > 1.5: big, small = (1344, 768), (832, 480)
    elif r < 0.67: big, small = (768, 1344), (480, 832)
    elif r > 1.15: big, small = (1152, 864), (768, 576)
    elif r < 0.87: big, small = (864, 1152), (576, 768)
    else: big, small = (1024, 1024), (640, 640)
    return (small if quality == "fast" else big), small


def read_frames(path, idx=None):
    cap = cv2.VideoCapture(path); F = []; i = 0; want = set(idx) if idx is not None else None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if want is None or i in want:
            F.append(f)
        i += 1
    cap.release()
    return F


def write_video(frames, path, crf=14):
    h, w = frames[0].shape[:2]; tmp = path + ".avi"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"FFV1"), FPS, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", path], check=True)
    os.remove(tmp)


def grid_len(n):
    """largest 17j+5 <= n (H3 renders 17j+5 frames)"""
    return max(22, ((n - 5) // 17) * 17 + 5)


def plan_segments(F, duration):
    if duration != "full" or F <= 124:  # one segment: round UP to H3's 17j+5 grid (the window repeats the last frame), the output is cut back to F
        return [{"k": 0, "start": 0, "length": 124 if F >= 124 else next(17 * j + 5 for j in range(1, 8) if 17 * j + 5 >= max(22, F))}]
    segs = [{"k": 0, "start": 0, "length": 124}]; k = 1
    while 119 * k + 5 < F:
        s = 119 * k; R = F - s
        L = 124 if R > 107 else max(22, next(17 * j + 5 for j in range(1, 8) if 17 * j + 5 >= R))
        segs.append({"k": k, "start": s, "length": L}); k += 1
    return segs


def plan_parts(F, cuts, duration="full"):
    """the people path's parts (2026-10-22 way: up to 124 frames rendered whole, the next part continues from the last 5 frames), except that a
    part whose last second holds a cut ENDS at that cut and the next part starts fresh there (2026-10-29, test-files case 4: a cut 0.4 s before
    the part's end - H3 cut to the shrine ~0.5 s early in both takes; on 2026-10-22 such tails kept the source people)"""
    if duration != "full" or F <= 124:
        return plan_segments(F, duration)
    cuts = sorted(c for c in set(cuts) if 0 < c < F); segs = []; s = 0; cont = False; k = 0
    while s < F:
        e = min(F, s + 124); at_cut = False
        if e < F:
            # a fresh start only on a shot of >= 32 frames (1.3 s): in a fast montage (test-files case 9, 4-11 frame shots) the part had nothing to
            # continue from and came out of order (score 0); case 6's 30-frame blurred transition after its cut at 108 became an invented split
            # screen (score 0) - the same part continued across the cut scored 10
            nxt = lambda c: min([x for x in cuts if x > c] + [F])
            near = [c for c in cuts if s + 48 <= c <= e and c > e - 45 and nxt(c) - c >= 32]   # last 45 frames (P04: a drone shot 40 frames before the end)
            if near:
                e = max(near); at_cut = True
        n = e - s
        L = 124 if n > 107 else next(17 * j + 5 for j in range(1, 8) if 17 * j + 5 >= max(22, n))
        segs.append({"k": k, "start": s, "length": L, "end": e, "cont": cont}); k += 1
        if e >= F:
            break
        if at_cut:   # a new shot: no continuation across a cut (a plain 124-frame end that happens to fall on a cut still continues)
            s, cont = e, False
        else:
            s, cont = e - 5, True
            if F - s <= 5:
                break
    return segs


def colour_steps(frames, w=3):
    """Hellinger distance between the colour histograms of the w frames before and the w frames from each index: a scene change (new place,
    new light) is a big step even inside fast motion or motion blur; motion within one shot barely moves it"""
    H = []
    for f in frames:
        hsv = cv2.cvtColor(cv2.resize(f, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256]).ravel(); H.append(h / max(1e-6, h.sum()))
    H = np.array(H); n = len(H); D = np.zeros(n)
    for i in range(1, n):
        a = H[max(0, i - w):i].mean(0); b = H[i:min(n, i + w)].mean(0)
        D[i] = float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(a * b))))))
    return D


def detect_cuts(path):
    """frame indices where a hard cut likely starts: a lone spike in pixel change, or a big colour step (scene change hidden in motion blur)"""
    cap = cv2.VideoCapture(path); g = []; raw = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        raw.append(f)
        g.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32))
    d = np.array([0.0] + [float(np.abs(g[i] - g[i - 1]).mean()) for i in range(1, len(g))])
    cuts = []
    for i in range(1, len(d)):
        nb = np.concatenate([d[max(1, i - 6):i], d[i + 1:i + 7]])
        if d[i] > max(18.0, 4.0 * (np.median(nb) if len(nb) else 0) + 4.0) and d[i] >= d[max(1, i - 2):i + 3].max():
            iso = d[i] / max(1e-3, max(d[max(1, i - 2)], d[min(len(d) - 1, i + 2)]))   # fast motion keeps its neighbours high; a cut is a lone spike
            if iso >= 3.0 or (d[i] >= 60.0 and iso >= 1.6):
                cuts.append(i)
    if len(raw) > 6:
        D = colour_steps(raw)
        for i in range(2, len(D) - 1):
            if D[i] >= 0.6 and D[i] >= D[max(1, i - 3):i + 4].max() and not any(abs(i - c) <= 3 for c in cuts):
                cuts.append(i)
        for i in range(2, len(d) - 2):   # a cut inside fast motion: the spike stays under the 4x-median gate, but it is still >= 2x its
            if any(abs(i - c) <= 3 for c in cuts):   # neighbours and the colours step at the same frame (13 missed cuts, 0 false on the test set)
                continue
            if d[i] >= 30.0 and d[i] >= d[i - 2:i + 3].max() and d[i] / max(1e-3, max(d[i - 2], d[i + 2])) >= 2.0 and D[max(1, i - 1):i + 2].max() >= 0.4:
                cuts.append(i)
                continue
            nb = np.concatenate([d[max(1, i - 6):i], d[i + 1:i + 7]])   # a lone ONE-frame spike between similar-looking shots (grey city to grey
            if (d[i] >= 30.0 and d[i] >= d[i - 2:i + 3].max() and d[i] / max(1e-3, max(d[i - 2], d[i + 2])) >= 3.0   # city, no colour step);
                    and d[i] >= 3.0 * float(np.median(nb)) and d[i] >= 2.5 * max(d[i - 1], d[i + 1])):           # a splash rises over 2 frames
                cuts.append(i)
    return sorted(cuts)


def detect_soft_cuts(path, hard=()):
    """transitions without a hard cut - a hand over the lens, a whip pan, a flash, a blurred push-in: the picture loses its detail for a few frames
    (a 'dip'), then a new view starts. Returns the frame where the picture is sharp again after each dip (a splice there hides behind the
    transition). The dip is measured against the sharpness of the shot BEFORE it."""
    cap = cv2.VideoCapture(path); sh = []; hs = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        small = cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA)
        sh.append(float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()))
        hh = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256]).ravel(); hs.append(hh / max(1e-6, hh.sum()))
    n = len(sh)
    def moved(a0, b0):   # Hellinger distance between the colours before the dip and after it (same scene through spray/blur: 0.07-0.25)
        a = np.mean(hs[max(0, a0 - 5):max(1, a0 - 1)], 0); b = np.mean(hs[b0:min(n, b0 + 4)], 0)
        return float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(a * b))))))
    if n < 10:
        return []
    sh = np.array(sh); out = []; i = 6
    while i < n - 3:
        ref = float(np.percentile(sh[max(0, i - 24):i], 75))               # how sharp the running shot is (its sharper frames: robust to a slow fade)
        if ref > 0 and sh[i] < 0.30 * ref:                                  # a dip starts
            j = i
            while j + 1 < n and sh[j + 1] < 0.45 * ref:
                j += 1
            if j - i + 1 >= 2:                                              # at least 2 frames of blur (not one noisy frame)
                after = sh[j + 1:min(n, j + 9)]
                k = next((j + 1 + q for q, v in enumerate(after) if v >= 0.85 * float(np.median(after))), None) if len(after) else None
                if (k is not None and k < n and not any(-3 <= k - h <= 10 for h in hard) and not any(abs(k - o) <= 6 for o in out)
                        and moved(i, k) >= 0.3):   # a new view after the dip (spray, a pan or motion blur inside one shot stays under 0.3)
                    out.append(int(k))
            i = j + 1
        else:
            i += 1
    return out


MAX_MERGED_CUTS = 2   # a segment of merged short shots holds at most this many cuts; the next short shot gets its own (padded) render


def plan_shot_segments(F, cuts):
    """segments that never cross a cut: every shot is rendered on its own (a shot longer than 124 frames in <=124-frame chunks with a 5-frame
    continuity overlap); shots under 22 frames are merged into a neighbour. Each segment: start, length (render length on H3's 17j+5 grid; the
    window is padded with the shot's last frame), end (last real frame + 1), cont (continues the previous segment inside the same shot), shot."""
    b = [0] + sorted(c for c in set(cuts) if 0 < c < F) + [F]
    shots = [[b[i], b[i + 1]] for i in range(len(b) - 1) if b[i + 1] > b[i]]
    merged = []; cuts_in = []   # cuts inside each merged segment: at most MAX_MERGED_CUTS (2026-10-29, test-files case 9: a 10-cut montage of
    for sh in shots:              # 4-27 frame shots was ONE segment and H3 rendered it out of step with the source - score 0 after the retake)
        if merged and (sh[1] - sh[0] < 22 or merged[-1][1] - merged[-1][0] < 22) and cuts_in[-1] < MAX_MERGED_CUTS:
            merged[-1][1] = sh[1]; cuts_in[-1] += 1
        else:
            merged.append(list(sh)); cuts_in.append(0)
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < 22 and cuts_in[-2] < MAX_MERGED_CUTS:
        merged[-2][1] = merged[-1][1]; merged.pop()
    segs = []; k = 0
    for si, (a, e) in enumerate(merged):
        n = e - a
        if n <= 124:
            L = next(17 * j + 5 for j in range(1, 8) if 17 * j + 5 >= max(22, n))
            segs.append({"k": k, "start": a, "length": L, "end": e, "cont": False, "shot": si}); k += 1
            continue
        c0 = -(-(n - 5) // 119)                                    # a long shot: EQUAL chunks with the 5-frame continuity overlap
        L = next(17 * j + 5 for j in range(1, 8) if 17 * j + 5 >= -(-(n + 5 * (c0 - 1)) // c0))   # (not 124 + a fragile 13-frame tail)
        for i, st in enumerate(range(a, e - 5, L - 5)):
            segs.append({"k": k, "start": st, "length": L, "end": min(e, st + L), "cont": i > 0, "shot": si}); k += 1
            if st + L >= e:
                break
    return segs


def join_segment(out, sg, frames, F):
    """put a rendered segment into the output frame list (out has F slots): a continuation cross-fades its first 5 frames, a new shot is a cut"""
    st = sg["start"]; end = min(F, sg.get("end", st + len(frames))); n = end - st
    cont = sg.get("cont", st > 0 and "end" not in sg)
    for i in range(min(n, len(frames))):
        if cont and i < 5 and out[st + i] is not None:
            a = (i + 1) / 6.0; out[st + i] = cv2.addWeighted(out[st + i], 1 - a, frames[i], a, 0)
        else:
            out[st + i] = frames[i]
    return out


def mentions_to_tags(text):
    text = re.sub(r"@\s?video\s?(\d+)", lambda m: f"<Video {m.group(1)}>", text, flags=re.I)
    return re.sub(r"@\s?(?:image|img|picture|pic)\s?(\d+)", lambda m: f"<Picture {m.group(1)}>", text, flags=re.I)


def swap_recipe_cfg():
    """data/swap_recipe.json, e.g. {"enabled": true, "sheet": true} - Objects swap through the object recipe (recipe/redoutline3.py); absent = on"""
    from .store import DATA as SDATA
    try:
        return json.load(open(os.path.join(SDATA, "swap_recipe.json")))
    except Exception:
        return {"enabled": True, "sheet": True}


def swap_recipe_on():
    return bool(swap_recipe_cfg().get("enabled"))


RECIPE_PERSON = r"\b(man|woman|person|boy|girl|child|kid|guy|lady|dancer|rider|people|couple|skater|player|character)\b"
RECIPE_ANIMAL = (r"\b(animals?|pets?|dogs?|pupp(?:y|ies)|corgis?|cats?|kittens?|horses?|ponies|pony|stallions?|foals?|birds?|parrots?|ducks?|chickens?|hens?|"
                 r"roosters?|cows?|calf|calves|pigs?|sheep|lambs?|goats?|rabbits?|bunn(?:y|ies)|bears?|lions?|tigers?|monkeys?|apes?|gorillas?|elephants?|"
                 r"giraffes?|zebras?|deer|fox(?:es)?|wol(?:f|ves)|squirrels?|hamsters?|turtles?|tortoises?|frogs?|snakes?|lizards?|sharks?|dolphins?|"
                 r"whales?|penguins?|owls?|eagles?|camels?|llamas?|alpacas?|kangaroos?|koalas?|pandas?|raccoons?|otters?|dinosaurs?)\b")


def recipe_skip(spec):
    """why an Objects swap does NOT go through the object recipe, or None (2026-10-29, the ~/test-files 'object swap' set = character and
    place swaps): redoutline3's shot_spec keeps only the replaced things (a location subject and its picture are dropped, so the place never
    changes) and autofix checks things only, never people. Character and place swaps go through the per-segment swap, whose Check & repair
    (identity_check, leftover people, place reverts) was built on that set."""
    subs = spec.get("subjects") or []
    if any(s.get("kind") == "location" for s in subs):
        return "a place"
    if any(s.get("kind") == "person" or (s.get("kind") == "animal" and re.search(RECIPE_PERSON, str(s.get("replaces") or ""), re.I)) for s in subs):
        return "a person"
    if any(s.get("kind") == "animal" and re.search(RECIPE_ANIMAL, str(s.get("replaces") or ""), re.I) for s in subs):
        return "an animal"   # an animal for an animal (a dog -> a tiger); an animal for a THING (motorcycle -> horse) stays with the object recipe
    return None


ROUTE_Q = """You see a user's instruction for editing a video and the reference pictures they added. List every replacement the edit makes.
Reply with JSON only: {"replacements": [{"original": "what is replaced in the video", "original_kind": "person" or "animal" or "place" or "thing", "new_kind": "person" or "animal" or "place" or "thing"}]}
"person" = a person or a character (human, cartoon or game character, robot, mascot); "animal" = a living animal; "place" = the location or
background; "thing" = an object - a phone, a fruit, a hat, a cup, a ball, a bag, a vehicle, a toy (a toy animal or a toy robot is a thing)."""


def people_swap_guess(job):
    """two swap paths (user 2026-10-29: "One for people swap and maybe animals and one for objects like phones, pineapples and hats"): True when
    an Objects swap replaces people, animals or a place (the 2026-10-22 way), False for things only (the object recipe), None when unsure.
    Asked BEFORE the plan because the two paths split the video differently; the spec written afterwards has the last word (run_create plans
    again when recipe_skip(spec) disagrees)."""
    try:
        txt = mentions_to_tags(job["prompt"]) if job.get("prompt_on") and job.get("prompt") else "none - decide from the pictures"
        content = [{"type": "text", "text": f"Instruction: {txt}\nReference pictures:"}]
        for i, p in enumerate(job["images"], 1):
            content += [{"type": "text", "text": f"<Picture {i}>:"}, llm.img_part(p, 384)]
        r = llm.ask(content, system=ROUTE_Q, max_tokens=300, temperature=0.0)
        reps = [x for x in (r.get("replacements") or []) if isinstance(x, dict)] if isinstance(r, dict) else []
    except Exception:
        return None
    if not reps:
        return None
    kind = lambda x, k: str(x.get(k) or "").strip().lower()
    return any(kind(x, "original_kind") in ("person", "place") or kind(x, "new_kind") == "place" or
               (kind(x, "original_kind") == "animal" and kind(x, "new_kind") in ("animal", "person")) for x in reps)


# ---------------------------------------------------------------------------- prompt assembly
def renumber(text, pic_map, subj_map):
    text = re.sub(r"<Picture (\d+)>", lambda m: f"<Picture {pic_map.get(int(m.group(1)), int(m.group(1)))}>", text)
    return re.sub(r"<Subject (\d+)>", lambda m: f"<Subject {subj_map.get(int(m.group(1)), int(m.group(1)))}>", text)


FIXTURES = re.compile(r"\b(doors?|doorways?|arch(?:way)?s?|walls?|rugs?|carpets?|clocks?|pictures?|paintings?|posters?|frames?|windows?|curtains?|"
                      r"blinds?|shel(?:f|ves)|cabinets?|cupboards?|refrigerators?|fridges?|sofas?|couch(?:es)?|floors?|ceilings?|lamps?|chandeliers?|"
                      r"stairs?|staircases?|counters?|hallways?|corridors?|rooms?|bookcases?|mirrors?|fireplaces?|radiators?|plants?|tiles?|furniture|"
                      r"whiteboards?|blackboards?|chalkboards?)\b", re.I)


STREET_FIXTURES = re.compile(r"\b(traffic lights?|traffic signals?|signals?|signs?|signposts?|poles?|lampposts?|street ?lights?|railings?|fences?|"
                             r"barriers?|buildings?|storefronts?|shopfronts?|shops?|billboards?|statues?|benches?)\b", re.I)


def old_fixture_items(kept, swap=False, new_desc=""):
    """the 'kept' items that belong to the replaced place (they are named in the location line as replaced): a fixture word the new place's
    description does not have (2026-10-30, case 1 re-run: 'the white cottage and stone building' - the VILLAGE's own - was listed as replaced)"""
    body = re.sub(r"\s*\b(stay|stays|remain|remains)\b.*$", "", str(kept or "").strip().rstrip("."), flags=re.I)
    items = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+(?=(?:the|a|an|his|her|their|its)\b)", body) if x.strip()]
    nd = str(new_desc or "").lower()
    def old(x):
        words = [m.group(0).lower() for m in FIXTURES.finditer(x)] + ([m.group(0).lower() for m in STREET_FIXTURES.finditer(x)] if swap else [])
        return bool(words) and any(w not in nd for w in words)
    return [re.sub(r"^(?:the|a|an)\s+", "", x, flags=re.I) for x in items if old(x)]


def kept_new_place(kept, swap=False):
    """the 'kept' line of a segment whose place is REPLACED: only what the people hold or use stays - never the old place's walls, doors, rugs,
    clocks or pictures (2026-10-29, test-files case 8: 'kept: the glass jar, the white marble table, the white door, the arched doorway, the
    geometric rug, the wall clock ...' in the playroom shots, and H3 kept the old beige room with its clock and door)"""
    body = re.sub(r"\s*\b(stay|stays|remain|remains)\b.*$", "", str(kept or "").strip().rstrip("."), flags=re.I)
    items = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+(?=(?:the|a|an|his|her|their|its)\b)", body) if x.strip()]   # 'red and black' stays one
    items = [x for x in items if not FIXTURES.search(x) and not (swap and STREET_FIXTURES.search(x))]
    return (", ".join(items) + ".") if items else ""   # the prompt line already says "Everything else stays exactly as in <Video 1>:"


COPY_WORDS = re.compile(r"\b(double[- ]exposures?|doubles?|copy|copies|clones?|reflections?|mirror(?:ed)? images?|twins?|second (?:blurred )?(?:figure|image))\b", re.I)


def kept_no_copies(kept):
    """a copy of a replaced person is never 'kept' (2026-10-29, test-files case 6: "everything else stays exactly as in <Video 1>: ... the
    double-exposure copy of the woman" - H3 kept the source woman as the blurred double; the 2026-10-22 hand-written prompt said both figures
    become the new woman and got it right)"""
    body = str(kept or "").strip()
    if not COPY_WORDS.search(body):
        return kept
    body = re.sub(r"\s*\b(stay|stays|remain|remains)\b.*$", "", body.rstrip("."), flags=re.I)
    items = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+(?=(?:the|a|an|his|her|their|its)\b)", body) if x.strip()]
    items = [x for x in items if not COPY_WORDS.search(x)]
    return ", ".join(items) if items else ""


def seg_keep(spec, seg, swap=False):
    """the subjects segment `seg` shows: its 'present' people/animals/objects + the new place. present == [] means nobody (2026-10-29, test-files
    case 8: an empty list fell back to ALL subjects, and the people-less debris shot became the playroom with the young woman in it); in an
    Objects swap whose spec says where the new place applies ('spot'), the place is only used in those segments (the example keeps the
    fiery close-up, the city street and the Earth shots as they are)"""
    subjects = spec.get("subjects", []); present = seg.get("present")
    locs = [s for s in subjects if s.get("kind") == "location"]
    spots = swap and any(str(g.get("spot") or "").strip() for g in spec.get("segments", []))
    loc_here = locs if (not spots or str(seg.get("spot") or "").strip()) else []
    if present is None or not isinstance(present, list):
        return [s for s in subjects if s.get("kind") != "location"] + loc_here
    who = [s for s in subjects if s.get("tag") in present and s.get("kind") != "location"]
    only_place = any(s.get("tag") in present and s.get("kind") == "location" for s in subjects)   # people10 P04: present = [the location] for a
    if present and not who and not only_place:   # tags that match nobody: as before, everyone       # people-less drone shot = nobody
        who = [s for s in subjects if s.get("kind") != "location"]
    return who + loc_here


def segment_prompt(spec, k, images, style, swap=False):
    """the structured H3 prompt for segment k, listing only the subjects present in it (pictures renumbered to match)"""
    seg = next((s for s in spec.get("segments", []) if int(s.get("index", -1)) == k), None) or {"shots": "", "present": None}
    subjects = spec.get("subjects", [])
    keep = seg_keep(spec, seg, swap) or subjects
    pics = []; pic_map = {}; subj_map = {}
    for i, s in enumerate(keep, 1):
        m = re.search(r"(\d+)", str(s.get("tag", ""))); subj_map[int(m.group(1)) if m else i] = i
        p = s.get("picture")
        if isinstance(p, int) and 1 <= p <= len(images) and p not in pic_map:
            pics.append(images[p - 1]); pic_map[p] = len(pics)
    def rn(t):
        return renumber(t or "", pic_map, subj_map)
    lines = []
    for i, s in enumerate(keep, 1):
        p = s.get("picture"); src = f" whose appearance comes from <Picture {pic_map[p]}>" if p in pic_map else ""
        kind = s.get("kind", "person"); rep = s.get("replaces") or ""
        if kind == "location":
            inc = ""
            if swap:   # the old place's fixtures named as replaced (the 2026-10-22 hand-written prompts did; H3 otherwise keeps them)
                fx = list(dict.fromkeys(old_fixture_items(seg.get("kept") or spec.get("kept"), swap, s.get("description")) + list(spec.get("_old_fixtures") or [])))[:5]
                fx = [x for x in fx if x.lower() not in str(s.get("description") or "").lower()]
                inc = (", including its " + ", ".join(fx[:-1]) + (" and " if len(fx) > 1 else "") + fx[-1]) if fx else ""
            line = (f"<Subject {i}> is the location{src}: {rn(s.get('description'))} It replaces {rn(rep) or 'the place in <Video 1>'}{inc}, in every shot, and never changes back."
                    if inc else f"<Subject {i}> is the location{src}: {rn(s.get('description'))} It replaces {rn(rep) or 'the place in <Video 1>'} in every shot and never changes back.")
        else:
            perf = f" and whose performance comes from {rn(rep)} in <Video 1>" if rep and rep.lower() != "new" else ""
            if kind == "person" and (seg.get("costume_what") or {}).get(s.get("tag")) and rep:
                perf = f" and whose performance comes from {rn(plain_who(rep))} in <Video 1>"
            elif swap and kind in ("person", "animal") and rep and rep.lower() != "new" and plain_who(rep) != rep and \
                    sum(1 for o in keep if o.get("kind") in ("person", "animal") and plain_who(o.get("replaces") or "") == plain_who(rep)) == 1:
                # the source person named WITHOUT their clothes when that is still unambiguous (2026-10-30, test-files case 4: "the young woman in
                # the black hoodie, cap and checkered shorts" -> H3 hung the hoodie's sleeve on the new woman as a white cloth; 10-29: "the young man
                # in the grey t-shirt" put the airport t-shirt back on him)
                perf = f" and whose performance comes from {rn(plain_who(rep))} in <Video 1>"
            desc_ = rn(s.get('description'))
            if kind == "person" and s.get("tag") in (seg.get("costume") or []) and not s.get("costume_shot"):   # the costume replaces the picture's outfit
                desc_ = re.split(r",?\s+(?:wearing|dressed in|clad in|in a|in an|in her|in his|in their)\b", desc_ or "", maxsplit=1)[0].rstrip(" ,.") + "."
            line = f"<Subject {i}> is the {kind}{src}{perf}: {desc_}"
            if swap and rep and rep.lower() != "new" and kind in ("person", "animal") and COPY_WORDS.search(str(seg.get("shots") or "") + " " + str(seg.get("kept") or "")):
                line += f" Every figure of {rn(rep)} in <Video 1> - also a double-exposure copy, a reflection or a clone of them - becomes <Subject {i}>."
            if kind == "person" and p in pic_map:
                if s.get("costume_shot"):
                    line += (" Preserve the reference character's facial features, hairstyle, skin tone and body proportions; in this shot the clothes are the "
                             "costume shown in <Picture %d> and worn in <Video 1>, not other clothes. %s" % (pic_map[p], KEEP))
                elif s.get("tag") in (seg.get("costume") or []) and (seg.get("costume_what") or {}).get(s.get("tag")):   # people path: named in words
                    line += (f" In this part of the video they wear the same costume as {rn(plain_who(rep)) if rep else 'the original person'} in <Video 1>: "
                             f"{seg['costume_what'][s['tag']]} - not the clothes of <Picture {pic_map[p]}>. Preserve the reference character's facial features, "
                             f"hairstyle, skin tone and body proportions. {KEEP}")
                elif s.get("tag") in (seg.get("costume") or []):   # add_costumes found a themed costume here (case 4's kimonos): not the picture's clothes
                    line += (" Preserve the reference character's facial features, hairstyle, skin tone and body proportions; in this shot the clothes are the "
                             f"costume worn in <Video 1>, not the clothes of <Picture {pic_map[p]}>. {KEEP}")
                else:
                    line += f" {PRES} {KEEP}"
        lines.append(line)
    kept = seg.get("kept") or spec.get("kept")   # this segment's own surroundings first (shot by shot: never the other shots' places)
    if kept and any(s_.get("kind") == "location" for s_ in keep):
        kept = kept_new_place(kept, swap)
    if kept and swap:
        kept = kept_no_copies(kept)
    if kept:
        lines.append(f"Everything else stays exactly as in <Video 1>: {rn(kept)}")
    ret = [rn(r) for r in spec.get("retention", []) if any(f"<Subject {j}>" in rn(r) for j in range(1, len(keep) + 1))] or [rn(r) for r in spec.get("retention", [])]
    keep_tags = {o.get("tag") for o in keep}
    subset = swap and any(s_.get("tag") not in keep_tags for s_ in subjects)
    if subset:   # a part that shows only some of the subjects (2026-10-30, people10 P04's drone shot; test-files case 8's one-person parts): the clip-wide
        # retention lines are kept only when ALL their tags are in this part - after renumbering an absent subject's '<Subject 1>' line read as this
        # part's place ("the man's face ... stay exactly as in <Picture 1>" in the place-only part, and H3 drew him)
        own = [rn(r) for r in spec.get("retention", []) if set(re.findall(r"<Subject \d+>", str(r))) and set(re.findall(r"<Subject \d+>", str(r))) <= keep_tags]
        ret = own or [f"<Subject {i}> (throughout this part): fully_preserved - its appearance from its picture." for i in range(1, len(keep) + 1)]
    # the clip-wide 'replace' line goes into EVERY segment: with several segments it named the other shots' places (2026-10-29, test-files case 2:
    # "the room, the street ... stay" in the indoor shot -> H3 drew a split frame with a street and the full-body reference sheet as an inset)
    rep_line = (rn(spec.get('replace', 'replace what the subjects describe')) if len(spec.get("segments", [])) <= 1 else
                "only the subjects defined above are replaced; everything else stays as in this shot of <Video 1>")
    plain_t = ("Preserve the source scene's visual treatment while retaining the replacement character's distinctive design. Adapt illumination, shading, "
               "and perspective to the source scene.")
    treatment = rn(spec.get("treatment") or plain_t)
    one_place = any(s_.get("kind") == "location" for s_ in subjects)   # a new place for EVERY shot: its light belongs in the look line
    if len(spec.get("segments", [])) > 1 and not one_place and not re.search(r"cartoon|anime|illustrat|\b3d\b|3-d|clay|styli[sz]ed|painted|comic|pixel|watercolou?r", treatment, re.I):
        treatment = plain_t   # the clip-wide look sentence names every shot's light and place (it put a neon-tunnel ceiling over c6's desert shot)
    if style and style != "none":
        treatment = STYLES[style][1] + " Apply this style to every subject and the whole scene, consistently in every shot."
    shots = rn(seg.get("shots", "")).strip()
    spot = str(seg.get("spot") or "").strip()
    if spot and spot.lower() not in ("none", "empty", "n/a") and any(s_.get("kind") == "location" for s_ in subjects):
        shots += f"\n\nStaging in the new place: {rn(spot)}"
    corr = spec.get("_corrections", {}).get(str(k))
    if corr:
        shots += f"\n\nImportant corrections: {corr}"
    instr = rn(spec.get('instruction', 'Edit <Video 1> as described.'))
    summ = rn(spec.get('summary', ''))
    if subset:   # the clip-wide instruction and summary name the absent subjects with picture numbers this part does not have: written for this part
        what = []
        for i, o in enumerate(keep, 1):
            po = o.get("picture"); pic = f"<Picture {pic_map[po]}>" if po in pic_map else "its description"
            if o.get("kind") == "location":
                what.append(f"the surroundings with the location shown in {pic}")
            else:
                what.append(f"{rn(plain_who(o.get('replaces') or '') if o.get('kind') in ('person', 'animal') else (o.get('replaces') or ''))} "
                            f"with the {o.get('kind', 'subject')} shown in {pic}")
        join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
        instr = "In <Video 1>, replace " + join(what) + ". Only these subjects are in this part of the video."
        summ = ("The target video is <Video 1> with " + join([w.split(" with the ")[0] for w in what]) + " replaced as defined above, keeping every "
                "movement, camera move, cut and the timing.")
    if swap and keep and all(o.get("kind") == "location" for o in keep):   # a part with nobody in it: the clip-wide
        instr = ("Replace the surroundings in <Video 1> with the location shown in <Picture 1>, seen exactly as <Video 1>'s camera sees it - "   # instruction
                 "the same height, angle, distance and movement; this part of <Video 1> shows only the place.")   # named the people (P04's drone shot got them)
    prompt = f"""{instr}

subject_definitions:
{chr(10).join(lines)}
<Video 1> is the source video being edited and supplies the shot composition, camera movement, cuts, timing, and sequence of actions.

summary:
[video editing + reference generation] The target video is an edited version of <Video 1>. {summ}

retention_analysis:
{chr(10).join(ret)}
<Video 1>: partially_preserved - retain composition, camera motion, cuts, action timing, and duration; {rep_line}.

detailed_description:
{treatment}

{shots}

{FOLLOW}

{TAIL}"""
    return prompt, pics


def template_spec(job, segs):
    """no LLM installed: a plain spec from the user's text and the pictures"""
    if job.get("tool") == "control":
        img = job.get("scene_used") == "image"
        instr = "Replace the main person in <Video 1> with the character shown in <Picture 1>, performing all of their movements." + (
            " Replace the whole place with the place shown in <Picture 2>." if img else " Everything else stays exactly as in <Video 1>.")
        if job.get("prompt"):
            instr += " " + mentions_to_tags(job["prompt"])
        subjects = [{"tag": "<Subject 1>", "picture": 1, "kind": "person", "replaces": "the main person", "description": "the character shown in <Picture 1>, with exactly that face, hair, body and outfit."}]
        if img:
            subjects.append({"tag": "<Subject 2>", "picture": 2, "kind": "location", "replaces": "the place in <Video 1>", "description": "the place shown in <Picture 2>."})
        return {"instruction": instr, "subjects": subjects, "kept": "" if img else "the place, other people, objects, light and camera of <Video 1>", "summary": instr,
                "retention": [f"<Subject {i + 1}> (throughout): fully_preserved." for i in range(len(subjects))], "replace": "replace the main person" + (" and the place" if img else ""),
                "treatment": "", "segments": [{"index": sg["k"], "shots": "", "present": None} for sg in segs], "_source": "template"}
    n = len(job["images"]); user = mentions_to_tags(job.get("prompt") or "") if job.get("prompt_on") else ""
    if job["mode"] == "swap":
        instr = user or "Replace the main subject of <Video 1> with the one shown in <Picture 1>."
    else:
        instr = user or ("Replace the people in <Video 1> with the characters shown in " + ", ".join(f"<Picture {i}>" for i in range(1, n + 1)) + ".")
    subjects = [{"tag": f"<Subject {i}>", "picture": i, "kind": "person", "replaces": "", "description": f"the character or thing shown in <Picture {i}>, with exactly its look."} for i in range(1, n + 1)]
    return {"instruction": instr, "subjects": subjects, "kept": "", "summary": instr, "retention": [f"<Subject {i}> (throughout): fully_preserved - appearance from <Picture {i}>." for i in range(1, n + 1)],
            "replace": "replace what the instruction says", "treatment": "", "segments": [{"index": s["k"], "shots": "", "present": None} for s in segs], "_source": "template"}


def spec_with_llm(job, src_path, segs):
    mode_txt = "Objects swap" if job["mode"] == "swap" else "Motion transfer"
    if job.get("tool") == "control":
        mode_txt = CONTROL_MODE[job.get("scene_used", "video")]
    content = [{"type": "text", "text": f"Mode: {mode_txt}.\n"
                f"User instruction: {mentions_to_tags(job['prompt']) if job.get('prompt_on') and job.get('prompt') else 'none - decide from the pictures and the source'}.\n"
                f"Style: {STYLES.get(job.get('style') or 'none', STYLES['none'])[0]}.\nReference pictures:"}]
    for i, p in enumerate(job["images"], 1):
        content += [{"type": "text", "text": f"<Picture {i}>:"}, llm.img_part(p, 640)]
    if not job["images"]:
        content.append({"type": "text", "text": "none - there are NO reference pictures, so never write <Picture n>; describe new things in words (\"picture\": null)."})
    content.append({"type": "text", "text": f"Source video <Video 1> frames, 2 per second ({len(segs)} segment(s)):"})
    cuts = detect_cuts(src_path); side = 640 if len(segs) == 1 else (448 if len(segs) <= 3 else 384)
    for s in segs:
        e = min(s.get("end") or s["start"] + s["length"], s["start"] + s["length"])   # shot by shot: only the shot's real frames (the render holds its last one)
        idx = list(range(s["start"], e, 12)); last = e - 1
        if last - idx[-1] >= 6:
            idx.append(last)
        frames = read_frames(src_path, idx)
        sc = [round((c - s["start"]) / FPS, 2) for c in cuts if s["start"] < c < e]
        content.append({"type": "text", "text": f"Segment {s['k']} (source {s['start'] / FPS:.1f}-{e / FPS:.1f} s)" +
                        ("; ONE continuous shot of the source (no cut inside it)" if "shot" in s and not sc else
                         f"; possible hard cuts at t = {', '.join(f'{x:.2f}' for x in sc)} s (big pixel jumps - fast motion near the camera causes them too, so check the frames)"
                         if sc else "; no big pixel jump found") + ":"})
        for i, f in zip(idx, frames):
            content += [{"type": "text", "text": f"t={(i - s['start']) / FPS:.1f}s"}, llm.img_part(f, side)]
    ctx = job.get("_context_frames") or []
    if ctx:
        content.append({"type": "text", "text": "Context only (NOT rendered): the source continues like this after the last segment - use it to tell who a "
                                                "partly visible person at the end is:"})
        content += [llm.img_part(f, 384) for f in ctx]
    spec = llm.ask(content, system=SPEC_SYSTEM, max_tokens=3500 if len(segs) <= 3 else min(6000, 3500 + 600 * (len(segs) - 3)))   # 6 parts ran out at 3500
    spec["_source"] = llm.NAME
    n = len(job["images"])  # drop references to pictures that don't exist
    fix = lambda t: re.sub(r"\s*(?:from|in|shown in|as in|like)?\s*<Picture (\d+)>", lambda m: m.group(0) if int(m.group(1)) <= n else "", t) if isinstance(t, str) else t
    for k in ("instruction", "kept", "summary", "replace", "treatment"):
        spec[k] = fix(spec.get(k))
    spec["retention"] = [fix(r) for r in spec.get("retention", [])]
    for sub in spec.get("subjects", []):
        if not isinstance(sub.get("picture"), int) or not (1 <= sub["picture"] <= n):
            sub["picture"] = None
        sub["description"] = fix(sub.get("description")); sub["replaces"] = fix(sub.get("replaces"))
        if isinstance(sub.get("replaces"), str):
            sub["replaces"] = re.sub(r"\s+(?:in|from|of)\s+<Video 1>\.?\s*$", "", sub["replaces"].strip())
    for g in spec.get("segments", []):
        g["shots"] = strip_subject_clothes(fix(g.get("shots")), spec); g["kept"] = fix(g.get("kept"))
        if job.get("mode") == "swap":
            g["shots"], gone = drop_old_fixtures(g["shots"], spec)
            if gone:
                spec.setdefault("_old_fixtures", []).extend(x.lower() for x in gone if x.lower() not in spec.get("_old_fixtures", []))
    return spec


GARMENT = (r"t-shirt|tee|shirt|top|blouse|tank top|camisole|cardigan|sweater|jumper|sweatshirt|hoodie|jacket|coat|blazer|dress|gown|jeans|trousers|"
           r"pants|shorts|skirt|jumpsuit|overalls|vest|outfit|clothes|clothing|tracksuit|cap|hat|beanie|scarf|headscarf|sneakers|boots|shoes|"
           r"headband|bow|glasses|sunglasses|necklace|earrings|bracelet|gloves|socks|belt|bag|backpack|choker|leggings|sleeve|sleeves")
THEMED = re.compile(r"\b(kimono|haori|hanbok|yukata|hakama|sari|uniform|armou?r|costume|robe|cloak|cape|spacesuit|wetsuit)\b", re.I)


FRAMING_Q = ('You see one frame of a video. Which parts of the main person are inside the frame? Reply with JSON only: {"framing": "full body" or '
             '"knees up" or "waist up" or "chest up" or "face" or "no person"}')


def add_framing(job, spec, src, segs):
    """waist-up / chest-up / face shots marked as close-ups (2026-10-29, test-files case 2: a waist-up talking shot with the picture's whole outfit
    in the prompt - jacket, jeans, sneakers - made H3 paste the reference sheet's full-body figure in as a picture-in-picture, 4 takes in a row; the
    spec left 'closeup' empty). A segment with people and no closeup gets one framing question on its middle source frame; closeup_refs then crops
    the pictures to those parts, as for feet close-ups (B1)."""
    if job.get("mode") != "swap" or not any(s.get("kind") == "person" for s in spec.get("subjects", [])):
        return spec
    F = len(read_frames(src))
    for g in spec.get("segments", []):
        if str(g.get("closeup") or "").strip() or not g.get("present"):
            continue
        sg = next((x for x in segs if str(x["k"]) == str(g.get("index"))), None)
        if not sg:
            continue
        e = min(F, sg.get("end", sg["start"] + sg["length"])); fr = read_frames(src, [sg["start"] + max(0, e - sg["start"]) // 2])
        if not fr:
            continue
        r = llm.ask([llm.img_part(fr[0], 640)], system=FRAMING_Q, max_tokens=60, temperature=0.0)
        f_ = str((r or {}).get("framing", "") if isinstance(r, dict) else "").strip().lower()
        if f_ in ("waist up", "chest up"):
            g["closeup"] = "face and upper body"; g["_framed"] = f_
        elif f_ == "face":
            g["closeup"] = "face"; g["_framed"] = f_
    return spec


COSTUME_ANY = """You get FRAMES FROM ACROSS A WHOLE VIDEO in time order. Does any person change into a THEMED costume later in the video - traditional dress (a kimono, a hanbok), a uniform, armour, a stage or fancy-dress costume - that differs from the outfit they wear when they first appear? Ordinary other clothes (another t-shirt, top, cardigan or jacket) are NOT a costume. Reply with JSON only: {"costume": true or false}"""
COSTUME_WHAT = ('You see one frame of a video. The person: {who}. Is that person visible in this frame (at least part of their body or clothes)? If yes, '
                'describe the outfit they wear IN THIS FRAME in one short phrase - the garments, colours and patterns (for example "a black haori jacket '
                'with white floral embroidery over a white kimono and grey striped hakama"). Reply with JSON only: {{"visible": true or false, "what": "..."}}')


def source_frames(job, n=8):
    cap = cv2.VideoCapture(job["video"]); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0; nf = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = min(nf / fps, MAX_SECONDS if job.get("duration") == "full" else 5.2); fr = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (i + 0.5) * dur / n * 1000.0); ok, f = cap.read()
        if ok:
            fr.append(f)
    cap.release(); return fr


def any_costume(job):
    """one question for the whole source before the per-shot costume checks of the people path (most videos have none). Unsure -> True."""
    try:
        fr = source_frames(job)
        r = llm.ask([{"type": "text", "text": "FRAMES FROM ACROSS THE WHOLE VIDEO (in time order):"}] +
                    [x for i, f in enumerate(fr, 1) for x in ({"type": "text", "text": f"{i}:"}, llm.img_part(f, 448))],
                    system=COSTUME_ANY, max_tokens=60, temperature=0.0)
        return bool(r.get("costume", True)) if isinstance(r, dict) else True
    except Exception:
        return True


COSTUME_WHO = """You see one frame of a video and the people an edit replaces:
{people}
Which of them can be seen in this frame (at least part of their body or clothes)? For each one you can see, describe the outfit they wear IN THIS
FRAME in one short phrase - the garments, colours and patterns (for example "a black haori jacket with white floral embroidery over a white kimono
and grey striped hakama"). Reply with JSON only: {{"visible": {{"A": true or false}}, "what": {{"A": "..."}}}}"""


def costume_seen(frame, people):
    """{tag: (visible, what)} for the replaced people in one frame - asked together, so a close-up of one person is not read as the other
    (test-files case 4: the man's haori close-up was described as the woman's outfit)"""
    labels = {chr(65 + i): s for i, s in enumerate(people)}
    txt = "\n".join(f"{L} = {plain_who(s.get('replaces') or s.get('tag'))}" for L, s in labels.items())
    try:
        r = llm.ask([llm.img_part(frame, 768)], system=COSTUME_WHO.format(people=txt), max_tokens=300, temperature=0.0)
        vis = r.get("visible") or {} if isinstance(r, dict) else {}; what = r.get("what") or {} if isinstance(r, dict) else {}
        out = {}
        for L, s in labels.items():   # 'visible' with nothing to describe = not there (a black frame, the other person's close-up)
            w = str(what.get(L) or "").strip().rstrip(".")[:300]; w = w[:1].lower() + w[1:]
            out[s["tag"]] = (bool(vis.get(L, True)) and bool(w), w)
        return out
    except Exception:
        return {s["tag"]: (False, "") for s in people}   # unsure: no costume (the picture's clothes, as before)


def plain_who(rep):
    """'the young woman in the black hoodie and checkered shorts' -> 'the young woman' (the outfit words made Qwen describe the airport
    clothes on a shrine frame where she was not even visible)"""
    out = re.sub(r",?\s+(?:in|wearing|dressed in|with)\s+.*?(?=,?\s+(?:who|that|which)\b|$)", "", str(rep or "the person"), count=1)   # a 'who ...' clause stays
    return re.sub(r"\s+(?:throughout|in)\s+<Video 1>\s*$", "", out).strip(" ,") or "the person"


def costume_what(frame, who):
    """(visible, what) for one person in one frame"""
    try:
        r = llm.ask([llm.img_part(frame, 768)], system=COSTUME_WHAT.format(who=plain_who(who)[:200]), max_tokens=120, temperature=0.0)
        if not isinstance(r, dict):
            return True, ""
        return bool(r.get("visible", True)), str(r.get("what") or "").strip().rstrip(".")[:300]
    except Exception:
        return True, ""


SAME_PLACE_Q = """You see two frames of one video. FRAME A is a shot without people; FRAME B is a shot of the video's main place. Is FRAME A the same
place as FRAME B - the same city, street, town, building, room or landscape, seen from another angle, height or distance (an aerial or drone
view of the same city or town counts as the same place)? Or is FRAME A clearly somewhere else - outer space or a planet seen from space, a
different world, a special-effect or dream scene, a clearly different landscape? Reply with JSON only: {"same_place": true or false, "why": "short"}"""


def add_place_scope(job, spec, src, segs):
    """a people-less part of a place swap that the spec left out of the new place (no 'spot'): decided in code, one frame against a frame of
    the people's place (2026-10-30, people10 P04: the spec put the snowy-city drone shot in the desert village in one run and kept the snowy
    city in the next; test-files case 8's Earth-from-space shot must stay). Same place -> the new place, seen as this shot's camera sees it."""
    if job.get("mode") != "swap":
        return spec
    loc = next((s for s in spec.get("subjects", []) if s.get("kind") == "location"), None)
    if not loc:
        return spec
    F = len(read_frames(src)); gs = spec.get("segments", [])
    ref = next((g for g in gs if str(g.get("spot") or "").strip() and any(x != loc.get("tag") for x in (g.get("present") or []))), None)
    rsg = next((x for x in segs if ref and str(x["k"]) == str(ref.get("index"))), None)
    if not rsg:
        return spec
    re_ = min(F, rsg.get("end", rsg["start"] + rsg["length"])); rf = read_frames(src, [min(F - 1, rsg["start"] + max(0, re_ - rsg["start"]) // 2)])
    for g in gs:
        if [x for x in (g.get("present") or []) if x != loc.get("tag")] or str(g.get("spot") or "").strip():
            continue
        sg = next((x for x in segs if str(x["k"]) == str(g.get("index"))), None)
        if not sg or not rf:
            continue
        e = min(F, sg.get("end", sg["start"] + sg["length"])); fr = read_frames(src, [min(F - 1, sg["start"] + max(0, e - sg["start"]) // 2)])
        if not fr:
            continue
        r = llm.ask([{"type": "text", "text": "FRAME A:"}, llm.img_part(fr[0], 640), {"type": "text", "text": "FRAME B:"}, llm.img_part(rf[0], 640)],
                    system=SAME_PLACE_Q, max_tokens=120, temperature=0.0)
        if isinstance(r, dict) and r.get("same_place") is True:
            cam = str(g.get("camera") or "").strip() or "the same height, angle and movement"
            g["spot"] = "the new place, seen exactly as this shot's camera sees it"; g["present"] = [loc["tag"]]
            g["shots"] = (f"[Shot 1] (0.0 to {(e - sg['start']) / FPS:.1f} seconds) {loc['tag']} seen exactly as <Video 1>'s camera sees this shot "
                          f"({cam}); nobody is in the shot.")
            g["_place_scope"] = str(r.get("why", ""))[:160]
    return spec


def add_costumes(job, spec, src, segs, part=False):
    """themed costumes found in code, not left to the spec writer (2026-10-29: under the themed-costume rule Qwen described test-files case 4's
    couple in their picture clothes at the shrine, where the source and the example show kimonos): each segment's people are checked against
    their first appearance (costume_confirm, themed only); a confirmed costume adds one sentence to that segment's shot text"""
    people = [s for s in spec.get("subjects", []) if s.get("kind") == "person"]
    if not people or job.get("mode") != "swap":
        return spec
    F = len(read_frames(src))
    if part:   # people path: a 5 s part can hold several shots - each shot checked, the costume named in words (as the 2026-10-22 prompts did)
        if not any_costume(job):
            return spec
        cuts = detect_cuts(src)
        for g in spec.get("segments", []):
            sg = next((x for x in segs if str(x["k"]) == str(g.get("index"))), None)
            if not sg:
                continue
            s0 = sg["start"]; e = min(F, sg.get("end", s0 + sg["length"])); b = [s0] + [c for c in cuts if s0 < c < e] + [e]
            pieces = [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] - b[i] >= 6]
            pres = [s for s in people if s.get("tag") in (g.get("present") or [])]
            if not pres:
                continue
            seen = {s["tag"]: 0 for s in pres}; hit = {s["tag"]: [] for s in pres}   # frames where each is visible; their shots in a costume
            for a, z in pieces:
                fr = read_frames(src, [(a + z) // 2])
                if not fr:
                    continue
                vw = costume_seen(fr[0], pres)
                for s in pres:
                    vis, what = vw.get(s["tag"], (True, ""))
                    if not vis:
                        continue
                    seen[s["tag"]] += z - a
                    if a > 0 and costume_confirm(job, fr[0], s.get("replaces") or s.get("tag")):   # the video's first shot = their own outfit
                        hit[s["tag"]].append((a, z, what))
            for s in pres:
                hs = hit[s["tag"]]
                if not hs:
                    continue
                what = next((w for _, _, w in hs if w), ""); wt = f"{what}, " if what else ""
                if sum(z - a for a, z, _ in hs) >= 0.8 * max(1, seen[s["tag"]]):
                    g["shots"] = (str(g.get("shots") or "").rstrip() + f" In this part {s['tag']} wears {wt}the costume the original person wears in "
                                  f"<Video 1> here, with the face, hair, skin and build of their picture.")
                    g.setdefault("costume", []).append(s["tag"]); g.setdefault("costume_what", {})[s["tag"]] = what or "the costume seen in <Video 1> here"
                else:
                    for a, z, w in hs:
                        g["shots"] = (str(g.get("shots") or "").rstrip() + f" From {(a - s0) / FPS:.1f} to {(z - s0) / FPS:.1f} seconds {s['tag']} wears "
                                      f"{(w or what) + ', ' if (w or what) else ''}the costume the original person wears in <Video 1> there, with the face, "
                                      f"hair, skin and build of their picture.")
                    g.setdefault("costume_part", []).append(s["tag"])
        return spec
    for g in spec.get("segments", []):
        sg = next((x for x in segs if str(x["k"]) == str(g.get("index"))), None)
        if not sg or sg["start"] == 0:
            continue
        e = min(F, sg.get("end", sg["start"] + sg["length"])); fr = read_frames(src, [sg["start"] + max(0, e - sg["start"]) // 2])
        if not fr:
            continue
        for s in people:
            if s.get("tag") not in (g.get("present") or []) or not costume_confirm(job, fr[0], s.get("replaces") or s.get("tag")):
                continue
            g["shots"] = (str(g.get("shots") or "").rstrip() + f" In this shot {s['tag']} wears the costume seen on the original person in <Video 1> here "
                          f"(its colours, patterns and cut), with the face, hair, skin and build of their picture.")
            g.setdefault("costume", []).append(s["tag"])
    return spec


def headwear_off(text, spec):
    """a replaced person's headwear named in a shot text that their picture does not have becomes their hair (2026-10-29, test-files case 1,
    people path: "adjusting a cap on his head", "hand on his cap" - H3 put the source man's cap on the new man, whose picture has none)"""
    desc = {s.get("tag"): str(s.get("description") or "").lower() for s in spec.get("subjects", []) if s.get("kind") in ("person", "animal")}
    out = []
    for sent in re.split(r"(?<=[.;!?])\s+", text):
        who = next((x for x in re.findall(r"<Subject \d+>", sent) if x in desc), None)
        if who and not THEMED.search(sent):
            def sub(m, who=who):
                if m.group("w").lower() in desc[who]:
                    return m.group(0)
                return f"{m.group('p2') or (m.group('p1').lower() if m.group('p1').lower() in ('his', 'her', 'their') else 'the')} hair"
            sent = re.sub(r"(?:the (?:brim|peak|visor) of )?(?P<p1>a|an|the|his|her|their)\s+(?:[\w-]+\s+){0,2}?(?P<w>cap|hat|beanie|helmet|headband|bandana|hood)s?\b"
                          r"(?:\s+on\s+(?P<p2>his|her|their)\s+head)?", sub, sent, flags=re.I)
        out.append(sent)
    return " ".join(out)


def drop_old_fixtures(text, spec):
    """in a place swap, a background sentence that names a fixture of the OLD place (one the new place's description does not have) is dropped,
    and the fixture is named in the location line as replaced (2026-10-29, test-files case 1: "The traffic light shows a red pedestrian symbol."
    in the village shots - H3 kept the city's traffic light; the 2026-10-22 hand-written prompt said "it replaces the original city street, its
    buildings, railings, traffic light and signs" and got it right) -> (text, the dropped fixture words)"""
    loc = next((s for s in spec.get("subjects", []) if s.get("kind") == "location"), None)
    if not loc or not isinstance(text, str) or not text:
        return text, []
    desc = str(loc.get("description") or "").lower(); out, gone = [], []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        hits = [m.group(0) for rx in (STREET_FIXTURES, FIXTURES) for m in rx.finditer(sent) if m.group(0).lower() not in desc]
        if hits and not re.search(r"<Subject \d+>|\[Shot \d+\]", sent):
            gone += hits; continue
        out.append(sent)
    return " ".join(out), gone


def strip_subject_clothes(text, spec):
    """shot texts never describe a REPLACED person by clothes (2026-10-29, test-files cases 2 and 4: "<Subject 1> in a white t-shirt", "<Subject 1>
    wears a grey t-shirt and dark cargo pants" - the source person's clothes; H3 dressed the new person in them or kept the source person, and the
    check then demanded them). The SPEC rule alone was ignored, so it is done here: "<Subject n> in/wearing ... <garment> (with ...)" loses the
    clothing words, a "<Subject n> wears ..." sentence is dropped; themed costumes (kimono, uniform, armour, costume ...) stay."""
    if not isinstance(text, str) or not text:
        return text
    people = {s.get("tag") for s in spec.get("subjects", []) if s.get("kind") in ("person", "animal")}
    def drop_mod(m):
        return m.group(0) if (m.group(1) not in people or THEMED.search(m.group(0))) else m.group(1)
    g_ = r"(?:" + GARMENT + r"|collar|print|pattern|stripes|buttons|zipper|hood|pocket|piping)s?\b"
    item = r"(?:a|an|the|his|her|their)?\s*(?:[\w'-]+\s+){0,4}?" + g_ + r"(?:\s+" + g_ + r")?"   # up to 4 describing words, then the garment
    text = re.sub(r"(<Subject \d+>)(?:'s)?,?\s+(?:who is\s+|is\s+)?(?:in|wearing|dressed in)\s+" + item + r"(?:(?:,\s*(?:and\s+)?|\s+and\s+|\s+with\s+)" + item + r")*,?",
                  drop_mod, text)
    def drop_sent(m):
        return m.group(0) if (m.group(2) not in people or THEMED.search(m.group(0))) else m.group(1)
    text = re.sub(r"(^|(?<=[.;!?])\s+)(<Subject \d+>) (?:wears|is wearing|is dressed in|has on)\b[^.;!?]*(?:[.;!?]|$)", drop_sent, text)
    def drop_clause(m):   # "<Subject 1> walking on a sidewalk during the day, wearing a green shirt and a white backpack" (test-files case 9)
        return m.group(0) if (m.group(2) not in people or THEMED.search(m.group(3))) else m.group(1)
    text = re.sub(r"((<Subject \d+>)[^.;!?<]*?),\s+(?:wearing|dressed in)\s+(" + item + r"(?:(?:,\s*(?:and\s+)?|\s+and\s+|\s+with\s+)" + item + r")*)",
                  drop_clause, text)
    text = headwear_off(re.sub(r"\s{2,}", " ", text), spec)
    return re.sub(r"\s{2,}", " ", text).replace(" ,", ",").strip()


def audit_with_llm(job, spec, win_path, out_frames):
    """win_path = this segment's source window (same frame count as the render)"""
    n = 5; L = len(out_frames)
    idx = [int(round(j * (L - 1) / (n - 1))) for j in range(n)]
    src = read_frames(win_path, idx)
    content = [{"type": "text", "text": "Edit spec (what the generated video must show):\n" + json.dumps(
        {k: spec.get(k) for k in ("instruction", "subjects", "kept", "treatment")}, ensure_ascii=False)[:6000] + "\nReference pictures:"}]
    for i, p in enumerate(job["images"], 1):
        content += [{"type": "text", "text": f"<Picture {i}>:"}, llm.img_part(p, 512)]
    content.append({"type": "text", "text": "Frame pairs at the same moments (SOURCE then GENERATED):"})
    for j, i in enumerate(idx):
        t = i / FPS
        content += [{"type": "text", "text": f"t={t:.1f}s SOURCE:"}, llm.img_part(src[j], 448), {"type": "text", "text": f"t={t:.1f}s GENERATED:"}, llm.img_part(out_frames[i], 640)]
    v = llm.ask(content, system=AUDIT_SYSTEM, max_tokens=1200)
    v["score"] = float(v.get("score", 0) or 0)
    v["pass"] = bool(v.get("pass")) and v["score"] >= 7 and not any(x.get("severity") == "critical" for x in v.get("issues", []) if isinstance(x, dict))
    return v


# ---------------------------------------------------------------------------- the job
class Job:
    def __init__(self, store, jid):
        self.store, self.id = store, jid

    @property
    def d(self):
        return self.store.get(self.id)

    def set(self, **kw):
        self.store.update(self.id, **kw)

    def log(self, msg):
        self.store.log(self.id, msg)

    def cancelled(self):
        return self.store.get(self.id).get("cancel")

    def run(self):
        tool = self.d.get("tool", "create")
        if tool in ("create", "edit", "control"):
            return self.run_create()
        if tool == "improve":
            return self.run_improve()
        if tool == "extend":
            return self.run_extend()
        return self.run_tool(tool)

    def run_swap_recipe(self, job, jdir, segs, src, F, t0):
        """Objects swap with the 2026-10-26 recipe (data/runs/redoutline3.py, after the user's ratings): each replaced object outlined in
        its own colour, Qwen3.8 picks and checks the meant instances, each shot's prompt names only its own objects, per-shot reference
        sheets made from the source frames, extra copies of the new objects removed. Runs as a helper process on this job's plan."""
        from .store import DATA as SDATA
        for sg in segs:   # the shots' source windows, as the per-segment loop writes them
            end_k = min(F, sg.get("end", sg["start"] + sg["length"])); real = read_frames(src, range(sg["start"], end_k))
            write_video(real + [real[-1]] * max(0, sg["length"] - len(real)), os.path.join(jdir, f"seg{sg['k']}_src.mp4"), crf=12)
            write_video(real, os.path.join(jdir, f"seg{sg['k']}_src_real.mp4"), crf=12)   # Improve compares a shot with its real-length window
        llm.down()
        cfg = swap_recipe_cfg(); outdir = os.path.join(jdir, "recipe"); os.makedirs(outdir, exist_ok=True)
        env = {**os.environ, "R3_DATA": SDATA, "R3_OUT": outdir, "R3_QUALITY": job.get("quality", "standard")}
        if cfg.get("sheet", True):
            env["R3_SHEET"] = "2"
        self.set(stage="Finding the objects to swap", progress=0.1)
        logp = os.path.join(jdir, "recipe.log"); n = max(1, len(segs)); done_n = 0
        with open(logp, "w") as lf:
            proc = subprocess.Popen([tools.PY["app"], "-u", os.path.join(config.ROOT, "recipe", "redoutline3.py"), self.id], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd=config.ROOT, start_new_session=True)
            for line in proc.stdout:
                lf.write(line)
                if re.search(r"seg\d+ (rendered|: nothing to replace)", line):
                    done_n += 1; self.set(stage=f"Swapping - shot {min(done_n, n)} of {n} done", progress=0.15 + 0.7 * done_n / n)
                elif "picks in" in line:
                    self.set(stage="Drawing the reference sheets and rendering", progress=0.15)
                elif "new objects" in line and "extra-copy" not in line:
                    pass
                elif "extra-copy check" in line or "newobj" in line:
                    self.set(stage="Removing extra copies of the new objects", progress=0.88)
                if self.cancelled():
                    try:
                        os.killpg(proc.pid, 15)
                    except OSError:
                        pass
                    raise comfy.Cancelled()
            proc.wait()
        run_root = os.path.join(outdir, f"redline3_{self.id}"); res = os.path.join(run_root, "result.mp4")
        if proc.returncode != 0 or not os.path.exists(res):
            raise RuntimeError("the swap failed: " + open(logp).read()[-400:])
        fixed, afx = None, {}
        if job.get("audit", True) and os.path.exists(os.path.join(run_root, "shots_meta.json")):
            # Check & auto-fix (2026-10-27): every shot checked for old objects left in place; a flagged shot is remade and the better take kept
            self.set(stage="Checking each shot", progress=0.9); alog = os.path.join(jdir, "autofix.log"); out_f = os.path.join(run_root, "result_fixed.mp4")
            with open(alog, "w") as lf:
                p2 = subprocess.Popen([tools.PY["app"], "-u", os.path.join(config.ROOT, "recipe", "autofix.py"), self.id, run_root, out_f],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd=config.ROOT,
                                      start_new_session=True)
                for line in p2.stdout:
                    lf.write(line)
                    if "remaking shot" in line:
                        self.set(stage="Remaking a shot where an object was not swapped", progress=0.92)
                    elif "comparing shot" in line:
                        self.set(stage="Comparing the two takes of the remade shot", progress=0.96)
                    if self.cancelled():
                        try:
                            os.killpg(p2.pid, 15)
                        except OSError:
                            pass
                        raise comfy.Cancelled()
                p2.wait()
            if p2.returncode == 0 and os.path.exists(out_f):
                fixed = out_f
                try:
                    afx = json.load(open(os.path.join(run_root, "autofix.json")))
                except Exception:
                    afx = {}
                self.log(f"auto-fix: {len(afx.get('remade') or [])} shot(s) remade {afx.get('remade') or ''}, {len(afx.get('pasted') or [])} stray "
                         f"object(s) painted out, {sum(len(x[1]) for x in afx.get('erased') or [])} erased, "
                         f"{len(afx.get('mixed_fixed') or [])} mixed area(s) redrawn, in {afx.get('seconds')} s")
            else:
                self.log("auto-fix failed - the first take is kept: " + open(alog).read()[-300:])
        final = os.path.join(jdir, "result.mp4"); src_v = fixed or res
        if job.get("keep_audio", True):
            durable.copy_file(src_v, final)
        else:   # the recipe keeps the source's sound; this job asked for none
            silent = os.path.join(jdir, "result_silent.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src_v, "-map", "0:v", "-c:v", "copy", silent], check=True); durable.copy_file(silent, final)
        fr0 = read_frames(final, [0])
        if fr0:
            cv2.imwrite(os.path.join(jdir, "poster.jpg"), fr0[0], [cv2.IMWRITE_JPEG_QUALITY, 88])
        for sg in segs:   # each shot kept as its takes (seg<k>_a0.mp4 first take, seg<k>_a1.mp4 auto-fix remake) - Improve repairs from them
            k = sg["k"]; L = min(sg["length"], sg.get("end", 10 ** 9) - sg["start"]); rec = {"attempt": 0, "recipe": "redoutline3"}; chosen = rec
            takes = [(res, "(the object swap recipe - see recipe.log)")]
            rm = ((afx.get("shots") or {}).get(str(k)) or {}).get("remake")
            if rm and os.path.exists(os.path.join(rm.get("folder", ""), "result_silent.mp4")):
                takes.append((os.path.join(rm["folder"], "result_silent.mp4"), f"(auto-fix remake: {rm.get('how')})"))
            for a_, (vid, why) in enumerate(takes):
                fr = read_frames(vid, range(sg["start"], sg["start"] + L))
                if not fr:
                    continue
                dst = os.path.join(jdir, f"seg{k}_a{a_}.mp4"); write_video(fr, dst, crf=12)
                r_ = {"attempt": a_, "recipe": "redoutline3", "seed": None, "video": dst, "prompt": why, "pictures": []}
                self.store.add_attempt(self.id, k, r_)
                if a_ == 0 or k in (afx.get("remade") or []):
                    chosen = r_
            self.store.seg_done(self.id, k, chosen)
        self.set(status="done", stage="Done", progress=1.0, result=final, poster=os.path.join(jdir, "poster.jpg"), seconds=round(time.time() - t0))

    def run_lowpoly(self, job, jdir, segs, src, F, t0, spec, seed0, meta):
        """Motion transfer, "Low-poly place" (the 2026-10-24 recipe, data/runs/fastvideo.py with no start stills): each shot's source is
        prepared before H3 sees it - the moving subjects stay real (SAM 3), the ground becomes a coarse low-poly mesh that keeps its slopes,
        steps and edges (MoGe-2 normals + lowpoly.py), everything else is flat grey - and H3 paints the new place into the grey, with no
        pinned first frame. Close-up shots describe the people only by the parts in view, and get one re-render if a head shows up."""
        from .lowpoly import lowpoly
        W, H = job["canvas"]
        loc = next((s_ for s_ in spec.get("subjects", []) if s_.get("kind") == "location" and isinstance(s_.get("picture"), int)
                    and 1 <= s_["picture"] <= len(job["images"])), None)
        subs = [s_ for s_ in spec.get("subjects", []) if isinstance(s_.get("picture"), int) and s_.get("kind") != "location"]
        names = []
        for s_ in subs:
            r = re.sub(r"\s*\b(in|from|of)\s+<Video \d+>.*$", "", rn_plain(s_.get("replaces") or s_.get("kind") or "subject")).strip(" ,.") or "the subject"
            names.append(r if re.match(r"^(the|a|an|both|two|his|her|their)\b", r, re.I) else "the " + r)
        who = (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]) if names else "the moving people and things"
        place = (f"the place from <Picture {loc['picture']}>", rn_plain(loc.get("description"))) if loc else \
                ("the place described in the prompt", rn_plain((spec.get("scene") or "")))
        shots = []
        for sg in segs:   # the shots' source windows at the canvas size, padded with the last frame
            k = sg["k"]; L = sg["length"]; end_k = min(F, sg.get("end", sg["start"] + L)); real = read_frames(src, range(sg["start"], end_k))
            S = [cover_resize(f, W, H) for f in real + [real[-1]] * max(0, L - len(real))][:L]
            win = os.path.join(jdir, f"seg{k}_src.mp4"); write_video(S, win, crf=12)
            write_video(real, os.path.join(jdir, f"seg{k}_src_real.mp4"), crf=12)
            shots.append(dict(k=k, L=L, n=end_k - sg["start"], S=S, win=win,
                              seg=next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), {})))
        # 1 - mattes: subjects (SAM 3, one load for every shot, every 2nd frame) and the ground (MoGe-2 normals), then the prepared videos
        tools.gpu_take(self)
        ws = [w for w in subject_words(spec).split("|") if w]; ws = [w for w in ws if not any(w != o and w in o for o in ws)] or ["person"]
        tools.run_proc(self, [tools.PY["sam3"], "-I", os.path.join(tools.TOOLS_DIR, "sam3_multi.py"), os.path.join(jdir, "lp_subj"), "2", "|".join(ws)]
                       + [sh["win"] for sh in shots], 0.09, 0.25, "Cutting out the moving subjects")
        tools.run_proc(self, [tools.PY["moge"], "-I", os.path.join(tools.TOOLS_DIR, "moge_ground.py"), os.path.join(jdir, "lp_ground")]
                       + [sh["win"] for sh in shots], 0.25, 0.32, "Finding the ground")
        self.set(stage="Turning the old place into low-poly", progress=0.33)
        for vi, sh in enumerate(shots):
            k = sh["k"]
            subj = [cv2.cvtColor(m, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for m in read_frames(os.path.join(jdir, "lp_subj", str(vi), "matte.mp4"))]
            grd = [cv2.cvtColor(m, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for m in read_frames(os.path.join(jdir, "lp_ground", str(vi), "matte.mp4"))]
            LP = lowpoly(sh["S"], None, 420); prep = []
            for i, f in enumerate(sh["S"]):
                a = cv2.GaussianBlur((cv2.resize(subj[min(i, len(subj) - 1)], (W, H)) > 0.4).astype(np.float32), (0, 0), 1.5)[..., None]
                g = cv2.GaussianBlur((cv2.resize(grd[min(i, len(grd) - 1)], (W, H)) > 0.4).astype(np.float32), (0, 0), 2.0)[..., None]
                b = 127.0 * (1 - g) + LP[i].astype(np.float32) * g
                prep.append((b * (1 - a) + f.astype(np.float32) * a).astype(np.uint8))
            sh["prep"] = os.path.join(jdir, f"seg{k}_lowpoly.mp4"); write_video(prep, sh["prep"], crf=12)
        self.log(f"low-poly place: {len(shots)} shot(s) prepared (subjects real: {'|'.join(ws)}; ground low-poly; the rest flat grey)")
        # 2 - close-ups: the people described only by the parts in view (a face in the text gets drawn into a feet close-up)
        use_llm = llm.available()
        for sh in shots:
            part = str(sh["seg"].get("closeup") or "").strip(); sh["desc"] = {}
            if not use_llm or not part or part.lower() in ("none", "null", "false", "no"):
                continue
            for s_ in subs:
                if s_.get("kind") not in ("person", "animal"):
                    continue
                try:
                    r = llm.ask([{"type": "text", "text": f"The full description: {rn_plain(s_.get('description'))}\nThe shot is a close-up of: {part}. "
                                                          "The reference picture (describe only that part of it):"}, llm.img_part(job["images"][s_["picture"] - 1], 512)],
                                system='Write ONE sentence describing only what a close-up of that part shows - clothes, shoes, colours, held or ridden '
                                       'things - starting "Only the ... are in the frame:". Never mention the face, head, hair or anything above the '
                                       'frame. Reply with JSON only: {"desc": "..."}', max_tokens=160, temperature=0.0)
                    if str(r.get("desc") or "").strip():
                        sh["desc"][s_["tag"]] = str(r["desc"]).strip()
                except Exception as e:
                    self.log(f"close-up description failed: {str(e)[:120]}")
        # 3 - H3 renders from the prepared videos (no pinned first frame)
        llm.down(); n = len(shots); out = []
        for i_, sh in enumerate(shots):
            if self.cancelled():
                raise comfy.Cancelled()
            k = sh["k"]; spec2 = copy.deepcopy(spec); part = str(sh["seg"].get("closeup") or "").strip()
            note = (f"In <Video 1> only {who} and the ground are shown; everything else is flat grey. Paint {place[0]} into the grey: {place[1]} - "
                    "photoreal; the ground becomes the ground of that place, with the same slopes, steps and edges.")
            if part and part.lower() not in ("none", "null", "false", "no"):
                note += (f" The whole shot is a close-up at the height of the {part}: only the {part} are in the frame; the upper body and the "
                         "face stay above the top edge of the frame, out of view, the whole time.")
            spec2["_corrections"] = {str(k): note}
            for s_ in spec2.get("subjects", []):
                if s_.get("tag") in sh["desc"]:
                    s_["description"] = sh["desc"][s_["tag"]]
            pics = closeup_refs(self, job, spec, k, job["images"])
            prompt, pics = segment_prompt(spec2, k, pics, job.get("style"))
            base = 0.35 + 0.55 * i_ / n
            self.set(stage=f"Rendering shot {i_ + 1} of {n}", progress=base)
            seed = seed0 + 1000 * k + 991
            files = comfy.run(h3.build(prompt, sh["prep"], pics, W, H, sh["L"], seed),
                              lambda st, x, *_: x is not None and self.set(stage=f"Rendering shot {i_ + 1} of {n}", progress=base + 0.55 / n * x), self.cancelled)
            sh["out"] = os.path.join(jdir, f"seg{k}_a0.mp4"); shutil.copy(next(f for f in files if f.endswith(".mp4")), sh["out"])
            sh["prompt"], sh["pics"], sh["seed"] = prompt, pics, seed
        # 3b - close-ups: a head or a small whole-body copy that must not be there -> one re-render with another seed
        again = []
        if use_llm:
            for sh in shots:
                part = str(sh["seg"].get("closeup") or "").strip()
                if not part or part.lower() in ("none", "null", "false", "no"):
                    continue
                fr_ = read_frames(sh["out"]); pick = [fr_[i] for i in (2, len(fr_) // 3, 2 * len(fr_) // 3, len(fr_) - 3)]
                try:
                    v = llm.ask([{"type": "text", "text": f"Frames of a close-up of {part}:"}] + [llm.img_part(f, 512) for f in pick],
                                system='Reply with JSON only: {"head": true if a face, a head or a small whole-body copy of a person can be seen in any frame, '
                                       'else false, "why": "short"}', max_tokens=120, temperature=0.0)
                except Exception:
                    v = {}
                if v.get("head") is True:
                    again.append(sh); self.log(f"shot {sh['k']}: a head showed up in a close-up ({str(v.get('why', ''))[:100]}) - one re-render")
            llm.down()
        for sh in again:
            sh["seed"] = seed0 + 1000 * sh["k"] + 7777; self.set(stage=f"Re-rendering close-up shot {sh['k'] + 1}", progress=0.92)
            files = comfy.run(h3.build(sh["prompt"], sh["prep"], sh["pics"], W, H, sh["L"], sh["seed"]), None, self.cancelled)
            shutil.copy(next(f for f in files if f.endswith(".mp4")), sh["out"])
        for sh in shots:
            out += read_frames(sh["out"])[:sh["n"]]
            rec = {"attempt": 0, "seed": sh["seed"], "video": sh["out"], "prompt": sh["prompt"], "pictures": sh["pics"], "lowpoly": sh["prep"]}
            self.store.add_attempt(self.id, sh["k"], rec); self.store.seg_done(self.id, sh["k"], rec)
        # 4 - join + sound
        self.set(stage="Finishing", progress=0.97)
        silent = os.path.join(jdir, "result_silent.mp4"); write_video(out, silent)
        final = os.path.join(jdir, "result.mp4")
        if job.get("keep_audio", True) and meta["audio"]:
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", job["video"], "-map", "0:v", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                                "-b:a", "192k", "-shortest", final])
            if r.returncode != 0:
                durable.copy_file(silent, final)
        else:
            durable.copy_file(silent, final)
        cv2.imwrite(os.path.join(jdir, "poster.jpg"), out[0], [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.set(status="done", stage="Done", progress=1.0, result=final, poster=os.path.join(jdir, "poster.jpg"), seconds=round(time.time() - t0))

    def run_create(self):
        job = self.d; jdir = self.store.dir(self.id); os.makedirs(jdir, exist_ok=True)
        t0 = time.time()
        meta = probe(job["video"])
        (W, H), (SW, SH) = canvas(meta["w"], meta["h"], job.get("quality", "standard"))
        if job.get("mode") == "swap" and job.get("quality", "standard") == "standard":
            SW, SH = W, H   # object swaps keep the source's fine detail (captions, small props): full-size reference, as in the 2026-10-22 tests
        self.set(status="running", stage="Preparing the source", progress=0.01, canvas=[W, H])
        src = os.path.join(jdir, "src24.mp4"); s = max(SW / meta["w"], SH / meta["h"])
        vf = f"fps={FPS},scale={round(meta['w'] * s / 2) * 2}:{round(meta['h'] * s / 2) * 2}:flags=lanczos,crop={SW}:{SH}"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", job["video"], "-t", str(MAX_SECONDS if job.get("duration") == "full" else 5.2),
                        "-vf", vf, "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", src], check=True)
        F = len(read_frames(src))
        if F < 22:
            raise ValueError("the video is shorter than 1 second")
        srcC = alpha = None
        if job.get("mask"):  # Draw mode: H3 regenerates only the painted area (Fun ControlNet inpaint), the rest is pasted back exactly
            srcC = os.path.join(jdir, "src24_canvas.mp4"); sC = max(W / meta["w"], H / meta["h"]); sw, sh = round(meta["w"] * sC / 2) * 2, round(meta["h"] * sC / 2) * 2
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", job["video"], "-t", str(MAX_SECONDS if job.get("duration") == "full" else 5.2),
                            "-vf", f"fps={FPS},scale={sw}:{sh}:flags=lanczos,crop={W}:{H}", "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", srcC], check=True)
            mk = cv2.imread(job["mask"], cv2.IMREAD_GRAYSCALE); mk = cv2.resize(mk, (sw, sh), interpolation=cv2.INTER_LINEAR)
            mk = mk[(sh - H) // 2:(sh - H) // 2 + H, (sw - W) // 2:(sw - W) // 2 + W]
            mk = (mk > 100).astype(np.uint8)
            if mk.sum() < 0.002 * W * H:
                raise ValueError("the painted area is empty - paint over what should change")
            grow = max(3, int(0.02 * max(W, H)))
            gen_mask = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
            alpha = cv2.GaussianBlur(gen_mask.astype(np.float32), (0, 0), grow * 0.6)[..., None]
            cv2.imwrite(os.path.join(jdir, "mask_canvas.png"), gen_mask * 255)
            self.log(f"draw mask: {100 * mk.mean():.1f} % of the frame painted, grown {grow} px")
        swap_create = job.get("mode") == "swap" and not srcC and job.get("tool", "create") == "create"
        recipe = swap_create and swap_recipe_on()
        # two swap paths (user 2026-10-29, after the ~/test-files run: "the older code was almost perfect on these examples" / "two different paths.
        # One for people swap and maybe animals and one for objects like phones, pineapples and hats"): people, animals and places go the
        # 2026-10-22 way - parts of up to 5 s, each rendered WHOLE so H3 follows the source's cuts itself (doubles, montages), no pasted first
        # frames, no shot repairs; things go through the object recipe
        people = people_swap_guess(job) if swap_create and llm.available() else None

        def plan(people_):
            if people_:
                hc = detect_cuts(src); return plan_parts(F, hc, job.get("duration"))
            if (job.get("duration") == "full" or recipe) and not srcC:   # shot by shot: segments start and end at the source's cuts
                Fp = F if job.get("duration") == "full" else min(F, 124)   # the swap recipe tracks each object inside one shot: a 5 s swap is split at its cuts too
                hc = detect_cuts(src); return plan_shot_segments(Fp, sorted(set(hc) | set(detect_soft_cuts(src, hc))))
            return plan_segments(F, job.get("duration"))
        segs = plan(people)
        out_len = F if job.get("duration") == "full" else min(F, 124)
        self.set(segments=[{**sg, "status": "waiting", "attempts": []} for sg in segs], frames=out_len)
        self.log(f"source {meta['w']}x{meta['h']} {meta['fps']} fps {meta['duration']} s -> {F} frames at 24 fps; canvas {W}x{H}; {len(segs)} segment(s)"
                 + ("; a swap of people, animals or a place: parts of up to 5 s" if people else ""))
        if job.get("tool") == "control":
            job = prepare_control(self, job, jdir, W, H)

        # the edit spec
        use_llm = llm.available()
        self.set(stage=f"Writing the prompt with {llm.NAME}" if use_llm else "Writing the prompt", progress=0.03)
        spec = None
        if use_llm:
            try:
                job["_context_frames"] = context_frames(job, src, F, F)
                spec = spec_with_llm(job, src, segs)
                if len(spec.get("segments") or []) < len(segs):   # 2026-10-29, test-files case 2: no "segments" at all; 2026-10-30, case 8: 1 of 6 - ask once more
                    self.log("the prompt came back without some shot descriptions - writing it again")
                    spec2 = spec_with_llm(job, src, segs)
                    if len(spec2.get("segments") or []) > len(spec.get("segments") or []):
                        spec = spec2
                if swap_create and bool(recipe_skip(spec)) != bool(people):   # the spec (it saw the video) disagrees with the first guess
                    people = bool(recipe_skip(spec))
                    self.log(f"the prompt replaces {recipe_skip(spec) or 'things only'} - planning the video again for that path")
                    segs = plan(people); self.set(segments=[{**sg, "status": "waiting", "attempts": []} for sg in segs], frames=out_len)
                    spec = spec_with_llm(job, src, segs)
                    if len(spec.get("segments") or []) < len(segs):
                        self.log("the prompt came back without some shot descriptions - writing it again")
                        spec2 = spec_with_llm(job, src, segs)
                        if len(spec2.get("segments") or []) > len(spec.get("segments") or []):
                            spec = spec2
                if not people:   # the people path keeps the whole pictures, as on 2026-10-22 (no crops)
                    try:
                        spec = add_framing(job, spec, src, segs)
                        for g in spec.get("segments", []):
                            if g.get("_framed"):
                                self.log(f"segment {g.get('index')}: {g['_framed']} shot - pictures cropped to the {g.get('closeup')}")
                    except Exception as e:
                        self.log(f"framing check failed: {str(e)[:160]}")
                try:
                    spec = add_place_scope(job, spec, src, segs)
                    for g in spec.get("segments", []):
                        if g.get("_place_scope"):
                            self.log(f"segment {g.get('index')}: a shot of the same place without people - the new place is used here too ({g['_place_scope'][:100]})")
                except Exception as e:
                    self.log(f"place check failed: {str(e)[:160]}")
                try:
                    spec = add_costumes(job, spec, src, segs, part=bool(people))
                    for g in spec.get("segments", []):
                        if g.get("costume") or g.get("costume_part"):
                            self.log(f"segment {g.get('index')}: costume kept for {', '.join((g.get('costume') or []) + (g.get('costume_part') or []))}"
                                     + (f" ({'; '.join(v for v in (g.get('costume_what') or {}).values())})" if g.get("costume_what") else ""))
                except Exception as e:
                    self.log(f"costume check failed: {str(e)[:160]}")
            except Exception as e:
                self.log(f"LLM prompt failed ({e}); using the template")
        spec = spec or template_spec(job, segs)
        self.set(spec=spec, progress=0.08)
        if job.get("tool") == "control" and job.get("scene_used") == "image":
            shots = sum(str(g.get("shots", "")).count("[Shot") for g in spec.get("segments", []))
            if shots > 1:
                self.set(note=f"The motion video has {shots} shots. The picture's scene holds best on one continuous shot; after a cut, the new shot may keep "
                              "the video's own place.")
        if self.cancelled():
            raise comfy.Cancelled()

        seed0 = int(job.get("seed") or random.randint(1, 2 ** 31)); self.set(seed=seed0)
        if swap_create and not people and swap_recipe_on():
            return self.run_swap_recipe(job, jdir, segs, src, F, t0)
        if job.get("mode") == "motion" and job.get("mt_style") == "lowpoly" and job.get("tool", "create") == "create" and not srcC:
            self.set(canvas=[W, H]); job["canvas"] = [W, H]
            return self.run_lowpoly(job, jdir, segs, src, F, t0, spec, seed0, meta)
        if people:
            self.log(f"swap of {recipe_skip(spec) or 'a person'}: the 2026-10-22 way - {len(segs)} part(s) of up to 5 s, each rendered whole (the "
                     "model follows the source's cuts itself), no pasted first frames, Check with one retake, no shot repairs")
        n = len(segs); out = []; prev = None; wseg = 0.92 / n
        out = [None] * F
        for sg in segs:
            k = sg["k"]; base = 0.08 + k * wseg; best = None
            end_k = min(F, sg.get("end", sg["start"] + sg["length"])); n_real = end_k - sg["start"]
            win = os.path.join(jdir, f"seg{k}_src.mp4")      # the source window: this shot only, padded with its last frame
            real = read_frames(src, range(sg["start"], end_k))
            write_video(real + [real[-1]] * max(0, sg["length"] - len(real)), win, crf=12)
            win_real = os.path.join(jdir, f"seg{k}_src_real.mp4"); write_video(real, win_real, crf=12)
            shot_guide = None; plate_win = None; render_win = win
            seg_k = next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), None)
            pass_k = (job.get("mode") == "swap" and not srcC and seg_k is not None and not seg_keep(spec, seg_k, swap=True)
                      and not re.search(r"<Subject \d+>", str(seg_k.get("shots") or "")))   # the shot text names nobody either
            first_pin = k == 0 and job.get("mode") == "motion" and not job.get("first_frame") and job.get("tool", "create") == "create"
            if "end" in sg and not sg.get("cont") and (k > 0 or first_pin) and use_llm and not pass_k and not people:   # a new shot after a cut (or a motion transfer's opening shot): pin a composited first frame (never on the people path)
                self.set(stage=f"Placing the subjects in shot {sg.get('shot', k) + 1}'s first frame", progress=base)
                try:
                    q0 = 0
                    shp = [float(cv2.Laplacian(cv2.cvtColor(cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()) for f in real]
                    if shp:
                        medp = float(np.median(shp)); q0 = next((i for i, v in enumerate(shp[:12]) if v >= 0.8 * medp), 0)
                    comp, _use = compose_frame(self, job, spec, cover_resize(real[q0], W, H), W, H, f"seg{k}_shot",
                                               kept=next((g.get("kept") for g in spec.get("segments", []) if str(g.get("index")) == str(k)), None),
                                               shots=next((g.get("shots") or "" for g in spec.get("segments", []) if str(g.get("index")) == str(k)), ""))
                    if comp and q0 == 0:
                        shot_guide = comp
                    elif comp:
                        self.log(f"segment {k}: the shot opens blurred - first-frame pin skipped")
                except comfy.Cancelled:
                    raise
                except Exception as e:
                    self.log(f"segment {k}: first-frame composite failed: {str(e)[:200]}")
            if pass_k:   # nobody to replace and no new place in this shot (case 8's debris and Earth shots): the source itself, no render
                frames = [cover_resize(f, W, H) for f in real]; frames = frames + [frames[-1]] * max(0, sg["length"] - len(frames))
                dst = os.path.join(jdir, f"seg{k}_a0.mp4"); write_video(frames, dst, crf=12)
                best = (10.0, frames, {"attempt": 0, "seed": None, "video": dst, "prompt": "(nothing to replace in this shot - the source is kept)",
                                       "pictures": [], "audit": {"score": 10.0, "pass": True, "issues": [], "pairs": [], "note": "source kept"}})
                self.store.add_attempt(self.id, k, best[2]); self.log(f"segment {k}: nothing to replace in this shot - the source is kept")
            for attempt in range(1 + (RETAKES if (job.get("audit") and use_llm) else 0)):
                if best is not None and best[2].get("prompt", "").startswith("(nothing to replace"):
                    break
                llm.down()
                if render_win == win and hair_off(spec) and not srcC:
                    try:
                        render_win = strip_hair(self, win, f"seg{k}")
                    except comfy.Cancelled:
                        raise
                    except Exception as e:
                        self.log(f"segment {k}: hair removal failed: {str(e)[:160]}"); render_win = win
                prompt, pics = segment_prompt(spec, k, job["images"] if people else closeup_refs(self, job, spec, k, job["images"]), job.get("style"),
                                              swap=job.get("mode") == "swap")
                seed = seed0 + 1000 * k + 7 * attempt
                guides = []
                continuation = sg.get("cont", prev is not None and "end" not in sg)
                if continuation and prev is not None:
                    gpath = os.path.join(jdir, f"seg{k}_guide.mp4"); write_video(prev[-5:], gpath, crf=10); guides = [{"video": gpath, "frame": 0}]
                elif shot_guide:
                    guides = [{"image": shot_guide, "frame": 0}]
                elif k == 0 and job.get("first_frame"):
                    guides = [{"image": job["first_frame"], "frame": 0}]
                label = f"Rendering {'segment ' + str(k + 1) + '/' + str(n) if n > 1 else 'the video'}" + (" (retake)" if attempt else "")
                self.set(stage=label, progress=base + 0.01)

                def prog(kind, frac, v, m, base=base, label=label):
                    if kind == "sampling" and frac is not None:
                        self.set(stage=f"{label} - step {v}/{m}", progress=base + wseg * 0.8 * min(0.97, frac))
                    elif kind == "queue" and v:
                        self.set(stage=f"{label} - waiting for the GPU ({v} in queue)")
                wf = h3.build(prompt, plate_win or render_win, pics, W, H, sg["length"], seed, guides)
                if srcC:
                    FCs = read_frames(srcC); nC = len(FCs)
                    winC = os.path.join(jdir, f"seg{k}_srcC.mp4"); mclip = os.path.join(jdir, f"seg{k}_mask.mp4")
                    write_video([FCs[min(i, nC - 1)] for i in range(sg["start"], sg["start"] + sg["length"])], winC, crf=12)
                    mimg = cv2.cvtColor(cv2.imread(os.path.join(jdir, "mask_canvas.png"), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
                    write_video([mimg] * sg["length"], mclip, crf=0)
                    h3.add_inpaint(wf, winC, mclip)
                files = comfy.run(wf, prog, self.cancelled)
                mp4 = next((f for f in files if f.endswith(".mp4")), None)
                if not mp4:
                    raise RuntimeError("the render produced no video")
                dst = os.path.join(jdir, f"seg{k}_a{attempt}.mp4"); shutil.copy(mp4, dst)
                frames = read_frames(dst)[:sg["length"]]
                if not srcC:
                    fl = keep_flashes(frames, real)
                    if fl is not frames:
                        frames = fl; write_video(frames, dst, crf=12)
                if srcC:  # exact paste-back outside the painted area
                    orig = read_frames(os.path.join(jdir, f"seg{k}_srcC.mp4"))
                    frames = [(o.astype(np.float32) * (1 - a) + g.astype(np.float32) * a).astype(np.uint8)
                              for o, g in zip(orig, frames) for a in [follow_alpha(o, g, alpha[..., 0])]]
                rec = {"attempt": attempt, "seed": seed, "video": dst, "prompt": prompt, "pictures": pics}
                if job.get("audit") and use_llm:
                    job["_context_frames"] = context_frames(job, src, F, sg["start"] + sg["length"])
                    self.set(stage=f"Checking {'segment ' + str(k + 1) if n > 1 else 'the result'} with {llm.NAME}", progress=base + wseg * 0.85)
                    try:
                        rec["audit"] = audit_dense(job, spec, win_real, frames[:n_real], seg=next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), None))
                    except Exception as e:
                        rec["audit"] = {"error": str(e)[:300]}
                self.store.add_attempt(self.id, k, rec)
                score = rec.get("audit", {}).get("score", 0) if "error" not in rec.get("audit", {}) else 0
                if best is None or take_rank(score, rec.get("audit")) > take_rank(best[0], best[2].get("audit")):
                    best = (score, frames, rec)
                a = rec.get("audit")
                if not a or a.get("pass") or "error" in a or attempt >= RETAKES:
                    break
                hc = detect_cuts(win)
                if not srcC and not people and localized(a, sorted(set(hc) | set(detect_soft_cuts(win, hc))), sg["length"]):
                    break   # the problems sit in a few shots: repair those shots instead of re-rendering everything (people path: the whole part again)
                fix = (a.get("fix") or "").strip()
                if not fix:  # the reviewer named problems but no fix: turn the failed checks and issues into one
                    probs = [f"{k}: {v}" for k, v in (a.get("checks") or {}).items() if isinstance(v, str) and v.strip().lower() not in ("ok", "")]
                    probs += [x.get("what", "") for x in a.get("issues", []) if isinstance(x, dict) and x.get("severity") in ("critical", "major")]
                    if probs:
                        fix = "Avoid these problems seen in an earlier attempt: " + "; ".join(probs)[:700] + ". Show only what the subject definitions describe."
                if fix:   # H3 draws every noun of the note: say what the frame must show, in the new place
                    fix = positive_fix(spec, [fix], seg=seg_k)
                if fix:
                    spec.setdefault("_corrections", {})[str(k)] = fix; self.set(spec=spec)
                self.log(f"segment {k}: audit score {a.get('score')} - retaking ({fix[:160]})")
                if people and not srcC and render_win == win and any(isinstance(x, dict) and x.get("severity") in ("critical", "major") and
                                                                     HAIR_LEAK.search(str(x.get("what", ""))) for x in a.get("issues", [])):
                    try:   # the original's long hair came through on the new person (test-files case 8's braids, people10 P02's hair flip): the
                        render_win = strip_hair(self, win, f"seg{k}")   # retake renders from the source with that hair erased; the check keeps the better take
                        self.log(f"segment {k}: the original's long hair leaked into the new person - retaking from the source with that hair erased")
                    except comfy.Cancelled:
                        raise
                    except Exception as e:
                        self.log(f"segment {k}: hair removal failed: {str(e)[:160]}"); render_win = win
                if shot_guide and any(isinstance(x, dict) and x.get("severity") in ("critical", "major") and issue_time(x) <= 0.3 for x in a.get("issues", [])):
                    shot_guide = None   # the first frame IS the pin: a pin that drops a rider or keeps a source person would fail every retake
                    self.log(f"segment {k}: the pinned first frame itself was flagged - retaking without the pin")
                if job.get("mode") == "motion" and not srcC and plate_win is None and (place_reverted(a) or float(a.get("score", 10)) <= 4) and any(
                        s_.get("kind") == "location" for s_ in spec.get("subjects", [])):
                    try:   # the old place keeps coming back: retake from a source that already shows the new place behind the subject
                        self.set(stage=f"Laying the new place into segment {k + 1}'s source")
                        plate_win = None
                        if REPAINT_PLACE:
                            try:
                                plate_win = repaint_plate(self, job, spec, k, render_win, sg["length"], W, H, f"seg{k}")
                            except comfy.Cancelled:
                                raise
                            except Exception as e:
                                self.log(f"segment {k}: re-painted place failed ({str(e)[:160]}) - using the place picture's view")
                        if plate_win is None:
                            try:
                                view = place_view(self, job, spec, k, cover_resize(real[0], W, H), W, H, f"seg{k}")
                            except comfy.Cancelled:
                                raise
                            except Exception as e:
                                self.log(f"segment {k}: place view failed: {str(e)[:160]}"); view = None
                            plate_win = plate_window(self, job, spec, render_win, sg["length"], W, H, f"seg{k}", view=view)
                        shot_guide = None
                        if "end" in sg and not sg.get("cont") and (k > 0 or first_pin):   # pin the people swapped into the plate's first frame
                            comp, _use = compose_frame(self, job, spec, cover_resize(read_frames(plate_win, [0])[0], W, H), W, H, f"seg{k}_pshot",
                                                       kept=next((g.get("kept") for g in spec.get("segments", []) if str(g.get("index")) == str(k)), None),
                                                       shots=next((g.get("shots") or "" for g in spec.get("segments", []) if str(g.get("index")) == str(k)), ""),
                                                       on_plate=True)
                            shot_guide = comp
                    except comfy.Cancelled:
                        raise
                    except Exception as e:
                        self.log(f"segment {k}: new-place plate failed: {str(e)[:200]}")
            if job.get("audit") and use_llm and not srcC and not people and best[2].get("audit") and not best[2]["audit"].get("pass") and "error" not in best[2]["audit"]:
                try:
                    sg_r = {**sg, "length": n_real} if "end" in sg else sg
                    best_r = (best[0], best[1][:n_real], best[2]) if "end" in sg else best
                    best = repair_segment(self, job, spec, sg_r, k, n, src, F, win_real if "end" in sg else os.path.join(jdir, f"seg{k}_src.mp4"),
                                          W, H, best_r, prev, seed0)
                except comfy.Cancelled:
                    raise
                except Exception as e:
                    self.log(f"segment {k}: shot repair failed: {str(e)[:300]}")
            frames = best[1]; self.store.seg_done(self.id, k, best[2])
            out = join_segment(out, sg, frames, F)
            prev = frames[:n_real] if n_real >= 5 else frames
        llm.down()
        out = [f for f in out[:out_len] if f is not None]
        self.set(stage="Finishing", progress=0.97)
        silent = os.path.join(jdir, "result_silent.mp4"); write_video(out, silent)
        final = os.path.join(jdir, "result.mp4")
        if job.get("keep_audio", True) and meta["audio"]:
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", job["video"], "-map", "0:v", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                                "-b:a", "192k", "-shortest", final])
            if r.returncode != 0:
                durable.copy_file(silent, final)
        else:
            durable.copy_file(silent, final)
        cv2.imwrite(os.path.join(jdir, "poster.jpg"), out[0], [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.set(status="done", stage="Done", progress=1.0, result=final, poster=os.path.join(jdir, "poster.jpg"), seconds=round(time.time() - t0))


EXT_SYSTEM = """You write the prompt for MiniMax H3, a reference-to-video model, to generate the {which} part of an existing video clip.
You see frames from the {edge} of the existing clip - the new part must join it seamlessly: the same people with the same faces, hair and
clothes, the same place, light, colours and camera style - plus optional reference pictures (<Picture n>) of people or things the user wants
in it, and the user's description of what happens.
Reply with JSON only:
{{"subjects": [{{"tag": "<Subject 1>", "picture": null, "description": "exact look as seen in the frames: face, hair, skin tone, build, every garment with colour"}}],
 "scene": "the place, light, colours and camera style seen in the frames",
 "shots": "[Shot 1] (0.0 to {secs:.1f} seconds) what happens, starting exactly from the {joint} and continuing naturally",
 "treatment": "one sentence on the look (match the clip)",
 "sound": "the ambient sound and any speech"}}
Rules: never change anyone's look; no cut at the join; for a picture the user added, set "picture" to its number and describe it from the picture;
follow the user's description; if there is none, continue the action in the most natural way."""

EXT_AUDIT = """You check an AI video continuation. You see frames of the ORIGINAL clip at the join, then frames of the NEW part in order, and the
user's request. Reply with JSON only: {"score": 0-10, "pass": true or false, "issues": [{"severity": "critical|major|minor", "what": "..."}],
"fix": "one or two imperative sentences that would fix the critical/major issues, or empty"}.
critical = a different person or face, different clothes, a different place, a hard cut or jump at the join, garbled faces or bodies.
major = the user's request not followed, objects appearing or vanishing for no reason. pass = score >= 7 and no critical issue."""


def _job_tool_methods():
    def run_tool(self, tool):
        job = self.d; jdir = self.store.dir(self.id); os.makedirs(jdir, exist_ok=True); t0 = time.time()
        self.set(status="running", stage="Preparing", progress=0.01)
        src = job["video"]; lim = tools.LIMITS.get(tool, 30)
        if os.path.splitext(src)[1].lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            meta = probe(src); self.set(canvas=[meta["w"], meta["h"]])
            if meta["duration"] > lim + 0.05:
                src = tools.trim(src, os.path.join(jdir, "input_trim.mp4"), lim); self.log(f"input trimmed to the first {lim} s")
        res = tools.RUNNERS[tool](self, job, jdir, src)
        out = res["result"]
        post = out if out.lower().endswith((".png", ".jpg")) else tools.poster(out, os.path.join(jdir, "poster.jpg"))
        self.set(status="done", stage="Done", progress=1.0, result=out, poster=post, extras=res.get("extras", []), note=res.get("note"),
                 seconds=round(time.time() - t0))

    def run_extend(self):
        job = self.d; P = job.get("params", {}); jdir = self.store.dir(self.id); os.makedirs(jdir, exist_ok=True); t0 = time.time()
        meta = probe(job["video"]); direction = P.get("direction", "sequel"); secs = float(P.get("duration", 5)); quality = P.get("quality", "standard")
        (W, H), _ = canvas(meta["w"], meta["h"], quality)
        self.set(status="running", stage="Preparing the clip", progress=0.01, canvas=[W, H])
        # full-quality copy at 24 fps (kept in the result) and a canvas-size copy (guide frames, the LLM)
        s = min(1.0, 1920 / max(meta["w"], meta["h"])); fw, fh = int(meta["w"] * s) // 2 * 2, int(meta["h"] * s) // 2 * 2
        srcF = os.path.join(jdir, "src24_full.mp4"); srcC = os.path.join(jdir, "src24_canvas.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", job["video"], "-t", "30", "-vf", f"fps={FPS},scale={fw}:{fh}:flags=lanczos", "-an", "-c:v", "libx264",
                        "-crf", "12", "-pix_fmt", "yuv420p", srcF], check=True)
        sC = max(W / fw, H / fh); cw, chh = round(fw * sC / 2) * 2, round(fh * sC / 2) * 2
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", srcF, "-vf", f"scale={cw}:{chh}:flags=lanczos,crop={W}:{H}", "-c:v", "libx264", "-crf", "12",
                        "-pix_fmt", "yuv420p", srcC], check=True)
        FC = read_frames(srcC); F = len(FC)
        if F < 24:
            raise ValueError("the clip is shorter than 1 second")
        L = 56 if secs <= 2.5 else 124; nseg = 1 if secs <= 5.3 else 2
        sequel = direction != "prequel"
        user = mentions_to_tags(job.get("prompt") or "")
        # the prompt
        use_llm = llm.available(); spec = None
        if use_llm:
            self.set(stage=f"Writing the prompt with {llm.NAME}", progress=0.04)
            edge = FC[-48:][::8] if sequel else FC[:48][::8]
            content = [{"type": "text", "text": f"User's description: {user or 'none - continue naturally'}.\nReference pictures:" if job["images"] else
                        f"User's description: {user or 'none - continue naturally'}."}]
            for i, pth in enumerate(job["images"], 1):
                content += [{"type": "text", "text": f"<Picture {i}>:"}, llm.img_part(pth, 512)]
            content.append({"type": "text", "text": f"Frames from the {'END' if sequel else 'START'} of the existing clip, in order:"})
            content += [llm.img_part(f, 640) for f in edge]
            sysm = EXT_SYSTEM.format(which="NEXT" if sequel else "PREVIOUS", edge="end" if sequel else "start", secs=nseg * L / FPS,
                                     joint="last frame of the clip" if sequel else "moment that leads into the clip's first frame")
            try:
                spec = llm.ask(content, system=sysm, max_tokens=1500); spec["_source"] = llm.NAME
            except Exception as e:
                self.log(f"LLM prompt failed ({e}); using the template")
        if not spec:
            spec = {"subjects": [], "scene": "", "shots": f"[Shot 1] (0.0 to {nseg * L / FPS:.1f} seconds) {user or 'The action continues naturally.'}",
                    "treatment": "", "sound": "", "_source": "template"}
        self.set(spec=spec)

        def ext_prompt(corr=None):
            subj = "\n".join(f"{x.get('tag', '<Subject>')} is {x.get('description', '')}" + (f" Appearance from <Picture {x['picture']}>." if x.get("picture") else "")
                             for x in spec.get("subjects", []))
            intro = ("This video continues an existing clip: its first frames are the last moments of that clip, and everything continues seamlessly from them."
                     if sequel else "This video leads into an existing clip: its last frames are the first moments of that clip, and everything leads seamlessly into them.")
            return f"""{intro} {user}

subject_definitions:
{subj or 'The people and objects exactly as they appear in the guide frames.'}

scene:
{spec.get('scene', '')}

detailed_description:
{spec.get('treatment') or 'Match the look of the existing clip exactly: same light, colours, lens and camera style.'}

{spec.get('shots', '')}
{('Important corrections: ' + corr) if corr else ''}
Keep every person's face, hair and clothing identical to the guide frames. No cut and no jump at the join.

overall_soundscape:
{spec.get('sound') or 'The same ambient sound as the clip.'}"""

        seed0 = random.randint(1, 2 ** 31); guide = FC[-5:] if sequel else FC[:5]; gens = []; corr = None
        for k in range(nseg):
            best = None
            for attempt in range(1 + (RETAKES if (job.get("audit") and use_llm) else 0)):
                llm.down()
                gpath = os.path.join(jdir, f"ext{k}_guide.mp4"); write_video(guide, gpath, crf=10)
                label = f"Generating {'part ' + str(k + 1) + '/' + str(nseg) if nseg > 1 else 'the ' + ('continuation' if sequel else 'lead-in')}" + (" (retake)" if attempt else "")
                base = 0.08 + k * (0.85 / nseg)
                self.set(stage=label, progress=base)

                def prog(kind, frac, v, m, base=base, label=label):
                    if kind == "sampling" and frac is not None:
                        self.set(stage=f"{label} - step {v}/{m}", progress=base + (0.85 / nseg) * 0.8 * min(0.97, frac))
                    elif kind == "queue" and v:
                        self.set(stage=f"{label} - waiting for the GPU ({v} in queue)")
                wf = h3.build_gen(ext_prompt(corr), job["images"], W, H, L, seed0 + 1000 * k + 7 * attempt,
                                  [{"video": gpath, "frame": 0 if sequel else L - 5}])
                files = comfy.run(wf, prog, self.cancelled)
                mp4 = next((f for f in files if f.endswith(".mp4")), None)
                if not mp4:
                    raise RuntimeError("the render produced no video")
                dst = os.path.join(jdir, f"ext{k}_a{attempt}.mp4"); shutil.copy(mp4, dst)
                gen = read_frames(dst)[:L]; rec = {"k": k, "attempt": attempt, "video": dst}
                if job.get("audit") and use_llm:
                    self.set(stage=f"Checking the join with {llm.NAME}", progress=base + (0.85 / nseg) * 0.85)
                    try:
                        c = [{"type": "text", "text": f"User's request: {user or 'continue naturally'}.\nORIGINAL clip at the join:"}] + [llm.img_part(f, 512) for f in (guide[::2])]
                        c += [{"type": "text", "text": "NEW part, in order:"}] + [llm.img_part(gen[i], 512) for i in (range(5, L, max(1, (L - 5) // 5)) if sequel else range(0, L - 5, max(1, (L - 5) // 5)))]
                        v = llm.ask(c, system=EXT_AUDIT, max_tokens=800); v["score"] = float(v.get("score", 0) or 0)
                        v["pass"] = bool(v.get("pass")) and v["score"] >= 7 and not any(x.get("severity") == "critical" for x in v.get("issues", []) if isinstance(x, dict))
                        rec["audit"] = v
                    except Exception as e:
                        rec["audit"] = {"error": str(e)[:300]}
                self.store.log(self.id, f"part {k}: take {attempt}" + (f", check {rec['audit'].get('score')}/10" if rec.get("audit", {}).get("score") is not None else ""))
                sc = rec.get("audit", {}).get("score", 0) or 0
                if best is None or sc > best[0]:
                    best = (sc, gen, rec)
                a = rec.get("audit")
                if not a or a.get("pass") or "error" in a or attempt >= RETAKES:
                    break
                corr = (a.get("fix") or "; ".join(x.get("what", "") for x in a.get("issues", []) if isinstance(x, dict))) or None
            gen = best[1]; gens.append(best[2]); self.set(ext_parts=gens)
            guide = gen[-5:] if sequel else gen[:5]
            if sequel:
                new = gen[5:] if k == 0 else gen[5:]
                self._newf = (getattr(self, "_newf", []) + new)
            else:
                self._newf = gen[:-5] + getattr(self, "_newf", [])
        llm.down()
        self.set(stage="Joining", progress=0.95)
        FF = read_frames(srcF); size = (FF[0].shape[1], FF[0].shape[0])
        G = [cv2.resize(f, size, interpolation=cv2.INTER_LANCZOS4) for f in self._newf]
        g5 = [cv2.resize(f, size, interpolation=cv2.INTER_LANCZOS4) for f in (FC[-5:] if sequel else FC[:5])]
        if sequel:  # cross-fade the original's last 5 frames into the (identical, guided) generated ones
            out = FF[:-5] + [cv2.addWeighted(FF[-5 + i], 1 - (i + 1) / 6.0, g5[i], (i + 1) / 6.0, 0) for i in range(5)] + G
        else:
            out = G + [cv2.addWeighted(g5[i], 1 - (i + 1) / 6.0, FF[i], (i + 1) / 6.0, 0) for i in range(5)] + FF[5:]
        silent = os.path.join(jdir, "result_silent.mp4"); write_video(out, silent)
        # audio: the clip's own sound + H3's generated sound for the new part
        dur_src = len(FF) / FPS; parts = [p["video"] for p in gens]; final = os.path.join(jdir, "result.mp4")
        args = ["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", job["video"]] + sum((["-i", p] for p in parts), [])
        fl = []; seq = []
        src_a = "[1:a]" if meta["audio"] else None
        if src_a:
            fl.append(f"[1:a]atrim=0:{dur_src:.3f},asetpts=PTS-STARTPTS,aresample=48000[a0]")
        else:
            fl.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{dur_src:.3f}[a0]")
        for i, _ in enumerate(parts):
            st, en = (5 / FPS, L / FPS) if sequel else (0, (L - 5) / FPS)
            fl.append(f"[{i + 2}:a]atrim={st:.3f}:{en:.3f},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[g{i}]")
        gl = "".join(f"[g{i}]" for i in (range(len(parts)) if sequel else reversed(range(len(parts)))))  # prequel parts: earliest first
        fl.append((f"[a0]{gl}concat=n={1 + len(parts)}:v=0:a=1[aout]") if sequel else (f"{gl}[a0]concat=n={1 + len(parts)}:v=0:a=1[aout]"))
        r = subprocess.run(args + ["-filter_complex", ";".join(fl), "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", final],
                           capture_output=True, text=True)
        if r.returncode:
            self.log("audio mix failed, kept the picture only: " + r.stderr[-300:]); shutil.copy(silent, final)
        post = tools.poster(final, os.path.join(jdir, "poster.jpg"))
        self.set(status="done", stage="Done", progress=1.0, result=final, poster=post, frames=len(out), seconds=round(time.time() - t0))

    Job.run_tool = run_tool
    Job.run_extend = run_extend


_job_tool_methods()


# ---------------------------------------------------------------------------- Motion Control
CONTROL_MODE = {
    "video": ("Motion control, scene from the VIDEO: <Picture 1> is the character who replaces the main person of <Video 1> and performs every one of "
              "their movements. Everything else - the place, other people, objects, light and camera - stays exactly as in <Video 1>. Use only the person "
              "from <Picture 1>, never its background"),
    "image": ("Motion control, scene from the IMAGE: <Picture 1> is the character who replaces the main person of <Video 1> and performs every one of "
              "their movements. The whole place comes from <Picture 1> (<Picture 2> shows that place without the person) and replaces <Video 1>'s place in "
              "every shot; other people from <Video 1> are left out. Restage every movement onto the solid, walkable ground of that place (its floor, path or "
              "deck) - the character never walks, crawls or lies in water, in the air or inside walls. Give <Picture 1> one person subject and <Picture 2> one "
              "location subject"),
}
SCENE_SYSTEM = """You look at a character reference picture for a video tool. Reply with JSON only: {"background": "scene" or "plain", "why": "short"}.
scene = the person is in a real place (a room, street, nature, a studio set with props) that could be a video's background.
plain = a plain white, grey or flat studio backdrop, a cut-out, a collage or a character sheet - nothing usable as a place."""
PICK_SYSTEM = """You rate candidate first frames for an AI motion-transfer video. You see A = the first frame of the motion video (the pose, position,
size in frame and camera angle to copy), B = the character picture (identity and outfit to copy), then numbered candidates.
The background rule: {bg}.
Rate EVERY candidate separately and strictly. Reply with JSON only:
{{"candidates": [{{"identity": 0-10 (same face, hair, skin tone, body, clothes, shoes, accessories as B), "pose": 0-10 (same pose, position, size in frame and camera angle as A),
"background_ok": true or false (follows the background rule), "artifacts": true or false (extra people, extra or missing limbs, warped face or hands)}}], "why": "short"}}"""
QWEN_FIRST = {
    "video": ("Replace the main person in Picture 1 with the person from Picture 2. Use Picture 2's exact face, hairstyle, hair colour, skin tone and "
              "body shape, and the exact clothing, shoes and accessories from Picture 2. Keep Picture 1's pose, body position, size in the frame, camera "
              "angle, lighting and the entire background exactly the same."),
    "image": ("Add the person from Picture 2 - the same face, hairstyle, skin tone and body shape, and the exact clothing, shoes and accessories - into "
              "Picture 1. Give them exactly the pose, body position, size in the frame and camera angle of the person in Picture 3 (shown on a plain grey "
              "background). Keep Picture 1's place, furniture, lighting and colours exactly as they are, with the person lit like Picture 1. The grey "
              "background of Picture 3 does not appear and no one else is in the picture."),
}
QWEN_PLATE = ("Remove the person from this picture. Show the same empty place from the same viewpoint, with every part of the background, the lighting "
              "and the colours exactly the same.")


def qwen_edit(job, paths, prompt, seed, tag):
    """Qwen-Image 2.1 edit on ComfyUI (the first picture is the one edited; the output keeps its aspect) -> a PNG in the job folder"""
    llm.down()
    wf = h3.W.qwen_image(prompt, 1024, 1024, [comfy.stage(p) for p in paths], seed, f"h3create/{tag}")
    files = comfy.run(wf, None, job.cancelled)
    png = next((f for f in files if f.lower().endswith(".png")), None)
    if not png:
        raise RuntimeError("the picture edit produced no image")
    dst = os.path.join(job.store.dir(job.id), f"{tag}.png"); shutil.copy(png, dst)
    return dst


def cover_resize(img, W, H):
    h, w = img.shape[:2]; s = max(W / w, H / h)
    r = cv2.resize(img, (max(W, int(round(w * s))), max(H, int(round(h * s)))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4)
    y = (r.shape[0] - H) // 2; x = (r.shape[1] - W) // 2
    return r[y:y + H, x:x + W]


def plain_backdrop(path):
    """no-LLM guess: a picture whose border is flat and bright is a studio backdrop"""
    im = cv2.imread(path, cv2.IMREAD_COLOR); h, w = im.shape[:2]; b = max(4, min(h, w) // 25)
    edge = np.concatenate([im[:b].reshape(-1, 3), im[-b:].reshape(-1, 3), im[:, :b].reshape(-1, 3), im[:, -b:].reshape(-1, 3)]).astype(np.float32)
    return float(edge.std(0).mean()) < 18 and float(edge.mean()) > 150


def pose_reference(job_obj, f0p, jdir):
    """the motion video's first frame with only the person kept (SAM 3), on flat mid-grey. None = no person in that frame;
    the frame itself if the cut-out failed for another reason"""
    try:
        llm.down(); comfy.free(); time.sleep(1)
        one = os.path.join(jdir, "motion_f0.mp4"); im = cv2.imread(f0p)
        write_video([im], one, crf=10)
        od = os.path.join(jdir, "pose_ref")
        tools.run_proc(job_obj, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_matte.py", one, od, "person", "matte"], 0.035, 0.04, "Reading the pose")
        rgba = cv2.imread(os.path.join(od, "cutout_first.png"), cv2.IMREAD_UNCHANGED)
        a = (rgba[..., 3:4].astype(np.float32) / 255.0)
        out = (rgba[..., :3].astype(np.float32) * a + 128.0 * (1 - a)).astype(np.uint8)
        dst = os.path.join(jdir, "pose_ref.png"); cv2.imwrite(dst, out); return dst
    except comfy.Cancelled:
        raise
    except Exception as e:
        if "nothing matching" in str(e):
            job_obj.log("nobody is in the motion video's first frame"); return None
        job_obj.log(f"pose cut-out failed ({str(e)[:200]}); using the whole frame")
        return f0p


def prepare_control(job_obj, job, jdir, W, H):
    """Motion Control: decide where the background comes from, make an empty plate of the picture's place (image scene), compose a first frame with
    the character in the motion video's opening pose (two Qwen-Image candidates, Qwen3.8 picks one or none), then hand over to the usual pipeline"""
    self = job_obj; char = job.get("char_image") or job["images"][0]; use_llm = llm.available()
    scene = job.get("scene", "auto")
    if scene == "auto":
        self.set(stage="Looking at your character picture", progress=0.02)
        try:
            if use_llm:
                r = llm.ask([{"type": "text", "text": "The character picture:"}, llm.img_part(char, 640)], system=SCENE_SYSTEM, max_tokens=120)
                scene = "image" if r.get("background") == "scene" else "video"
                self.log(f"scene auto: picture background = {r.get('background')} ({r.get('why', '')})")
            else:
                scene = "video" if plain_backdrop(char) else "image"; self.log(f"scene auto (no LLM): {scene}")
        except Exception as e:
            scene = "video"; self.log(f"scene check failed ({e}); using the video's scene")
    images = [char]; plate = None
    if scene == "image":
        self.set(stage="Making an empty plate of the picture's place", progress=0.03)
        try:
            plate = qwen_edit(self, [char], QWEN_PLATE, 4242, "plate"); images.append(plate)
        except comfy.Cancelled:
            raise
        except Exception as e:
            self.log(f"plate failed ({e}); the place is described from the character picture only")
    first = None
    if job.get("anchor", True):
        self.set(stage="Placing your character in the first frame", progress=0.04)
        cap = cv2.VideoCapture(job["video"]); ok, f0 = cap.read(); cap.release()
        f0p = os.path.join(jdir, "motion_f0.png"); cv2.imwrite(f0p, cover_resize(f0, W, H))
        pose = pose_reference(self, f0p, jdir)   # the video's person cut out onto plain grey (None: nobody in frame 0)
        base = os.path.join(jdir, "place_canvas.png")
        if scene == "image":
            cv2.imwrite(base, cover_resize(cv2.imread(plate or char, cv2.IMREAD_COLOR), W, H))
        if pose is None:      # the video starts without its person: pin the empty place itself
            first = base if scene == "image" else f0p
            self.log("first frame: the video starts empty, so it is pinned as the " + ("picture's empty place" if scene == "image" else "video's own first frame"))
        if scene == "image":  # edit the empty place (canvas shape): the person is added into it, posed like the video's first frame
            pics = [base, char, pose]
        else:                 # edit the video's first frame: the person is swapped, everything else stays
            pics = [f0p, char]
        prompt = QWEN_FIRST[scene]
        cands = []
        for i, seed in enumerate((11, 23) if pose is not None else ()):
            try:
                c = qwen_edit(self, pics, prompt, seed, f"first_c{i + 1}")
                cv2.imwrite(c, cover_resize(cv2.imread(c, cv2.IMREAD_COLOR), W, H)); cands.append(c)
            except comfy.Cancelled:
                raise
            except Exception as e:
                self.log(f"first-frame candidate {i + 1} failed: {e}")
        if cands and use_llm:
            self.set(stage=f"Choosing the first frame with {llm.NAME}", progress=0.05)
            try:
                c = [{"type": "text", "text": "A (motion video, first frame):"}, llm.img_part(f0p, 640), {"type": "text", "text": "B (character):"}, llm.img_part(char, 640)]
                for i, cp in enumerate(cands, 1):
                    c += [{"type": "text", "text": f"Candidate {i}:"}, llm.img_part(cp, 640)]
                bg = ("the background must be A's background, unchanged" if scene == "video" else
                      "the background must be the place from B; if a candidate still shows A's background (A's street, room or landscape), background_ok is false")
                r = llm.ask(c, system=PICK_SYSTEM.format(bg=bg), max_tokens=400)
                rated = r.get("candidates") or []
                ok = [(float(x.get("identity", 0)) + float(x.get("pose", 0)), i) for i, x in enumerate(rated[:len(cands)])
                      if x.get("background_ok") and not x.get("artifacts") and float(x.get("identity", 0)) >= 6 and float(x.get("pose", 0)) >= 5]
                best = max(ok)[1] + 1 if ok else 0
                self.log(f"first frame: candidate {best or 'none'} {json.dumps(rated)[:400]} ({r.get('why', '')})")
                first = cands[best - 1] if best else None
            except Exception as e:
                self.log(f"first-frame check failed ({e}); using candidate 1"); first = cands[0]
        elif cands:
            first = cands[0]
    upd = {"scene_used": scene, "images": images, "char_image": char, "first_frame": first,
           "image_thumbs": job.get("image_thumbs", [])[:1] + ([plate] if plate else [])}
    self.set(**upd); job.update(upd)
    return job


# ---------------------------------------------------------------------------- dense, shot-aware audit (2026-10-22: aim for 10/10)
AUDIT2_SYSTEM = """You are a strict QA reviewer for AI video edits. You get the edit spec (who or what is replaced by which picture, and what must stay),
the reference pictures, the source's shot changes, and PAIRS of frames at the same moment: SOURCE then GENERATED.
The SOURCE frames are the ground truth for everything the spec does not replace. For EVERY pair decide whether the GENERATED frame is a perfect edit:
 1. same shot: the same framing, camera angle, positions and moment of the action as the source frame (another shot, or the shot change happening
    earlier or later, is a timing error)
 2. every person, animal, object or place the spec replaces appears as its picture (face, hair, skin, build, clothes, colours) wherever the source shows
    the original - in every shot, even a very short one at the very end; a source person who should have been replaced but is still visible is critical.
    A THEMED costume change in the source (the same person later in traditional dress such as a kimono, a uniform or a stage costume) is kept:
    there the replacement has the picture's face, hair, skin and build but wears that costume. Ordinary clothes that differ between scenes are
    not a costume: there the replacement wears its picture's outfit. Use the context frames after the clip to tell who a partly visible person is.
    Leftovers of the original on the new subject also count: human hair, braids or a ponytail on an animal, robot, creature or helmeted figure
    whose picture has none (hair flying out in spins and jumps is the usual leak), or human skin, hands or a face where the picture shows fur,
    metal, paws or a visor. A replaced object sits exactly where the original is, at its size and angle and held the same way: a whole copy of
    the picture's object placed somewhere else (beside the person, standing up, floating) while parts of the original stay visible is critical.
 3. everything the spec does not replace (other people, props, the place unless a location picture is given) matches the source
 4. the same hand actions and props as the source at that moment; nothing invented; clothing is not a prop
 5. on-screen text and captions: the same words, style and position as the source (compare word by word). A caption may appear or change up to
    0.15 s (about 3 frames) earlier or later than in the source - single frames are only snapshots, so a caption that is one word ahead or behind
    in ONE pair next to a caption change is fine; flag a wrong word, style or position, or a caption that is off in two or more pairs
 6. bodies supported like the source person (in the air only if the source person is; a galloping or running animal or a sprinting person
    has normal moments with every foot off the ground between strides - that is not a problem), no warped faces or hands, no extra limbs, no smears
A cartoon or illustrated reference character that keeps its own art style is correct unless the user asked for another style.
Reply with JSON only:
{"pairs": [{"t": seconds, "ok": true or false, "problem": "" or what differs}],
 "issues": [{"severity": "critical|major|minor", "time": seconds, "what": "..."}],
 "score": 0-10,
 "fix": "imperative corrections that would fix every issue, or an empty string"}
Scoring: 10 = every pair ok; 9 = one minor issue; 7-8 = minor issues only; 5-6 = a major issue; 0-4 = a critical issue.
critical = a source person who should be replaced still visible, a wrong or merged identity, the place reverting, a missing or duplicated main subject,
a different shot or action at that moment, a garbled face in a close shot, an unsupported body. major = clothes or colours different from the picture,
an invented object, a wrong caption word, a shot change off by more than a quarter second, leftover hair or skin of the replaced original.
minor = a small detail."""


def audit_times(L, cuts, offset=0):
    """frame indices to check: every 12 frames (from `offset`), both sides of every cut, and the last frame"""
    idx = set(range(offset, L, 12)) | {L - 1}
    for c in cuts:
        idx |= ({max(0, c - 2), min(L - 1, c + 2)} if offset == 0 else {max(0, c - 4), min(L - 1, c + 1)})
    out = []
    for i in sorted(idx):  # drop near-duplicates (keep the cut-adjacent ones)
        if out and i - out[-1] < 3 and not any(abs(i - (c + 2)) < 1 or abs(i - (c - 2)) < 1 for c in cuts):
            continue
        out.append(i)
    return out[:22]


VERIFY_SYSTEM = """A reviewer flagged a problem in ONE frame of an AI video edit by comparing it with the source frame at the same moment.
You get the GENERATED frame and SOURCE frames from 0.17 s before to 0.17 s after that moment, plus the reviewer's note and the reference pictures.
The edit is SUPPOSED to replace what the edit spec says; only judge the flagged problem. Decide whether it is REAL:
- not real if the generated frame matches one of the nearby source frames in the flagged respect (a caption, a shot change or a gesture that is only
  a few frames early or late), or if the note is wrong about what the frame shows
- real if no nearby source frame matches (an invented or missing object, a wrong word, a wrong person, a different shot)
Reply with JSON only: {"real": true or false, "why": "short"}"""


SRC_COUNT = ('You see one frame of a video. Is any ONE person shown twice or more in it - the same face, hair and clothes appearing as two figures, '
             'for example a blurred double, a double exposure, a clone, a mirror reflection or a split screen? Two or more DIFFERENT people standing '
             'together do NOT count. Reply with JSON only: {"same_person_twice": true or false, "why": "short"}')
DUP_WORDS = re.compile(r"\b(duplicate|second (?:full-body |blurred )?(?:copy|figure|woman|man|person|version)|cop(?:y|ies) of|clones?|twice|"
                       r"two (?:copies|versions|figures) of|double of)\b", re.I)


def source_doubles(src_f):
    """True when the SOURCE frame itself shows one person twice (a double exposure, a clone or mirror effect) - a copy in the edit is then the
    source's own effect (2026-10-30, test-files case 6 re-run: the spec text named no double, the pair check and the physics check both called
    the swapped double 'a duplicate'; the example keeps both figures)"""
    try:
        r = llm.ask([llm.img_part(src_f, 768)], system=SRC_COUNT, max_tokens=40, temperature=0.0)
        return bool(r.get("same_person_twice")) if isinstance(r, dict) else False
    except Exception:
        return False


def blank_jitter(out_f, src, i, d=4):
    """the generated frame is all black or all white and so is a source frame within +-d frames: a cut to black / a flash 1-4 frames off, not a
    missing subject (2026-10-30, test-files case 4 re-run: the cut to black came 2 frames early -> 'missing main subject' + 'anatomy: missing',
    score 0 and a retake)"""
    def blank(f):
        m, s = float(f.mean()), float(f.std())
        return (m < 18 and s < 12) or (m > 225 and s < 25)
    if out_f is None or not src or not blank(out_f):
        return False
    return any(blank(src[min(len(src) - 1, max(0, i + k))]) for k in range(-d, d + 1))


def source_doubles_near(src, i):
    """source_doubles() at the flagged frame and 0.25 s either side (test-files case 6: the source's double enters a few frames after the time
    the render's copy was flagged). Test 8/9 single frames (two different people never counted; the miss = a frame before the double enters)"""
    for d in (0, 6, -6):
        if src and source_doubles(src[min(len(src) - 1, max(0, i + d))]):
            return True
    return False


def verify_issues(job, spec, src, out_frames, v):
    """drop flagged problems that are only frame-level timing jitter (checked against source frames +-4 frames); identity issues stay"""
    L = len(out_frames); keep = []; dropped = []
    for x in v.get("issues", []) or []:
        if not isinstance(x, dict) or x.get("time") is None or str(x.get("what", "")).startswith("shot "):
            keep.append(x); continue
        try:
            i = min(L - 1, max(0, int(round(float(x["time"]) * FPS))))
        except (TypeError, ValueError):
            keep.append(x); continue
        if blank_jitter(out_frames[i], src, i):
            dropped.append({**x, "dropped": "a black or white frame the source also has within 4 frames (cut timing jitter)"}); continue
        if DUP_WORDS.search(str(x.get("what", ""))) and source_doubles_near(src, i):
            dropped.append({**x, "dropped": "the source shows the same person twice here (its own double / clone effect)"}); continue
        c = [{"type": "text", "text": f"Reviewer's note: {x.get('what')}"}, {"type": "text", "text": "Edit spec: " + json.dumps(
            {"instruction": spec.get("instruction"), "kept": spec.get("kept")}, ensure_ascii=False)[:1500]}]
        c += [{"type": "text", "text": f"GENERATED frame at t={i / FPS:.2f}s:"}, llm.img_part(out_frames[i], 512)]
        for d in (-4, -2, 0, 2, 4):
            j = min(len(src) - 1, max(0, i + d))
            c += [{"type": "text", "text": f"SOURCE t={j / FPS:.2f}s:"}, llm.img_part(src[j], 384)]
        try:
            r = llm.ask(c, system=VERIFY_SYSTEM, max_tokens=200, temperature=0.0)
            if r.get("real", True):
                keep.append(x)
            else:
                dropped.append({**x, "dropped": r.get("why", "")})
        except Exception:
            keep.append(x)
    if dropped:
        v["issues"] = keep; v["dropped"] = dropped
        dt = {round(float(d["time"]), 2) for d in dropped}
        for pr in v.get("pairs", []) or []:
            if isinstance(pr, dict) and not pr.get("ok", True) and pr.get("t") is not None and round(float(pr["t"]), 2) in dt:
                pr["ok"] = True; pr["problem"] = ""
        sev = [x.get("severity") for x in keep if isinstance(x, dict)]
        bad_pairs = [pr for pr in v.get("pairs", []) or [] if isinstance(pr, dict) and not pr.get("ok", True)]
        v["score"] = 10.0 if not keep and not bad_pairs else (4.0 if "critical" in sev else 6.0 if "major" in sev else 8.0 if keep or bad_pairs else 10.0)
    return v


def spec_for_part(spec, seg):
    """the spec as this part shows it: a person in a themed costume here (add_costumes, people path) is checked against that costume, not their
    picture's clothes (2026-10-30, test-files case 4 re-run: the shrine part was scored down for the kimonos and the retake note asked for the
    picture's green t-shirt and red polo)"""
    what = (seg or {}).get("costume_what") or {}; part = (seg or {}).get("costume_part") or []
    if not what and not part:
        return spec
    sp = json.loads(json.dumps(spec))
    for s in sp.get("subjects", []):
        w = what.get(s.get("tag"))
        if w:
            base = re.split(r",?\s+(?:wearing|dressed in|clad in|in a|in an|in her|in his|in their)\b", str(s.get("description") or ""), maxsplit=1)[0].rstrip(" ,.")
            s["description"] = f"{base}. In this part of the video they wear {w} - the costume the original person wears here - not the clothes of their picture."
        elif s.get("tag") in part:
            s["description"] = (str(s.get("description") or "").rstrip(" .") + ". In the shots of this part where the original person wears a costume "
                                "(see the source frames), they wear that costume instead of their picture's clothes.")
    return sp


def audit_dense(job, spec, win_path, out_frames, cuts=None, seg=None):
    """two passes with offset sampling points (one pass misses things), merged, then the who-is-who check and the jitter verification"""
    spec = spec_for_part(spec, seg)
    L = len(out_frames)
    if cuts is None:
        cuts = detect_cuts(win_path); cuts = sorted(set(cuts) | set(detect_soft_cuts(win_path, cuts)))
    src = read_frames(win_path)
    va = audit_pass(job, spec, src, out_frames, cuts, 0)
    vb = audit_pass(job, spec, src, out_frames, cuts, 6)
    seen = set(); issues = []
    for x in (va.get("issues") or []) + (vb.get("issues") or []):
        if isinstance(x, dict):
            key = (round(float(x.get("time") or 0), 1), str(x.get("what", ""))[:60])
            if key not in seen:
                seen.add(key); issues.append(x)
    v = {"pairs": (va.get("pairs") or []) + (vb.get("pairs") or []), "issues": issues, "score": min(va["score"], vb["score"]),
         "times": sorted(set(va["times"]) | set(vb["times"])), "fix": " ".join(x for x in (va.get("fix"), vb.get("fix")) if x)}
    try:  # who-is-who per shot (catches a short shot at the end that kept or swapped the wrong person)
        idi = recheck_missing(job, spec, out_frames, identity_check(job, spec, src, out_frames, cuts))
        if idi:
            v["issues"].extend(idi); v["score"] = min(v["score"], 4.0); v["identity"] = idi
    except Exception as e:
        v["identity_error"] = str(e)[:200]
    if v.get("issues"):
        v = verify_issues(job, spec, src, out_frames, v)
    try:   # focused physical check - confirmed by its own step, so it skips the timing-jitter filter above
        ph = physics_check(job, spec, src, out_frames, cuts, seg=seg)
        if ph:
            v["issues"] = (v.get("issues") or []) + ph; v["physics"] = ph
            v["score"] = min(v["score"], 4.0 if any(x["severity"] == "critical" for x in ph) else 6.0)
            fixes = " ".join(x["fix"] for x in ph if x.get("fix"))
            if fixes:
                v["fix"] = ((v.get("fix") or "") + " " + fixes).strip()
    except Exception as e:
        v["physics_error"] = str(e)[:200]
    bad = [x for x in v.get("issues", []) if isinstance(x, dict)]
    v["pass"] = v["score"] >= 10 and not bad and all(p.get("ok", True) for p in v.get("pairs", []) if isinstance(p, dict))
    return v


SEEN_SYSTEM = """You look at ONE frame of a video and at reference pictures of some subjects. For each listed subject, say whether ANY part of it
can be seen in the frame - even only its legs, feet, hands, an arm, its back or the side of its head, or the thing it rides or holds (a bike, a
board). Reply with JSON only: {"visible": ["<Subject n>", ...]}"""


def recheck_missing(job, spec, out_frames, issues):
    """the who-is-who check can miss a subject that is only partly in frame (a pedal close-up shows just legs and the bike): every 'the edit
    shows none of them' issue gets one focused question about that frame and is dropped when every missing subject is in fact visible"""
    out = []
    for x in issues:
        m = re.match(r"shot \d+: the source shows (\[.*?\]|none of the subjects) but the edit shows (\[.*?\]|none of them)", str(x.get("what", "")))
        if not m or "; also" in str(x.get("what", "")):   # leftover / invented people are never dropped
            out.append(x); continue
        tags = lambda t: set(re.findall(r"<Subject \d+>", t))
        miss = tags(m.group(1)) - tags(m.group(2))
        subs = [s_ for s_ in spec.get("subjects", []) if s_.get("tag") in miss and isinstance(s_.get("picture"), int) and 1 <= s_["picture"] <= len(job["images"])]
        if not miss or len(subs) != len(miss):
            out.append(x); continue
        try:
            fi = min(len(out_frames) - 1, int(round(float(x.get("time") or 0) * FPS)))
            content = []
            for s_ in subs:
                content += [{"type": "text", "text": f"{s_['tag']} ({(s_.get('description') or '')[:200]}):"}, llm.img_part(job["images"][s_["picture"] - 1], 384)]
            content += [{"type": "text", "text": "The frame:"}, llm.img_part(out_frames[fi], 640)]
            r = llm.ask(content, system=SEEN_SYSTEM, max_tokens=120, temperature=0.0)
            if miss <= set(r.get("visible") or []):
                continue      # it is there after all (partly in frame)
        except Exception:
            pass
        out.append(x)
    return out

def audit_pass(job, spec, src, out_frames, cuts, offset):
    """one strict pass over frame pairs sampled from `offset`"""
    L = len(out_frames); idx = audit_times(L, cuts, offset)
    subj = [{k: s.get(k) for k in ("tag", "picture", "kind", "replaces", "description")} for s in spec.get("subjects", [])]
    content = [{"type": "text", "text": "Edit spec:\n" + json.dumps({"instruction": spec.get("instruction"), "subjects": subj, "kept": spec.get("kept"),
                                                                    "treatment": spec.get("treatment")}, ensure_ascii=False)[:5000]}]
    if job.get("prompt"):
        content.append({"type": "text", "text": f"The user's request: {job['prompt']}"})
    content.append({"type": "text", "text": "Reference pictures:"})
    for i, p in enumerate(job["images"], 1):
        content += [{"type": "text", "text": f"<Picture {i}>:"}, llm.img_part(p, 512)]
    content.append({"type": "text", "text": ("Source shot changes at t = " + ", ".join(f"{c / FPS:.2f}" for c in cuts) + " s.") if cuts else "The source is one continuous shot."})
    ctx = job.get("_context_frames") or []
    if ctx:
        content.append({"type": "text", "text": "Context only (NOT part of the clip): the source continues like this right after the clip's last frame:"})
        content += [llm.img_part(f, 384) for f in ctx]
    content.append({"type": "text", "text": "Frame pairs (SOURCE then GENERATED):"})
    for i in idx:
        content += [{"type": "text", "text": f"t={i / FPS:.2f}s SOURCE:"}, llm.img_part(src[min(i, len(src) - 1)], 384),
                    {"type": "text", "text": f"t={i / FPS:.2f}s GENERATED:"}, llm.img_part(out_frames[i], 448)]
    v = llm.ask(content, system=AUDIT2_SYSTEM, max_tokens=2200, temperature=0.0)
    v["score"] = float(v.get("score", 0) or 0); v["times"] = [round(i / FPS, 2) for i in idx]
    return v


# ---------------------------------------------------------------------------- shot repair (re-render only the shots the audit flags)
REPAIR_ROUNDS = 2
COMPOSE_PICK = """You rate edited video frames. A = the original frame from the video. Then the reference pictures, then numbered candidates.
The edit asked for: {edit}
A good candidate keeps A's framing, camera angle, poses, hand positions, held objects, on-screen text and lighting exactly - every person on the
SAME side of the frame as in A (a mirrored or swapped left/right arrangement is same_as_A 3 or less) at the same size - keeps every person,
animal and object of A that the edit does not replace (a rider stays seated on the new animal or vehicle), and applies the edit exactly (identity,
hair and clothes from the pictures), with no warped faces or hands and no extra or missing people. A candidate that drops someone the edit does not
replace is edit_ok false, and so is one where the original thing is still there with a small copy of the picture's thing pasted on top.
Reply with JSON only: {{"candidates": [{{"edit_ok": true or false, "identity": 0-10, "same_as_A": 0-10, "artifacts": true or false}}], "why": "short"}}"""


KEEP_OTHERS = "Every person, animal and object in Picture 1 that is not named above stays exactly where it is, in the same pose."


def keep_line(kept, text=""):
    """the composite's keep sentence: everyone not replaced stays, plus this segment's own kept list. Qwen-Image takes every noun literally (an
    example 'rider on the new animal' put a woman on a horse in a meeting room), so the rider sentence appears only when the shot has a rider."""
    kept = re.sub(r"<Subject \d+>", "the new subject", kept or "").strip().rstrip(".")
    line = KEEP_OTHERS
    if re.search(r"\b(rider|riders|riding|rides|ridden|mounted)\b", f"{kept} {text}", re.I):
        line += " The rider stays seated on the new animal or vehicle, in the same riding pose."
    return line + (f" These stay in the picture: {kept}." if kept else "")


def issue_time(x):
    try:
        return float(x.get("time") or 0)
    except (TypeError, ValueError):
        return 99.0


FIX_REWRITE = """You rewrite correction notes for a video generator that does not understand negation: every thing a note names gets DRAWN
('the skatepark ramps are still visible' and 'airborne high above the ramp' both put ramps back in). The edit is given below. Rewrite the notes as
1-4 short sentences that describe only what the EDITED video must show: the new characters, objects and place exactly as the edit defines them, the
poses, and the timing ('at 0.5 s he is high in the air above the walkway'). Keep every concrete side, position and time the notes give ('on the
right side of the walkway stands only the brick parapet wall'). Name the characters briefly ('the skateboarder') - never repeat their full
descriptions, the prompt already has them. Never name a mistake. Never write 'no ...', 'without ...', 'instead of ...' or 'remove ...'.
{place}Reply with JSON only: {{"fix": "the rewritten notes"}}"""


def positive_fix(spec, notes, seg=None):
    """correction notes -> positive sentences in the NEW place (Qwen3.8); the notes unchanged when the LLM fails. The checks' notes name the
    mistake ('the skatepark ramps are still visible') or describe the SOURCE frame ('a concrete path in a park, with trees'), and H3 draws every
    noun: M9 v2's retake and repair of the toward-camera shot rebuilt the skatepark from exactly those words"""
    notes = [str(n).strip() for n in notes if str(n or "").strip()]
    if not notes:
        return ""
    spec = spec_for_part(spec, seg)
    loc = next((s for s in spec.get("subjects", []) if s.get("kind") == "location"), None)
    place = (f"The edit moves everything to a new place ({loc.get('description') or ''}). Describe that place only - never name the old place it replaces "
             f"({loc.get('replaces') or 'the original location'}) or any of the old place's features, buildings, plants or ground, not even to say where "
             f"something was. " if loc else "")
    subs = spec.get("subjects", []); only = ""
    if seg and isinstance(seg.get("present"), list):   # only the people of THIS part (2026-10-29, test-files case 8: the retake note of a part with
        subs = [s for s in subs if s.get("kind") == "location" or s.get("tag") in seg["present"]]   # only the young woman said "The older woman
        only = ("\nOnly these subjects are in this part of the video: " + (", ".join(s.get("tag") for s in subs if s.get("kind") != "location") or "nobody")
                + ". Never add anyone else.")                                                          # stands in the playroom")
    subj = [{k: s.get(k) for k in ("tag", "kind", "description")} for s in subs]
    try:
        r = llm.ask([{"type": "text", "text": "The edit: " + json.dumps({"instruction": spec.get("instruction"), "subjects": subj}, ensure_ascii=False)[:2500]
                      + only + "\n\nNotes:\n- " + "\n- ".join(notes)[:2500]}],
                    system=FIX_REWRITE.format(place=place), max_tokens=400, temperature=0.0)
        fix = str((r or {}).get("fix") or "").strip()
        return fix[:700] if len(fix) > 20 else "; ".join(notes)[:600]
    except Exception:
        return "; ".join(notes)[:600]


def shot_bounds(cuts, L):
    b = [0] + sorted(c for c in cuts if 0 < c < L) + [L]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] > b[i]]


def flagged_shots(a, cuts, L):
    """the shots (frame ranges) that hold the audit's problems"""
    bounds = shot_bounds(cuts, L); ts = []
    for x in a.get("issues", []) or []:
        if isinstance(x, dict) and x.get("time") is not None:
            ts.append(x["time"])
    for p in a.get("pairs", []) or []:
        if isinstance(p, dict) and not p.get("ok", True) and p.get("t") is not None:
            ts.append(p["t"])
    bad = set()
    for t in ts:
        try:
            f = min(L - 1, max(0, int(round(float(t) * FPS))))
        except (TypeError, ValueError):
            continue
        for si, (s0, s1) in enumerate(bounds):
            if s0 <= f < s1:
                bad.add(si)
    return bounds, sorted(bad)


def localized(a, cuts, L):
    bounds, bad = flagged_shots(a, cuts, L)
    return bool(bad) and sum(bounds[i][1] - bounds[i][0] for i in bad) <= 0.6 * L


def shot_text(spec, k, a, b):
    seg = next((s for s in spec.get("segments", []) if int(s.get("index", -1)) == k), None) or {}
    parts = re.findall(r"\[Shot \d+\]\s*\(([\d.]+)\s*to\s*([\d.]+)\s*seconds?\)\s*(.*?)(?=\[Shot \d+\]|$)", seg.get("shots", "") or "", flags=re.S)
    mid = (a + b) / 2 / FPS
    for s0, s1, txt in parts:
        if float(s0) - 0.05 <= mid <= float(s1) + 0.05:
            return txt.strip()
    return min(parts, key=lambda p: abs((float(p[0]) + float(p[1])) / 2 - mid))[2].strip() if parts else ""


SHOT_SYSTEM = """You describe ONE shot of a video for a video model that re-renders it. Subjects (each replaces something in the source):
{subjects}
You see frames of the source shot in order. Write one paragraph: the framing and camera (e.g. extreme close-up of the face, low-angle wide shot,
camera tilting up), what is in the frame, and exactly what each visible subject does, naming them by their tags and describing them as their
replacement looks (keep a costume they wear in this shot). Mention on-screen text word for word if any. Reply with JSON only: {{"text": "..."}}"""


def describe_shot(job, spec, frames):
    subs = [s for s in spec.get("subjects", [])]
    desc = "\n".join(f"{s.get('tag')}: {s.get('kind')} - replaces {s.get('replaces') or '-'}; looks like: {(s.get('description') or '')[:200]}" for s in subs)
    content = [{"type": "text", "text": "Source shot, frames in order:"}] + [llm.img_part(f, 512) for f in frames]
    r = llm.ask(content, system=SHOT_SYSTEM.format(subjects=desc), max_tokens=400, temperature=0.0)
    return (r.get("text") or "").strip()


PLATE_PLACE = ("Picture 1 already shows the place from Picture {n}: keep the whole place - the ground, walls, buildings, background and sky - exactly "
               "as it is in Picture 1.")


def compose_prompt(job, use, costume=(), on_plate=False, extra="", kept=None, shots=""):
    """compose_frame's edit prompt: Picture 1 = the frame, then one picture per subject in `use` -> (prompt, subject pictures, edit lines)"""
    pics = [None]; lines = []
    for s in use:
        pics.append(job["images"][s["picture"] - 1]); n = len(pics)
        if s.get("kind") == "location" and on_plate:
            lines.append(PLATE_PLACE.format(n=n))
        elif s.get("kind") == "location":
            lines.append(f"Replace the surrounding place with the place from Picture {n}, as Picture 1's camera would see it there: the same camera angle, "
                             f"distance and framing - in a close-up only the nearby floor or wall of that place and an out-of-focus background in its colours, "
                             f"never the whole view of Picture {n}.")
        else:
            if s["tag"] in costume:
                lines.append(f"Replace the person wearing that outfit with the {s.get('kind', 'person')} from Picture {n} - the same face, hair, skin tone and "
                             f"body as Picture {n}, but keep the clothes the person in Picture 1 wears.")
            elif s.get("kind") == "animal" and re.search(r"\b(motorcycle|motorbike|bike|bicycle|scooter|car|quad|atv|jet ?ski|snowmobile|vehicle|horse)\b", str(s.get("replaces") or ""), re.I):
                lines.append(f"Replace {s.get('replaces')} with the {s.get('kind')} from Picture {n}, ridden the way a real rider rides it: the rider sits "
                             f"astride it in the saddle, legs down both its sides, hands holding the reins just above its shoulders, and its neck and head "
                             f"are well ahead of the rider at their natural distance (a horse's head is about one and a half metres in front of the "
                             f"rider's hands) - in a close-up of the rider's hands only the mane and the top of the neck are in front of the hands and the "
                             f"head is far ahead or out of the frame. Its body is as big as a real {s.get('kind')} of its kind next to the rider.")
            else:
                lines.append(f"Replace {s.get('replaces') or 'the ' + s.get('kind', 'person')} with the {s.get('kind', 'person')} from Picture {n} - the same face, "
                             f"hair, skin tone, body and clothes as Picture {n}." if s.get("kind") != "object" else
                             f"Replace {s.get('replaces') or 'the object'} with the object from Picture {n}: the same shape, colours and materials as Picture {n}, "
                             f"at the size, position and angle of the original in Picture 1. If only part of the original is visible (a close-up), turn exactly "
                             f"that part into the matching part of Picture {n}'s object - never add a small whole copy of it.")
    edit = " ".join(lines)
    prompt = ("Edit Picture 1, a frame from a video. " + edit + (" " + extra.strip() if extra else "") + " " + keep_line(kept, shots) +
              " Keep everything else in Picture 1 exactly the same: the framing, camera angle, poses, hand positions, held objects, on-screen "
              "text, lighting and colours.")
    return prompt, pics[1:], edit


def compose_frame(job_obj, job, spec, frame, W, H, tag, kept=None, shots="", on_plate=False, extra="", seeds=(101, 202)):
    """the shot's first source frame with the spec's replacements applied (Qwen-Image, two candidates, Qwen3.8 picks) -> path or None.
    on_plate: the frame comes from a new-place plate (the new place is already there, only the people / things are swapped)"""
    jdir = job_obj.store.dir(job_obj.id); fp = os.path.join(jdir, f"{tag}_src.png"); cv2.imwrite(fp, frame)
    subs = [s for s in spec.get("subjects", []) if isinstance(s.get("picture"), int) and 1 <= s["picture"] <= len(job["images"])]
    if not subs:
        return None, []
    try:
        r = llm.ask([{"type": "text", "text": "Which of these can be seen in the frame, even partly? " + json.dumps(
            [{"tag": s["tag"], "what": s.get("replaces") or s.get("description")} for s in subs if s.get("kind") != "location"], ensure_ascii=False)},
            llm.img_part(fp, 640)] + ([{"type": "text", "text": "Context (the source right after this moment):"}] + [llm.img_part(f, 384) for f in job.get("_context_frames", [])[:2]]
                                       if job.get("_context_frames") else []),
            system='Reply with JSON only: {"visible": [the tags of the listed things that can be seen in the frame], "costume": {tag: true if that '
                   'person wears a clearly different outfit here than the one described (a costume change)}}', max_tokens=250)
        vis = set(r.get("visible") or []); costume = {k2 for k2, v2 in (r.get("costume") or {}).items() if v2}
    except Exception:
        vis = {s["tag"] for s in subs}; costume = set()
    use = [s for s in subs if s["tag"] in vis or s.get("kind") == "location"]
    if not use:
        return None, []
    prompt, sub_pics, edit = compose_prompt(job, use, costume, on_plate, extra, kept, shots)
    pics = [fp] + sub_pics
    cands = []
    for i, seed in enumerate(seeds):
        try:
            c = qwen_edit(job_obj, pics, prompt, seed, f"{tag}_c{i + 1}")
            cv2.imwrite(c, cover_resize(cv2.imread(c, cv2.IMREAD_COLOR), W, H)); cands.append(c)
        except comfy.Cancelled:
            raise
        except Exception as e:
            job_obj.log(f"{tag}: composite {i + 1} failed: {e}")
    if not cands:
        return None, use
    try:
        c = [{"type": "text", "text": "A (original frame):"}, llm.img_part(fp, 640)]
        for i, p in enumerate(pics[1:], 2):
            c += [{"type": "text", "text": f"Picture {i}:"}, llm.img_part(p, 448)]
        for i, cp in enumerate(cands, 1):
            c += [{"type": "text", "text": f"Candidate {i}:"}, llm.img_part(cp, 640)]
        r = llm.ask(c, system=COMPOSE_PICK.format(edit=edit + (" " + extra.strip() if extra else "")), max_tokens=400, temperature=0.0)
        rated = r.get("candidates") or []
        place = "Replace the surrounding place" in edit   # a new place: 'identity' says nothing about the place, so the edit itself must be ok
        ok = [(float(x.get("identity", 0)) + float(x.get("same_as_A", 0)), i) for i, x in enumerate(rated[:len(cands)])
              if (x.get("edit_ok") or (float(x.get("identity", 0)) >= 8 and not place)) and not x.get("artifacts")
              and float(x.get("identity", 0)) >= 6 and float(x.get("same_as_A", 0)) >= 6]
        job_obj.log(f"{tag}: composite ratings {json.dumps(rated)[:300]} ({str(r.get('why') or '')[:200]})")
        return (cands[max(ok)[1]] if ok else None), use
    except Exception as e:
        job_obj.log(f"{tag}: composite check failed ({e}); using candidate 1")
        return cands[0], use


def inpaint_masks(job_obj, job, spec2, p, cur, L, W, H):
    """SAM 3 outlines of the people / things to replace in the SOURCE shot (every instance, also reflections and double exposures) -> per-frame
    alpha (grown + feathered), the current video's frames for the clip (what stays outside the outlines) and the mask clip for the inpaint"""
    self = job_obj; jd = self.store.dir(self.id); Lr, s0 = p["Lr"], p["s0"]
    kinds = {x.get("kind", "person") for x in spec2.get("subjects", []) if x.get("kind") != "location"}
    words = "|".join(sorted({"person" if k_ == "person" else "animal" if k_ == "animal" else re.sub(r"\s*\b(in|of|from)\s+<Video \d+>|<[^>]+>", "", (x.get("replaces") or "object").split(",")[0]).strip()[:40] or "object"
                             for x in spec2.get("subjects", []) for k_ in [x.get("kind", "person")] if k_ != "location"})) or "person"
    canvas = [cover_resize(f, W, H) for f in p["clip"]]
    cwinC = os.path.join(jd, f"{p['tag']}_srcC.mp4"); write_video(canvas, cwinC, crf=12)
    tools.gpu_take(self)
    od = os.path.join(jd, f"{p['tag']}_sam")
    tools.run_proc(self, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_matte.py", cwinC, od, words, "matte"], 0.5, 0.5, f"Outlining {words.replace('|', ', ')}")
    M = read_frames(os.path.join(od, "matte.mp4"))
    grow = max(4, int(0.025 * max(W, H))); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    alpha, masks = [], []
    for i in range(Lr):
        m = cv2.cvtColor(M[min(i, len(M) - 1)], cv2.COLOR_BGR2GRAY) if M else np.zeros((H, W), np.uint8)
        m = cv2.dilate((m > 100).astype(np.uint8), ker)
        masks.append(cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR))
        alpha.append(cv2.GaussianBlur(m.astype(np.float32), (0, 0), grow * 0.5))
    if not any(float(a_.mean()) > 0.002 for a_ in alpha):
        raise RuntimeError(f"nothing found for '{words}'")
    curC = [cover_resize(cur[min(L - 1, s0 + i)], W, H) for i in range(Lr)]
    cur_clip = os.path.join(jd, f"{p['tag']}_cur.mp4"); write_video(curC, cur_clip, crf=12)
    mclip = os.path.join(jd, f"{p['tag']}_mask.mp4"); write_video(masks, mclip, crf=0)
    self.log(f"{p['tag']}: masked repair of '{words}' ({100 * float(np.mean([a_.mean() for a_ in alpha])):.0f} % of the frame)")
    return alpha, cur_clip, mclip


PLACE_ISSUE = re.compile(r"\b(location|place|background|setting|scene|environment|surroundings|room|bedroom|rink|street|park|skatepark|gym|beach|sea|ocean|city|interior|floor|ground)\b[^.;]{0,90}"
                         r"\b(not (?:been )?replaced|remains?|remaining|still|revert\w*|original|source|instead of|missing|fails?|failed|incomplete|incorrect|wrong|unchanged)\b|"
                         r"\b(revert\w*|still visible|not replaced|leftover|left over|original)\b[^.;]{0,90}\b(location|place|background|room|setting|scene|environment)\b", re.I)


def place_reverted(a):
    """did the check see the OLD place (a critical / major issue about the location not being replaced or coming back)?"""
    return any(isinstance(x, dict) and x.get("severity") in ("critical", "major") and (PLACE_ISSUE.search(str(x.get("what", "")))
               or str(x.get("what", "")).startswith("out_of_place:"))     # things of the old place left in the new one = the old place is partly back
               for x in (a or {}).get("issues", []))


def camera_path(frames, keep):
    """3x3 similarity transforms taking frame-0 coordinates to frame-i coordinates, from ORB features on the BACKGROUND only (keep = 1 on the
    subject, which is masked out); chained frame to frame, identity where it can't be estimated"""
    orb = cv2.ORB_create(1500); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True); A = [np.eye(3)]
    def feats(f, m):
        return orb.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), ((m < 0.5) * 255).astype(np.uint8))
    kp0, d0 = feats(frames[0], keep[0])
    for i in range(1, len(frames)):
        kp1, d1 = feats(frames[i], keep[i]); T = np.eye(3)
        if d0 is not None and d1 is not None and len(kp0) >= 12 and len(kp1) >= 12:
            mt = bf.match(d0, d1)
            if len(mt) >= 10:
                p0 = np.float32([kp0[x.queryIdx].pt for x in mt]); p1 = np.float32([kp1[x.trainIdx].pt for x in mt])
                M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None and abs(np.linalg.det(M[:, :2]) - 1) < 0.3:
                    T = np.vstack([M, [0, 0, 1]])
        A.append(T @ A[-1]); kp0, d0 = kp1, d1
    return A


def damp_path(A, W, H, mg, my_room=None):
    """scale a camera path (frame-0 -> frame-i similarity transforms) down so the frame centre never moves more than ~90 % of the plate margin:
    each transform is rebuilt about the frame centre from its centre shift, rotation and zoom, all damped by the same factor"""
    c0 = np.array([W / 2.0, H / 2.0, 1.0]); parts = []
    for T in A:
        d = (T @ c0)[:2] - c0[:2]; sc = float(np.sqrt(abs(np.linalg.det(T[:2, :2])))) or 1.0
        parts.append((d, float(np.arctan2(T[1, 0], T[0, 0])), sc))
    mx, my = max(abs(p_[0][0]) for p_ in parts), max(abs(p_[0][1]) for p_ in parts)
    f = min(1.0, 0.9 * mg * W / max(mx, 1e-3), 0.9 * (mg * H if my_room is None else my_room) / max(my, 1e-3))
    out = []
    for d, ang, sc in parts:
        a2, s2 = ang * f, float(np.clip(sc ** f, 0.8, 1.6)); c, s_ = np.cos(a2) * s2, np.sin(a2) * s2
        T2 = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]], np.float64)
        pre = np.array([[1, 0, -c0[0]], [0, 1, -c0[1]], [0, 0, 1]]); post = np.array([[1, 0, c0[0] + f * d[0]], [0, 1, c0[1] + f * d[1]], [0, 0, 1]])
        out.append(post @ T2 @ pre)
    return out

VIEW_PICK = """You choose the best background view for one shot of a video that moves to a new place. You get: SOURCE (the shot's first
frame, in the old place), PLACE (a picture of the new place) and numbered candidate views of the new place.
A good candidate looks like PLACE (same buildings, ground, materials, colours, light), is seen from the same camera as SOURCE (the same
height above the ground, the same angle, the horizon at the same height), is empty (no people, animals or vehicles), and has open
ground, path, road or water where the SOURCE subject is, running the way the subject travels ({travel}) - not a wall or a drop.
Reply with JSON only: {{"candidates": [{{"same_place": 0-10, "camera_match": 0-10, "open_path": true or false, "empty": true or false}}], "why": "short"}}"""


REPAINT_MAP = """A video moves to a NEW PLACE but keeps its camera. You get OLD (the shot's first frame, the moving subjects greyed out) and a
picture of the NEW PLACE. Write 2-5 short sentences for a painter. Begin with the camera: how high above the ground it is (judge by the
size of the ground texture and the things on it: "Seen from just above the floor, a few centimetres high", "seen from about one metre high")
and where the horizon sits. Then what each main part of OLD becomes in the NEW PLACE, keeping its position and the way it runs across the
frame: the ground the subject moves on (where it is, which way it runs: left to right, toward the
camera...), any wall, ledge, rail or ramp beside or behind it (what it becomes, where, which way it runs), and the far background. Keep
every thing's SIZE and height: a knee-high ledge or kerb stays knee-high (a low stone step or bench of the new place - never a tall wall
the subject would hit), a rail stays a rail-sized thing, a tall wall stays tall. A ledge, ramp, rail or box the subject could ride onto or
land on stays a solid thing standing ON the ground in front of the far background, with ground behind it - never the top edge of a wall
with a drop or open sky behind it (the subject would ride off into the air). The subject travels {travel}; the ground must stay open
that way. Name only things of the NEW PLACE in the result (example of the style: "The flat
ground in the lower half becomes the grey stone walkway, running left to right across the frame. The low wall behind it becomes the brick
parapet, running left to right parallel to the walkway. The trees and houses above become hazy mountains and sky."). Never mention people.
Reply with JSON only: {{"mapping": "..."}}"""

REPAINT_TXT = ("Re-paint Picture 1 as a different place, keeping its exact camera and layout: the same camera height and angle, the same lens, "
               "the horizon at the same height, and every path, ledge and wall at the same position and running the same way across the frame. "
               "{mapping} The new place: {place} The flat grey patch is a hole: fill it with the ground and walls around it. The scene is empty. "
               "Photorealistic, sharp, natural daylight.")
FILL_HOLE = ("Remove the flat grey shape from Picture 1 and fill its area with the scene around it: continue the ground, walls and background "
             "exactly, in the same perspective and light, as if nothing had ever stood there. Change nothing else in Picture 1.")
EXTEND_TXT = ("Picture 1 shows a place with flat grey bands around it: they are unseen parts of the same place. Extend the place into them "
              "seamlessly - continue the ground, path, walls, horizon and sky exactly, in the same perspective and light; a path or wall that "
              "runs across the picture keeps running across it at the same distance, height and size. Change nothing in the part that is "
              "already painted.")


REPAINT_PICK = """You check re-painted video frames. SOURCE = a shot's first frame in the OLD place (the moving subjects may be missing). PLACE =
a picture of the NEW place. Each candidate = the same frame re-painted as the new place. The new place may look nothing like the old one -
judge only these: camera_match 0-10 = the same camera as SOURCE: the same height above the ground (a floor-level camera stays at floor level:
judge by how big the ground texture is near the bottom of the frame), the ground ends at about the same height in the frame (the horizon or
the far edge of the floor), the same lens and angle; same_place 0-10 = it shows the NEW place of PLACE (its kind of ground, buildings,
landscape, materials); old_left = true if anything of the OLD place is still there (furniture, rooms, skatepark ramps, rails, houses...);
open_path = the ground is open where the subject travels ({travel}) - not a wall or a drop right in front of it; empty = no people,
animals or vehicles. Reply with JSON only: {{"candidates": [{{"camera_match": 0-10, "same_place": 0-10, "old_left": true or false,
"open_path": true or false, "empty": true or false}}], "why": "short"}}"""


def place_repaint(job_obj, job, spec, k, image, hole, crop, tag):
    """the new place painted INTO the shot's own picture (2026-10-23, user: 'if the skater is skating east west we need a location at the
    Great Wall that allows for east west'): `image` is the shot's first frame or its whole-shot background mosaic, `hole` (0/1, same size)
    marks what is unseen or covered by the subjects (flat grey), `crop` = (x, y, w, h) of the first frame inside it. Qwen3.8 says what each
    part of the old place becomes (ground -> walkway running left to right, knee-high ledge -> low stone step...), Qwen-Image re-paints from
    TEXT ONLY (shown the place picture it slides back to that picture's postcard angle), a second pass fills the grey. Two candidates; VIEW_PICK
    judges their first-frame crops against the source. -> path of the re-painted image, same size as `image` (or None)"""
    jd = job_obj.store.dir(job_obj.id); dst = os.path.join(jd, f"{tag}_repaint.png")
    if os.path.exists(dst):
        return dst
    loc = next((s_ for s_ in spec.get("subjects", []) if s_.get("kind") == "location" and isinstance(s_.get("picture"), int)
                and 1 <= s_["picture"] <= len(job["images"])), None)
    if not loc:
        return None
    seg = next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), {}); travel = str(seg.get("travel") or "as in the video")
    Hc, Wc = image.shape[:2]; x0, y0, w0, h0 = crop
    empty = image.copy(); empty[hole > 0] = 127
    ep = os.path.join(jd, f"{tag}_repaint_in.png"); cv2.imwrite(ep, empty); place = job["images"][loc["picture"] - 1]
    wide = (Wc, Hc) != (w0, h0)
    try:
        r = llm.ask([{"type": "text", "text": "OLD" + (" (everything the moving camera sees during the shot, stitched into one wide picture):" if wide else ":")},
                     llm.img_part(ep, 900 if wide else 640), {"type": "text", "text": "NEW PLACE:"}, llm.img_part(place, 640)]
                    + ([{"type": "text", "text": f"Staging in the new place: {seg.get('spot')}"}] if seg.get("spot") else []),
                    system=REPAINT_MAP.format(travel=travel), max_tokens=400, temperature=0.0)
        mapping = str(r.get("mapping") or "").strip()
    except Exception as e:
        job_obj.log(f"{tag}: re-paint plan failed ({str(e)[:120]})"); mapping = ""
    prompt = REPAINT_TXT.format(mapping=mapping, place=rn_plain(loc.get("description")))
    cands = []
    for seed in (31, 47):
        try:
            c = os.path.join(jd, f"{tag}_repaint_s{seed}_f.png")
            if not os.path.exists(c):   # a retry of the same shot reuses its re-paints
                c = qwen_edit(job_obj, [ep], prompt, seed, f"{tag}_repaint_s{seed}")
                c = qwen_edit(job_obj, [c], FILL_HOLE, seed, f"{tag}_repaint_s{seed}_f")
            cv2.imwrite(c, cv2.resize(cv2.imread(c, cv2.IMREAD_COLOR), (Wc, Hc), interpolation=cv2.INTER_LANCZOS4)); cands.append(c)
        except comfy.Cancelled:
            raise
        except Exception as e:
            job_obj.log(f"{tag}: re-paint {seed} failed: {str(e)[:160]}")
    if not cands:
        return None
    try:
        fp = os.path.join(jd, f"{tag}_repaint_src.png"); cv2.imwrite(fp, image[y0:y0 + h0, x0:x0 + w0])
        c = [{"type": "text", "text": "SOURCE:"}, llm.img_part(fp, 640), {"type": "text", "text": "PLACE:"}, llm.img_part(place, 640)]
        for i, cp in enumerate(cands, 1):
            cc = os.path.join(jd, f"{tag}_repaint_c{i}_crop.png"); cv2.imwrite(cc, cv2.imread(cp, cv2.IMREAD_COLOR)[y0:y0 + h0, x0:x0 + w0])
            c += [{"type": "text", "text": f"Candidate {i}:"}, llm.img_part(cc, 768)]
        r = llm.ask(c, system=REPAINT_PICK.format(travel=travel), max_tokens=400, temperature=0.0)
        rated = r.get("candidates") or []
        ok = [(float(x.get("same_place", 0)) + 2 * float(x.get("camera_match", 0)), i) for i, x in enumerate(rated[:len(cands)])
              if x.get("open_path") and x.get("empty", True) and not x.get("old_left") and float(x.get("same_place", 0)) >= 5
              and float(x.get("camera_match", 0)) >= 3]   # the re-paint keeps the layout by construction and the judge reads camera height
                                                          # poorly (it called floor-level desert roads "high-angle"): only a clear miss fails
        job_obj.log(f"{tag}: re-painted views {json.dumps(rated)[:300]} ({str(r.get('why', ''))[:160]}); plan: {mapping[:300]}")
        if not ok:
            return None
        shutil.copy(cands[max(ok)[1]], dst)
    except Exception as e:
        job_obj.log(f"{tag}: re-paint check failed ({str(e)[:120]}); using view 1"); shutil.copy(cands[0], dst)
    return dst


def source_mosaic(frames, keep, A, off, Wc, Hc, step=2):
    """everything the moving camera sees during the shot, in one picture (frame-0 coordinates shifted by `off`): each canvas pixel from the
    first frame that sees it without a subject in front (frame 0 first, so its area is exactly frame 0). -> (mosaic, unseen 0/1 mask)"""
    mos = np.full((Hc, Wc, 3), 127, np.uint8); filled = np.zeros((Hc, Wc), bool)
    H_, W_ = frames[0].shape[:2]; g = max(4, W_ // 60); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    for i in sorted(set(range(0, len(frames), step)) | {len(frames) - 1}):
        M = A[i] @ off
        f = cv2.warpPerspective(frames[i], M, (Wc, Hc), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        free = (cv2.dilate((keep[i] > 0.02).astype(np.uint8), ker) == 0).astype(np.uint8)
        v = cv2.warpPerspective(free, M, (Wc, Hc), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        v = cv2.erode(v.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        new = v & ~filled; mos[new] = f[new]; filled |= new
    return mos, (~filled).astype(np.uint8)


def repaint_set(job_obj, job, spec, k, win, Lr, W, H, tag, view_tag=None, mosaic=False):
    """the new place as a SET for one shot (see repaint_plate): -> dict(E = the re-painted place covering everywhere the camera looks,
    A = source camera path (frame 0 -> frame i), off = canvas -> frame-0 shift, keep = per-frame subject alphas, frames = the source
    window at W x H, words, L_, T_). Raises when no re-painted view passes the check."""
    self = job_obj; jd = self.store.dir(self.id); words = subject_words(spec)
    frames = [cover_resize(f, W, H) for f in read_frames(win)][:Lr]
    tools.gpu_take(self); od = os.path.join(jd, f"{tag}_rp_sam")
    tools.run_proc(self, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_matte.py", win, od, words, "matte"], 0.5, 0.5, f"Outlining {words.replace('|', ', ')}")
    M = read_frames(os.path.join(od, "matte.mp4")); grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    keep = [cv2.resize(cv2.cvtColor(M[min(i, len(M) - 1)], cv2.COLOR_BGR2GRAY), (W, H)).astype(np.float32) / 255.0 if M else np.zeros((H, W), np.float32)
            for i in range(len(frames))]
    keep = [cv2.GaussianBlur(cv2.dilate((m > 0.15).astype(np.float32), grow), (0, 0), 2.0) for m in keep]
    A = camera_path(frames, [cv2.dilate((m > 0.05).astype(np.float32), grow) for m in keep])
    corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], np.float64).T; xs, ys = [0.0, W], [0.0, H]
    for T in A:   # where every frame looks, in frame-0 coordinates
        q = np.linalg.inv(T) @ corners; q = q[:2] / q[2:]; xs += list(q[0]); ys += list(q[1])
    pad = 0.08   # every side gets at least this margin: a frame that looks past the canvas would show a reflected seam
    L_ = min(1.5 * W, max(0, -min(xs)) + pad * W); R_ = min(1.5 * W, max(0, max(xs) - W) + pad * W)
    T_ = min(0.6 * H, max(0, -min(ys)) + pad * H); B_ = min(0.6 * H, max(0, max(ys) - H) + pad * H)
    L_, R_, T_, B_ = [int(round(v / 16.0)) * 16 for v in (L_, R_, T_, B_)]
    Wc, Hc = W + L_ + R_, H + T_ + B_; off = np.array([[1, 0, -L_], [0, 1, -T_], [0, 0, 1]], np.float64)
    vt = view_tag or tag   # one view per shot start: retakes and repairs reuse it
    if mosaic:
        mos, unseen = source_mosaic(frames, keep, A, off, Wc, Hc); cv2.imwrite(os.path.join(jd, f"{tag}_mosaic.png"), mos)
        view = place_repaint(self, job, spec, k, mos, unseen, (L_, T_, W, H), vt)
        if not view:
            raise RuntimeError("no re-painted view passed the check")
        E = cv2.resize(cv2.imread(view, cv2.IMREAD_COLOR), (Wc, Hc), interpolation=cv2.INTER_LANCZOS4)
    else:
        hole = cv2.dilate((keep[0] > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(4, W // 60) + 1,) * 2))
        view = place_repaint(self, job, spec, k, frames[0], hole, (0, 0, W, H), vt)
        if not view:
            raise RuntimeError("no re-painted view passed the check")
        V = cv2.resize(cv2.imread(view, cv2.IMREAD_COLOR), (W, H), interpolation=cv2.INTER_LANCZOS4); E = V
        ext = os.path.join(jd, f"{vt}_repaint_ext.png")
        if (Wc, Hc) != (W, H):   # room for the camera move: extend the view only where the camera goes
            canvas = np.full((Hc, Wc, 3), 127, np.uint8); canvas[T_:T_ + H, L_:L_ + W] = V
            cp = os.path.join(jd, f"{tag}_repaint_ext_in.png"); cv2.imwrite(cp, canvas)
            try:
                if not os.path.exists(ext):
                    shutil.copy(qwen_edit(self, [cp], EXTEND_TXT, 31, f"{tag}_repaint_ext_raw"), ext)
                E = cv2.resize(cv2.imread(ext, cv2.IMREAD_COLOR), (Wc, Hc), interpolation=cv2.INTER_LANCZOS4)
                fb = max(8, W // 24)   # the chosen frame goes back in exactly, feathered into the extension (no sky seam)
                wgt = np.zeros((Hc, Wc), np.float32); wgt[T_:T_ + H, L_:L_ + W] = 1.0
                wgt = cv2.erode(wgt, np.ones((2 * fb + 1, 2 * fb + 1), np.uint8)); wgt = cv2.GaussianBlur(wgt, (0, 0), fb / 2.0)
                wgt[T_ + fb:T_ + H - fb, L_ + fb:L_ + W - fb] = 1.0
                full = E.astype(np.float32).copy(); full[T_:T_ + H, L_:L_ + W] = V
                E = (full * wgt[..., None] + E.astype(np.float32) * (1 - wgt[..., None])).astype(np.uint8)
            except comfy.Cancelled:
                raise
            except Exception as ex:
                self.log(f"{tag}: extending the view failed ({str(ex)[:120]}) - the camera move is kept inside the frame")
                E = V; L_ = T_ = 0; off = np.eye(3); _, A, _f = fit_path(A, W, H, 0.0, 0.0)
    return dict(E=E, A=A, off=off, keep=keep, frames=frames, words=words, L_=L_, T_=T_, mosaic=mosaic)


def set_frames(st, W, H):
    """the plate frames of a set: the place moved with the source camera, the source subjects on top"""
    out = []
    for fr_, a_, T in zip(st["frames"], st["keep"], st["A"]):
        bg = cv2.warpPerspective(st["E"], T @ st["off"], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out.append((fr_.astype(np.float32) * a_[..., None] + bg.astype(np.float32) * (1 - a_[..., None])).astype(np.uint8))
    return out


def set_edit(st, i, edited, W, H, thr=28):
    """put a picture edit of plate frame i back into the set's place: where the edit changed the BACKGROUND (a stone ledge added under
    the board), its pixels are mapped into the place through the camera path, so every frame that sees that spot shows it"""
    cur = set_frames({**st, "frames": st["frames"][i:i + 1], "keep": st["keep"][i:i + 1], "A": st["A"][i:i + 1]}, W, H)[0]
    ed = cv2.resize(edited, (W, H), interpolation=cv2.INTER_AREA)
    d = (np.abs(ed.astype(np.int16) - cur.astype(np.int16)).max(axis=2) > thr).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); d = cv2.dilate(d, np.ones((9, 9), np.uint8))
    d = (d.astype(np.float32) * (1 - np.clip(st["keep"][i] * 3, 0, 1)))             # never the subject itself
    M = st["A"][i] @ st["off"]; Hc, Wc = st["E"].shape[:2]
    edw = cv2.warpPerspective(ed, M, (Wc, Hc), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    dw = cv2.GaussianBlur(cv2.warpPerspective(d, M, (Wc, Hc), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP), (0, 0), 3.0)[..., None]
    st["E"] = (st["E"].astype(np.float32) * (1 - dw) + edw.astype(np.float32) * dw).astype(np.uint8)
    return float(d.mean())


def repaint_plate(job_obj, job, spec, k, win, Lr, W, H, tag, view_tag=None, mosaic=False):
    """new-place plate re-painted from the shot itself (place_repaint): by default the shot's FIRST FRAME is re-painted (its camera,
    layout and the way the path runs stay), then extended only where the source camera goes (its path mapped back to frame 0) and moved
    with the full camera path - no zoom, no slowed pans, frame 0 of the plate IS the re-painted frame - the source subjects laid on top.
    mosaic=True re-paints a stitch of everything the camera sees instead (catches a ledge that appears mid-shot, but Qwen-Image loses the
    exact layout on a wide curved strip - M9 test 2026-10-23). -> plate path (raises when no re-painted view passes the check)"""
    st = repaint_set(job_obj, job, spec, k, win, Lr, W, H, tag, view_tag, mosaic); jd = job_obj.store.dir(job_obj.id)
    out = set_frames(st, W, H)
    dst = os.path.join(jd, f"{tag}_plate.mp4"); write_video(out, dst, crf=12); cv2.imwrite(os.path.join(jd, f"{tag}_plate_first.png"), out[0])
    job_obj.log(f"{tag}: new place re-painted from the shot's own {'background' if mosaic else 'first frame'} under '{st['words']}' "
                f"(canvas {st['E'].shape[1]}x{st['E'].shape[0]}, frame 0 at {st['L_']},{st['T_']})")
    return dst


def place_view(job_obj, job, spec, k, first_frame, W, H, tag):
    """the new place drawn from THIS shot's camera, wide enough for its camera move (Qwen-Image: a blank canvas of the wanted shape as
    Picture 1, the place picture as Picture 2), two candidates, Qwen3.8 keeps the one whose camera and free path match the source shot
    (2026-10-23: the raw place picture made cars and jet skis float and sent a skater into the Great Wall's parapet). -> path or None"""
    jd = job_obj.store.dir(job_obj.id); dst = os.path.join(jd, f"{tag}_view.png")
    if os.path.exists(dst):
        return dst
    loc = next((s_ for s_ in spec.get("subjects", []) if s_.get("kind") == "location" and isinstance(s_.get("picture"), int)
                and 1 <= s_["picture"] <= len(job["images"])), None)
    if not loc:
        return None
    seg = next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), {})
    camera = str(seg.get("camera") or "the same camera height and angle as the video").strip()
    spot = str(seg.get("spot") or "").strip(); travel = str(seg.get("travel") or "none").strip()
    aspect = (W / H) * (1.6 if W >= H else 1.25)                    # wider than the frame: room for the camera move
    cw = 1536; chh = int(round(cw / aspect / 16) * 16)
    canvas = os.path.join(jd, f"{tag}_canvas.png"); cv2.imwrite(canvas, np.full((chh, cw, 3), 128, np.uint8))
    path_txt = {"left-to-right": "The path, road, walkway or water runs left to right across the picture.",
                "right-to-left": "The path, road, walkway or water runs right to left across the picture.",
                "toward-camera": "The path, road, walkway or water runs straight toward the camera.",
                "away-from-camera": "The path, road, walkway or water runs straight away from the camera."}.get(travel.lower(), "")
    prompt = (f"Picture 1 is a blank canvas. Fill it completely with a photograph of the place shown in Picture 2, seen from this camera: {camera}. "
              + (f"{spot}. " if spot and spot.lower() not in ("none", "empty") else "") + path_txt +
              " The place is empty: no people, no animals, no vehicles, nothing moving. Keep the exact look of Picture 2: its buildings, ground, "
              "materials, colours, light and time of day. Wide angle, sharp, photorealistic.")
    place = job["images"][loc["picture"] - 1]; cands = []
    for i, seed in enumerate((31, 47)):
        try:
            c = qwen_edit(job_obj, [canvas, place], prompt, seed, f"{tag}_view_c{i + 1}"); cands.append(c)
        except comfy.Cancelled:
            raise
        except Exception as e:
            job_obj.log(f"{tag}: place view {i + 1} failed: {str(e)[:160]}")
    if not cands:
        return None
    try:
        fp = os.path.join(jd, f"{tag}_view_src.png"); cv2.imwrite(fp, first_frame)
        c = [{"type": "text", "text": "SOURCE:"}, llm.img_part(fp, 640), {"type": "text", "text": "PLACE:"}, llm.img_part(place, 640)]
        for i, cp in enumerate(cands, 1):
            c += [{"type": "text", "text": f"Candidate {i}:"}, llm.img_part(cp, 768)]
        r = llm.ask(c, system=VIEW_PICK.format(travel=travel), max_tokens=400, temperature=0.0)
        rated = r.get("candidates") or []
        ok = [(float(x.get("same_place", 0)) + float(x.get("camera_match", 0)), i) for i, x in enumerate(rated[:len(cands)])
              if x.get("open_path") and x.get("empty", True) and float(x.get("same_place", 0)) >= 6 and float(x.get("camera_match", 0)) >= 5]
        job_obj.log(f"{tag}: place views {json.dumps(rated)[:300]} ({str(r.get('why', ''))[:160]})")
        if not ok:
            return None
        shutil.copy(cands[max(ok)[1]], dst)
    except Exception as e:
        job_obj.log(f"{tag}: place view check failed ({str(e)[:120]}); using view 1"); shutil.copy(cands[0], dst)
    return dst


def fit_path(A, W, H, room_x, room_y):
    """center a camera path (frame-0 -> frame-i similarity transforms) inside a plate with room_x / room_y of spare size on each side:
    -> (x0, y0) of the view's top-left in the plate at frame 0, and the path (scaled down only if its move is wider than the room)"""
    c0 = np.array([W / 2.0, H / 2.0, 1.0]); parts = []
    for T in A:
        d = (T @ c0)[:2] - c0[:2]; sc = float(np.sqrt(abs(np.linalg.det(T[:2, :2])))) or 1.0
        parts.append((d, float(np.arctan2(T[1, 0], T[0, 0])), sc))
    xs = [p_[0][0] for p_ in parts]; ys = [p_[0][1] for p_ in parts]
    mx, my = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    f = min(1.0, 0.95 * 2 * room_x / max(max(xs) - min(xs), 1e-3), 0.95 * 2 * room_y / max(max(ys) - min(ys), 1e-3))
    out = []
    for d, ang, sc in parts:
        dd = np.array([mx + f * (d[0] - mx), my + f * (d[1] - my)])
        a2, s2 = ang * f, float(np.clip(sc ** f, 0.8, 1.6)); cs, sn = np.cos(a2) * s2, np.sin(a2) * s2
        out.append(np.array([[1, 0, c0[0] + dd[0]], [0, 1, c0[1] + dd[1]], [0, 0, 1]]) @ np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
                   @ np.array([[1, 0, -c0[0]], [0, 1, -c0[1]], [0, 0, 1]]))
    return (room_x + mx, room_y + my), out, f


def subject_words(spec):
    """SAM 3 words for everything that moves with the subjects (not the place): the people / animals / objects being replaced, plus what
    they ride or hold (a skater's board is part of the skater)"""
    words = "|".join(sorted({"person" if k_ == "person" else "animal" if k_ == "animal" else re.sub(r"\s*\b(in|of|from)\s+<Video \d+>|<[^>]+>", "", (x.get("replaces") or "object").split(",")[0]).strip()[:40] or "object"
                             for x in spec.get("subjects", []) for k_ in [x.get("kind", "person")] if k_ != "location"})) or "person"
    gear = re.findall(r"\b(jet ?ski|quad bike|atv|golf cart|kayak|canoe|rowboat|speedboat|pedal boat|raft|paddleboard|motorcycle|motorbike|vespa|"
                      r"snowmobile|dune buggy|go-?karts?|tuk-?tuk|rickshaw|jeep|paraglider|toboggan|sled|tandem bicycle|"
                      r"skateboard|surfboard|snowboard|bicycle|bike|scooter|board|skis?|kite|ball|rope|horse|wheelchair|frisbee|staff|guitar|drums?|"
                      r"drum kit|microphone|umbrella|leash|rollerblades)\b",
                      " ".join(str(x.get("description") or "") + " " + str(x.get("replaces") or "") for x in spec.get("subjects", [])) + " " +
                      " ".join(str(g.get("shots") or "") for g in spec.get("segments", [])), re.I)
    for g_ in sorted({g.lower() for g in gear}):   # what the subject rides or holds is part of the subject: cut it out with them
        if g_ not in words:
            words += "|" + g_
    return words


def plate_window(job_obj, job, spec, win, Lr, W, H, tag, view=None):
    """motion transfer to a new place, fallback when the old place keeps coming back: the shot's source window with its old place REPLACED
    by the new place's picture - the subject cut out by SAM 3 and laid over the picture, which is moved with the source camera (pans, tilts,
    zooms) - so the swap LoRA, which keeps the source's background, keeps the new place. A close-up gets a soft crop of the place's lower
    middle (its floor) instead of the whole view. Returns the path of the new window (same length as `win`)."""
    self = job_obj; jd = self.store.dir(self.id)
    loc = next((s_ for s_ in spec.get("subjects", []) if s_.get("kind") == "location" and isinstance(s_.get("picture"), int)
                and 1 <= s_["picture"] <= len(job["images"])), None)
    if not loc:
        raise RuntimeError("no place picture")
    words = subject_words(spec)
    frames = [cover_resize(f, W, H) for f in read_frames(win)][:Lr]
    tools.gpu_take(self)
    od = os.path.join(jd, f"{tag}_plate_sam")
    tools.run_proc(self, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_matte.py", win, od, words, "matte"], 0.5, 0.5, f"Outlining {words.replace('|', ', ')}")
    M = read_frames(os.path.join(od, "matte.mp4"))
    keep = [cv2.resize(cv2.cvtColor(M[min(i, len(M) - 1)], cv2.COLOR_BGR2GRAY), (W, H)).astype(np.float32) / 255.0 if M else np.zeros((H, W), np.float32)
            for i in range(len(frames))]
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    keep = [cv2.GaussianBlur(cv2.dilate((m > 0.15).astype(np.float32), grow), (0, 0), 2.0) for m in keep]   # SAM 3 marks fast fabric (a tutu) as half-transparent: keep it solid
    cover = float(np.mean([m.mean() for m in keep]))
    A = camera_path(frames, [cv2.dilate((m > 0.05).astype(np.float32), grow) for m in keep])
    P = cv2.imread(job["images"][loc["picture"] - 1], cv2.IMREAD_COLOR)
    close = cover > 0.30
    if close:   # a close-up sees only a small, out-of-focus part of the place: its floor area
        h_, w_ = P.shape[:2]; cw, ch = int(w_ * 0.42), int(h_ * 0.42); x0 = (w_ - cw) // 2; y0 = min(h_ - ch, max(0, int(h_ * 0.62) - ch // 2))
        P = cv2.GaussianBlur(P[y0:y0 + ch, x0:x0 + cw], (0, 0), 2.5)
    if view is not None and not close:   # the place drawn from this shot's camera: keep the whole camera move inside it, unslowed
        V = cv2.imread(view, cv2.IMREAD_COLOR); BH = int(H * 1.24); BW = max(int(W * 1.24), int(V.shape[1] * BH / V.shape[0]))
        plate = cover_resize(V, BW, BH); (x0, y0), A, f = fit_path(A, W, H, (BW - W) / 2.0, (BH - H) / 2.0)
        off = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64); out = []
        for fr_, a_, T in zip(frames, keep, A):
            bg = cv2.warpPerspective(plate, T @ off, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            out.append((fr_.astype(np.float32) * a_[..., None] + bg.astype(np.float32) * (1 - a_[..., None])).astype(np.uint8))
        dst = os.path.join(jd, f"{tag}_plate.mp4"); write_video(out, dst, crf=12); cv2.imwrite(os.path.join(jd, f"{tag}_plate_first.png"), out[0])
        self.log(f"{tag}: new-place plate from this shot's view under '{words}' ({100 * cover:.0f} % subject, camera move kept at {100 * f:.0f} %)")
        return dst
    mg = 0.5; BW, BH = int(W * (1 + 2 * mg)), int(H * (1 + 2 * mg)); plate = cover_resize(P, BW, BH)
    dn = 0.0 if close else 0.24 * H   # the view starts low in the picture (its ground is usually in the lower third), so feet stand on it
    off = np.array([[1, 0, -mg * W], [0, 1, -(mg * H + dn)], [0, 0, 1]], np.float64)
    A = damp_path(A, W, H, mg, mg * H - dn)   # a pan wider than the picture: the far place moves less (like parallax), view stays inside
    out = []
    for f, a_, T in zip(frames, keep, A):
        bg = cv2.warpPerspective(plate, T @ off, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out.append((f.astype(np.float32) * a_[..., None] + bg.astype(np.float32) * (1 - a_[..., None])).astype(np.uint8))
    dst = os.path.join(jd, f"{tag}_plate.mp4"); write_video(out, dst, crf=12)
    cv2.imwrite(os.path.join(jd, f"{tag}_plate_first.png"), out[0])
    self.log(f"{tag}: new-place plate under '{words}' ({100 * cover:.0f} % subject{', close-up crop' if close else ''}, camera moved "
             f"{float(np.hypot(A[-1][0, 2], A[-1][1, 2])):.0f} px)")
    return dst


HAIR_LEAK = re.compile(r"\b(leftover|left-over|retains?|retaining|keeps?|kept|still has|original'?s?)\b[^.;]{0,80}\b(hair|braids?|ponytail)\b|"
                       r"\blong\b[^.;]{0,30}\b(hair|braids?|ponytail)\b[^.;]{0,60}\b(instead|leftover|original|should|not)\b", re.I)


def hair_off(spec):
    """the source's long hair erased before rendering - only when no replacement is an ordinary person (2026-10-29, test-files case 2: the spec
    flag alone turned it on for a short-haired PERSON; the hair was painted out as a flat column that smeared the captions, and H3 drew the
    reference sheet into it as a picture-in-picture). Helmeted / robot / masked figures still count."""
    v = spec.get("hair_off")
    if not (v is True or str(v).strip().lower() == "true"):
        return False
    subs = [s for s in spec.get("subjects", []) if s.get("kind") != "location"]
    plain = [s for s in subs if s.get("kind") == "person" and not re.search(r"\b(helmet|helmeted|robot|android|cyborg|astronaut|mascot|mask|armou?r)\b",
                                                                         str(s.get("description") or ""), re.I)]
    return bool(subs) and not plain


def strip_hair(job_obj, win, tag):
    """every person is replaced by an animal / robot / helmeted figure and someone has long hair or braids: erase the hair from the SOURCE
    window (SAM 3 'hair' outlines, grown a little, filled from the surroundings) so H3 has no flying braids to copy (C3). -> new window path"""
    jd = job_obj.store.dir(job_obj.id); dst = os.path.join(jd, f"{tag}_nohair.mp4")
    if os.path.exists(dst):
        return dst
    od = os.path.join(jd, f"{tag}_hair_sam"); tools.gpu_take(job_obj)
    tools.run_proc(job_obj, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_matte.py", win, od, "hair", "matte"], 0.5, 0.5, "Outlining hair")
    M = read_frames(os.path.join(od, "matte.mp4")); fr = read_frames(win)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)); out = []
    for i, f in enumerate(fr):
        m = cv2.cvtColor(M[min(i, len(M) - 1)], cv2.COLOR_BGR2GRAY) if M else np.zeros(f.shape[:2], np.uint8)
        m = cv2.dilate((cv2.resize(m, (f.shape[1], f.shape[0])) > 60).astype(np.uint8) * 255, ker)
        out.append(cv2.inpaint(f, m, 9, cv2.INPAINT_TELEA) if m.any() else f)
    write_video(out, dst, crf=12); job_obj.log(f"{tag}: erased the hair of the people being replaced (long hair would leak into the new subjects)")
    return dst


def closeup_refs(job_obj, job, spec, k, images):
    """a segment that is a close-up of just part of a person or animal ('feet and lower legs'): the reference pictures of the people and
    animals are cropped to those parts (SAM 3 boxes, one crop per sheet panel), so H3 has no full figure to paste in as a small extra copy
    (B1's miniature couple). Cached per picture and part. -> the images list with the crops swapped in"""
    seg = next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), {})
    part = str(seg.get("closeup") or "").strip()
    if not part or part.lower() in ("none", "null", "false", "no"):
        return images
    words = "|".join(w.strip() for w in re.split(r",| and |/|;", part) if w.strip())[:120] or part
    jd = job_obj.store.dir(job_obj.id); out = list(images); slug = re.sub(r"[^a-z]+", "-", part.lower()).strip("-")[:30]
    for sub in spec.get("subjects", []):
        p = sub.get("picture")
        if sub.get("kind") not in ("person", "animal") or not isinstance(p, int) or not 1 <= p <= len(images):
            continue
        dst = os.path.join(jd, f"ref{p}_closeup_{slug}.png")
        if not os.path.exists(dst):
            try:
                img = cv2.imread(images[p - 1], cv2.IMREAD_COLOR); one = os.path.join(jd, f"ref{p}_still.mp4"); write_video([img] * 3, one, crf=10)
                od = os.path.join(jd, f"ref{p}_{slug}_sam"); tools.gpu_take(job_obj)
                tools.run_proc(job_obj, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_matte.py", one, od, words, "boxes"], 0.5, 0.5, f"Finding the {part} in picture {p}")
                B = (json.load(open(os.path.join(od, "boxes.json"))).get("boxes") or [[]])[0]
            except comfy.Cancelled:
                raise
            except Exception as e:
                job_obj.log(f"close-up crop of picture {p} failed: {str(e)[:160]}"); continue
            if not B:
                job_obj.log(f"close-up: no {part} found in picture {p} - using the whole picture"); continue
            H_, W_ = img.shape[:2]; groups = []
            for b in sorted(B, key=lambda b: b[0]):          # one group per sheet panel: boxes that overlap left-right
                if groups and b[0] < groups[-1][2]:
                    g = groups[-1]; groups[-1] = [min(g[0], b[0]), min(g[1], b[1]), max(g[2], b[2]), max(g[3], b[3])]
                else:
                    groups.append(list(b))
            crops = []
            for x0, y0, x1, y1 in groups[:3]:
                pw, ph = 0.12 * (x1 - x0) + 8, 0.12 * (y1 - y0) + 8
                X0, Y0, X1, Y1 = max(0, int(x0 - pw)), max(0, int(y0 - ph)), min(W_, int(x1 + pw)), min(H_, int(y1 + ph))
                if X1 - X0 > 16 and Y1 - Y0 > 16:
                    crops.append(img[Y0:Y1, X0:X1])
            if not crops:
                continue
            hh = max(c.shape[0] for c in crops)
            cv2.imwrite(dst, np.concatenate([np.pad(cv2.resize(c, (max(1, int(c.shape[1] * hh / c.shape[0])), hh)), ((0, 0), (0, 12), (0, 0)), constant_values=235)
                                             for c in crops], 1))
            job_obj.log(f"segment {k}: close-up of the {part} - picture {p} cropped to those parts ({len(crops)} view(s))")
        out[p - 1] = dst
    return out


def keep_flashes(frames, src_frames):
    """the source's white flash frames (a cut's flash: test-files cases 7, 8, 9) copied into the render as they are - H3 leaves them out and the
    check then fails the shot ('the white flash transition present in the source is missing'); a flash shows nothing to swap. Returns the same
    list object when there is no flash."""
    hit = []
    for i in range(min(len(frames), len(src_frames))):
        g = cv2.cvtColor(src_frames[i], cv2.COLOR_BGR2GRAY)
        if float(g.mean()) > 225 and float(g.std()) < 25 and float(np.percentile(g, 2)) > 160:
            hit.append(i)
    if not hit:
        return frames
    out = list(frames)
    for i in hit:
        s_ = src_frames[i]; out[i] = s_ if s_.shape == out[i].shape else cv2.resize(s_, (out[i].shape[1], out[i].shape[0]))
    return out


def take_rank(score, audit):
    """a take's rank when the best try is kept: the check score, then how bad its issues are (2026-10-29, test-files case 4: both takes scored 4 -
    the first left the source man unswapped (five critical who-is-who issues), the retake swapped both people and had one 'hands' note - and the
    tie kept the first). critical = 3 (2 for a physics note), major = 1, other = 0.3"""
    w = 0.0
    for x in (audit or {}).get("issues") or []:
        if isinstance(x, dict):
            sev = x.get("severity")
            w += (2.0 if x.get("check") == "physics" else 3.0) if sev == "critical" else (1.0 if sev == "major" else 0.3)
    return (float(score or 0), -w)


def repair_segment(job_obj, job, spec, sg, k, n, src, F, win, W, H, best, prev, seed0):
    """re-render only the flagged shots (each as its own short clip starting at its cut, pinned to a composited first frame), splice them in
    at the cuts, re-audit; keep the result only if the score improves. Batched per round so each model loads once: LLM (who is visible,
    shot descriptions) -> Qwen-Image (composites) -> LLM (ratings) -> H3 (all clips) -> LLM (audit). Returns the new best (score, frames, rec)."""
    self = job_obj; L = sg["length"]; jd = self.store.dir(self.id)
    cuts = detect_cuts(win); cuts = sorted(set(cuts) | set(detect_soft_cuts(win, cuts)))
    subs = [s for s in spec.get("subjects", []) if isinstance(s.get("picture"), int) and 1 <= s["picture"] <= len(job["images"])]
    if "end" in sg:   # a shot-aligned segment: only this shot's frames, padded with its last frame (never the next shot)
        S = read_frames(src, range(sg["start"], min(F, sg["end"]))); S = S + [S[-1]] * (L + 140 - len(S))
    else:
        S = read_frames(src, range(sg["start"], min(F, sg["start"] + L + 140)))      # the source around this segment, read once
    def sfr(i):
        return S[min(i, len(S) - 1)]
    seg = next((s for s in spec.get("segments", []) if int(s.get("index", -1)) == k), {})
    work = best   # what the next round builds on: round 1's output carries into round 2 even when its score did not rise (the masked round
    for rnd in range(1, REPAIR_ROUNDS + 1):   # fixes the people that round 1's re-render left, on round 1's corrected shots)
        score, frames, rec = work; a = rec.get("audit") or {}
        if a.get("pass") or "error" in a:
            break
        bounds, bad = flagged_shots(a, cuts, L)
        if not bad:
            break
        if not localized(a, cuts, L):
            tries = [x for x in next((s_ for s_ in (self.d.get("segments") or []) if s_.get("k") == k), {}).get("attempts", [])
                     if isinstance(x.get("audit"), dict) and x["audit"].get("score") is not None]
            sc = [float(x["audit"]["score"]) for x in tries]
            if len(sc) >= 2 and sc[-1] <= max(sc[:-1]):   # the last whole-segment try did not beat the best: another one repeats the same
                self.log(f"segment {k}: the last re-render did not improve it - keeping the best try ({max(sc):g})")   # cause (it won 0 of
                break                                                                                              # 10 times; saves ~8 min)
            self.log(f"segment {k}: the problems cover most of the segment - re-rendering it with corrections"); todo = [(-1, 0, L)]
        else:
            todo = [(si, bounds[si][0], bounds[si][1]) for si in bad]
        plan = []
        for si, s0, s1 in todo:
            if si >= 0 and s1 - s0 > 8:   # a shot that opens blurred (hand over the lens, whip pan): repair from its first clear frame
                shp = [float(cv2.Laplacian(cv2.cvtColor(cv2.resize(sfr(i), (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
                       for i in range(s0, s1)]
                medp = float(np.median(shp)); q0 = next((i for i, v in enumerate(shp[:12]) if v >= 0.8 * medp), 0)
                if 0 < q0 < (s1 - s0) - 4:
                    self.log(f"seg{k} shot {si + 1}: opens blurred - repairing from its first clear frame (+{q0})"); s0 += q0
            Lr = next(17 * j + 5 for j in range(1, 8) if 17 * j + 5 >= max(22, s1 - s0)); tag = f"seg{k}_r{rnd}_" + (f"s{si}" if si >= 0 else "all")
            clip = [sfr(s0 + i) for i in range(Lr)]; cwin = os.path.join(jd, f"{tag}_src.mp4"); write_video(clip, cwin, crf=12)
            if job.get("mode") == "motion" and (place_reverted(a) or float(a.get("score", 10)) <= 4) and any(s_.get("kind") == "location" for s_ in subs):
                try:   # the old place keeps coming back: re-render from a window that already shows the new place behind the subject
                    pw = None
                    if REPAINT_PLACE:
                        try:
                            pw = repaint_plate(self, job, spec, k, cwin, Lr, W, H, tag, view_tag=f"seg{k}" if s0 == 0 else None)
                        except comfy.Cancelled:
                            raise
                        except Exception as e:
                            self.log(f"{tag}: re-painted place failed ({str(e)[:160]}) - using the place picture's view")
                    if pw is None:
                        try:
                            view = place_view(self, job, spec, k, cover_resize(clip[0], W, H), W, H, f"seg{k}")
                        except comfy.Cancelled:
                            raise
                        except Exception as e:
                            self.log(f"{tag}: place view failed: {str(e)[:160]}"); view = None
                        pw = plate_window(self, job, spec, cwin, Lr, W, H, tag, view=view)
                    cwin = pw; clip = [cover_resize(f, W, H) for f in read_frames(cwin)]
                except comfy.Cancelled:
                    raise
                except Exception as e:
                    self.log(f"{tag}: new-place plate failed: {str(e)[:200]}")
            inside = lambda t: t is not None and s0 <= int(round(float(t) * FPS)) < s1
            probs = [x.get("fix") or x.get("what", "") for x in a.get("issues", []) if isinstance(x, dict) and inside(x.get("time"))]
            probs += [p.get("problem", "") for p in a.get("pairs", []) if isinstance(p, dict) and not p.get("ok", True) and inside(p.get("t"))]
            if si == -1:
                probs = [x.get("fix") or x.get("what", "") for x in a.get("issues", []) if isinstance(x, dict)] + [a.get("fix", "")]
            g = []
            if s0 == 0 and prev is not None and sg.get("cont", "end" not in sg):
                gp = os.path.join(jd, f"{tag}_guide.mp4"); write_video(prev[-5:], gp, crf=10); g = [{"video": gp, "frame": 0}]
            elif s0 == 0 and job.get("first_frame"):
                g = [{"image": job["first_frame"], "frame": 0}]
            plan.append(dict(si=si, s0=s0, s1=s1, Lr=Lr, tag=tag, clip=clip, cwin=cwin, probs=[p for p in probs if p], guides=g, use=None, text="", fix="",
                             cands=[], lines=[], costume=set(), looks={},
                             mode="inpaint" if (rnd >= 2 and si >= 0 and any(x.get("kind") != "location" for x in subs)) else "pin"))
        # phase 1 - LLM: who is visible (+ costume) in each shot's first frame, and a fresh description of each shot
        self.set(stage=f"Planning the repair of {len(plan)} shot(s) with {llm.NAME} (round {rnd})")
        for p in plan:
            p["fix"] = positive_fix(spec, list(dict.fromkeys(p["probs"])))   # H3 draws every noun of a note: what the frame must show, in the new place
            if p["si"] >= 0:
                try:
                    p["text"] = describe_shot(job, spec, [p["clip"][i] for i in sorted({0, (p["s1"] - p["s0"]) // 2, max(0, p["s1"] - p["s0"] - 1)})])
                except Exception:
                    p["text"] = ""
                p["text"] = p["text"] or shot_text(spec, k, p["s0"], p["s1"])
            if p["guides"] or not subs:
                continue
            sh = [float(cv2.Laplacian(cv2.cvtColor(cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
                  for f in p["clip"][:max(1, p["s1"] - p["s0"])]]
            med = float(np.median(sh)); q = next((i for i, v in enumerate(sh[:12]) if v >= 0.8 * med), 0)   # first clear frame of the shot
            fp = os.path.join(jd, f"{p['tag']}_first.png"); cv2.imwrite(fp, cover_resize(p["clip"][q], W, H)); p["first"] = fp
            if q:
                self.log(f"{p['tag']}: the shot opens blurred - composited its first clear frame (+{q})")
            try:
                r = llm.ask([{"type": "text", "text": "Which of these can be seen in the frame, even partly? " + json.dumps(
                    [{"tag": s["tag"], "what": s.get("replaces") or s.get("description")} for s in subs if s.get("kind") != "location"], ensure_ascii=False)},
                    llm.img_part(fp, 640)] + ([{"type": "text", "text": "Context (the source right after this moment):"}] +
                                              [llm.img_part(f, 384) for f in job.get("_context_frames", [])[:2]] if job.get("_context_frames") and p["s1"] >= L else []),
                    system='Reply with JSON only: {"visible": [the tags of the listed things that can be seen in the frame], "costume": {tag: true if that '
                           'person wears a clearly different outfit here than the one described (a costume change)}, "looks": {tag: how that ORIGINAL person or '
                           'thing looks in THIS frame - clothes, hair, pose, where in the frame, seen from which side}}', max_tokens=400, temperature=0.0)
                tnum = lambda t: (re.findall(r"\d+", str(t)) or ["?"])[0]          # "<Subject 1>", "Subject 1", "1" -> "1"
                vis = {tnum(t) for t in (r.get("visible") or [])}
                p["costume"] = {f"<Subject {tnum(k2)}>" for k2, v2 in (r.get("costume") or {}).items() if v2}
                p["looks"] = {f"<Subject {tnum(k2)}>": str(v2)[:240] for k2, v2 in (r.get("looks") or {}).items() if v2}
            except Exception:
                tnum = lambda t: (re.findall(r"\d+", str(t)) or ["?"])[0]; vis = {tnum(s["tag"]) for s in subs}
            p["use"] = [s for s in subs if tnum(s["tag"]) in vis or s.get("kind") == "location"]
            self.log(f"{p['tag']}: visible {sorted(vis)}, costume change {sorted(p['costume'])}")
            pics = [fp]; lines = []
            for s in p["use"]:
                pics.append(job["images"][s["picture"] - 1]); m = len(pics)
                if s.get("kind") == "location" and "_plate" in p["cwin"]:
                    lines.append(PLATE_PLACE.format(n=m))
                elif s.get("kind") == "location":
                    lines.append(f"Replace the surrounding place with the place from Picture {m}, as Picture 1's camera would see it there: the same camera angle, "
                             f"distance and framing - in a close-up only the nearby floor or wall of that place and an out-of-focus background in its colours, "
                             f"never the whole view of Picture {m}.")
                elif s["tag"] in p["costume"]:
                    lines.append(f"Replace the person wearing that outfit with the {s.get('kind', 'person')} from Picture {m} - the same face, hair, skin tone and "
                                 f"body as Picture {m}, but keep the clothes the person in Picture 1 wears.")
                else:
                    lines.append(f"Replace {s.get('replaces') or 'the ' + s.get('kind', 'person')} - every copy of them in the frame, including reflections, "
                                 f"mirror images and double-exposed copies - with the {s.get('kind', 'person')} from Picture {m}: the same face, hair, skin tone, "
                                 f"body and clothes as Picture {m}." + (f" If only part of the original is visible (a close-up), turn exactly that part into the matching part of Picture {m}'s object, at "
                                                       f"the original's size - never add a small whole copy of it." if s.get("kind") == "object" else ""))
            p["pics"] = pics; p["lines"] = lines
        # phase 2 - Qwen-Image: two composite candidates per shot that needs a pinned first frame
        for p in plan:
            if p["guides"] or not p.get("lines") or p["mode"] == "inpaint":
                continue
            if self.cancelled():
                raise comfy.Cancelled()
            self.set(stage=f"Placing the characters in shot {p['si'] + 1}'s first frame (round {rnd})")
            prompt = ("Edit Picture 1, a frame from a video. " + " ".join(p["lines"]) + " " + keep_line(seg.get("kept"), seg.get("shots", "")) + " Keep everything else in "
                      "Picture 1 exactly the same: the framing, camera angle, poses, hand positions, held objects, on-screen text, lighting and colours.")
            for i, seed in enumerate((101, 202)):
                try:
                    c = qwen_edit(self, p["pics"], prompt, seed, f"{p['tag']}_c{i + 1}")
                    cv2.imwrite(c, cover_resize(cv2.imread(c, cv2.IMREAD_COLOR), W, H)); p["cands"].append(c)
                except comfy.Cancelled:
                    raise
                except Exception as e:
                    self.log(f"{p['tag']}: composite {i + 1} failed: {e}")
        # phase 3 - LLM: rate the candidates, pin the best
        for p in plan:
            if not p["cands"]:
                continue
            try:
                c = [{"type": "text", "text": "A (original frame):"}, llm.img_part(p["first"], 640)]
                for i, pic in enumerate(p["pics"][1:], 2):
                    c += [{"type": "text", "text": f"Picture {i}:"}, llm.img_part(pic, 448)]
                for i, cp in enumerate(p["cands"], 1):
                    c += [{"type": "text", "text": f"Candidate {i}:"}, llm.img_part(cp, 640)]
                r = llm.ask(c, system=COMPOSE_PICK.format(edit=" ".join(p["lines"])), max_tokens=400, temperature=0.0)
                rated = r.get("candidates") or []
                place = any("Replace the surrounding place" in ln for ln in p["lines"])
                ok = [(float(x.get("identity", 0)) + float(x.get("same_as_A", 0)), i) for i, x in enumerate(rated[:len(p["cands"])])
                      if (x.get("edit_ok") or (float(x.get("identity", 0)) >= 8 and not place)) and not x.get("artifacts")
                      and float(x.get("identity", 0)) >= 6 and float(x.get("same_as_A", 0)) >= 6]
                self.log(f"{p['tag']}: composite ratings {json.dumps(rated)[:300]} ({str(r.get('why') or '')[:200]})")
                if ok:
                    p["guides"] = [{"image": p["cands"][max(ok)[1]], "frame": 0}]
            except Exception as e:
                self.log(f"{p['tag']}: composite check failed ({e}); using candidate 1"); p["guides"] = [{"image": p["cands"][0], "frame": 0}]
        # phase 4 - H3: render every clip back to back
        new = [f.copy() for f in frames]; fixed = []
        for p in plan:
            if self.cancelled():
                raise comfy.Cancelled()
            s0, s1, Lr, si = p["s0"], p["s1"], p["Lr"], p["si"]
            self.set(stage=f"Re-rendering {'the segment' if si < 0 else f'shot {si + 1} ({s0 / FPS:.1f}-{s1 / FPS:.1f} s)'}, round {rnd}")
            spec2 = copy.deepcopy(spec)
            if si < 0:
                spec2["segments"] = [{"index": 0, "shots": seg.get("shots", ""), "present": seg.get("present"), "kept": seg.get("kept")}]
            else:
                present = [s["tag"] for s in p["use"]] if p["use"] else seg.get("present")
                spec2["segments"] = [{"index": 0, "shots": f"[Shot 1] (0.0 to {(s1 - s0) / FPS:.2f} seconds) {p['text']} [Shot 2] ({(s1 - s0) / FPS:.2f} to "
                                                            f"{Lr / FPS:.1f} seconds) whatever follows in <Video 1>.", "present": present}]
            spec2["_corrections"] = {"0": p["fix"] or (("Fix these problems seen in an earlier attempt: " + "; ".join(p["probs"])[:600]) if p["probs"] else "")}
            if si >= 0:   # one shot: the clip-wide 'kept' text can name another shot's place - keep THIS shot's surroundings instead
                spec2["kept"] = "the place, background, light and everything else in this shot of <Video 1>, exactly as they are"
                spec2["retention"] = []          # the clip-wide lines name other shots' outfits and places
                spec2["summary"] = "Re-render this one shot of <Video 1> with the subjects exactly as defined above, keeping every movement, the camera and the timing."
                for sub in spec2.get("subjects", []):
                    if sub.get("tag") in p.get("looks", {}):   # whom to replace, as they look in THIS shot
                        sub["replaces"] = p["looks"][sub["tag"]]; sub["looks_here"] = p["looks"][sub["tag"]]
                    if sub.get("tag") in p["costume"]:
                        sub["description"] = (sub.get("description") or "") + " In this shot the person wears the costume seen in <Video 1> instead of the picture's clothes."
                spec2["instruction"] = " ".join(f"Replace {sub.get('replaces')} in <Video 1> with the {sub.get('kind', 'person')} shown in <Picture {sub.get('picture')}>."
                                                for sub in spec2.get("subjects", []) if sub.get("tag") in (p.get("looks") or {}) and isinstance(sub.get("picture"), int)) \
                    or spec2.get("instruction")
            images2 = closeup_refs(self, job, spec, k, list(job["images"]))
            if si >= 0 and p["costume"] and p["guides"] and p["guides"][0].get("image"):
                for sub in spec2.get("subjects", []):   # the composite shows the face AND this shot's costume: it becomes that subject's picture
                    if sub.get("tag") in p["costume"] and isinstance(sub.get("picture"), int) and 1 <= sub["picture"] <= len(images2):
                        images2[sub["picture"] - 1] = p["guides"][0]["image"]; sub["costume_shot"] = True
                        sub["description"] = (sub.get("looks_here") or "") + " The face, hair, skin tone and build of the person in the picture; the costume they wear in this shot."
            prompt, pics = segment_prompt(spec2, 0, images2, job.get("style"))
            alpha = None
            if p["mode"] == "inpaint":   # regenerate only inside the source people's outlines; everything else follows the current video
                try:
                    alpha, cur_clip, mclip = inpaint_masks(self, job, spec2, p, new, L, W, H)
                except comfy.Cancelled:
                    raise
                except Exception as e:
                    self.log(f"{p['tag']}: masking failed ({str(e)[:160]}) - plain re-render"); p["mode"] = "pin"
            llm.down()
            if hair_off(spec) and not p["cwin"].endswith("_nohair.mp4") and "_plate" not in p["cwin"]:
                try:
                    p["cwin"] = strip_hair(self, p["cwin"], p["tag"])
                except comfy.Cancelled:
                    raise
                except Exception as e:
                    self.log(f"{p['tag']}: hair removal failed: {str(e)[:160]}")
            wf = h3.build(prompt, p["cwin"], pics, W, H, Lr, seed0 + 1000 * k + 97 * rnd + max(0, si), [] if alpha is not None else p["guides"])
            if alpha is not None:
                h3.add_inpaint(wf, cur_clip, mclip)
            files = comfy.run(wf, None, self.cancelled)
            mp4 = next((f for f in files if f.endswith(".mp4")), None)
            if not mp4:
                continue
            dst = os.path.join(jd, f"{p['tag']}.mp4"); shutil.copy(mp4, dst); rf = read_frames(dst)
            if len(rf) < s1 - s0:
                continue
            for i in range(s1 - s0):
                g = cv2.resize(rf[i], (new[s0 + i].shape[1], new[s0 + i].shape[0])) if rf[i].shape != new[s0 + i].shape else rf[i]
                if alpha is not None:
                    a_ = follow_alpha(new[s0 + i], g, cv2.resize(alpha[i], (g.shape[1], g.shape[0])))
                    g = (new[s0 + i].astype(np.float32) * (1 - a_) + g.astype(np.float32) * a_).astype(np.uint8)
                new[s0 + i] = g
            fixed.append(f"{s0 / FPS:.2f}-{s1 / FPS:.2f}s" + (" (masked)" if alpha is not None else " (pinned)" if p["guides"] else ""))
        if not fixed:
            break
        # phase 5 - LLM: re-audit the repaired segment
        new = keep_flashes(new, S)
        out_mp4 = os.path.join(jd, f"seg{k}_repair{rnd}.mp4"); write_video(new, out_mp4, crf=12)
        self.set(stage=f"Checking the repaired segment with {llm.NAME}")
        try:
            v = audit_dense(job, spec, win, new, cuts, seg=next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), None))
        except Exception as e:
            v = {"error": str(e)[:300], "score": 0}
        r2 = {"attempt": 10 + rnd, "repair": fixed, "seed": None, "video": out_mp4, "prompt": "(shot repair) " + ", ".join(fixed), "pictures": [], "audit": v}
        self.store.add_attempt(self.id, k, r2)
        self.log(f"segment {k}: repair round {rnd} ({', '.join(fixed)}) -> audit {v.get('score')}")
        if v.get("score", 0) > best[0]:
            best = (v.get("score", 0), new, r2)
        work = (v.get("score", 0), new, r2)
        if rnd >= 2 and best[2] is not r2:
            break
    return best


def context_frames(job, src, F, end):
    """source frames just after a clip (end = 24 fps frame index): +0.4, +0.9, +1.4 s - from the prepared source, else from the original upload"""
    out = []
    for dt in (0.4, 0.9, 1.4):
        i = end + int(dt * FPS)
        if i < F:
            out.append(read_frames(src, [i])[0])
        else:
            cap = cv2.VideoCapture(job["video"]); cap.set(cv2.CAP_PROP_POS_MSEC, i / FPS * 1000.0); ok, f = cap.read(); cap.release()
            if ok:
                out.append(f)
    return out


def _job_improve_method():
    def run_improve(self):
        """audit a finished video segment by segment and repair the flagged shots (no full re-render) - the 'Improve' action"""
        job = self.d; jdir = self.store.dir(self.id); os.makedirs(jdir, exist_ok=True); t0 = time.time()
        sid = job["from_job"]; srcj = self.store.get(sid); sdir = self.store.dir(sid)
        if not srcj or srcj.get("status") != "done" or not srcj.get("segments"):
            raise ValueError("the video to improve isn't a finished Create / Edit / Motion Control video")
        if not llm.available():
            raise ValueError(f"{llm.NAME} is not installed")
        spec = copy.deepcopy(srcj["spec"]); W, H = srcj["canvas"]; meta = probe(srcj["video"])
        jl = copy.deepcopy(srcj)                      # what the audit / composites read (pictures, prompt, video)
        root = srcj                                   # an Improve of an Improve: the source clips live with the original job
        while root.get("tool") == "improve" and root.get("from_job") and self.store.get(root["from_job"]):
            root = self.store.get(root["from_job"])
        rdir = self.store.dir(root["id"])
        src = os.path.join(rdir, "src24.mp4"); F = len(read_frames(src)); segs = srcj["segments"]; n = len(segs)
        self.set(status="running", stage="Checking the video", progress=0.02, canvas=[W, H], spec=spec, frames=srcj.get("frames"),
                 segments=[{"k": s["k"], "start": s["start"], "length": s["length"], "status": "waiting", "attempts": []} for s in segs])
        out = [None] * F; prev = None
        for sg in segs:
            k = sg["k"]; L = min(sg["length"], sg.get("end", 10 ** 9) - sg["start"])
            win = os.path.join(rdir, f"seg{k}_src_real.mp4" if "end" in sg else f"seg{k}_src.mp4"); ch = sg.get("chosen", 0)
            vid = next((a.get("video") for a in sg["attempts"] if a["attempt"] == ch), None)
            if not os.path.exists(win):   # an object swap made by the recipe before it kept the real-length windows
                win = os.path.join(jdir, os.path.basename(win)); write_video(read_frames(src, range(sg["start"], min(F, sg["start"] + L))), win, crf=12)
            if not vid or not os.path.exists(vid):   # no take kept (an object swap made by the recipe before 2026-10-26, or by a batch script): cut the shot from the result
                fr = read_frames(srcj.get("result") or "", range(sg["start"], sg["start"] + L))
                if not fr or (fr[0].shape[1], fr[0].shape[0]) != (W, H):
                    raise ValueError("this video was made outside the app's own steps (a test batch) - use Retry to make it again, then Improve that one")
                vid = os.path.join(jdir, f"seg{k}_from_result.mp4"); write_video(fr, vid, crf=12)
            frames = read_frames(vid)[:L]
            jl["_context_frames"] = context_frames(jl, src, F, sg["start"] + L)
            self.set(stage=f"Checking {'segment ' + str(k + 1) + '/' + str(n) if n > 1 else 'the video'} with {llm.NAME}", progress=0.05 + 0.9 * k / n)
            a = audit_dense(jl, spec, win, frames, seg=next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), None))
            rec = {"attempt": 0, "seed": None, "video": vid, "prompt": "(the video being improved)", "pictures": [], "audit": a}
            self.store.add_attempt(self.id, k, rec); best = (a.get("score", 0), frames, rec)
            self.log(f"segment {k}: audit {a.get('score')}" + ("" if a.get("pass") else " - repairing the flagged shots"))
            if not a.get("pass"):
                best = repair_segment(self, jl, spec, {**sg, "length": L}, k, n, src, F, win, W, H, best, prev, int(srcj.get("seed") or 1))
            frames = best[1]; self.store.seg_done(self.id, k, best[2])
            out = join_segment(out, sg, frames, F)
            prev = frames
        llm.down()
        out = [f for f in out[:srcj.get("frames") or F] if f is not None]
        self.set(stage="Finishing", progress=0.97)
        silent = os.path.join(jdir, "result_silent.mp4"); write_video(out, silent); final = os.path.join(jdir, "result.mp4")
        if srcj.get("keep_audio", True) and meta["audio"]:
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", srcj["video"], "-map", "0:v", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                                "-b:a", "192k", "-shortest", final])
            if r.returncode != 0:
                durable.copy_file(silent, final)
        else:
            durable.copy_file(silent, final)
        cv2.imwrite(os.path.join(jdir, "poster.jpg"), out[0], [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.set(status="done", stage="Done", progress=1.0, result=final, poster=os.path.join(jdir, "poster.jpg"), seconds=round(time.time() - t0))

    Job.run_improve = run_improve


_job_improve_method()


ID_SYSTEM = """You check WHO appears in each shot of an AI video edit. The subjects (each replaces someone or something of the source):
{subjects}
For each numbered shot you get one SOURCE frame and one GENERATED frame; for the last shot also context frames showing how the source continues
(use them to recognise a partly visible person - e.g. only a chin and an outfit).
For every shot give: "source" = the tags whose ORIGINAL (the 'replaces' description - same person even in another costume) is visible in the SOURCE
frame, even partly; "generated" = the tags whose PICTURE person or thing is visible in the GENERATED frame, even partly; "others" = anyone in the
GENERATED frame who is neither (an original source person left unreplaced, or someone invented) - or an empty string; "costume" = the tags whose
original wears clearly DIFFERENT clothes in this SOURCE frame than the 'replaces' description (a costume change, e.g. a kimono instead of a t-shirt);
"picture_clothes" = the tags who wear their PICTURE's clothes in the GENERATED frame.
Reply with JSON only: {{"shots": [{{"shot": n, "source": [tags], "generated": [tags], "others": "", "costume": [tags], "picture_clothes": [tags]}}]}}"""


COSTUME_CONFIRM = """You check one claim about a SOURCE video (before any edit): that in one shot a person wears a COSTUME - a THEMED outfit (traditional dress such as a kimono or hanbok, a uniform, armour, a stage or fancy-dress costume) that differs from the outfit they wear when they first appear in the video (for example a kimono later, after a t-shirt and shorts). Ordinary clothes that differ between scenes (another t-shirt, top, cardigan or jacket) are NOT a costume. The person: {who}.
You get FRAMES FROM ACROSS THE WHOLE SOURCE VIDEO in time order, then THE SHOT'S FRAME. The first of those frames where that person is visible shows their own outfit. Reply with JSON only: {{"first_seen": the number (1, 2, ...) of the first frame across the video where that person is visible, or 0, "different": true or false (in THE SHOT'S FRAME that person wears clearly different clothes than in that first frame; true only for a themed costume; ordinary other clothes, taking off a cap or glasses, opening a jacket, another camera angle or other light is NOT different)}}"""


def costume_confirm(job, frame, who, n=8):
    """True when the person really wears other clothes in this shot than elsewhere in the source (os4: the airport couple in kimonos after the
    transition) - the identity check compared the shot with the 'replaces' TEXT, so a short one ('the guy by the motorcycle') made any outfit a
    'costume' (test-files case 1: the repair would have dressed the new man in the old one's cap and jersey). Unsure -> True (as before)."""
    try:
        cap = cv2.VideoCapture(job["video"]); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0; nf = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        dur = min(nf / fps, MAX_SECONDS if job.get("duration") == "full" else 5.2); fr = []
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_MSEC, (i + 0.5) * dur / n * 1000.0); ok, f = cap.read()
            if ok:
                fr.append(f)
        cap.release()
        if len(fr) < 2:
            return True
        content = [{"type": "text", "text": "FRAMES FROM ACROSS THE WHOLE SOURCE VIDEO (in time order):"}] + \
                  [x for i, f in enumerate(fr, 1) for x in ({"type": "text", "text": f"{i}:"}, llm.img_part(f, 384))] + \
                  [{"type": "text", "text": "THE SHOT'S FRAME:"}, llm.img_part(frame, 640)]
        r = llm.ask(content, system=COSTUME_CONFIRM.format(who=str(who)[:200]), max_tokens=150, temperature=0.0)
    except Exception:
        return True
    if not isinstance(r, dict):
        return True
    return bool(r.get("different")) and str(r.get("first_seen", "1")).strip() not in ("0", "")


def identity_check(job, spec, src, out_frames, cuts):
    """per shot: are the same subjects present in the source and the generated frame? -> list of critical issues"""
    subs = [s for s in spec.get("subjects", []) if s.get("kind") in ("person", "animal") and isinstance(s.get("picture"), int) and 1 <= s["picture"] <= len(job["images"])]
    if not subs:   # people and animals only: the who-is-who check misses small held objects (a phone in a hand) - objects are covered by the pair audit
        return []
    L = len(out_frames); bounds = shot_bounds(cuts, L)
    shots = [(s0, s1, min(L - 1, (s0 + s1) // 2 if s1 - s0 > 4 else s0 + (s1 - s0) // 2)) for s0, s1 in bounds]
    content = [{"type": "text", "text": "Pictures:"}]
    for s in subs:
        content += [{"type": "text", "text": f"{s['tag']} = <Picture {s['picture']}>:"}, llm.img_part(job["images"][s["picture"] - 1], 448)]
    for n, (s0, s1, m) in enumerate(shots, 1):
        content += [{"type": "text", "text": f"Shot {n} ({s0 / FPS:.2f}-{s1 / FPS:.2f} s) SOURCE:"}, llm.img_part(src[min(m, len(src) - 1)], 448),
                    {"type": "text", "text": f"Shot {n} GENERATED:"}, llm.img_part(out_frames[m], 448)]
    if job.get("_context_frames"):
        content.append({"type": "text", "text": "Context after the last shot (the source continues like this):"})
        content += [llm.img_part(f, 384) for f in job["_context_frames"]]
    desc = "\n".join(f"{s['tag']}: replaces {s.get('replaces') or '?'}; looks like <Picture {s['picture']}>" for s in subs)
    r = llm.ask(content, system=ID_SYSTEM.format(subjects=desc), max_tokens=900, temperature=0.0)
    issues = []
    for row in r.get("shots", []) or []:
        try:
            n = int(row.get("shot")); s0, s1, m = shots[n - 1]
        except (TypeError, ValueError, IndexError):
            continue
        tn = lambda xs: {f"<Subject {(re.findall(r'\d+', str(x)) or ['?'])[0]}>" for x in (xs or [])}
        a, b = tn(row.get("source")), tn(row.get("generated")); oth = (row.get("others") or "").strip()
        if a != b or oth:
            what = f"shot {n}: the source shows {sorted(a) or 'none of the subjects'} but the edit shows {sorted(b) or 'none of them'}" + (f"; also {oth}" if oth else "")
            issues.append({"severity": "critical", "time": round(m / FPS, 2), "what": what})
        wrong = tn(row.get("costume")) & tn(row.get("picture_clothes")) & b
        if wrong:   # confirmed against the whole source (2026-10-29: a short 'replaces' - 'the guy by the motorcycle' - made his everyday cap a 'costume')
            wrong = {t_ for t_ in wrong if costume_confirm(job, src[min(m, len(src) - 1)], next((s.get("replaces") for s in subs if s["tag"] == t_), None) or t_)}
        if wrong:
            issues.append({"severity": "major", "time": round(m / FPS, 2),
                           "what": f"shot {n}: {sorted(wrong)} should keep the costume worn in the source in this shot but wear their picture's clothes"})
    return issues


# ---------------------------------------------------------------------------- focused physical check (2026-10-23, from the user's own ratings)
PHYS_SYSTEM = """You check AI video edits for physical mistakes a viewer notices at once. You get what the edit changed and PAIRS of frames at the same moment: SOURCE (the original video) then GENERATED (the edit). Compare each GENERATED frame with its SOURCE frame and report only clear, visible mistakes of these kinds:
- floating: a person, animal or vehicle that touches the ground, a floor or water in the SOURCE frame touches no surface in the GENERATED frame (hovering above the road or the water). Being in the air is right only when the SOURCE subject is in the air at that moment too.
- hands: a hand that grips or holds something in the SOURCE holds nothing in the GENERATED frame (fingers closed around air) or holds the wrong thing.
- held_object: a replaced object is not where the original is or is not used the same way (hanging on the back, floating, beside the person, a second copy), or part of the original object is still there.
- anatomy: a person or animal with an extra or missing arm, leg, foot, head or tail, or two limbs merged into one. Unusual poses that a dancer, gymnast or athlete can make (arms over the head, legs in the air, twisted bodies) are not mistakes.
- copies: an extra copy of a subject - often a small full-body figure in the background or between the legs in a close-up.
- leftover: hair, skin, clothes or a face of the replaced original on the new subject (long human hair on an animal or a robot).
- out_of_place: an object from the old place that does not belong in the new place and that nobody uses (a book lying on a desert road, a sofa on a beach). A ledge, rail, ramp, step or box the subject rides, jumps onto, grinds or lands on is NOT out_of_place, even if it looks like the old place: it carries the action.
- path: the subject heads straight into a wall, cliff or obstacle it would hit, while the SOURCE subject has open space ahead.
- off_edge: the subject rides, runs, drives or jumps off the top of a wall, parapet or edge into open air or a drop, or lands on nothing, while the SOURCE subject lands on or rolls along a ledge, ramp or the ground.
Do not report anything the edit asks for (the new character, object or place), differences of style, light or framing, motion blur, or anything you are not sure of.
Reply with JSON only: {"frames": [{"t": seconds, "problems": [{"type": "floating|hands|held_object|anatomy|copies|leftover|out_of_place|path|off_edge", "what": "short and specific: who or what, where in the frame"}]}]} with one entry per pair (an empty list when the frame is fine)."""

PHYS_CONFIRM = """You verify one reported mistake in ONE frame of an AI video edit. You get the SOURCE frame (the original video at that moment), the GENERATED frame and the report. Look closely. The mistake is real only if it is clearly visible in the GENERATED frame and the SOURCE frame does not show the same thing (a subject in the air in both frames is not floating; a figure that is in the source too is not an extra copy; an unusual but possible pose - a dance freeze, a split, an arm over the head - is not an anatomy mistake: anatomy means a wrong number of limbs or limbs merged together; when the edit replaces a PERSON, that person's own outfit and accessories - a cap, hat, glasses, gloves, jewellery, a bag they wear - go with them: the replacement wears and carries what its picture shows, so a hand that touched or held the old person's cap or gloves now touching nothing, its own hair or its own clothes is right. Things used in the action (a phone, a cup, a ball, reins, handlebars, a tool) must still be held).
You also get each replacement's own picture: what a replacement's picture shows on or with it (its clothes, a backpack, jewellery, a hat, a bag) belongs to it and is not a mistake, even when the edit's text does not name it. A replaced OBJECT must still sit where the original was and be used the same way.
Reply with JSON only: {"real": true or false, "why": "short", "fix": "if real: one sentence saying what the frame must show instead, written positively and without naming the mistake (for example: 'His gloved fists grip the black reins just above the horse's neck.'). It describes the EDITED video - the new characters, objects and place the edit asks for - never the source's old place, people or objects"}"""

PHYS_SEV = {"floating": "critical", "hands": "critical", "held_object": "critical", "anatomy": "critical", "copies": "critical",
            "off_edge": "critical", "path": "major", "leftover": "major", "out_of_place": "major"}


def phys_confirm(job, spec, typ, what, src_f, out_f):
    """one reported physical mistake checked on a bigger view - with the replacements' own pictures (2026-10-29, test-files case 7: the
    character's grey bear backpack, shown in the picture's back view but missing from the spec's description, was called 'a backpack not present
    on the original character' and the retake was told to show his bare back)"""
    subj = [{k: s.get(k) for k in ("tag", "kind", "replaces", "description")} for s in spec.get("subjects", [])]
    pics = []
    for s in spec.get("subjects", []):
        p = s.get("picture")
        if s.get("kind") != "location" and isinstance(p, int) and 1 <= p <= len(job.get("images") or []):
            pics += [{"type": "text", "text": f"{s.get('tag')} (a replacement) looks like this picture:"}, llm.img_part(job["images"][p - 1], 448)]
    return llm.ask([{"type": "text", "text": "The edit: " + json.dumps({"instruction": spec.get("instruction"), "subjects": subj}, ensure_ascii=False)[:1800]}] + pics +
                   [{"type": "text", "text": f"Reported mistake ({typ}): {what}"}, {"type": "text", "text": "SOURCE frame:"},
                    llm.img_part(src_f, 640), {"type": "text", "text": "GENERATED frame:"}, llm.img_part(out_f, 896)],
                   system=PHYS_CONFIRM, max_tokens=300, temperature=0.0)


def physics_check(job, spec, src, out_frames, cuts=None, trace=None, seg=None):
    """the mistakes the user scored down that the general check passed: floating, empty hands, misplaced objects, extra limbs, tiny copies,
    leftover hair, props from the old place, heading into a wall. Detect on SOURCE|GENERATED pairs every 0.5 s (6 pairs per call), then
    confirm each finding (one per kind per second) on a bigger view of that frame. -> issues with a positive 'fix' sentence"""
    L = len(out_frames)
    if not L or not src:
        return []
    idx = sorted(set(range(3, L, 12)) | {L - 1})
    subj = [{k: s.get(k) for k in ("tag", "kind", "replaces", "description")} for s in spec.get("subjects", [])]
    head = [{"type": "text", "text": "The edit: " + json.dumps({"instruction": spec.get("instruction"), "subjects": subj}, ensure_ascii=False)[:2500]}]
    found = []
    for c0 in range(0, len(idx), 6):
        content = list(head) + [{"type": "text", "text": "Frame pairs (SOURCE then GENERATED):"}]
        for i in idx[c0:c0 + 6]:
            content += [{"type": "text", "text": f"t={i / FPS:.2f}s SOURCE:"}, llm.img_part(src[min(i, len(src) - 1)], 384),
                        {"type": "text", "text": f"t={i / FPS:.2f}s GENERATED:"}, llm.img_part(out_frames[i], 576)]
        try:
            r = llm.ask(content, system=PHYS_SYSTEM, max_tokens=1200, temperature=0.0)
        except Exception:
            continue
        for fr in (r.get("frames") or []) if isinstance(r, dict) else []:
            for p in fr.get("problems") or [] if isinstance(fr, dict) else []:
                if isinstance(p, dict) and p.get("type") in PHYS_SEV:
                    try:
                        found.append((float(fr.get("t")), p["type"], str(p.get("what", ""))[:240]))
                        if trace is not None:
                            trace.append({"step": "detect", "t": float(fr.get("t")), "type": p["type"], "what": str(p.get("what", ""))[:240]})
                    except (TypeError, ValueError):
                        pass
    try:
        found += physics_probes(job, spec, src, out_frames, trace, seg=seg)
    except Exception as e:
        if trace is not None:
            trace.append({"step": "probe_error", "error": str(e)[:200]})
    issues, tried = [], set()
    for t, typ, what in sorted(found):
        i = min(L - 1, max(0, int(round(t * FPS))))
        if (typ, i // 24) in tried or len(tried) >= 10:
            continue
        tried.add((typ, i // 24))
        if blank_jitter(out_frames[i], src, i):
            if trace is not None:
                trace.append({"step": "blank_jitter", "t": round(i / FPS, 2), "type": typ})
            continue
        try:
            r = phys_confirm(job, spec, typ, what, src[min(i, len(src) - 1)], out_frames[i])
        except Exception:
            continue
        if trace is not None:
            trace.append({"step": "confirm", "t": round(i / FPS, 2), "type": typ, "real": bool(isinstance(r, dict) and r.get("real")), "why": str((r or {}).get("why", ""))[:200]})
        if isinstance(r, dict) and r.get("real") and typ == "floating" and float_same(src[min(i, len(src) - 1)], out_frames[i], trace, round(i / FPS, 2)):
            continue
        if isinstance(r, dict) and r.get("real") and typ in ("copies", "anatomy") and same_in_source(src[min(i, len(src) - 1)], out_frames[i], f"{typ}: {what}", trace, round(i / FPS, 2)):
            continue
        if isinstance(r, dict) and r.get("real") and typ == "copies" and ((seg and COPY_WORDS.search(str(seg.get("shots") or ""))) or source_doubles_near(src, i)):
            if trace is not None:   # the part's own description has the source's double / clone / reflection (2026-10-29, test-files case 6 on the
                trace.append({"step": "copy_effect", "t": round(i / FPS, 2), "why": "the source shows the person twice in this part"})   # people path:
            continue   # same_in_source missed it, and the retake note asked for her 'alone in the spotlight' - against the source and the example
        if isinstance(r, dict) and r.get("real"):
            issues.append({"severity": PHYS_SEV[typ], "time": round(i / FPS, 2), "what": f"{typ}: {what}", "fix": str(r.get("fix") or "")[:300], "check": "physics"})
    return issues


FLOAT_CMP = """You see ONE moment of a video twice, side by side at the same size: LEFT = the SOURCE (the original video), RIGHT = the GENERATED edit of it. The edit may change who or what is there and the place; the camera and the movement are the same. Look only at the main subject's lowest points (feet, shoes, wheels, hooves, board or hull) and the ground, floor, water or sand right under them, in each half.
Reply with JSON only: {"source": "touching" or "in the air", "generated": "touching" or "in the air", "gap": "bigger in GENERATED" or "same" or "bigger in SOURCE"} - "same" when the subject's lowest point is about as far above its ground in both halves (both standing or riding on it, or both in the air in the same pose of a jump or a stride)."""


SAME_CMP = """You see ONE moment of a video twice, side by side at the same size: LEFT = the SOURCE (the original video), RIGHT = the GENERATED edit of it. The edit replaces some people, things or the place; poses, camera, framing and effects (blur, reflections, double exposures, mirrored or ghosted copies, split screens) stay as in the source.
A checker reported this mistake in the GENERATED frame: {what}
Does the SOURCE frame show the same thing at the same place - the same extra or ghosted figure, reflection or copy, the same pose and limb position? Reply with JSON only: {{"same_in_source": true or false, "why": "short"}}"""


def side_by_side(src_f, out_f, h=720):
    """SOURCE | GENERATED in one labelled image, both at height h"""
    a = cv2.resize(src_f, (int(round(src_f.shape[1] * h / src_f.shape[0])), h)); b = cv2.resize(out_f, (int(round(out_f.shape[1] * h / out_f.shape[0])), h))
    pair = np.hstack([a, np.full((h, 10, 3), 255, np.uint8), b])
    for x0, lab in ((0, "SOURCE"), (a.shape[1] + 10, "GENERATED")):
        cv2.rectangle(pair, (x0, 0), (x0 + 16 * len(lab) + 14, 34), (0, 0, 0), -1)
        cv2.putText(pair, lab, (x0 + 7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return pair


def same_in_source(src_f, out_f, what, trace=None, t=None):
    """True when a confirmed 'copies' / 'anatomy' report shows something the SOURCE has too (2026-10-29, test-files case 6: the source's own
    blurred double of the dancer in the foreground was called 'a second copy of the woman'); one side-by-side image like float_same. Test: case 6
    dropped, B1's miniature couple (user-rated) kept twice. Not for leftovers or props of the old place - those ARE in the source by nature."""
    try:
        r = llm.ask([{"type": "text", "text": "LEFT = SOURCE, RIGHT = GENERATED:"}, llm.img_part(side_by_side(src_f, out_f), 1400)],
                    system=SAME_CMP.format(what=str(what)[:300]), max_tokens=200, temperature=0.0)
    except Exception:
        return False
    same = bool(isinstance(r, dict) and r.get("same_in_source") is True)
    if trace is not None:
        trace.append({"step": "same_in_source", "t": t, "same": same, "why": str((r or {}).get("why", "") if isinstance(r, dict) else "")[:200]})
    return same


def float_same(src_f, out_f, trace=None, t=None):
    """True when a confirmed 'floating' report is not a difference from the source (2026-10-29: test-files case 3 - a leap in the source
    was called 'feet touching the asphalt' and the matching leap in the edit 'floating'; M9's ollie and M7's surfboard close-up the same):
    both frames in ONE side-by-side image, asked which one's gap under the subject is bigger"""
    h = 720
    a = cv2.resize(src_f, (int(round(src_f.shape[1] * h / src_f.shape[0])), h)); b = cv2.resize(out_f, (int(round(out_f.shape[1] * h / out_f.shape[0])), h))
    pair = np.hstack([a, np.full((h, 10, 3), 255, np.uint8), b])
    for x0, lab in ((0, "SOURCE"), (a.shape[1] + 10, "GENERATED")):
        cv2.rectangle(pair, (x0, 0), (x0 + 16 * len(lab) + 14, 34), (0, 0, 0), -1)
        cv2.putText(pair, lab, (x0 + 7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    try:
        r = llm.ask([{"type": "text", "text": "LEFT = SOURCE, RIGHT = GENERATED:"}, llm.img_part(pair, 1400)], system=FLOAT_CMP, max_tokens=200, temperature=0.0)
    except Exception:
        return False
    gap = str((r or {}).get("gap", "") if isinstance(r, dict) else "").lower()
    same = bool(gap) and "generated" not in gap
    if trace is not None:
        trace.append({"step": "float_cmp", "t": t, "gap": gap, "source": str((r or {}).get("source", "")), "generated": str((r or {}).get("generated", "")), "dropped": same})
    return same


# focused probes: one physical question at a time, asked as facts about the SOURCE and the GENERATED frame (more reliable than "find
# problems" for empty hands, a misplaced object, an extra leg, a gap under the wheels and a wall ahead - the misses of the first pass)
PROBE_HEAD = """You compare two frames of the same moment: SOURCE (the original video) and GENERATED (an AI edit of it; the edit: {edit}).
Changes the edit asks for are fine. Answer only the question below, for every pair, by describing what you actually see in each frame.
Reply with JSON only: {{"pairs": [{{"t": seconds, "source": "...", "generated": "...", "verdict": "{verdicts}"}}]}}
"""
PROBES = {
    "hands": ("ok|empty|wrong", "hands", "QUESTION: what does each visible hand hold? Describe it for SOURCE and for GENERATED ('no hands visible' if none). "
              "verdict = empty when a hand that holds something in SOURCE holds nothing in GENERATED (the fingers close around air, or the thing "
              "is beside the hand instead of in it); wrong when it holds something else than the edit says; ok otherwise."),
    "object": ("ok|moved|double|missing", "held_object", "QUESTION: count every visible PIECE of the thing the edit replaces ({replaces}) in "
               "SOURCE and of its replacement ({desc}) in GENERATED - e.g. bodies, necks and headstocks of a guitar, cups, handles - and for each "
               "piece say where it is in the frame (left / centre / right, top / middle / bottom), which way it points and what holds it. Pieces "
               "that differ in colour or material (a dark wooden neck next to a golden bread body) are separate things, not one. verdict = double "
               "when GENERATED shows more pieces than SOURCE (a second neck, the original's neck or handle next to the replacement); moved when the "
               "replacement is in another part of the frame, points another way or is held another way than the original (hanging at the hip or "
               "on the back while the original is held across the body); missing when nothing replaces it; ok otherwise."),
    "legs": ("ok|extra|missing", "anatomy", "QUESTION: count the legs and feet of every person and animal you can see, in SOURCE and in GENERATED "
             "('2 legs, 2 feet'). verdict = extra when one body in GENERATED has more legs or feet than SOURCE shows for it; missing when a leg "
             "that SOURCE shows is gone; ok otherwise (unusual poses are fine)."),
    "contact": ("ok|floating", "floating", "QUESTION: where does the main subject (and anything it rides) meet the ground, floor or water - look at the "
                "wheels, feet, hooves or hull and the surface right under them. In SOURCE, does it touch the surface or is it in the air? In "
                "GENERATED? Also: is there ground, a ledge or a ramp right under and ahead of it, or open air / a drop behind a wall top? verdict = floating "
                "when it touches the surface in SOURCE but hovers with a gap above the surface in GENERATED, or when GENERATED has it riding or "
                "jumping off the top of a wall, parapet or edge into open air while SOURCE has it on a ledge, ramp or the ground; ok otherwise."),
    "path": ("ok|blocked", "path", "The main subject travels {travel} in the frame. QUESTION: in the direction it travels, what is close ahead of "
             "it - open space (a path, road, walkway, slope or water) or a wall, railing, parapet, cliff edge or obstacle? Describe it for SOURCE and "
             "for GENERATED. verdict = blocked when GENERATED has a wall, railing, edge or obstacle close ahead in that direction that SOURCE does "
             "not have; ok otherwise."),
}
HELD_RE = re.compile(r"\b(hold|holds|holding|grip|grips|hand|hands|fist|reins|handlebar|steering|guitar|cup|mug|bag|briefcase|ball|phone|bat|"
                     r"racket|board|skateboard|surfboard|bike|bicycle|motorcycle|scooter|jet ski|paddle|knife|pan|bottle|book)\b", re.I)


def physics_probes(job, spec, src, out_frames, trace=None, seg=None):
    """-> [(t, type, what)] candidates from the focused probes that apply to this edit"""
    L = len(out_frames)
    if not L or not src:
        return []
    subs = spec.get("subjects", [])
    edit = (spec.get("instruction") or "")[:600]
    text = " ".join(str(s.get("replaces") or "") + " " + str(s.get("description") or "") for s in subs) + " " + \
           " ".join(str(g.get("shots") or "") for g in spec.get("segments", []))
    objs = [s for s in subs if s.get("kind") == "object"]
    use = ["contact"]
    if any(s.get("kind") in ("person", "animal") for s in subs):
        use.append("legs")
    if HELD_RE.search(text):
        use.append("hands")
    if objs:
        use.append("object")
    if job.get("mode") == "motion" and any(s.get("kind") == "location" for s in subs):
        use.append("path")
    idx = sorted(set(range(3, L, 12)) | {L - 1}); found = []
    for name in use:
        verdicts, typ, q = PROBES[name]
        if name == "path":
            q = q.format(travel=str((seg or {}).get("travel") or "across the frame").replace("-", " "))
        if name == "object":
            q = q.format(replaces="; ".join(str(o.get("replaces") or "") for o in objs)[:300], desc="; ".join(str(o.get("description") or "")[:160] for o in objs)[:400])
        system = PROBE_HEAD.format(edit=edit, verdicts=verdicts) + q
        for c0 in range(0, len(idx), 6):
            content = [{"type": "text", "text": "Frame pairs (SOURCE then GENERATED):"}]
            for i in idx[c0:c0 + 6]:
                content += [{"type": "text", "text": f"t={i / FPS:.2f}s SOURCE:"}, llm.img_part(src[min(i, len(src) - 1)], 448),
                            {"type": "text", "text": f"t={i / FPS:.2f}s GENERATED:"}, llm.img_part(out_frames[i], 640)]
            try:
                r = llm.ask(content, system=system, max_tokens=1500, temperature=0.0)
            except Exception:
                continue
            for pr in (r.get("pairs") or []) if isinstance(r, dict) else []:
                v = str((pr or {}).get("verdict", "ok")).strip().lower() if isinstance(pr, dict) else "ok"
                if v and v != "ok" and v in verdicts.split("|"):
                    try:
                        t = float(pr.get("t"))
                    except (TypeError, ValueError):
                        continue
                    what = f"{v}: in the source {str(pr.get('source', ''))[:110]}; in the edit {str(pr.get('generated', ''))[:110]}"
                    found.append((t, typ, what))
                    if trace is not None:
                        trace.append({"step": "probe", "probe": name, "t": t, "type": typ, "what": what})
    return found


# ---------------------------------------------------------------------------- ideas from the open-Genjutsu repos (2026-10-23 evening)
REPAINT_PLACE = False    # new-place plates from the shot's own re-painted first frame (repaint_plate) before the place_view plate
GREY_HAIR = False        # hair of the replaced people: flat grey instead of filled from the surroundings (A/B before switching on)
MASKED_OBJECTS = False   # object swaps: grey the old object + re-render only inside its outline (A/B before switching on)
DEPTH_PLACE = False      # new place drawn from the source's own depth before the plate fallback (test before switching on)


def follow_alpha(orig, gen, alpha, diff=18):
    """Ex8-ca/openjutsu's follow_regeneration: keep the render where it really differs from the original, inside the mask grown by ~4.5 %
    of the frame (plus the mask itself), grown a little and feathered - a replacement bigger than the original stays whole instead of being
    cut along the old outline. orig/gen: HxWx3 uint8, alpha: HxW float 0..1 -> HxWx1 float"""
    H_, W_ = orig.shape[:2]; g = max(3, int(0.045 * max(W_, H_)))
    big = cv2.dilate((alpha > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    d = (np.abs(orig.astype(np.int16) - gen.astype(np.int16)).max(axis=2) > diff).astype(np.uint8)
    m = cv2.dilate(d & big, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(np.float32)
    return cv2.GaussianBlur(np.maximum(m, alpha.astype(np.float32)), (0, 0), 2.5)[..., None]


def pan_speed(frames, step=2):
    """median camera speed (px per frame at 320 px wide) from optical flow in the top and bottom 28 % of the frame (backlot's
    orbit_direction idea): the background bands, mostly away from the subject"""
    g = [cv2.cvtColor(cv2.resize(f, (320, max(2, int(320 * f.shape[0] / f.shape[1])))), cv2.COLOR_BGR2GRAY) for f in frames]
    if len(g) <= step:
        return 0.0
    b = max(2, int(0.28 * g[0].shape[0])); v = []
    for i in range(step, len(g), step):
        fl = cv2.calcOpticalFlowFarneback(g[i - step], g[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        band = np.concatenate([fl[:b].reshape(-1, 2), fl[-b:].reshape(-1, 2)])
        v.append(float(np.hypot(np.median(band[:, 0]), np.median(band[:, 1]))) / step)
    return float(np.median(v))


def pan_check(src, out, min_speed=1.5, ratio=0.7):
    """the camera of the result moves much slower than the source's (a slowed plate pan: 'he is not really moving like in the source')"""
    s, o = pan_speed(src), pan_speed(out)
    if s >= min_speed and o < ratio * s:
        return [{"severity": "major", "time": round(len(out) / 2 / FPS, 2), "check": "pan",
                 "what": f"camera: the view moves at {100 * o / max(s, 1e-6):.0f} % of the source camera's speed",
                 "fix": "The camera travels exactly as fast as in <Video 1>; the background streams past at the same speed."}]
    return []


def selectivity(orig, out, masks):
    """Ex8-ca's check for masked edits: mean change inside the mask / outside (low = nothing really changed where it should)"""
    ins, outs = [], []
    for o, g, m in zip(orig, out, masks):
        d = np.abs(o.astype(np.int16) - g.astype(np.int16)).mean(axis=2); mm = m > 0
        if mm.any() and (~mm).any():
            ins.append(float(d[mm].mean())); outs.append(float(d[~mm].mean()))
    return (np.mean(ins) / max(np.mean(outs), 1e-3)) if ins else 0.0


def grey_regions(job_obj, win, words, tag, grow_frac=0.012):
    """SydFloyd/openjutsu's trick: paint what must disappear (the old object, long hair) flat grey (RGB 127) in the SOURCE window through a
    slightly grown SAM 3 mask, so the model has no old colours or shapes to copy. -> (grey window path, per-frame 0/1 masks)"""
    jd = job_obj.store.dir(job_obj.id); dst = os.path.join(jd, f"{tag}_grey.mp4"); od = os.path.join(jd, f"{tag}_grey_sam")
    tools.gpu_take(job_obj)
    tools.run_proc(job_obj, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_matte.py", win, od, words, "matte"], 0.5, 0.5, f"Outlining {words.replace('|', ', ')}")
    M = read_frames(os.path.join(od, "matte.mp4")); fr = read_frames(win); out, masks = [], []
    for i, f in enumerate(fr):
        m = cv2.cvtColor(M[min(i, len(M) - 1)], cv2.COLOR_BGR2GRAY) if M else np.zeros(f.shape[:2], np.uint8)
        g = max(2, int(grow_frac * max(f.shape[:2])))
        m = cv2.dilate((cv2.resize(m, (f.shape[1], f.shape[0])) > 127).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
        f2 = f.copy(); f2[m > 0] = 127; out.append(f2); masks.append(m)
    write_video(out, dst, crf=12)
    return dst, masks


def depth_place_plate(job_obj, job, spec, k, win, Lr, W, H, tag, strength=1.8, seed=4242):
    """fxjutsu's order for a new place: the SOURCE's own depth (MoGe-2), with the subjects filled out of it, drives H3 (Fun ControlNet
    Union) to draw the NEW place - so ground, layout and camera move are the source's by construction (nothing floats, nothing runs into a
    wall, pans keep their speed; the new place keeps the old one's shape) - then the subjects are laid back on top. -> plate path"""
    jd = job_obj.store.dir(job_obj.id)
    loc = next((s_ for s_ in spec.get("subjects", []) if s_.get("kind") == "location" and isinstance(s_.get("picture"), int)
                and 1 <= s_["picture"] <= len(job["images"])), None)
    if not loc:
        raise RuntimeError("no place picture")
    words = subject_words(spec)
    od = os.path.join(jd, f"{tag}_dp_sam"); dd = os.path.join(jd, f"{tag}_dp_depth"); tools.gpu_take(job_obj)
    tools.run_proc(job_obj, [tools.PY["sam3"], "-I", f"{tools.TOOLS_DIR}/sam3_matte.py", win, od, words, "matte"], 0.5, 0.5, f"Outlining {words.replace('|', ', ')}")
    tools.run_proc(job_obj, [tools.PY["moge"], "-I", f"{tools.TOOLS_DIR}/moge_depth.py", win, dd, "gray", "auto", str(Lr)], 0.5, 0.5, "Measuring the shot's depth (MoGe-2)")
    F = [cover_resize(f, W, H) for f in read_frames(win)][:Lr]; M = read_frames(os.path.join(od, "matte.mp4")); Dp = read_frames(os.path.join(dd, "depth.mp4"))
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.03 * max(W, H)) + 1,) * 2); ctl, alphas = [], []
    for i in range(len(F)):
        m = cv2.resize(cv2.cvtColor(M[min(i, len(M) - 1)], cv2.COLOR_BGR2GRAY), (W, H)) if M else np.zeros((H, W), np.uint8)
        d = cv2.resize(cv2.cvtColor(Dp[min(i, len(Dp) - 1)], cv2.COLOR_BGR2GRAY), (W, H))
        d2 = cv2.inpaint(d, cv2.dilate((m > 40).astype(np.uint8) * 255, grow), 15, cv2.INPAINT_TELEA)   # the subjects filled out of the depth
        ctl.append(cv2.cvtColor(d2, cv2.COLOR_GRAY2BGR))
        alphas.append(cv2.GaussianBlur(cv2.dilate((m > 100).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))), (0, 0), 1.5))
    ctl_p = os.path.join(jd, f"{tag}_dp_control.mp4"); write_video(ctl, ctl_p, crf=12)
    seg = next((g for g in spec.get("segments", []) if str(g.get("index")) == str(k)), {})
    spot = str(seg.get("spot") or "").strip()
    prompt = (f"A video of the place shown in <Picture 1>: {rn_plain(loc.get('description'))} "
              + (f"{spot}. " if spot and spot.lower() not in ("none", "empty") else "") +
              "The camera moves through the place exactly as the depth shows. The place is empty: no people, no animals, no vehicles. "
              "Photorealistic, with the look, materials and light of <Picture 1>.")
    llm.down()
    wf = h3.build_gen(prompt, [job["images"][loc["picture"] - 1]], W, H, Lr, seed, []); h3.add_control(wf, ctl_p, strength)
    files = comfy.run(wf, None, job_obj.cancelled); mp4 = next((f for f in files if f.endswith(".mp4")), None)
    if not mp4:
        raise RuntimeError("the place render produced no video")
    place_v = os.path.join(jd, f"{tag}_dp_place.mp4"); shutil.copy(mp4, place_v); Pv = read_frames(place_v)
    out = []
    for i, f in enumerate(F):
        a = alphas[i][..., None]; p = cv2.resize(Pv[min(i, len(Pv) - 1)], (W, H))
        out.append((f.astype(np.float32) * a + p.astype(np.float32) * (1 - a)).astype(np.uint8))
    dst = os.path.join(jd, f"{tag}_plate.mp4"); write_video(out, dst, crf=12); cv2.imwrite(os.path.join(jd, f"{tag}_plate_first.png"), out[0])
    job_obj.log(f"{tag}: new place drawn from the shot's own depth (MoGe-2 + Fun ControlNet {strength}), subjects laid back on top")
    return dst


def rn_plain(t):
    return re.sub(r"<Subject \d+>", "the new subject", str(t or "")).strip()
