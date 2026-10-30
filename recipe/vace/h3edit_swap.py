#!/usr/bin/env python3
"""h3edit_swap.py ROOT K OUT [options] : H3 masked-edit swap of one shot of a redoutline3 run folder (test tool).

H3 (the app's own Ref2VA recipe: character-swap LoRA + 8-step turbo, er_sde) starts from the SOURCE clip and redraws only a region
around the picked old objects (H3StudioMaskedEditLatent, ComfyUI/custom_nodes/h3studio_masked_edit.py). Everything outside the region
stays the source by construction: no stray copies, no outline leaks, no changes to people or background.
--video   also cite the source as <Video 1> with the shot's edit prompt (outline words removed); default: the pictures only and the
          shot's scene sentence (D3voz/openjutsu: citing the source can make H3 return it unchanged)
--region  box (per-frame box of each old object, grown by --grow-rel x its size) or mask (the mask grown the same way)
OUT gets region.mp4, h3_raw.mp4, result.mp4 (source pixels outside the redrawn token area), panel.mp4 and h3edit.json."""
import argparse, json, os, re, shutil, subprocess, sys, time
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import comfy, h3
import vace_swap as V

FPT = (1, 4, 4, 4, 4)  # H3: latent frame k covers FPT[k % 5] video frames


def token_area(R, W, H):
    """the area H3 really redraws: the mask max-pooled over each latent frame's video frames and 32 px tokens"""
    n = len(R); out = np.zeros_like(R); s = 0; k = 0
    while s < n:
        e = min(n, s + FPT[k % 5]); m = R[s:e].any(0)
        if m.any():
            hh, ww = (H + 31) // 32, (W + 31) // 32
            pad = np.zeros((hh * 32, ww * 32), bool); pad[:H, :W] = m
            blk = pad.reshape(hh, 32, ww, 32).any((1, 3))
            m = np.kron(blk, np.ones((32, 32), bool))[:H, :W]
        out[s:e] = m; s = e; k += 1
    return out


def plain_prompt(meta, members):
    """a swap prompt in H3's structured form for a shot the recipe did not render (no seg<k>_prompt.txt: L05's shot 1 had every pick
    dropped by the close-up check, so it stayed the source shot - autofix's box redraw then crashed on the missing prompt)"""
    subs, defs, summ = [], [], []
    for j, oi in enumerate(members, 1):
        o = meta["objs"][oi]; old = re.sub(r"<[^>]+>", "", str(o.get("replaces") or ("the " + o["word"]))).strip()
        subs.append(f"{old} in <Video 1> with the object from <Picture {j}>")
        defs.append(f"<Subject {j}> is the object whose appearance comes from <Picture {j}> and whose performance comes from {old} in <Video 1>: "
                    f"{o['full']}")
        summ.append(f"{old} replaced by the {o['full']} (<Subject {j}>)")
    return ("Replace " + " and ".join(subs) + ", each at exactly the place, size and angle of its original, held and used exactly the same way."
            "\n\nsubject_definitions:\n" + "\n".join(defs) + "\n<Video 1> is the source video being edited and supplies the shot composition, "
            "camera movement, cuts, timing, and sequence of actions.\n\nsummary:\n[video editing + reference generation] The target video is "
            "<Video 1> with " + " and ".join(summ) + ", keeping every movement, camera move, cut and the timing.")


def edit_prompt(root, k, meta=None, members=()):
    """the shot's own swap prompt without the outline words (the source is not outlined here)"""
    if not os.path.exists(f"{root}/seg{k}_prompt.txt") and meta is not None:
        return plain_prompt(meta, list(members))
    s = open(f"{root}/seg{k}_prompt.txt").read()
    s = re.sub(r" outlined in \w+", "", s)
    s = re.sub(r"[^.\n]*\b[Oo]utline[^.\n]*\.\s*", "", s)
    return s


def add_objects(prompt, extra):
    """objects the shot's prompt does not swap, added: [(picture number, obj)] -> a 'Replace ...' sentence first and a <Subject> line"""
    lines = []; defs = []
    for n_, o in extra:
        old = re.sub(r"<[^>]+>", "", str(o.get("replaces") or ("the " + o["word"]))).strip()
        lines.append(f"Also replace {old} in <Video 1> with the object from <Picture {n_}>, at exactly its place, size and angle.")
        defs.append(f"<Subject {n_}> is the object whose appearance comes from <Picture {n_}> and whose performance comes from {old} in "
                    f"<Video 1>: {o['full']}")
    head, sep, rest = prompt.partition("\n\nsubject_definitions:\n")
    if not sep:
        return " ".join(lines) + "\n\n" + prompt
    body = rest.split("\n"); end = next((i for i, l in enumerate(body) if not l.strip()), len(body))   # subject_definitions only
    last = max((i for i, l in enumerate(body[:end]) if l.startswith("<Subject")), default=-1)
    body[last + 1:last + 1] = defs
    return head + " " + " ".join(lines) + sep + "\n".join(body)


def scene_prompt(root, k, meta, members):
    """pictures-only prompt: the shot's scene sentence with each <Subject i> named after its <Picture i>"""
    s = open(f"{root}/seg{k}_prompt.txt").read(); i = s.find("detailed_description:")
    paras = [x.strip() for x in s[i + len("detailed_description:"):].split("\n\n") if x.strip()] if i >= 0 else []
    txt = next((x for x in paras if "<Subject" in x and not x.startswith("Important")), "")
    for j, oi in enumerate(members):
        o = meta["objs"][oi]
        txt = txt.replace(f"<Subject {j + 1}>", f"{o['full']} from <Picture {j + 1}>", 1).replace(f"<Subject {j + 1}>", o["full"])
    txt = re.sub(r"<Video \d+>", "the video", txt)
    names = "; ".join(f"the {meta['objs'][oi]['full']} looks exactly like <Picture {j + 1}>" for j, oi in enumerate(members))
    return (txt + f" Photorealistic live-action footage, natural light and motion blur; {names}, at its real size, "
            "held and moved naturally.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("k", type=int); ap.add_argument("out")
    ap.add_argument("--refs", default="", help="object pictures in shots_meta 'objs' order (default: the shot's reference sheets)")
    ap.add_argument("--video", action="store_true"); ap.add_argument("--region", default="auto", choices=["auto", "box", "mask"],
                    help="auto: a box, but a thin or slanted object (it fills < 30 %% of its box: a pool cue, a pole) gets its own grown "
                         "outline - its box covered the player's head and H3 redrew him as another man (T22)")
    ap.add_argument("--extra-region", default="", help="a mask video added to the region (make_paste's pasted_mask.mp4)")
    ap.add_argument("--grow-rel", type=float, default=0.25); ap.add_argument("--tpad", type=int, default=1)
    ap.add_argument("--grow-rels", default="", help="per object, e.g. '1:0.5' (a pasted object made bigger than the old one needs the room)")
    ap.add_argument("--strength", type=float, default=1.0); ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--swap-lora", type=float, default=1.0, help="character-swap LoRA strength (0 = off)")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--ref-video", default="plain", choices=["plain", "grey", "blur", "outlined"],
                    help="with --video: the old objects in the cited source are left as they are, painted grey or blurred")
    ap.add_argument("--pick", default="", help="override picks, e.g. '0:obj_0+obj_3;1:obj_2'")
    ap.add_argument("--fallback-refs", default="", help="pictures (objs order) used where the run has no reference sheet")
    ap.add_argument("--start", default="", help="video the masked latent starts from (default: the source) - e.g. make_paste.py's "
                    "pasted.mp4 with --strength < 1: H3 refines a pasted new object instead of inventing it")
    ap.add_argument("--region-video", default="", help="exactly this region (white) is redrawn instead of the picked objects' (autofix's fix of a "
                    "new object drawn on another one's place: the stray's own area, on the take)")
    ap.add_argument("--only-picked", action="store_true", help="the prompt and pictures name only the objects that have picks (a plain "
                    "prompt): the stray's own object is left out, so nothing asks H3 to draw it there again")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    meta = json.load(open(f"{a.root}/shots_meta.json"))
    srcp = f"{a.root}/seg{a.k}_src.mp4"; src = V.read_frames(srcp); n = len(src); H, W = src[0].shape[:2]
    orig_members = list(meta["shots"][str(a.k)].get("members", []))
    if a.pick:  # picks given (autofix: a missed instance added; K03 test: a corrected pick)
        sh_ = meta["shots"][str(a.k)]
        for part in a.pick.split(";"):
            oi_s, names = part.split(":")
            sh_["pick"][oi_s] = [x for x in names.split("+") if x]
            if sh_["pick"][oi_s] and int(oi_s) not in sh_.setdefault("members", []):
                sh_["members"].append(int(oi_s))
    masks = V.masks_for_shot(a.root, meta, a.k, n, H, W)
    if not masks:
        raise SystemExit("no picked instances in this shot")
    # pictures in the order the shot's prompt numbers them (its members), objects added by --pick after them (G09 5 s: the wheelbarrow the
    # pick step had dropped was no member of the shot - its prompt never mentions it)
    added = [oi for oi in sorted(masks) if oi not in orig_members]
    members = [oi for oi in orig_members if oi in masks or os.path.exists(f"{a.root}/sheet2_s{a.k}_{oi}.png")] + added
    if a.only_picked:
        members = sorted(masks); added = []
    R = np.zeros((n, H, W), bool)
    gr = {int(x.split(":")[0]): float(x.split(":")[1]) for x in a.grow_rels.split(";") if ":" in x}
    for oi in (masks if not a.region_video else []):
        box = a.region == "box"
        if a.region == "auto":
            fill = [m.sum() / max(1, (np.ptp(np.nonzero(m)[0]) + 1) * (np.ptp(np.nonzero(m)[1]) + 1)) for m in masks[oi] if m.sum() > 20]
            box = not fill or float(np.median(fill)) >= 0.3
            if not box:
                log_ = f"object {oi} is thin or slanted (fills {np.median(fill) * 100:.0f}% of its box): its own outline, grown"
                print(time.strftime("%H:%M:%S"), log_, flush=True)
        g_ = gr.get(oi, a.grow_rel)
        R |= V.grow_rel(masks[oi], g_ if box else max(g_, 0.6), box, a.tpad)
    if a.region_video:
        rv_ = V.read_frames(a.region_video)[:n]
        for i in range(min(n, len(rv_))):
            R[i] = cv2.resize(rv_[i], (W, H), interpolation=cv2.INTER_NEAREST)[..., 1] > 127
    if a.extra_region and os.path.exists(a.extra_region):
        ex = V.read_frames(a.extra_region)[:n]
        for i in range(min(n, len(ex))):
            e_ = cv2.resize(ex[i], (W, H), interpolation=cv2.INTER_NEAREST)[..., 1] > 127
            R[i] |= cv2.dilate(e_.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    G = token_area(R, W, H)
    V.write_video(f"{a.out}/region.mp4", [m.astype(np.uint8) * 255 for m in R], crf=0, gray=True)
    refs = a.refs.split(",") if a.refs else []
    pics = []
    for oi in members:
        sh = f"{a.root}/sheet2_s{a.k}_{oi}.png"
        plain = f"{a.root}/sheet2_{oi}_plain.png"; fb = a.fallback_refs.split(",") if a.fallback_refs else []
        pics.append(refs[oi] if refs else (sh if os.path.exists(sh) else plain if os.path.exists(plain) or not fb else fb[oi]))
    if a.video and a.ref_video == "outlined":  # the recipe's own outlined clip and prompt (red/blue tell H3 which object is which)
        prompt = a.prompt or open(f"{a.root}/seg{a.k}_prompt.txt").read()
    else:
        prompt = a.prompt or ((plain_prompt(meta, members) if a.only_picked else edit_prompt(a.root, a.k, meta, members)) if a.video
                              else scene_prompt(a.root, a.k, meta, members))
    if not os.path.exists(f"{a.root}/seg{a.k}_prompt.txt"):
        added = []   # the plain prompt already names every object
    if added and not a.prompt:
        prompt = add_objects(prompt, [(members.index(oi) + 1, meta["objs"][oi]) for oi in added])
    open(f"{a.out}/prompt.txt", "w").write(prompt)
    log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
    log(f"shot {a.k}: {n} frames {W}x{H}; region {R.mean() * 100:.1f}% (token area {G.mean() * 100:.1f}%); "
        f"{'with' if a.video else 'without'} <Video 1>; pictures {[os.path.basename(p) for p in pics]}")

    refv = f"{a.root}/seg{a.k}_outlined.mp4" if a.ref_video == "outlined" else srcp
    if a.video and a.ref_video in ("grey", "blur"):  # hide the old objects from the cited source so H3 takes only the motion from it
        Mo = np.zeros((n, H, W), bool)
        for oi in masks:
            Mo |= V.grow(masks[oi], max(3, W // 100), 1)
        rv = []
        for i in range(n):
            f = src[i].copy()
            if a.ref_video == "grey":
                f[Mo[i]] = 127
            else:
                b = cv2.GaussianBlur(src[i], (0, 0), max(8, W // 40)); f[Mo[i]] = b[Mo[i]]
            rv.append(f)
        refv = f"{a.out}/ref_video.mp4"; V.write_video(refv, rv)
    wf = h3.build(prompt, refv, pics, W, H, n, a.seed) if a.video else h3.build_gen(prompt, pics, W, H, n, a.seed)
    if a.video:
        wf["swap"]["inputs"]["strength_model"] = float(a.swap_lora)
    wf["msrc"] = {"class_type": "LoadVideo", "inputs": {"file": comfy.stage(a.start or srcp)}}
    wf["msrcc"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["msrc", 0]}}
    wf["mreg"] = {"class_type": "LoadVideo", "inputs": {"file": comfy.stage(f"{a.out}/region.mp4")}}
    wf["mregc"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["mreg", 0]}}
    wf["mregm"] = {"class_type": "ImageToMask", "inputs": {"image": ["mregc", 0], "channel": "red"}}
    wf["medit"] = {"class_type": "H3StudioMaskedEditLatent", "inputs": {"vae": ["vvae", 0], "images": ["msrcc", 0], "width": W,
                                                                         "height": H, "strength": float(a.strength), "mask": ["mregm", 0]}}
    wf["sample"]["inputs"]["latent_image"] = ["medit", 0]
    json.dump(wf, open(f"{a.out}/workflow_api.json", "w"), indent=1)
    t1 = time.time(); last = [0]

    def prog(stage, frac, v, mx):
        if stage == "sampling" and time.time() - last[0] > 15:
            last[0] = time.time(); log(f"  sampling {v}/{mx}")
    files = comfy.run(wf, on_progress=prog)
    vid = next(f for f in files if f.endswith(".mp4"))
    gen = V.read_frames(vid)[:n]
    shutil.move(vid, f"{a.out}/h3_raw.mp4")   # (a scratch run folder can be on another filesystem than ComfyUI's output)
    log(f"H3 done ({time.time() - t1:.0f} s), {len(gen)} frames")
    if len(gen) < n:
        raise SystemExit(f"H3 gave {len(gen)} frames for {n}")
    gen = [cv2.resize(g, (W, H)) if g.shape[:2] != (H, W) else g for g in gen]
    # how far the kept area drifted from the source (should be ~0: it is pinned)
    base = src   # outside the redrawn area: the start video when given (the first take for a fix of one object, or the pasted source)
    if a.start:
        bf = V.read_frames(a.start)[:n]; bf = [cv2.resize(f, (W, H)) if f.shape[:2] != (H, W) else f for f in bf]
        base = bf + [bf[-1]] * max(0, n - len(bf))
    out_diff = float(np.mean([np.abs(gen[i].astype(int) - base[i].astype(int))[~G[i]].mean() for i in range(n) if (~G[i]).any()]))
    res = []
    for i in range(n):
        al = cv2.GaussianBlur(G[i].astype(np.float32), (0, 0), 4)
        al = (np.minimum(al * 1.8, 1.0) * G[i])[..., None]  # soft only inside the redrawn area
        res.append((base[i].astype(np.float32) * (1 - al) + gen[i].astype(np.float32) * al).astype(np.uint8))
    V.write_video(f"{a.out}/result.mp4", res)
    V.write_video(f"{a.out}/panel.mp4", [np.hstack([src[i], res[i]]) for i in range(n)], crf=16)
    info = dict(root=a.root, k=a.k, video=a.video, ref_video=a.ref_video, pick=a.pick, start=a.start, region=a.region, grow_rel=a.grow_rel, tpad=a.tpad, strength=a.strength,
                seed=a.seed, swap_lora=a.swap_lora, pictures=pics, frames=n, region_pct=round(float(R.mean()) * 100, 2),
                token_pct=round(float(G.mean()) * 100, 2), kept_area_diff=round(out_diff, 2), seconds=round(time.time() - t0))
    json.dump(info, open(f"{a.out}/h3edit.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f} s; kept area differs from the source by {out_diff:.2f}/255 -> {a.out}/result.mp4")


if __name__ == "__main__":
    main()
