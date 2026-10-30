#!/usr/bin/env python3
"""vace_swap.py ROOT K OUT [options] : erase-then-insert object swap of one shot of a redoutline3 run folder (test tool).

1. erase  : MiniMax-Remover (its own venv, CC BY-NC 4.0) takes the old objects out inside their SAM 3 masks, grown;
2. insert : Wan VACE in ComfyUI draws the new objects inside a region that follows the old ones, the object pictures as reference;
3. joint  : the source outside the region stays pixel-exact; inside it the VACE frames, with a soft edge.

ROOT needs seg<K>_src.mp4, shots_meta.json, obj/<vi>/obj_*.mp4 (what redoutline3 writes). Pictures: --refs a.png,b.png in the
order of shots_meta "objs" (only the shot's members are used).
--control grey  : the region is grey for VACE (plain VACE inpainting; the eraser only gives the joint a clean plate)
          plate : VACE sees the erased plate inside the region too (it has to add the object to a clean background)
--region mask   : the old masks grown by --igrow px;  box : their per-frame boxes grown by --igrow px
OUT gets plate.mp4, region.mp4, vace_raw.mp4, result.mp4, panel.mp4 (source | plate | result) and vace_swap.json."""
import argparse, glob, json, os, re, shutil, subprocess, sys, time
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from server import comfy

HERE = os.path.dirname(os.path.abspath(__file__))
from server.config import PY_REMOVER as REMOVER_PY
NEG = ("过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
       "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
MODELS = {
    "14b": dict(unet="wan2.1_vace_14B_fp16.safetensors", dtype="fp8_e4m3fn_fast",
                lora="Wan21_CausVid_14B_T2V_lora_rank32.safetensors", lora_w=0.3, steps=6, cfg=1.0, shift=5.0),
    "14b-full": dict(unet="wan2.1_vace_14B_fp16.safetensors", dtype="fp8_e4m3fn_fast", lora=None, steps=20, cfg=6.0, shift=5.0),
    "1.3b": dict(unet="wan2.1_vace_1.3B_fp16.safetensors", dtype="default",
                 lora="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors", lora_w=0.5, steps=6, cfg=1.0, shift=5.0),
    "22": dict(high="wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors", low="wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors",
               lora_h="wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
               lora_l="wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors", lora_w=1.0, steps=6, switch=3, cfg=1.0, shift=8.0),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_frames(p):
    cap = cv2.VideoCapture(p); F = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        F.append(f)
    return F


def write_video(path, frames, fps=24, crf=8, gray=False):
    tmp = path + ".avi"; H, W = frames[0].shape[:2]
    w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"FFV1"), fps, (W, H))
    for f in frames:
        w.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f)
    w.release()
    pix = "yuv444p" if gray else "yuv420p"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", pix, "-crf", str(crf), path], check=True)
    os.remove(tmp)


def masks_for_shot(root, meta, k, n, H, W):
    """per member object: the union of its picked SAM 3 instances, frame by frame (bool n x H x W)"""
    sh = meta["shots"][str(k)]; vi = sh.get("vi", k); out = {}
    for oi_s, names in sh.get("pick", {}).items():
        oi = int(oi_s)
        if oi not in sh.get("members", []) or not names:
            continue
        M = np.zeros((n, H, W), bool)
        for nm in names:
            p = f"{root}/obj/{vi}/{nm}.mp4"
            if not os.path.exists(p):
                log("no mask file", p); continue
            for i, f in enumerate(read_frames(p)[:n]):
                if f.shape[:2] != (H, W):
                    f = cv2.resize(f, (W, H), interpolation=cv2.INTER_NEAREST)
                M[i] |= f[..., 1] > 127
        if M.any():
            out[oi] = M
    return out


def grow(M, px, tpad=1):
    """grow each frame's mask by px (ellipse) and over +-tpad frames"""
    if px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        M = np.stack([cv2.dilate(m.astype(np.uint8), k) > 0 for m in M])
    if tpad:
        G = M.copy()
        for d in range(1, tpad + 1):
            G[d:] |= M[:-d]; G[:-d] |= M[d:]
        M = G
    return M


def grow_rel(M, rel, box=False, tpad=0, minpx=3):
    """grow each frame's mask by rel x its own size (sqrt of the area for masks, box side for boxes); then +-tpad frames"""
    out = np.zeros_like(M); H, W = M.shape[1:]
    for i, m in enumerate(M):
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        if box:
            bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            gy, gx = max(minpx, int(rel * bh)), max(minpx, int(rel * bw))
            out[i, max(0, ys.min() - gy):min(H, ys.max() + gy + 1), max(0, xs.min() - gx):min(W, xs.max() + gx + 1)] = True
        else:
            g = max(minpx, int(rel * np.sqrt(len(xs))))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
            out[i] = cv2.dilate(m.astype(np.uint8), k) > 0
    return grow(out, 0, tpad) if tpad else out


def boxes(M, px):
    B = np.zeros_like(M); H, W = M.shape[1:]
    for i, m in enumerate(M):
        ys, xs = np.nonzero(m)
        if len(xs):
            B[i, max(0, ys.min() - px):min(H, ys.max() + px + 1), max(0, xs.min() - px):min(W, xs.max() + px + 1)] = True
    return B


def ref_sheet(paths, W, H):
    """the new objects side by side on white, at the video's size (WanVaceToVideo centre-crops to W x H)"""
    can = np.full((H, W, 3), 255, np.uint8); n = len(paths); cw = W // n; pad = int(min(cw, H) * 0.06)
    for j, p in enumerate(paths):
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[2] == 4:
            a = im[..., 3:] / 255.0; im = (im[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
        # crop to the object (non-white)
        nz = np.argwhere((im < 235).any(2))
        if len(nz):
            (y0, x0), (y1, x1) = nz.min(0), nz.max(0) + 1; im = im[y0:y1, x0:x1]
        s = min((cw - 2 * pad) / im.shape[1], (H - 2 * pad) / im.shape[0])
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
        y = (H - im.shape[0]) // 2; x = j * cw + (cw - im.shape[1]) // 2
        can[y:y + im.shape[0], x:x + im.shape[1]] = im
    return can


def vace_prompt(root, meta, k):
    """the shot's scene sentence from redoutline3's prompt, with the new objects named"""
    objs = meta["objs"]; txt = ""
    p = f"{root}/seg{k}_prompt.txt"
    if os.path.exists(p):
        s = open(p).read(); i = s.find("detailed_description:")
        if i >= 0:
            paras = [x.strip() for x in s[i + len("detailed_description:"):].split("\n\n") if x.strip()]
            txt = next((x for x in paras if "<Subject" in x and not x.startswith("Important")), paras[1] if len(paras) > 1 else "")
    if not txt:
        txt = "; ".join(f"the {o['replaces']} is now a {o['full']}" for o in objs)
    for j, o in enumerate(objs):
        txt = txt.replace(f"<Subject {j + 1}>", o["full"])
    txt = re.sub(r"<[^>]+>", "", txt)
    return "Realistic video, natural light and real camera footage. " + txt


def comfy_workflow(cfg, ctrl_name, mask_name, ref_name, prompt, W, H, L, seed, prefix, neg_extra="", strength=1.0):
    wf = {}
    nid = [0]

    def node(cls, **inputs):
        nid[0] += 1; wf[str(nid[0])] = {"class_type": cls, "inputs": inputs}; return str(nid[0])

    clip = node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default")
    pos = node("CLIPTextEncode", clip=[clip, 0], text=prompt)
    neg = node("CLIPTextEncode", clip=[clip, 0], text=NEG + ("，" + neg_extra if neg_extra else ""))
    vae = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    cv = node("GetVideoComponents", video=[node("LoadVideo", file=ctrl_name), 0])
    mv = node("GetVideoComponents", video=[node("LoadVideo", file=mask_name), 0])
    mk = node("ImageToMask", image=[mv, 0], channel="red")
    ref = node("LoadImage", image=ref_name)
    vace = node("WanVaceToVideo", positive=[pos, 0], negative=[neg, 0], vae=[vae, 0], width=W, height=H, length=L, batch_size=1,
                strength=strength, control_video=[cv, 0], control_masks=[mk, 0], reference_image=[ref, 0])

    def model(unet, dtype, lora, lw):
        m = node("UNETLoader", unet_name=unet, weight_dtype=dtype)
        if lora:
            m = node("LoraLoaderModelOnly", model=[m, 0], lora_name=lora, strength_model=lw)
        return node("ModelSamplingSD3", model=[m, 0], shift=cfg["shift"])

    if "high" in cfg:
        mh = model(cfg["high"], "default", cfg.get("lora_h"), cfg["lora_w"])
        ml = model(cfg["low"], "default", cfg.get("lora_l"), cfg["lora_w"])
        s1 = node("KSamplerAdvanced", model=[mh, 0], add_noise="enable", noise_seed=seed, steps=cfg["steps"], cfg=cfg["cfg"],
                  sampler_name="euler", scheduler="simple", positive=[vace, 0], negative=[vace, 1], latent_image=[vace, 2],
                  start_at_step=0, end_at_step=cfg["switch"], return_with_leftover_noise="enable")
        lat = node("KSamplerAdvanced", model=[ml, 0], add_noise="disable", noise_seed=seed, steps=cfg["steps"], cfg=cfg["cfg"],
                   sampler_name="euler", scheduler="simple", positive=[vace, 0], negative=[vace, 1], latent_image=[s1, 0],
                   start_at_step=cfg["switch"], end_at_step=10000, return_with_leftover_noise="disable")
    else:
        m = model(cfg["unet"], cfg["dtype"], cfg.get("lora"), cfg.get("lora_w", 1.0))
        lat = node("KSampler", model=[m, 0], seed=seed, steps=cfg["steps"], cfg=cfg["cfg"], sampler_name="uni_pc", scheduler="simple",
                   positive=[vace, 0], negative=[vace, 1], latent_image=[vace, 2], denoise=1.0)
    trim = node("TrimVideoLatent", samples=[lat, 0], trim_amount=[vace, 3])
    img = node("VAEDecode", samples=[trim, 0], vae=[vae, 0])
    node("SaveImage", images=[img, 0], filename_prefix=prefix)
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("k", type=int); ap.add_argument("out")
    ap.add_argument("--refs", required=True, help="object pictures in shots_meta 'objs' order, comma separated")
    ap.add_argument("--model", default="14b", choices=sorted(MODELS))
    ap.add_argument("--control", default="grey", choices=["grey", "plate", "src"])
    ap.add_argument("--region", default="mask", choices=["mask", "box"])
    ap.add_argument("--egrow", type=int, default=0, help="erase mask growth px (0 = W/48)")
    ap.add_argument("--igrow", type=int, default=0, help="insert region growth px (0 = W/40)")
    ap.add_argument("--igrow-rel", type=float, default=0, help="grow the insert region per object by this x its size (0 = fixed px)")
    ap.add_argument("--itpad", type=int, default=2, help="insert region also covers +-N frames")
    ap.add_argument("--sep-erase", action="store_true",
                    help="do not add the erase mask to the insert region: the erased plate covers the rest of the old object")
    ap.add_argument("--steps", type=int, default=0); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-erase", action="store_true", help="skip the eraser (joint uses the source)")
    ap.add_argument("--prompt", default="", help="override the VACE prompt")
    ap.add_argument("--strength", type=float, default=1.0, help="VACE context strength")
    ap.add_argument("--cfg", type=float, default=0); ap.add_argument("--lora-w", type=float, default=-1)
    ap.add_argument("--neg-old", action="store_true", help="name the old objects in the negative prompt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    meta = json.load(open(f"{a.root}/shots_meta.json"))
    src = read_frames(f"{a.root}/seg{a.k}_src.mp4"); n = len(src); H, W = src[0].shape[:2]
    if W % 16 or H % 16:
        raise SystemExit(f"size {W}x{H} is not a multiple of 16")
    refs = a.refs.split(",")
    masks = masks_for_shot(a.root, meta, a.k, n, H, W)
    if not masks:
        raise SystemExit("no picked instances in this shot")
    members = sorted(masks)
    log(f"shot {a.k}: {n} frames {W}x{H}, objects", [meta["objs"][oi]["word"] + " -> " + meta["objs"][oi]["full"] for oi in members])
    U = np.zeros((n, H, W), bool)
    for oi in members:
        U |= masks[oi]
    E = grow(U, a.egrow or W // 48, 1)
    if a.igrow_rel:
        I = np.zeros_like(U)
        for oi in members:  # per object, so a small ball gets a small margin and a racket a bigger one
            I |= grow_rel(masks[oi], a.igrow_rel, a.region == "box", a.itpad)
    else:
        I = grow(U, a.igrow or W // 40, a.itpad) if a.region == "mask" else grow(boxes(U, a.igrow or W // 40), 0, a.itpad)
    if not a.sep_erase:
        I |= E

    # 1. erase
    plate_p = f"{a.out}/plate.mp4"
    if a.no_erase or a.control == "src":
        plate = src
    else:
        if not os.path.exists(plate_p):
            write_video(f"{a.out}/erase_mask.mp4", [m.astype(np.uint8) * 255 for m in E], crf=0, gray=True)
            comfy.free(); time.sleep(2)
            log("erasing with MiniMax-Remover")
            subprocess.run([REMOVER_PY, f"{HERE}/remove_run.py", f"{a.root}/seg{a.k}_src.mp4", f"{a.out}/erase_mask.mp4", plate_p],
                           check=True)
        plate = read_frames(plate_p)[:n]
    log(f"erase done ({time.time() - t0:.0f} s)")

    # 2. insert
    cfg = dict(MODELS[a.model])
    if a.steps:
        cfg["steps"] = a.steps
    if a.cfg:
        cfg["cfg"] = a.cfg
    if a.lora_w >= 0:
        cfg["lora_w"] = a.lora_w
        if a.lora_w == 0:
            cfg["lora"] = None; cfg.pop("lora_h", None); cfg.pop("lora_l", None)
    neg_extra = ", ".join(dict.fromkeys(meta["objs"][oi]["samw"] for oi in members)) if a.neg_old else ""
    L = ((n - 1 + 3) // 4) * 4 + 1
    base = src if a.control == "src" else plate
    ctrl = []
    for i in range(L):
        f = base[min(i, n - 1)].copy()
        if a.control in ("grey", "src"):
            f[I[min(i, n - 1)]] = 128
        ctrl.append(f)
    write_video(f"{a.out}/vace_control.mp4", ctrl, crf=4)
    write_video(f"{a.out}/region.mp4", [I[min(i, n - 1)].astype(np.uint8) * 255 for i in range(L)], crf=0, gray=True)
    cv2.imwrite(f"{a.out}/reference.png", ref_sheet([refs[oi] for oi in members], W, H))
    prompt = a.prompt or vace_prompt(a.root, meta, a.k)
    open(f"{a.out}/vace_prompt.txt", "w").write(prompt)
    prefix = f"vace/{os.path.basename(os.path.normpath(a.out))}_{int(time.time())}"
    wf = comfy_workflow(cfg, comfy.stage(f"{a.out}/vace_control.mp4"), comfy.stage(f"{a.out}/region.mp4"),
                        comfy.stage(f"{a.out}/reference.png"), prompt, W, H, L, a.seed, prefix, neg_extra, a.strength)
    json.dump(wf, open(f"{a.out}/workflow_api.json", "w"), indent=1)
    log(f"VACE {a.model}: {L} frames, {cfg['steps']} steps")
    t1 = time.time(); last = [0]

    def prog(stage, frac, v, mx):
        if stage == "sampling" and time.time() - last[0] > 20:
            last[0] = time.time(); log(f"  sampling {v}/{mx}")
    files = comfy.run(wf, on_progress=prog)
    files = sorted(f for f in files if f.endswith(".png"))
    gen = [cv2.imread(f) for f in files][:n]
    for f in files:
        os.remove(f)
    if len(gen) < n:
        raise SystemExit(f"VACE gave {len(gen)} frames for {n}")
    gen = [cv2.resize(g, (W, H)) if g.shape[:2] != (H, W) else g for g in gen]
    write_video(f"{a.out}/vace_raw.mp4", gen)
    t_vace = time.time() - t1
    log(f"VACE done ({t_vace:.0f} s)")

    # 3. joint: source outside the region, VACE inside (soft edge inside the region)
    res = []
    for i in range(n):
        al = cv2.GaussianBlur(I[i].astype(np.float32), (0, 0), max(2, W // 160))
        al = np.minimum(al * 1.6, 1.0) * I[i]  # soft only on the inner side of the region edge
        al = cv2.GaussianBlur(al, (0, 0), 1.5)[..., None]
        bg = src[i].astype(np.float32)
        if plate is not src and (E[i] & ~I[i]).any():  # erased plate where the old object was but nothing is drawn
            ae = cv2.GaussianBlur(E[i].astype(np.float32), (0, 0), 2.0)[..., None]
            bg = bg * (1 - ae) + plate[i].astype(np.float32) * ae
        res.append((bg * (1 - al) + gen[i].astype(np.float32) * al).astype(np.uint8))
    write_video(f"{a.out}/result_silent.mp4", res)
    srcv = f"{a.root}/seg{a.k}_src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{a.out}/result_silent.mp4", "-i", srcv, "-map", "0:v", "-map", "1:a?",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", f"{a.out}/result.mp4"], check=True)
    panel = [np.hstack([src[i], plate[i] if plate is not src else src[i], res[i]]) for i in range(n)]
    write_video(f"{a.out}/panel.mp4", panel, crf=16)
    info = dict(root=a.root, k=a.k, model=a.model, cfg=cfg, control=a.control, region=a.region, frames=n, L=L, neg_extra=neg_extra,
                igrow_rel=a.igrow_rel, itpad=a.itpad, egrow=a.egrow, sep_erase=a.sep_erase,
                strength=a.strength,
                members=members, prompt=prompt, seconds=round(time.time() - t0), vace_seconds=round(t_vace))
    json.dump(info, open(f"{a.out}/vace_swap.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f} s -> {a.out}/result.mp4")


if __name__ == "__main__":
    main()
