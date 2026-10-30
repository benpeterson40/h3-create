"""h3.py : the MiniMax H3 Ref2VA workflow used for both Motion transfer and Objects swap (the recipe from the 2026-10-22 tests):
H3 Ref2VA int8 + akatz Character Swap LoRA v1.5 (strength 1) + fl2v turbo 8-step LoRA, SigmaShift 12/3, er_sde + simple, 8 steps, cfg 1.
<Video 1> = the 24 fps motion clip, <Picture n> = reference pictures; guides pin a picture or a short clip at a frame (MiniMaxH3AddGuide).
The graph builders are in workflows.py (native ComfyUI nodes only)."""
import importlib.util, os, uuid
from . import comfy

from . import workflows as W

SWAP_LORA = "MiniMaxH3/CharacterSwap/h3_character_swap_v1.5.safetensors"
TURBO_FL8 = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def _relink(wf, old, new, skip):
    for k, node in wf.items():
        if k == skip:
            continue
        for name, v in node["inputs"].items():
            if v == old:
                node["inputs"][name] = new


DMAD = "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors"


def _speed(wf):
    """H3_SPEED=dmad (a test switch, 2026-10-27; off unless set): Kijai's DMAD 4-step distillation LoRA in place of the 8-step turbo, as its
    model card runs it - lcm sampler, simple scheduler, 4 steps, shift 12/2 (H3_SPEED_STEPS / H3_SPEED_SAMPLER override)"""
    if os.environ.get("H3_SPEED", "") == "dmad":
        wf["lora"]["inputs"]["lora_name"] = DMAD
        wf["sampler"]["inputs"]["sampler_name"] = os.environ.get("H3_SPEED_SAMPLER", "lcm")
        wf["sched"]["inputs"]["steps"] = int(os.environ.get("H3_SPEED_STEPS", "4"))
        if "shift" in wf:
            wf["shift"]["inputs"]["shift_audio"] = 2.0
    return wf


def build(prompt, video, images, w, h, length, seed, guides=(), swap_strength=1.0):
    """video / images / guide files are local paths (staged into ComfyUI's input dir here)."""
    wf = W.h3_reference(prompt, w, h, length, [comfy.stage(p) for p in images], [], seed, True, "match", f"h3create/{uuid.uuid4().hex[:8]}",
                        videos=[comfy.stage(video)], turbo_version="v1.0-8step", shift=(12.0, 3.0), steps=8)
    wf["lora"]["inputs"]["lora_name"] = TURBO_FL8
    wf["swap"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["lora", 0], "lora_name": SWAP_LORA, "strength_model": float(swap_strength)}}
    _relink(wf, ["lora", 0], ["swap", 0], "swap")
    wf["sampler"]["inputs"]["sampler_name"] = "er_sde"
    wf["sched"]["inputs"]["scheduler"] = "simple"
    cond = ["r2v", 0]
    for i, g in enumerate(guides):
        if g.get("video"):
            wf[f"gvid{i}"] = {"class_type": "LoadVideo", "inputs": {"file": comfy.stage(g["video"])}}
            wf[f"gimg{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"gvid{i}", 0]}}
        else:
            wf[f"gimg{i}"] = {"class_type": "LoadImage", "inputs": {"image": comfy.stage(g["image"])}}
        wf[f"guide{i}"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": cond, "vae": ["vvae", 0], "latent": ["r2v", 1],
                                                                          "image": [f"gimg{i}", 0], "frame_idx": int(g.get("frame", 0))}}
        cond = [f"guide{i}", 0]
    if cond != ["r2v", 0]:
        wf["guider"]["inputs"]["conditioning"] = cond
    return _speed(wf)


FUN_PATCH = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"


def add_inpaint(wf, source_video, mask_video, strength=1.0):
    """H3 Fun ControlNet Union inpaint: only the white part of mask_video is regenerated; the rest follows source_video
    (both clips the render's length and canvas). Wraps whatever model the sampler already uses (LoRAs included)."""
    model = wf["guider"]["inputs"]["model"]
    for n, path in (("isrc", source_video), ("imask", mask_video)):
        wf[f"v_{n}"] = {"class_type": "LoadVideo", "inputs": {"file": comfy.stage(path)}}
        wf[f"c_{n}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"v_{n}", 0]}}
    wf["m_imask"] = {"class_type": "ImageToMask", "inputs": {"image": ["c_imask", 0], "channel": "red"}}
    wf["patch"] = {"class_type": "ModelPatchLoader", "inputs": {"name": FUN_PATCH}}
    wf["fun"] = {"class_type": "MiniMaxH3FunControlNetApply", "inputs": {"model": model, "model_patch": ["patch", 0], "vae": ["vvae", 0], "strength": float(strength),
                                                                          "start_percent": 0.0, "end_percent": 1.0, "mask": ["m_imask", 0], "source_video": ["c_isrc", 0]}}
    wf["guider"]["inputs"]["model"] = ["fun", 0]
    if "model" in wf.get("sched", {}).get("inputs", {}):
        wf["sched"]["inputs"]["model"] = ["fun", 0]
    return wf


def build_gen(prompt, images, w, h, length, seed, guides=()):
    """plain H3 generation (no reference video) from the prompt + pictures, with guide clips/pictures pinned at frames - Extend Video"""
    wf = W.h3_reference(prompt, w, h, length, [comfy.stage(p) for p in images], [], seed, True, "match", f"h3create/{uuid.uuid4().hex[:8]}",
                        videos=[], turbo_version="v1.0-8step", shift=(12.0, 3.0), steps=8)
    wf["lora"]["inputs"]["lora_name"] = TURBO_FL8
    wf["sampler"]["inputs"]["sampler_name"] = "er_sde"
    wf["sched"]["inputs"]["scheduler"] = "simple"
    cond = ["r2v", 0]
    for i, g in enumerate(guides):
        if g.get("video"):
            wf[f"gvid{i}"] = {"class_type": "LoadVideo", "inputs": {"file": comfy.stage(g["video"])}}
            wf[f"gimg{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"gvid{i}", 0]}}
        else:
            wf[f"gimg{i}"] = {"class_type": "LoadImage", "inputs": {"image": comfy.stage(g["image"])}}
        wf[f"guide{i}"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": cond, "vae": ["vvae", 0], "latent": ["r2v", 1],
                                                                          "image": [f"gimg{i}", 0], "frame_idx": int(g.get("frame", 0))}}
        cond = [f"guide{i}", 0]
    if cond != ["r2v", 0]:
        wf["guider"]["inputs"]["conditioning"] = cond
    return _speed(wf)


def add_control(wf, control_video, strength=1.0):
    """H3 Fun ControlNet Union with a control video (depth, edges or pose): the render follows its structure. Wraps the sampler's model."""
    model = wf["guider"]["inputs"]["model"]
    wf["v_ctl"] = {"class_type": "LoadVideo", "inputs": {"file": comfy.stage(control_video)}}
    wf["c_ctl"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["v_ctl", 0]}}
    wf["patch_ctl"] = {"class_type": "ModelPatchLoader", "inputs": {"name": FUN_PATCH}}
    wf["fun_ctl"] = {"class_type": "MiniMaxH3FunControlNetApply", "inputs": {"model": model, "model_patch": ["patch_ctl", 0], "vae": ["vvae", 0],
                                                                              "strength": float(strength), "start_percent": 0.0, "end_percent": 1.0,
                                                                              "control_video": ["c_ctl", 0]}}
    wf["guider"]["inputs"]["model"] = ["fun_ctl", 0]
    if "model" in wf.get("sched", {}).get("inputs", {}):
        wf["sched"]["inputs"]["model"] = ["fun_ctl", 0]
    return wf
