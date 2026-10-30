"""ComfyUI API-format workflow builders for MiniMax H3 and Qwen-Image 2.1.

Mirrors the official Comfy-Org templates (video_minimax_h3_r2v / t2v / i2v,
image_qwen_image_2_1_t2i / image_edit) using only native ComfyUI nodes.
"""
import math

H3_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
H3_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
H3_REF_TURBO = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
H3_REF_TURBO8 = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"  # LightX2V, 768p family (trained shift 6/3)
REF_TURBOS = {"v0.1": (H3_REF_TURBO, 4), "v1.0-8step": (H3_REF_TURBO8, 8)}
H3_FL_TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
DEFAULT_STEPS = 20  # 16 garbled speech in testing (the official guide: speech converges last); 16 is the "fast" preset

QWEN_MODEL = "qwen_image_2.1_int8_convrot.safetensors"
QWEN_TEXT_ENCODER = "qwen3vl_8b_int8_convrot.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
QWEN_TURBO = "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"  # official 8-step Turbo; 3x faster, same identity in testing
QWEN_STEPS = 8

REQUIRED_MODELS = {
    "diffusion_models": [H3_REF2VA, H3_FL2VA, QWEN_MODEL],
    "text_encoders": [H3_TEXT_ENCODER, QWEN_TEXT_ENCODER],
    "vae": [H3_VIDEO_VAE, H3_AUDIO_VAE, QWEN_VAE],
    "loras": [H3_REF_TURBO, H3_FL_TURBO, QWEN_TURBO],
}

ASPECTS = {
    "16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3),
    "3:4": (3, 4), "21:9": (21, 9), "3:2": (3, 2), "2:3": (2, 3),
}


def snap(v, m=32):
    return max(m, int(round(v / m)) * m)


# H3's native canvas at 768p is 1344x768 (0.98 MP); 1376x768 exceeds its pixel-area cap (official ComfyUI guide), so
# sizes are floored to the 32-px grid and, for very wide or tall ratios, shrunk to fit 1.75 x short_edge^2.
def _fit(ratio, short_edge):
    """(long, short) edge for a long/short ratio >= 1."""
    cap = 1.75 * short_edge * short_edge
    long_ = max(32, int(short_edge * ratio) // 32 * 32)
    if long_ * short_edge > cap:
        trimmed = int(cap / short_edge) // 32 * 32
        if trimmed / short_edge >= ratio * 0.97:  # e.g. 1376x768 -> 1344x768: a 2% narrower crop keeps the full height
            return trimmed, short_edge
        short_edge = max(32, int(math.sqrt(cap / ratio)) // 32 * 32)
        long_ = max(32, int(short_edge * ratio) // 32 * 32)
    return long_, short_edge


def size_for(aspect, short_edge):
    """Width/height (multiples of 32) for an aspect ratio and short-edge size."""
    a, b = ASPECTS.get(aspect, (16, 9))
    long_, short = _fit(max(a, b) / min(a, b), short_edge)
    return (long_, short) if a >= b else (short, long_)


def size_from_image(w, h, short_edge):
    long_, short = _fit(max(w, h) / min(w, h), short_edge)
    return (long_, short) if w >= h else (short, long_)


def frames_for(seconds):
    """Same formula as the official template: snap to H3's 17k+5 frame grid at 24 fps."""
    f = max(5, round(seconds * 24))
    return f + (5 - f % 17) % 17


H3_LATENT_UPSCALER = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"  # LBH-123-AI, via the xmarre Plus node


def _h3_upscale_refine(wf, model_ref, cond_ref, latent_ref, width, height, steps, denoise, seed):
    """Learned latent upscale of pass 1 to width x height, then a short low-noise H3 pass at full size
    (Comfyui_Minimax_h3_latent_Upscaler-Plus). Pass-1 audio is kept exactly (lock_audio)."""
    wf["sched2"] = {"class_type": "BasicScheduler",
                    "inputs": {"model": model_ref, "scheduler": "simple", "steps": steps, "denoise": denoise}}
    wf["noise2"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 1}}
    wf["upref"] = {"class_type": "MinimaxH3LatentUpscaler3DRefineHandoff", "inputs": {
        "latent": latent_ref, "noise": ["noise2", 0], "sampler": ["sampler", 0], "sigmas": ["sched2", 0],
        "model_name": H3_LATENT_UPSCALER, "mode": "target dimensions", "scale": 1.5, "width": width, "height": height,
        "megapixels": 1.0, "align": 32, "keep_proportion": False, "lock_audio": True, "cfg": 1.0,
        "device": "cuda", "precision": "fp16", "offload_after_upscale": False,
        "model": model_ref, "positive": cond_ref}}
    return ["upref", 0]


def _h3_tail(wf, model_ref, cond_ref, latent_ref, steps, seed, prefix, refine=None):
    wf["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    wf["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    wf["sched"] = {"class_type": "BasicScheduler",
                   "inputs": {"model": model_ref, "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    wf["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": cond_ref}}
    wf["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
        "sigmas": ["sched", 0], "latent_image": latent_ref}}
    final = ["sample", 0]
    if refine:  # dict(cond=..., width=..., height=..., steps=..., denoise=...)
        final = _h3_upscale_refine(wf, model_ref, refine["cond"], final, refine["width"], refine["height"],
                                   refine["steps"], refine["denoise"], seed)
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": final, "vae": ["vvae", 0]}}
    wf["decode_audio"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": final, "vae": ["avae", 0]}}
    wf["video"] = {"class_type": "CreateVideo",
                   "inputs": {"images": ["decode", 0], "audio": ["decode_audio", 0], "fps": 24.0}}
    wf["save"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["video", 0], "filename_prefix": prefix, "format": "auto", "format.codec": "auto"}}
    return wf


def _h3_base(unet, turbo_lora, turbo, attention=None, shift=None):
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": H3_TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
    }
    model = ["unet", 0]
    if turbo:
        wf["lora"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model, "lora_name": turbo_lora, "strength_model": 1.0}}
        model = ["lora", 0]
    if shift:  # (video, audio) flow shifts, e.g. (6, 3) for the 768p turbo family
        wf["shift"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": float(shift[0]), "shift_audio": float(shift[1])}}
        model = ["shift", 0]
    if attention:  # per-render backend (custom_nodes/h3studio_attention.py): pytorch | sage | sage3
        wf["attn"] = {"class_type": "H3StudioAttention", "inputs": {"model": model, "backend": attention}}
        model = ["attn", 0]
    return wf, model


def h3_reference(prompt, width, height, length, images, audios, seed, turbo, ref_image_size, prefix, videos=(), steps=None, attention=None,
                 upscale=None, refine_steps=6, refine_denoise=0.35, turbo_version="v0.1", shift=None):
    """upscale (e.g. 1.5): pass 1 runs at width/upscale x height/upscale, then the learned latent upscaler and a short
    refine pass bring it to width x height; the refine pass gets its own full-size reference conditioning."""
    """Reference-to-video (Ref2VA): up to 9 images and 3 audio clips (filenames in ComfyUI input dir)."""
    lora, turbo_steps = REF_TURBOS.get(turbo_version, REF_TURBOS["v0.1"])
    wf, model = _h3_base(H3_REF2VA, lora, turbo, attention, shift)
    r2v = {"clip": ["clip", 0], "vae": ["vvae", 0], "audio_vae": ["avae", 0], "prompt": prompt,
           "width": width, "height": height, "length": length, "ref_image_size": ref_image_size}
    for i, name in enumerate(images):
        wf[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        r2v[f"ref_images.ref_image_{i}"] = [f"img{i}", 0]
    for i, name in enumerate(audios):
        wf[f"aud{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        r2v[f"ref_audios.ref_audio_{i}"] = [f"aud{i}", 0]
    for i, name in enumerate(videos):  # motion references (frames only)
        wf[f"vid{i}"] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        wf[f"vidc{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"vid{i}", 0]}}
        r2v[f"ref_videos.ref_video_{i}"] = [f"vidc{i}", 0]
    wf["r2v"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": r2v}
    steps = (steps or turbo_steps) if turbo else (steps or DEFAULT_STEPS)
    if not upscale:
        return _h3_tail(wf, model, ["r2v", 0], ["r2v", 1], steps, seed, prefix)
    lw, lh = (max(256, int(v / upscale) // 32 * 32) for v in (width, height))
    wf["r2v_lo"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {**r2v, "width": lw, "height": lh}}
    return _h3_tail(wf, model, ["r2v_lo", 0], ["r2v_lo", 1], steps, seed, prefix,
                    refine={"cond": ["r2v", 0], "width": width, "height": height, "steps": refine_steps, "denoise": refine_denoise})


def h3_continuum(prompts, width, height, chunk_seconds, images, audios, seed, prefix, steps=None, attention=None,
                 continuity="Balanced — 22 frames", turbo=False, turbo_version="v1.0-8step", shift=None,
                 reference_size="Match Output"):
    """A multi-chunk story in one job with ComfyUI-H3-Continuum (Ref2VA): each chunk continues from the previous chunk's
    last 22 frames of video and its audio (instead of a single last frame), with the same reference pictures and voices
    for every chunk. prompts: one full H3 prompt per chunk (List mode, separated by ---)."""
    lora, turbo_steps = REF_TURBOS.get(turbo_version, REF_TURBOS["v1.0-8step"])
    wf, model = _h3_base(H3_REF2VA, lora, turbo, attention, shift)
    steps = steps or (turbo_steps if turbo else None)
    refs = {"reference_use": "All chunks", **{f"reference_r{i}_chunks": "off" for i in range(1, 10)}}
    for i, name in enumerate(images[:9]):
        wf[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        refs[f"reference_image_{i + 1}"] = [f"img{i}", 0]
    wf["refs"] = {"class_type": "H3ContinuumReferenceImagesV39", "inputs": refs}
    auds = {"reference_audio_vae": ["avae", 0]}
    for i, name in enumerate(audios[:3]):
        wf[f"aud{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        auds[f"reference_audio_{i + 1}"] = [f"aud{i}", 0]
    wf["auds"] = {"class_type": "H3ContinuumReferenceAudios", "inputs": auds}
    wf["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    wf["sched"] = {"class_type": "BasicScheduler",
                   "inputs": {"model": model, "scheduler": "simple", "steps": steps or DEFAULT_STEPS, "denoise": 1.0}}
    wf["cont"] = {"class_type": "H3ContinuumSamplerV39", "inputs": {
        "model": model, "clip": ["clip", 0], "video_vae": ["vvae", 0], "sampler": ["sampler", 0], "sigmas": ["sched", 0],
        "sequence_prompt": "\n---\n".join(p.strip() for p in prompts), "prompt_mode": "List",
        "chunks": len(prompts), "chunk_seconds": float(chunk_seconds),
        "aspect": "Landscape 16:9", "preset": "Custom", "custom_mp": round(width * height / 1048576, 3),
        "continuity": continuity, "base_seed": seed, "audio_continuity": True, "diagnostics": "Basic",
        "reroll_from_chunk": "Auto", "reroll_nonce": 0, "strict_compatibility": True, "debug": False, "show_preview": False,
        "run_storage": "Off", "run_name": "", "reference_size": reference_size, "project_id": "",
        "video_reference_size": "Efficient - 0.4 MP", "continuation_backend": "Standard", "generation_mode": "Full Run",
        "review_action": "Continue / Next", "take_group": 0, "take_revision_id": "", "take_action": "Automatic",
        "size_source": "Manual", "width": width, "height": height,
        "audio_vae": ["avae", 0], "audio_references": ["auds", 0], "reference_images": ["refs", 0]}}
    wf["decode"] = {"class_type": "H3DecodeCacheHelper", "inputs": {
        "cache_mode": "Off", "ram_budget_mb": 256, "disk_budget_gb": 8, "reset_token": 0,
        "video_samples": ["cont", 0], "video_vae": ["vvae", 0], "audio_samples": ["cont", 1], "audio_vae": ["avae", 0]}}
    wf["final"] = {"class_type": "H3ContinuumAssembleSeamV35", "inputs": {
        "images": ["decode", 0], "audio": ["decode", 1], "assembly_plan": ["cont", 2], "exact_total_duration": True,
        "audio_seam": "Auto", "video_seam": "Auto", "buffer_backend": "Auto", "diagnostics": "Basic"}}
    wf["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["final", 0], "audio": ["final", 1], "fps": 24.0}}
    wf["save"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["video", 0], "filename_prefix": prefix, "format": "auto", "format.codec": "auto"}}
    return wf


def h3_first_frame(prompt, width, height, length, first_frame, seed, turbo, prefix, steps=None, attention=None, last_frame=None):
    """Text-to-video (no first_frame) or image-to-video (FL2VA); with last_frame the clip also ends on that picture."""
    wf, model = _h3_base(H3_FL2VA, H3_FL_TURBO, turbo, attention)
    i2v = {"clip": ["clip", 0], "vae": ["vvae", 0], "prompt": prompt,
           "width": width, "height": height, "length": length}
    if first_frame:
        wf["first"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        i2v["first_frame"] = ["first", 0]
    if last_frame:
        wf["last"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
        i2v["last_frame"] = ["last", 0]
    wf["i2v"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": i2v}
    steps = 6 if turbo else (steps or DEFAULT_STEPS)
    return _h3_tail(wf, model, ["i2v", 0], ["i2v", 1], steps, seed, prefix)


def qwen_image(prompt, width, height, ref_images, seed, prefix, negative="", steps=None, turbo=True):
    """Qwen-Image 2.1 text-to-image, or multi-reference edit when ref_images is non-empty.

    turbo: the official Turbo LoRA at 8 steps, CFG 1, linear_quadratic (matches the checkpoint's baked sigmas, no shift);
    off: the base model at 25 steps with the simple scheduler.
    """
    steps = steps or (QWEN_STEPS if turbo else 25)
    enc = {"clip": ["clip", 0], "prompt": prompt, "negative_prompt": negative,
           "resolution": 1024, "vae": ["vae", 0]}
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_MODEL, "weight_dtype": "default"}},
        "cache": {"class_type": "QwenImage21Cache",
                  "inputs": {"model": ["lora", 0] if turbo else ["unet", 0], "device": "auto", "dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": QWEN_TEXT_ENCODER, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
    }
    if turbo:
        wf["lora"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": ["unet", 0], "lora_name": QWEN_TURBO, "strength_model": 1.0}}
    for i, name in enumerate(ref_images[:16]):
        wf[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        enc[f"images.image_{i + 1}"] = [f"img{i}", 0]
    wf["enc"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
    if ref_images:
        latent = ["enc", 2]  # edit: sample from the encoder's latent, as the docs advise
    else:
        wf["empty"] = {"class_type": "EmptyLatentImage",
                       "inputs": {"width": width, "height": height, "batch_size": 1}}
        latent = ["empty", 0]
    wf["ksampler"] = {"class_type": "KSampler", "inputs": {
        "model": ["cache", 0], "positive": ["enc", 0], "negative": ["enc", 1], "latent_image": latent,
        "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
        "scheduler": "linear_quadratic" if turbo else "simple",
        "denoise": 1.0}}
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ksampler", 0], "vae": ["vae", 0]}}
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return wf
