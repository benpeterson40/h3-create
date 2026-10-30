#!/usr/bin/env python3
"""check_setup.py : checks everything H3 Create needs and says what is missing. Reads the same settings as the app (.env / environment).
  python scripts/check_setup.py"""
import json, os, shutil, subprocess, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_file = os.path.join(ROOT, ".env")
if os.path.exists(env_file):   # same rules as run.sh: KEY=VALUE lines, # comments
    for line in open(env_file):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.split(" #")[0].strip().strip('"').strip("'"))
sys.path.insert(0, ROOT)
from server import config as C

ok_all = True


def report(ok, what, hint=""):
    global ok_all
    ok_all &= bool(ok) or what.startswith("(optional)")
    print(("  OK   " if ok else "  --   ") + what + ("" if ok or not hint else "\n         -> " + hint))


def pyhas(py, code):
    try:
        return subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=600).returncode == 0
    except Exception:
        return False


print("Programs")
report(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg + ffprobe", "install FFmpeg (e.g. sudo apt install ffmpeg)")
report(os.path.exists(C.LLAMA_SERVER), f"llama-server: {C.LLAMA_SERVER}", "build llama.cpp with CUDA (README) or set H3C_LLAMA_SERVER")

print("ComfyUI")
try:
    with urllib.request.urlopen(C.COMFY_URL + "/object_info", timeout=60) as r:
        nodes = json.loads(r.read())
    report(True, f"running at {C.COMFY_URL}")
    need = ["UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "MiniMaxH3SigmaShift", "MiniMaxH3ReferenceToVideo", "MiniMaxH3AddGuide",
            "LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo", "SamplerCustomAdvanced", "BasicGuider", "BasicScheduler", "KSamplerSelect",
            "RandomNoise", "VAEDecodeAudio"]
    miss = [n for n in need if n not in nodes]
    report(not miss, "native MiniMax H3 + video nodes" + (": missing " + ", ".join(miss) if miss else ""), "update ComfyUI (git pull) - H3 support is built in")
    report("MiniMaxH3FunControlNetApply" in nodes and "ModelPatchLoader" in nodes, "(optional) H3 Fun ControlNet nodes (Edit Video 'Draw' mode)")
    report("H3StudioMaskedEditLatent" in nodes, "H3 masked-edit node (object fixes: H3 redraws only the boxes around a missed object)",
           "cp comfyui_nodes/h3studio_masked_edit.py <ComfyUI>/custom_nodes/ and restart ComfyUI")
except Exception as e:
    report(False, f"ComfyUI at {C.COMFY_URL}: {e}", "start ComfyUI (python main.py --listen 127.0.0.1 --port 8188) or set H3C_COMFY_URL")
report(os.path.isdir(os.path.join(C.COMFY_DIR, "input")), f"ComfyUI folder: {C.COMFY_DIR}", "set H3C_COMFY_DIR to your ComfyUI folder")

print("Models (python scripts/download_models.py fetches any that are missing)")
man = json.load(open(os.path.join(ROOT, "scripts", "models.json")))["files"]
for f in man:
    base = {"comfy": C.COMFY_DIR, "llm": None, "remover": C.REMOVER_DIR}[f["to"]]
    p = os.path.join(base, f["dest"]) if base else (C.LLM_MODEL if f["dest"].endswith("IQ4_XS.gguf") else C.LLM_MMPROJ)
    have = os.path.exists(p) and os.path.getsize(p) == f["size"]
    tag = "" if f["group"] == "core" else f"(optional, {f['group']}) "
    report(have, tag + p)

print("Python helpers")
report(pyhas(C.PY_APP, "import cv2, numpy, fastapi, uvicorn, websockets, multipart"), f"app: {C.PY_APP}", "pip install -r requirements.txt")
report(pyhas(C.PY_SAM3, "import torch, cv2; from transformers import Sam3VideoModel, Sam3VideoProcessor; assert torch.cuda.is_available()"),
       f"SAM 3 (Objects swap, Remove Background, Reframe): {C.PY_SAM3}", "install torch (CUDA) + pip install -r requirements-sam3.txt, or set H3C_PY_SAM3")
report(pyhas(C.PY_SAM3, "from huggingface_hub import hf_hub_download as d; d('facebook/sam3', 'config.json')"), "SAM 3 weights reachable (facebook/sam3 is gated)",
       "request access at https://huggingface.co/facebook/sam3, then run: hf auth login")
report(pyhas(C.PY_REMOVER, "import sys, os; sys.path.insert(0, %r); import torch, diffusers, cv2, einops; import transformer_minimax_remover" % C.REMOVER_DIR)
       and os.path.isdir(os.path.join(C.REMOVER_DIR, "weights", "transformer")),
       f"(optional) MiniMax-Remover (automatic object fixes): {C.REMOVER_DIR}", "see README: MiniMax-Remover")
report(pyhas(C.PY_MOGE, "from moge.model.v2 import MoGeModel"), f"(optional) MoGe-2 (Depth Map tool): {C.PY_MOGE}", "pip install -r requirements-moge.txt")

print("\nAll required parts are in place - start the app with ./run.sh" if ok_all else "\nSome required parts are missing (marked --).")
sys.exit(0 if ok_all else 1)
