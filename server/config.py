"""config.py : every machine-specific setting in one place. Each value can be overridden with an environment variable
(run.sh loads a .env file next to it first; see .env.example). Paths may start with ~."""
import os, sys


def _path(var, default):
    return os.path.abspath(os.path.expanduser(os.environ.get(var) or default))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # the repo folder
DATA = _path("H3C_DATA", os.path.join(ROOT, "data"))                         # uploads, jobs, logs (created on first run)
LIBRARY = _path("H3C_LIBRARY", os.path.join(ROOT, "examples", "library"))    # clips offered in the Motion Library (*.mp4)

# ComfyUI: the app talks to a running ComfyUI over its HTTP + websocket API and copies inputs into its input folder
COMFY_URL = (os.environ.get("H3C_COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = _path("H3C_COMFY_DIR", "~/ComfyUI")

# the director / checker LLM: Qwen3.8-27B (vision) served by llama.cpp's llama-server, started on demand
LLAMA_SERVER = _path("H3C_LLAMA_SERVER", "~/llama.cpp/build/bin/llama-server")
LLM_MODEL = _path("H3C_LLM_MODEL", "~/models/qwen38/Qwen3.8-27B-UD-IQ4_XS.gguf")
LLM_MMPROJ = _path("H3C_LLM_MMPROJ", "~/models/qwen38/mmproj-Q8_0.gguf")
LLM_PORT = int(os.environ.get("H3C_LLM_PORT") or 8091)

# Python interpreters for the helper processes (one environment can serve all of them; see README)
PY_APP = _path("H3C_PY", sys.executable)                 # the app's own environment (OpenCV, NumPy)
PY_SAM3 = _path("H3C_PY_SAM3", PY_APP)                   # needs torch + transformers with SAM 3 (facebook/sam3)
PY_MOGE = _path("H3C_PY_MOGE", PY_APP)                   # needs torch + MoGe-2 (Depth Map tool only)
REMOVER_DIR = _path("H3C_REMOVER_DIR", "~/minimax-remover")                          # MiniMax-Remover checkout (object fixes)
PY_REMOVER = _path("H3C_PY_REMOVER", os.path.join(REMOVER_DIR, ".venv", "bin", "python"))
