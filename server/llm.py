"""llm.py : Qwen3.8-27B (vision) through llama.cpp's llama-server, started on demand. The 5080 can't hold the LLM and H3 at once, so
up() frees ComfyUI's models first and down() stops the server before the next render. ask() sends text + images (OpenAI format) and
returns the reply (JSON when json_mode)."""
import base64, json, os, subprocess, threading, time, urllib.request
import cv2
from . import comfy

from .config import LLAMA_SERVER as BIN, LLM_MODEL as MODEL, LLM_MMPROJ as MMPROJ, LLM_PORT as PORT, DATA   # mmproj Q8_0: the F16 one leaves no room for the vision buffers on 16 GB
URL = f"http://127.0.0.1:{PORT}"
CTX = int(os.environ.get("H3C_LLM_CTX", "16384"))
LOG = os.path.join(DATA, "llm.log")
_lock = threading.RLock()
_proc = None
NAME = "Qwen3.8-27B"


def available():
    return all(os.path.exists(p) for p in (BIN, MODEL, MMPROJ))


def _health():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def up(timeout=300):
    """start llama-server (after freeing ComfyUI's VRAM) and wait until the model is loaded"""
    global _proc
    with _lock:
        if _proc and _proc.poll() is None and _health():
            return
        t_wait = time.time()
        while time.time() - t_wait < 3 * 3600:  # never load while ComfyUI is rendering something (H3 Studio shares the card)
            try:
                if comfy.busy() == 0:
                    break
            except Exception:
                break
            time.sleep(5)
        comfy.free(); time.sleep(2)
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        # no -ngl: llama.cpp's fit puts as many layers on the 5080 as fit with a 700 MiB margin (all-on-GPU runs out of memory at 16k context)
        args = [BIN, "-m", MODEL, "--mmproj", MMPROJ, "-c", str(CTX), "--fit-target", "700", "--host", "127.0.0.1", "--port", str(PORT),
                "-fa", "on", "--jinja", "-np", "1", "-ctk", "q8_0", "-ctv", "q8_0"]
        _proc = subprocess.Popen(args, stdout=open(LOG, "ab"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if _proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({_proc.returncode}); see {LOG}")
            if _health():
                return
            time.sleep(1)
        down(); raise RuntimeError("llama-server did not become ready")


def down():
    """stop llama-server so the GPU is free for H3"""
    global _proc
    with _lock:
        if _proc and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _proc.kill(); _proc.wait()
        _proc = None


def img_part(img, max_side=768):
    """an image (path or BGR array) -> OpenAI image_url part (JPEG, longest side <= max_side)"""
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]; s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}}


def ask(content, system=None, max_tokens=3000, json_mode=True, temperature=0.2):
    """content: list of parts ({'type':'text','text':...} / img_part(...)) or a string. Returns text (or parsed JSON)."""
    up()
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    text = out["choices"][0]["message"]["content"].strip()
    if not json_mode:
        return text
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[1] if "\n" in text else text
    a, b = text.find("{"), text.rfind("}")
    return json.loads(text[a:b + 1])
