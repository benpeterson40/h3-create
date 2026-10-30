"""comfy.py : small ComfyUI client for the video server (:8188) - stage inputs, run a workflow with live sampler progress over the
websocket, fetch the output file, free VRAM, interrupt."""
import json, os, shutil, time, uuid, urllib.request
from websockets.sync.client import connect

from .config import COMFY_URL as API, COMFY_DIR
WS = API.replace("https://", "wss://").replace("http://", "ws://") + "/ws"


class Cancelled(Exception):
    pass


def _post(path, body):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def _get(path):
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return json.loads(r.read())


def stage(path):
    """copy a local file into ComfyUI/input under a unique name; returns the name the loader nodes use"""
    name = f"h3c_{uuid.uuid4().hex[:10]}{os.path.splitext(path)[1].lower()}"
    shutil.copy(path, os.path.join(COMFY_DIR, "input", name))
    return name


def free():
    """unload ComfyUI's cached models so another GPU program (the LLM) has the card"""
    try:
        _post("/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass


def interrupt(prompt_id=None):
    try:
        if prompt_id:
            _post("/queue", {"delete": [prompt_id]})
        _post("/interrupt", {})
    except Exception:
        pass


def busy():
    q = _get("/queue")
    return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))


def _outputs(entry):
    files = []
    for nd in entry.get("outputs", {}).values():
        for k in ("videos", "gifs", "images"):
            for x in nd.get(k, []):
                if x.get("type", "output") != "output":
                    continue
                files.append(os.path.join(COMFY_DIR, "output", x.get("subfolder", ""), x["filename"]))
    return files


def run(wf, on_progress=None, cancelled=lambda: False, timeout=4 * 3600):
    """queue wf, report (stage, fraction) via on_progress, return the output file paths. Raises Cancelled / RuntimeError."""
    cid = "h3create-" + uuid.uuid4().hex[:8]
    with connect(f"{WS}?clientId={cid}", max_size=None, open_timeout=30) as ws:
        pid = _post("/prompt", {"prompt": wf, "client_id": cid})["prompt_id"]
        t0 = time.time(); last_poll = 0
        while True:
            if cancelled():
                interrupt(pid); raise Cancelled()
            if time.time() - t0 > timeout:
                interrupt(pid); raise RuntimeError("timed out")
            try:
                msg = ws.recv(timeout=5)
            except TimeoutError:
                msg = None
            if isinstance(msg, str):
                m = json.loads(msg); d = m.get("data", {})
                if m["type"] == "progress" and d.get("prompt_id") == pid and on_progress:
                    on_progress("sampling", d["value"] / max(1, d["max"]), d["value"], d["max"])
                elif m["type"] == "status" and on_progress:
                    q = d.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
                    on_progress("queue", None, q, None)
                elif m["type"] == "execution_error" and d.get("prompt_id") == pid:
                    raise RuntimeError(f"{d.get('exception_type', 'error')}: {d.get('exception_message', '')}"[:600])
                elif m["type"] == "executing" and d.get("prompt_id") == pid and d.get("node") is None:
                    break
                elif m["type"] == "execution_interrupted" and d.get("prompt_id") == pid:
                    raise Cancelled()
            if time.time() - last_poll > 15:  # websocket messages can be missed; the history is the truth
                last_poll = time.time(); h = _get(f"/history/{pid}")
                if pid in h:
                    break
    for _ in range(30):
        h = _get(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                msgs = [x for x in st.get("messages", []) if x[0] == "execution_error"]
                raise RuntimeError((msgs[-1][1].get("exception_message", "error") if msgs else "error")[:600])
            return _outputs(h[pid])
        time.sleep(1)
    raise RuntimeError("finished but no history entry")
