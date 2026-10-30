"""shotcheck.py : catch generated test shots that play BACKWARDS. The shot text says which way the camera travels (following from
behind / point of view = forward, so the ground streams toward the camera; leading a subject that comes toward the camera = backward,
so the ground recedes); optical flow on the two lower corners of the frame (away from the subject) says which way it really moves.
Only clear cases are judged: a shot with no camera-travel words, or a flow under 0.3 px per frame, passes."""
import re
import cv2, numpy as np

FORWARD = re.compile(r"rear-following|following shot from behind|from behind|follows? .{0,40}from behind|point of view|first-person|\bpov\b|front-seat", re.I)
BACKWARD = re.compile(r"toward(?:s)? the camera|front[- ]tracking|front-facing tracking", re.I)


def expected(text):
    f, b = bool(FORWARD.search(text or "")), bool(BACKWARD.search(text or ""))
    return "forward" if f and not b else "backward" if b and not f else None


def ground_flow(path, step=2):
    """median vertical optical flow (px/frame at 320 px wide) of the lower corners: + = the ground streams down, toward the camera"""
    cap = cv2.VideoCapture(path); F = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        F.append(cv2.cvtColor(cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY))
    if len(F) < 3:
        return 0.0
    h = F[0].shape[0]; vals = []
    for i in range(step, len(F), step):
        fl = cv2.calcOpticalFlowFarneback(F[i - step], F[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        band = fl[int(h * .78):, :, 1]
        vals.append(float(np.median(np.concatenate([band[:, :80].ravel(), band[:, 240:].ravel()]))) / step)
    return float(np.median(vals))


def check(path, text, thr=0.3):
    """-> (ok, expected direction or None, measured flow)"""
    exp = expected(text); v = ground_flow(path)
    bad = (exp == "forward" and v < -thr) or (exp == "backward" and v > thr)
    return (not bad), exp, round(v, 2)


ANATOMY = """Count body parts in frames of an AI-generated video. For every frame and every person or animal in it: count each separate LEG you
can see entering the frame or attached to the body (a thigh or shin you can follow, bare or clothed, kneeling or standing) and each separate
FOOT. One person has at most 2 legs and 2 feet; an animal at most 4. Reply with JSON only:
{"frames": [{"i": index, "legs": n, "feet": n, "people": n, "too_many": true or false, "how": "which legs and feet you counted"}]}"""


def anatomy(path, n=4):
    """-> list of (frame index, problem) for frames where a body has extra or merged limbs (Qwen3.8 via the H3 Create server's llm module)"""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from server import llm
    cap = cv2.VideoCapture(path); F = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        F.append(f)
    if not F:
        return []
    idx = sorted({int(round(j * (len(F) - 1) / (n - 1))) for j in range(n)})
    content = []
    for i in idx:
        content += [{"type": "text", "text": f"Frame {i}:"}, llm.img_part(F[i], 768)]
    r = llm.ask(content, system=ANATOMY, max_tokens=600, temperature=0.0)
    return [(int(x.get("i", -1)), str(x.get("how"))[:200]) for x in (r.get("frames") or []) if isinstance(x, dict) and x.get("too_many") is True]


def stop_orphan_llm():
    """a script that used the llm module left llama-server running (llm.down() only stops a server THIS process started): stop servers
    whose parent is gone (parent pid 1) - the app's own server keeps its parent"""
    import subprocess, os, signal
    out = subprocess.run(["ps", "-eo", "pid,ppid,cmd"], capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1] == "1" and "llama-server" in parts[2]:
            try:
                os.kill(int(parts[0]), signal.SIGTERM)
            except OSError:
                pass
