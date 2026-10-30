#!/usr/bin/env python3
"""download_models.py : fetch the model files H3 Create was tested with, straight from their Hugging Face pages (resumable; size and
sha256 checked). Nothing is downloaded that is already in place with the right size.

  python scripts/download_models.py --comfy ~/ComfyUI --llm ~/models/qwen38                  # core: H3 + Character Swap LoRA + Qwen3.8 (~59 GB)
  python scripts/download_models.py --comfy ~/ComfyUI --groups edit,control                   # optional extras
  python scripts/download_models.py --remover ~/minimax-remover --groups remover              # MiniMax-Remover weights (object fixes)
  python scripts/download_models.py --list                                                    # show what would be fetched
  python scripts/download_models.py --check --comfy ~/ComfyUI --llm ~/models/qwen38           # report only, no downloads

Groups: core (any swap), edit (Edit Video 'Draw' mode: H3 Fun ControlNet), control (Motion Control image scene: Qwen-Image 2.1),
remover (MiniMax-Remover weights, CC BY-NC 4.0). By downloading you accept each model's license - see THIRD_PARTY.md.
SAM 3 (facebook/sam3) is gated on Hugging Face: request access there and run `hf auth login`; transformers downloads it on first use."""
import argparse, hashlib, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--comfy", help="ComfyUI folder (its models/ folder receives the H3 files)")
ap.add_argument("--llm", help="folder for the Qwen3.8-27B GGUF + mmproj")
ap.add_argument("--remover", help="MiniMax-Remover checkout (its weights/ folder)")
ap.add_argument("--groups", default="core", help="comma list: core,edit,control,remover")
ap.add_argument("--list", action="store_true")
ap.add_argument("--check", action="store_true", help="only report which files are in place")
ap.add_argument("--no-sha", action="store_true", help="skip the sha256 check (faster)")
a = ap.parse_args()
groups = set(a.groups.split(","))
roots = {"comfy": a.comfy, "llm": a.llm, "remover": a.remover}
items = [f for f in json.load(open(os.path.join(HERE, "models.json")))["files"] if f["group"] in groups]


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


total = sum(f["size"] for f in items)
if a.list:
    for f in items:
        print(f"{f['group']:8} {f['size'] / 1e9:7.2f} GB  {f['repo']}/{f['path']}  ->  <{f['to']}>/{f['dest']}")
    print(f"total {total / 1e9:.1f} GB"); sys.exit(0)
missing = sorted({f["to"] for f in items if not roots[f["to"]]})
if missing:
    sys.exit("give the folder for: " + ", ".join("--" + m for m in missing))
bad = 0
for f in items:
    final = os.path.join(os.path.expanduser(roots[f["to"]]), f["dest"])
    if os.path.exists(final) and os.path.getsize(final) == f["size"]:
        log("have", f["dest"]); continue
    if a.check:
        log("MISSING" if not os.path.exists(final) else "WRONG SIZE", final); bad += 1; continue
    os.makedirs(os.path.dirname(final), exist_ok=True)
    part = final + ".part"
    url = f"https://huggingface.co/{f['repo']}/resolve/main/{f['path']}"
    hdr = ["-H", "Authorization: Bearer " + os.environ["HF_TOKEN"]] if os.environ.get("HF_TOKEN") else []
    for attempt in range(6):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have == f["size"]:
            break
        log(f"get {f['dest']} ({f['size'] / 1e9:.2f} GB, have {have / 1e9:.2f} GB, try {attempt + 1})")
        if subprocess.call(["curl", "-L", "--fail", "-sS", "-C", "-", "--retry", "5", "--retry-delay", "10", *hdr, "-o", part, url]) != 0:
            time.sleep(10)
    if not os.path.exists(part) or os.path.getsize(part) != f["size"]:
        log("FAILED (size)", f["dest"]); bad += 1; continue
    if f.get("sha256") and not a.no_sha:
        log("checking", f["dest"])
        if sha256(part) != f["sha256"]:
            log("FAILED (sha256)", f["dest"]); os.replace(part, part + ".bad"); bad += 1; continue
    os.replace(part, final); log("ok", final)
log("done,", bad, "failed")
sys.exit(1 if bad else 0)
