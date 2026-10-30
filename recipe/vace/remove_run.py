"""remove_run.py VIDEO MASK OUT [--iterations N] [--steps N] : MiniMax-Remover (H3C_REMOVER_DIR, default ~/minimax-remover; CC BY-NC 4.0) - the masked object out of
the clip, background filled in. Frames read with OpenCV (832x480 expected), padded to 4k+1 frames (Wan), mask white = remove."""
import sys, os, argparse, cv2, numpy as np, torch
R = os.path.abspath(os.path.expanduser(os.environ.get("H3C_REMOVER_DIR") or "~/minimax-remover")); sys.path.insert(0, R)
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from transformer_minimax_remover import Transformer3DModel
from pipeline_minimax_remover import Minimax_Remover_Pipeline
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("mask"); ap.add_argument("out")
ap.add_argument("--iterations", type=int, default=6); ap.add_argument("--steps", type=int, default=12); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
def read(p):
    cap = cv2.VideoCapture(p); F = []
    while True:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    return F
V, M = read(a.video), read(a.mask); n = min(len(V), len(M)); H, W = V[0].shape[:2]
L = ((n - 1 + 3) // 4) * 4 + 1                                                  # 124 -> 125
idx = [min(i, n - 1) for i in range(L)]
images = torch.from_numpy(np.stack([cv2.cvtColor(V[i], cv2.COLOR_BGR2RGB) for i in idx])).float() / 127.5 - 1.0
masks = torch.from_numpy(np.stack([(M[i][..., :1] > 127).astype(np.float32) for i in idx]))
dev = torch.device("cuda:0")
vae = AutoencoderKLWan.from_pretrained(f"{R}/weights/vae", torch_dtype=torch.float16)
tr = Transformer3DModel.from_pretrained(f"{R}/weights/transformer", torch_dtype=torch.float16)
sch = UniPCMultistepScheduler.from_pretrained(f"{R}/weights/scheduler")
pipe = Minimax_Remover_Pipeline(transformer=tr, vae=vae, scheduler=sch).to(dev)
out = pipe(images=images, masks=masks, num_frames=L, height=H, width=W, num_inference_steps=a.steps,
           generator=torch.Generator(device=dev).manual_seed(a.seed), iterations=a.iterations).frames[0]
fr = [(np.clip(np.asarray(f), 0, 1) * 255).astype(np.uint8) if np.asarray(f).dtype != np.uint8 else np.asarray(f) for f in out][:n]
tmp = a.out + ".avi"; w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"FFV1"), 24, (W, H))
for f in fr: w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
w.release(); os.system(f"ffmpeg -v error -y -i {tmp} -c:v libx264 -pix_fmt yuv420p -crf 12 {a.out} && rm {tmp}")
print(f"{n} frames ({L} run), peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB -> {a.out}")
