#!/usr/bin/env python3
"""erase_strays.py VIDEO MASK OUT : a stray new object taken out of a finished shot (2026-10-28 evening, autofix's stray step).

When the source cannot be pasted back over a stray (the person moved there - the ring around it differs from the source, or it sits next to
another replaced object), MiniMax-Remover erases it from the shot itself and fills in what is behind it (L02: the left woman's mug handle
was drawn as a carrot in 74 frames, next to the pineapple that replaced the mug). MASK = white where the stray is (autofix grows it); only
the erased area changes (make_paste.erase: soft edge, big frames through a scaled copy, long shots in chunks). OUT = the shot with it erased."""
import os, sys
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vace_swap as V
import make_paste as MP


def main():
    vid, mpath, out = sys.argv[1:4]
    fr = V.read_frames(vid); n = len(fr); H, W = fr[0].shape[:2]
    M = V.read_frames(mpath)
    E = np.zeros((n, H, W), bool)
    for j in range(min(n, len(M))):
        m = M[j] if M[j].shape[:2] == (H, W) else cv2.resize(M[j], (W, H), interpolation=cv2.INTER_NEAREST)
        E[j] = m[..., 1] > 127
    if not E.any():
        raise SystemExit("empty mask")
    wd = os.path.splitext(out)[0] + "_work"; os.makedirs(wd, exist_ok=True)
    plate = MP.erase(vid, fr, E, wd)
    V.write_video(out, plate, crf=8)
    print(f"erased {int(E.any((1, 2)).sum())} of {n} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
