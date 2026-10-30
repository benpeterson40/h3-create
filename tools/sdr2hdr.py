"""sdr2hdr.py IN OUT PEAK STRENGTH [FORMAT] : SDR (BT.709) video -> HDR, 10-bit HEVC, audio kept. FORMAT hdr10 (PQ, BT.2020, mastering
metadata; default) | hlg (ARIB STD-B67 / BT.2100 HLG, nominal 1000-nit display, system gamma 1.2).
Inverse tone mapping: BT.1886 decode -> SDR white at 203 nits (BT.2408 reference white) -> highlight expansion that leaves shadows and
mid-tones nearly unchanged and lifts the brightest SDR values toward PEAK nits (L = 203*Y*(1 + a*Y^g)) applied to luminance (hue kept)
-> BT.709 to BT.2020 primaries -> PQ. STRENGTH subtle | standard | vivid sets how much of the range is expanded.
Prints 'PROGRESS' lines. Run with the app's Python."""
import json, subprocess, sys
import cv2, numpy as np

src, out, peak, strength = sys.argv[1], sys.argv[2], float(sys.argv[3]), (sys.argv[4] if len(sys.argv) > 4 else "standard")
FMT = sys.argv[5] if len(sys.argv) > 5 else "hdr10"
G = {"subtle": 6.0, "standard": 4.0, "vivid": 2.8}[strength]       # higher g = only the very top end is expanded
REF_WHITE = 203.0
a = peak / REF_WHITE - 1.0
M709_2020 = np.array([[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]], np.float32)
m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875

def hlg(rgb_nits):  # display light (nits, BT.2020) -> HLG signal 0..1: inverse BT.2100 OOTF (Lw 1000, gamma 1.2), then the HLG OETF
    Lw, gamma = 1000.0, 1.2
    Yd = np.maximum(0.2627 * rgb_nits[..., 0] + 0.6780 * rgb_nits[..., 1] + 0.0593 * rgb_nits[..., 2], 1e-6)
    E = np.clip(rgb_nits / Lw, 0, 1) * ((Yd / Lw) ** ((1 - gamma) / gamma))[..., None]
    E = np.clip(E, 0, 1)
    a_, b_, c_ = 0.17883277, 0.28466892, 0.55991073
    return np.where(E <= 1 / 12, np.sqrt(3 * E), a_ * np.log(np.maximum(12 * E - b_, 1e-6)) + c_)


def pq(L):  # nits -> PQ code value 0..1
    y = np.clip(L / 10000.0, 0, 1) ** m1
    return ((c1 + c2 * y) / (1 + c3 * y)) ** m2

probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                                   "-of", "json", src], capture_output=True, text=True).stdout)["streams"][0]
W, H = int(probe["width"]), int(probe["height"]); fr = probe.get("r_frame_rate", "24/1"); total = int(probe.get("nb_frames") or 0)
has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", src],
                                capture_output=True, text=True).stdout.strip())
if FMT == "hlg":
    peak = min(peak, 1000.0); a = peak / REF_WHITE - 1.0
    x265 = "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:repeat-headers=1"
    TRC, TRCZ = "arib-std-b67", "arib-std-b67"
else:
    x265 = ("colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:hdr10=1:repeat-headers=1:"
            f"master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L({int(peak * 10000)},1):max-cll={int(peak)},{int(min(peak, 400))}")
    TRC, TRCZ = "smpte2084", "smpte2084"
enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{W}x{H}", "-r", fr, "-i", "-"]
if has_audio:
    enc += ["-i", src, "-map", "0:v", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
enc += ["-vf", f"zscale=transferin={TRCZ}:primariesin=2020:rangein=full:matrix=2020_ncl:transfer={TRCZ}:primaries=2020:range=limited,format=yuv420p10le",
        "-c:v", "libx265", "-preset", "medium", "-crf", "16", "-x265-params", x265, "-tag:v", "hvc1",
        "-color_primaries", "bt2020", "-color_trc", TRC, "-colorspace", "bt2020nc", "-color_range", "tv", "-shortest", out]
p = subprocess.Popen(enc, stdin=subprocess.PIPE)
cap = cv2.VideoCapture(src); i = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lin = rgb ** 2.4                                                       # BT.1886 display EOTF (relative to SDR white)
    Y = np.maximum(0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2], 1e-6)
    Lhdr = REF_WHITE * Y * (1.0 + a * Y ** G)                              # nits
    scale = (Lhdr / Y)[..., None]
    hdr = lin * scale
    sat = np.clip((Y - 0.7) / 0.3, 0, 1)[..., None] * 0.25                 # tame colour blow-out in the brightest areas
    hdr = hdr * (1 - sat) + (Lhdr[..., None]) * sat
    hdr = hdr @ M709_2020.T
    code = hlg(np.clip(hdr, 0, peak)) if FMT == "hlg" else pq(np.clip(hdr, 0, peak))
    p.stdin.write((code * 65535.0 + 0.5).astype(np.uint16).tobytes())
    i += 1
    if i % 12 == 0:
        print(f"PROGRESS {min(0.97, 0.03 + 0.94 * i / max(1, total)):.3f} converting {i}/{total or '?'}", flush=True)
p.stdin.close(); p.wait()
if p.returncode:
    print("ERROR the HDR encode failed", flush=True); sys.exit(2)
print(f"{'HLG' if FMT == 'hlg' else 'HDR10'} written: {W}x{H}, {i} frames, peak {peak:.0f} nits, {strength}", flush=True)
print("PROGRESS 1.0 done", flush=True)
