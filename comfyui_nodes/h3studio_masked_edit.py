"""H3 Studio masked edit latent (MiniMax H3): the source clip as the starting AV latent plus a noise mask, so H3 redraws only the
masked area and keeps everything else - and the soundtrack - from the source. The H3 model handles masked rows natively
(comfy/ldm/minimax/model.py: a row with mask m runs at sigma = m * sigma, m = 0 rows stay the source; masks are max-pooled to the
32 px token grid by model_base.MiniMaxH3). Feed the LATENT to the sampler instead of MiniMaxH3ReferenceToVideo's empty one, with the
same width / height / length."""
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_extras.nodes_minimax_h3 import _encode_ref_audio, _resize, temporal_shape


def _fit_frames(x, n):
    """pad (repeating the last frame) or cut a [F, ...] batch to n frames"""
    if x.shape[0] >= n:
        return x[:n]
    return torch.cat([x, x[-1:].repeat((n - x.shape[0],) + (1,) * (x.ndim - 1))], 0)


class H3StudioMaskedEditLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "images": ("IMAGE",),
                             "width": ("INT", {"default": 832, "min": 32, "max": 8192, "step": 32}),
                             "height": ("INT", {"default": 480, "min": 32, "max": 8192, "step": 32}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                                    "tooltip": "how strongly the masked area is redrawn (1 = from scratch)"})},
                "optional": {"mask": ("MASK", {"tooltip": "per-frame mask, white = redraw; none = redraw everything"}),
                             "audio": ("AUDIO", {"tooltip": "the source soundtrack, kept as it is"}),
                             "audio_vae": ("VAE",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "build"
    CATEGORY = "latent/minimax"

    def build(self, vae, images, width, height, strength, mask=None, audio=None, audio_vae=None):
        frame_count, latent_t, audio_t = temporal_shape(images.shape[0])
        frames = _fit_frames(_resize(images, width, height, "center"), frame_count)
        video = vae.encode(frames)  # [1, 24, T, H/16, W/16]
        if video.shape[2] != latent_t:
            raise ValueError(f"H3 masked edit: {frame_count} frames gave {video.shape[2]} latent frames, expected {latent_t}")
        lh, lw = video.shape[3], video.shape[4]

        if mask is None:
            vm = torch.ones((1, 1, latent_t, lh, lw))
        else:
            m = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).float()
            m = F.interpolate(m[:, None], size=(height, width), mode="bilinear")[:, 0]
            m = _fit_frames(m, frame_count)
            # latent frame k covers FRAME_PER_TOKEN[k % 5] pixel frames (1, 4, 4, 4, 4): any coverage there counts
            per_t, s = [], 0
            for k in range(latent_t):
                e = s + FRAME_PER_TOKEN[k % 5]
                per_t.append(m[s:e].amax(0))
                s = e
            m = torch.stack(per_t)
            m = F.max_pool2d(m[:, None], kernel_size=16, stride=16, ceil_mode=True)[:, 0, :lh, :lw]
            vm = (m.clamp(0.0, 1.0) * strength)[None, None]

        if audio is not None and audio_vae is not None:
            a, _ = _encode_ref_audio(audio_vae, audio)  # [1, 32, 2, Ta]
            a = a[..., :audio_t] if a.shape[-1] >= audio_t else F.pad(a, (0, audio_t - a.shape[-1]))
            am = torch.zeros((1, 1, 2, audio_t))  # keep the source sound
        else:
            a = torch.zeros((1, 32, 2, audio_t))
            am = torch.ones((1, 1, 2, audio_t))  # no source sound: H3 makes it

        dev = comfy.model_management.intermediate_device()
        samples = comfy.nested_tensor.NestedTensor((video.to(dev), a.to(dev)))
        noise_mask = comfy.nested_tensor.NestedTensor((vm.to(dev), am.to(dev)))
        return ({"samples": samples, "noise_mask": noise_mask},)


NODE_CLASS_MAPPINGS = {"H3StudioMaskedEditLatent": H3StudioMaskedEditLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"H3StudioMaskedEditLatent": "H3 Studio masked edit latent (MiniMax H3)"}
