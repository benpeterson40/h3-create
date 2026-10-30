# Third-party software and models

H3 Create's own code is MIT (see `LICENSE`). It ships **no model weights and no third-party code**. Everything below is
installed or downloaded separately, from its own page, under its own license. Please read each license before you use it.

## Models

| Used for | Model | Where | License |
|---|---|---|---|
| Video generation (every swap, Motion transfer, Edit, Extend) | MiniMax H3 Ref2VA (int8), text encoder, video + audio VAEs, 8-step turbo LoRA, Fun ControlNet Union 2.0 | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (repackaged from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)) | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) + Acceptable Use Policy. **Read the "Applicable Territory" section before use.** The text encoder is based on Qwen3-VL-32B (Apache-2.0). |
| Swapping a person / animal / object into the source performance | H3 Character Swap LoRA v1.5 by Akatz Labs | [akatz-ai/MiniMax-H3-Character-Swap-LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA) | MiniMax H3 Community License (adapter of MiniMax H3) |
| The director and checker: watches the video, writes the shot-by-shot plan and prompts, checks every part | Qwen3.8-27B (vision), GGUF UD-IQ4_XS + mmproj Q8_0 | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) | Apache-2.0 |
| Object outlines (Objects swap), Remove Background, Reframe | SAM 3 | [facebook/sam3](https://huggingface.co/facebook/sam3) (gated: request access) | SAM License (Meta) |
| Automatic object fixes (erase a leftover object before H3 refines the spot) | MiniMax-Remover | [zibojia/MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover), [weights](https://huggingface.co/zibojia/minimax-remover) | CC BY-NC 4.0 (non-commercial) |
| Motion Control "image scene" (optional) | Qwen-Image 2.1 (int8) + its text encoder, VAE and turbo LoRA | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | [Qwen research license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) |
| Depth Map tool (optional) | MoGe-2 ViT-L normal | [microsoft/MoGe](https://github.com/microsoft/MoGe), [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) | MIT |

## Software

| Used for | Project | License |
|---|---|---|
| Runs every generation graph (native nodes, HTTP + websocket API) | [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | GPL-3.0 (H3 Create only talks to it over its API) |
| Serves Qwen3.8-27B | [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) | MIT |
| Video cutting, joining, audio | [FFmpeg](https://ffmpeg.org) | LGPL/GPL |
| Web server | FastAPI, Uvicorn, Pydantic, python-multipart, websockets | MIT / BSD |
| Frames and masks | OpenCV, NumPy | Apache-2.0 / BSD |
| SAM 3 and MiniMax-Remover runtimes | PyTorch, Hugging Face Transformers, Diffusers | BSD / Apache-2.0 |

## Example media

Everything in `examples/` and `media/` was generated on one machine with Qwen-Image 2.1 (the people, animal and place pictures) and
MiniMax H3 (the videos). Every person, animal and place in them is synthetic; no real person's likeness was used. They are offered
under CC BY 4.0 for testing and showing this app, subject to the licenses of the models that made them.
