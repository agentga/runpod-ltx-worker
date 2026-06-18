"""RunPod Serverless worker — Wan 2.2 TI2V-5B text-to-video (diffusers).
Принимает {prompt, negative_prompt, width, height, frames, steps, guidance, fps, seed} -> {mp4_base64}.
Модель кэшируется на Network Volume (HF_HOME=/runpod-volume/hf), грузится один раз на воркер.
Vertical 9:16 по умолчанию (704x1280), 24fps, num_frames=4k+1 (Wan VAE temporal x4).
"""
import os
os.environ.setdefault('HF_HOME', '/runpod-volume/hf')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import base64
import tempfile
import runpod
import torch
from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video

MODEL = os.environ.get('WAN_MODEL', 'Wan-AI/Wan2.2-TI2V-5B-Diffusers')
FLOW_SHIFT = float(os.environ.get('WAN_FLOW_SHIFT', '5.0'))  # 5.0 для 720p, 3.0 для 480p

# Рекомендованный Wan negative (повышает качество и убирает артефакты)
DEFAULT_NEG = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, "
    "many people in the background, walking backwards, watermark, text overlay"
)

_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder='vae', torch_dtype=torch.float32)
        _pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
        _pipe.scheduler = UniPCMultistepScheduler.from_config(_pipe.scheduler.config, flow_shift=FLOW_SHIFT)
        _pipe.to('cuda')
        try:
            _pipe.vae.enable_tiling()
            _pipe.vae.enable_slicing()
        except Exception:
            pass
    return _pipe


def _round32(v, default):
    try:
        v = int(v)
    except Exception:
        v = default
    return max(16, (v // 16) * 16)  # кратно 16 для Wan VAE


def handler(event):
    i = event.get('input', {}) or {}
    prompt = i.get('prompt')
    if not prompt:
        return {'error': 'prompt is required'}
    pipe = get_pipe()

    gen = None
    if i.get('seed') is not None:
        gen = torch.Generator(device='cuda').manual_seed(int(i['seed']))

    frames_n = int(i.get('frames', 121))
    if (frames_n - 1) % 4 != 0:                  # Wan VAE: num_frames = 4k+1
        frames_n = ((frames_n - 1) // 4) * 4 + 1

    result = pipe(
        prompt=prompt,
        negative_prompt=i.get('negative_prompt', DEFAULT_NEG),
        width=_round32(i.get('width', 704), 704),
        height=_round32(i.get('height', 1280), 1280),
        num_frames=frames_n,
        num_inference_steps=int(i.get('steps', 40)),
        guidance_scale=float(i.get('guidance', 5.0)),
        generator=gen,
    ).frames[0]

    out = tempfile.mktemp(suffix='.mp4')
    export_to_video(result, out, fps=int(i.get('fps', 24)))
    with open(out, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    try:
        os.remove(out)
    except OSError:
        pass
    return {'mp4_base64': data, 'fps': int(i.get('fps', 24)), 'model': MODEL}


runpod.serverless.start({'handler': handler})
