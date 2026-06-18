"""RunPod Serverless worker — LTX-Video-13B-distilled (diffusers).
Принимает {prompt, width, height, frames, steps, fps} → возвращает {mp4_base64}.
Модель кэшируется на Network Volume (HF_HOME=/runpod-volume/hf), грузится один раз на воркер.
"""
import os
os.environ.setdefault('HF_HOME', '/runpod-volume/hf')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import base64
import tempfile
import runpod
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

MODEL = os.environ.get('LTX_MODEL', 'Lightricks/LTX-Video-0.9.7-distilled')
_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = LTXPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        _pipe.enable_model_cpu_offload()          # надёжно и на 24GB, и на 48GB
        try:
            _pipe.vae.enable_tiling()
            _pipe.vae.enable_slicing()
        except Exception:
            pass
    return _pipe


def handler(event):
    i = event.get('input', {}) or {}
    prompt = i.get('prompt')
    if not prompt:
        return {'error': 'prompt is required'}
    pipe = get_pipe()
    frames = pipe(
        prompt=prompt,
        negative_prompt=i.get('negative_prompt', 'worst quality, blurry, distorted, watermark, text, low resolution'),
        width=int(i.get('width', 704)),       # кратно 32
        height=int(i.get('height', 1216)),    # 9:16-ish, в пайплайне кропим в 1080x1920
        num_frames=int(i.get('frames', 97)),  # (n-1)%8==0 → ~4с@24fps
        num_inference_steps=int(i.get('steps', 8)),  # distilled = мало шагов
    ).frames[0]
    out = tempfile.mktemp(suffix='.mp4')
    export_to_video(frames, out, fps=int(i.get('fps', 24)))
    with open(out, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    try:
        os.remove(out)
    except OSError:
        pass
    return {'mp4_base64': data, 'fps': int(i.get('fps', 24))}


runpod.serverless.start({'handler': handler})
