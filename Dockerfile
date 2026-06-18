# RunPod Serverless worker — LTX-Video (diffusers). torch cu124 тянет CUDA сам.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# torch с CUDA 12.4
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
