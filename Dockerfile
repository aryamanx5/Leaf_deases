# Backend image for the Patra API (FastAPI + PyTorch, CPU). Hugging Face Docker Spaces expect port 7860.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/home/user/.cache/torch \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /app

COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY --chown=user configs configs
COPY --chown=user models models
COPY --chown=user results results
COPY --chown=user src src
COPY --chown=user api api

# Fetch the frozen backbones at build time so the first request does not need the network.
RUN python -c "from leafdx.robust.backbones import load_backbone; \
load_backbone('dinov2_vits14', 'cpu'); load_backbone('openclip_vitb32', 'cpu')"

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
