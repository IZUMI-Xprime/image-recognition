# ─────────────────────────────────────────────────────────────────────────────
# Wildlife Sentinel — Production Cloud Dockerfile
# OpenCV AI Competition 2026 (Powered by AWS)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Prevent interactive prompts and write logs unbuffered
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    USE_WAITRESS=1 \
    FLASK_DEBUG=0

# Install system dependencies required for OpenCV, FFmpeg, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications
COPY requirements.txt .

# Install CPU PyTorch first to avoid massive CUDA overhead on cloud instances
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY . .

# Ensure storage directories exist
RUN mkdir -p videos uploads

# Expose web service port
EXPOSE 5000

# Healthcheck for container orchestrators (AWS ECS / Render / Fly.io)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:5000/ || exit 1

# Launch application with production Waitress WSGI server
CMD ["python", "webcam_full_v5.py"]
