# Dockerfile for coPPeR local inference server.
# Builds a container with FastAPI + PyTorch (CPU or CUDA selectable).
# Usage:
#   docker build -t copper-server -f docker/Dockerfile .
#   docker compose up

FROM python:3.12-slim

LABEL maintainer="coPPeR Local" \
      description="Local inference server for coPPeR -- screen-aware document editing"

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends         libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project source and app code
COPY src/ /app/src/
COPY app/ /app/app/
COPY scripts/ /app/scripts/

# Install Python dependencies
RUN pip install --no-cache-dir         fastapi \
        uvicorn[standard] \
        torch --index-url https://download.pytorch.org/whl/cpu \
        safetensors \
        tokenizers \
        requests \
        pydantic

# Expose inference server port (bound to 127.0.0.1 in docker-compose)
EXPOSE 8088

# Default: run the FastAPI inference server
CMD ["python", "-m", "app.inference_server"]
