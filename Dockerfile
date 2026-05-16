# syntax=docker/dockerfile:1
# GPU image for LiveActionAOV Colab lane (commercial preset extras).
# Build: docker build -t laov-colab:latest .
# Run:  docker run --gpus all --rm -e LAOV_DRIVE_MOUNT=/data -v /path/to/MyDrive:/data \
#          -v /path/to/job.json:/config/job.json:ro laov-colab:latest --job-json /config/job.json

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LAOV_DRIVE_MOUNT=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e ".[matte,dsine]"

ENTRYPOINT ["python", "/app/scripts/laov_colab_run.py"]
CMD ["--help"]
