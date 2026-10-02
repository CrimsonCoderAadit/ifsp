FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000 YOLO_OFFLINE=true ORT_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng libgl1 libglib2.0-0 libgomp1 libportaudio2 tesseract-ocr openssl fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt && python -m pip install ultralytics onnx
COPY . .
RUN python scripts/download_models.py && python -m transit.gtfs_loader
EXPOSE 8000
CMD ["python", "-m", "app.server", "--host", "0.0.0.0"]
