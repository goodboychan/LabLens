FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    LABLENS_USE_ADC=true \
    GOOGLE_CLOUD_LOCATION=global \
    GEMINI_MODEL_ID=gemini-4-argon

WORKDIR /app

# Install system libraries required for PyAV / Pillow frame decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code, reference saved analyses, and dataset assets
COPY backend /app/backend
COPY frontend /app/frontend
COPY tests /app/tests
COPY storage/saved_reports /app/storage/saved_reports
COPY lablens_so101_episodes /app/lablens_so101_episodes

EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
