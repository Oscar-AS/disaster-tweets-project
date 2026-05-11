FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better Docker cache)
COPY API/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy source
COPY API /app/API

# Hugging Face Spaces listens on 7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "7860"]
