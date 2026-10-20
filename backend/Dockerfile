FROM python:3.12-slim

WORKDIR /app

# System dependencies for PyMuPDF and python-docx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# Configuration (incl. ADMIN_PASSWORD / AUTH_SECRET_KEY / API keys) comes from the
# environment at runtime (docker-compose env_file, or your host's env settings) --
# never baked into the image.

RUN mkdir -p /app/logs /app/documents

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
