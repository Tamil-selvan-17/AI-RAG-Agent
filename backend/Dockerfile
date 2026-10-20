FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# All dependencies (incl. PyMuPDF) ship prebuilt wheels, so no compiler toolchain is needed.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
# Configuration (incl. ADMIN_PASSWORD / AUTH_SECRET_KEY / API keys) comes from the
# environment at runtime (docker-compose env_file, or your host's env settings) --
# never baked into the image.

# Run as an unprivileged user; it owns only the folders the app writes to.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs /app/documents \
    && chown -R appuser:appuser /app/logs /app/documents
USER appuser

EXPOSE 8000

# /health/live calls no backing services, so it reports whether *this* process is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
