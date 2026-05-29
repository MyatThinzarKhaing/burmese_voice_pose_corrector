FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

RUN set -eux; \
    apt-get update; \
    if apt-cache show libasound2 >/dev/null 2>&1; then \
        ASOUND_PACKAGE=libasound2; \
    else \
        ASOUND_PACKAGE=libasound2t64; \
    fi; \
    if apt-cache show libglib2.0-0 >/dev/null 2>&1; then \
        GLIB_PACKAGE=libglib2.0-0; \
    else \
        GLIB_PACKAGE=libglib2.0-0t64; \
    fi; \
    apt-get install -y --no-install-recommends \
        libgl1 \
        "$GLIB_PACKAGE" \
        "$ASOUND_PACKAGE" \
        libportaudio2 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers=1 --threads=8 --timeout=120 --access-logfile - --error-logfile - week3:app"]
