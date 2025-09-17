# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral) - newest installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy only project manifests first for better Docker layer caching
COPY pyproject.toml uv.lock ./

# Ensure uv links the environment to the system Python in this image
ENV UV_PYTHON=/usr/local/bin/python3.12

# Sync dependencies into a virtualenv managed by uv (no dev deps)
RUN /root/.local/bin/uv sync --frozen --no-dev --python /usr/local/bin/python3.12

# Ensure venv binaries are preferred at runtime
ENV PATH="/app/.venv/bin:${PATH}"

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Database URL is read by Hydra from env var DATABASE_URL (see conf/config.yaml)
ENV DATABASE_URL=""

# Preserve original import style in server/app.py
# Ensure top-level imports for `controller` (under server/) and `src/` resolve
ENV PYTHONPATH="/app:/app/server"

# Default command: run uvicorn directly from the venv
CMD ["uvicorn", "server.asgi:app", "--host", "0.0.0.0", "--port", "8000"]


