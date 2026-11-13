# 1. Base image: slim Python matching your local version
FROM python:3.13-slim

# 2. Bring in uv (pin a version later for reproducible builds)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 3. Install dependencies first (cached unless pyproject/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 4. Copy the code and trained artifacts, then install the project itself
COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/
RUN uv sync --frozen --no-dev

# 5. Document the port and start the API
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]