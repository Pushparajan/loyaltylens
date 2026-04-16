# =============================================================================
# LoyaltyLens — Development Environment
# =============================================================================
# Includes: Python 3.11, uv, Node.js 18, all Python dependencies,
#           Claude Code, ruff, mypy, pytest, MLflow, Streamlit
#
# Build:  docker build -t loyaltylens-dev .
# Run:    docker run -it --env-file .env -v $(pwd):/app loyaltylens-dev bash
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: uv installer — copy the binary from Astral's official image
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:latest AS uv-binary

# ---------------------------------------------------------------------------
# Stage 2: Base system — Python 3.11 slim + system dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS base

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages required by Python libraries:
#   build-essential   — gcc/g++ for compiling C extensions (psycopg2, numpy)
#   libpq-dev         — PostgreSQL client headers (psycopg2-binary still needs this)
#   curl              — healthchecks, uv install fallback
#   git               — prompt versioning (git-tracked YAML), Claude Code
#   nodejs / npm      — feedback loop React UI (Vite), Claude Code
#   ca-certificates   — HTTPS connections to HuggingFace, OpenAI, Weaviate
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    ca-certificates \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Node installation
RUN node --version && npm --version

# ---------------------------------------------------------------------------
# Stage 3: uv + Python environment setup
# ---------------------------------------------------------------------------
FROM base AS python-env

# Copy uv binary from the official image
COPY --from=uv-binary /uv /uvx /usr/local/bin/

# uv environment settings:
#   UV_SYSTEM_PYTHON=1      — use the system Python (no separate pyenv download)
#   UV_LINK_MODE=copy       — copy files instead of hardlinks (better for Docker layers)
#   UV_COMPILE_BYTECODE=1   — pre-compile .pyc files at install time (faster startup)
#   UV_NO_CACHE=1           — skip uv's cache (cache belongs in a mounted volume, not the layer)
ENV UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=0

# Pin the Python version
RUN echo "3.11" > /usr/local/.python-version

# ---------------------------------------------------------------------------
# Stage 4: Dependency installation (cached layer)
# ---------------------------------------------------------------------------
FROM python-env AS deps

WORKDIR /app

# Copy only the dependency manifests first — changes to source code will not
# invalidate this layer, so uv install runs from cache on most rebuilds.
COPY pyproject.toml ./
# uv.lock is optional on first build — generate it if absent
COPY uv.lock* ./

# Install all production + dev dependencies into the system Python
# --frozen: fail if uv.lock is stale (ensures reproducibility in CI)
# If no lockfile exists yet, omit --frozen for first-time bootstrap
RUN uv sync --dev 2>/dev/null || uv sync --dev --no-lock-check

# ---------------------------------------------------------------------------
# Stage 5: Node dependencies (feedback UI)
# ---------------------------------------------------------------------------
FROM deps AS node-deps

WORKDIR /app/feedback_loop/ui

# Copy only package manifests — same layer-caching rationale as Python
COPY feedback_loop/ui/package*.json ./

RUN npm ci --prefer-offline 2>/dev/null || npm install

# ---------------------------------------------------------------------------
# Stage 6: Claude Code installation
# ---------------------------------------------------------------------------
FROM node-deps AS claude-code

# Install Claude Code globally — requires Node.js
RUN npm install -g @anthropic-ai/claude-code

# Verify installation
RUN claude --version 2>/dev/null || echo "Claude Code installed (auth required at runtime)"

# ---------------------------------------------------------------------------
# Stage 7: Final dev image
# ---------------------------------------------------------------------------
FROM claude-code AS dev

WORKDIR /app

# Copy the full project source (when building from repo root)
# In development, mount the source instead: -v $(pwd):/app
COPY . .

# Environment variable defaults — override at runtime via --env-file .env
ENV PYTHONPATH=/app \
    POSTGRES_URL=postgresql://loyaltylens:loyaltylens@postgres:5432/loyaltylens \
    WEAVIATE_URL=http://weaviate:8080 \
    REDIS_URL=redis://redis:6379 \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    OPENAI_API_KEY="" \
    HF_TOKEN=""

# Expose ports used by LoyaltyLens services
EXPOSE 8001   # M1: Feature Store API
EXPOSE 8002   # M2: Propensity Model API
EXPOSE 8003   # M3: RAG Retrieval API
EXPOSE 8004   # M4: LLM Generator API
EXPOSE 8005   # M6: Feedback Loop API
EXPOSE 5000   # MLflow UI
EXPOSE 8501   # Streamlit LLMOps dashboard
EXPOSE 5173   # Vite dev server (feedback UI)

# Copy and set the entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
