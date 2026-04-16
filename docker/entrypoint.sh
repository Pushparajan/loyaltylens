#!/usr/bin/env bash
# =============================================================================
# LoyaltyLens — Container Entrypoint
# =============================================================================
# Runs on every container start. Handles:
#   - Waiting for dependent services to be ready
#   - One-time database setup (pgvector extension, schema)
#   - Executing the passed command (default: bash)
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[loyaltylens]${NC} $*"; }
warn() { echo -e "${YELLOW}[loyaltylens]${NC} $*"; }
err()  { echo -e "${RED}[loyaltylens]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Wait for PostgreSQL
# ---------------------------------------------------------------------------
wait_for_postgres() {
    local max_attempts=30
    local attempt=0
    log "Waiting for PostgreSQL at ${POSTGRES_URL:-localhost:5432}..."
    until uv run python -c "
import psycopg2, os, sys
try:
    psycopg2.connect(os.environ.get('POSTGRES_URL', 'postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens'))
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$max_attempts" ]; then
            err "PostgreSQL not ready after ${max_attempts} attempts. Continuing anyway."
            return 1
        fi
        sleep 2
    done
    log "PostgreSQL is ready."
}

# ---------------------------------------------------------------------------
# Wait for Weaviate
# ---------------------------------------------------------------------------
wait_for_weaviate() {
    local weaviate_url="${WEAVIATE_URL:-http://localhost:8080}"
    local max_attempts=20
    local attempt=0
    log "Waiting for Weaviate at ${weaviate_url}..."
    until curl -sf "${weaviate_url}/v1/.well-known/ready" > /dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$max_attempts" ]; then
            warn "Weaviate not ready after ${max_attempts} attempts. Continuing anyway."
            return 0
        fi
        sleep 3
    done
    log "Weaviate is ready."
}

# ---------------------------------------------------------------------------
# Wait for Redis
# ---------------------------------------------------------------------------
wait_for_redis() {
    local redis_url="${REDIS_URL:-redis://localhost:6379}"
    local host port
    host=$(echo "$redis_url" | sed 's|redis://||' | cut -d: -f1)
    port=$(echo "$redis_url" | sed 's|redis://||' | cut -d: -f2)
    port="${port:-6379}"
    local max_attempts=20
    local attempt=0
    log "Waiting for Redis at ${host}:${port}..."
    until uv run python -c "
import redis, os, sys
url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
try:
    r = redis.from_url(url, socket_connect_timeout=2)
    r.ping()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$max_attempts" ]; then
            warn "Redis not ready after ${max_attempts} attempts. Continuing anyway."
            return 0
        fi
        sleep 2
    done
    log "Redis is ready."
}

# ---------------------------------------------------------------------------
# One-time database initialisation
# ---------------------------------------------------------------------------
init_database() {
    log "Running database initialisation..."
    uv run python -c "
import psycopg2, os

url = os.environ.get('POSTGRES_URL', 'postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens')
try:
    conn = psycopg2.connect(url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS offer_embeddings (
            id UUID PRIMARY KEY,
            title TEXT,
            category TEXT,
            channel TEXT,
            min_propensity FLOAT,
            embedding vector(384)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            offer_id TEXT,
            customer_id TEXT,
            generated_copy TEXT,
            rating INTEGER,
            thumbs TEXT,
            prompt_version INTEGER,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    ''')
    conn.close()
    print('Database initialised.')
except Exception as e:
    print(f'DB init skipped: {e}')
" 2>/dev/null || warn "Database init skipped (will retry on next start)."
}

# ---------------------------------------------------------------------------
# Print environment summary
# ---------------------------------------------------------------------------
print_summary() {
    log "============================================================"
    log "  LoyaltyLens Dev Environment"
    log "============================================================"
    log "  Python:      $(python --version 2>&1)"
    log "  uv:          $(uv --version 2>&1)"
    log "  Node.js:     $(node --version 2>&1)"
    log "  Postgres:    ${POSTGRES_URL:-not configured}"
    log "  Weaviate:    ${WEAVIATE_URL:-not configured}"
    log "  Redis:       ${REDIS_URL:-not configured}"
    log "  OpenAI key:  ${OPENAI_API_KEY:+set}${OPENAI_API_KEY:-NOT SET}"
    log "  HF token:    ${HF_TOKEN:+set}${HF_TOKEN:-not set (local models only)}"
    log "------------------------------------------------------------"
    log "  Services:"
    log "    Feature API   →  http://localhost:8001"
    log "    Propensity    →  http://localhost:8002"
    log "    RAG           →  http://localhost:8003"
    log "    LLM Generator →  http://localhost:8004"
    log "    Feedback API  →  http://localhost:8005"
    log "    MLflow UI     →  http://localhost:5000"
    log "    Streamlit     →  http://localhost:8501"
    log "    Vite (UI)     →  http://localhost:5173"
    log "============================================================"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    # Only wait for services if running inside docker-compose
    # (skip if POSTGRES_URL points to localhost, which means running standalone)
    if echo "${POSTGRES_URL:-}" | grep -q "@postgres:"; then
        wait_for_postgres && init_database || true
        wait_for_weaviate || true
        wait_for_redis    || true
    fi

    print_summary

    # Execute the command passed to the container (default: bash)
    exec "$@"
}

main "$@"
