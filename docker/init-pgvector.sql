-- LoyaltyLens — PostgreSQL initialisation
-- Runs automatically on first container start via docker-entrypoint-initdb.d

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
