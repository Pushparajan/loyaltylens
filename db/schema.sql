CREATE TABLE IF NOT EXISTS transactions (
    transaction_id UUID        PRIMARY KEY,
    customer_id    UUID        NOT NULL,
    amount         NUMERIC     NOT NULL,
    currency       VARCHAR(10) NOT NULL DEFAULT 'USD',
    store_id       VARCHAR(255) NOT NULL,
    items          JSONB       NOT NULL DEFAULT '[]',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id  UUID         PRIMARY KEY,
    email        VARCHAR(255) NOT NULL,
    tier         VARCHAR(50)  NOT NULL DEFAULT 'standard',
    total_spend  NUMERIC      NOT NULL DEFAULT 0,
    visit_count  INTEGER      NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
