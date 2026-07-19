1) SQL Schema

The transformed recommendation data is stored in a structured warehouse layer with four main tables: user features, item features, interaction features, and item co-occurrence features.

-- User-level aggregated features
CREATE TABLE IF NOT EXISTS user_features (
    visitorid TEXT PRIMARY KEY,
    total_interactions INTEGER NOT NULL,
    unique_items INTEGER NOT NULL,
    views INTEGER NOT NULL,
    addtocarts INTEGER NOT NULL,
    transactions INTEGER NOT NULL,
    avg_interaction_weight REAL,
    active_days REAL,
    activity_frequency_per_day REAL,
    transaction_rate REAL,
    recency_days REAL,
    avg_rating_per_user REAL,
    avg_price_seen REAL,
    first_interaction_ts TEXT,
    last_interaction_ts TEXT
);

-- Item-level aggregated features
CREATE TABLE IF NOT EXISTS item_features (
    itemid TEXT PRIMARY KEY,
    total_interactions INTEGER NOT NULL,
    unique_users INTEGER NOT NULL,
    views INTEGER NOT NULL,
    addtocarts INTEGER NOT NULL,
    transactions INTEGER NOT NULL,
    avg_interaction_weight REAL,
    popularity_score REAL,
    avg_price_per_item REAL,
    avg_rating_per_item REAL,
    avg_rating_count REAL,
    first_seen_ts TEXT,
    last_seen_ts TEXT
);

-- Row-level enriched interaction data
CREATE TABLE IF NOT EXISTS interaction_features (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    visitorid TEXT NOT NULL,
    itemid TEXT NOT NULL,
    event TEXT NOT NULL,
    interaction_weight INTEGER NOT NULL,
    price REAL,
    rating_rate REAL,
    category TEXT,
    title TEXT,
    visitorid_encoded INTEGER,
    itemid_encoded INTEGER
);

-- Item-to-item co-occurrence features
CREATE TABLE IF NOT EXISTS item_cooccurrence_features (
    itemid_a TEXT NOT NULL,
    itemid_b TEXT NOT NULL,
    co_occurrence_count INTEGER NOT NULL,
    support_a INTEGER NOT NULL,
    support_b INTEGER NOT NULL,
    co_occurrence_score REAL NOT NULL,
    PRIMARY KEY (itemid_a, itemid_b)
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_user_features_visitorid
    ON user_features(visitorid);

CREATE INDEX IF NOT EXISTS idx_item_features_itemid
    ON item_features(itemid);

CREATE INDEX IF NOT EXISTS idx_interaction_features_visitorid
    ON interaction_features(visitorid);

CREATE INDEX IF NOT EXISTS idx_interaction_features_itemid
    ON interaction_features(itemid);