from __future__ import annotations

import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()


def _build_db_url_from_parts() -> str | None:
    user = os.getenv("user")
    password = os.getenv("password")
    host = os.getenv("host")
    port = os.getenv("port")
    dbname = os.getenv("dbname")

    if not all([user, password, host, port, dbname]):
        return None

    # Require SSL for remote hosts; local Postgres typically doesn't support SSL by default.
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    sslmode = "" if host in local_hosts else "?sslmode=require"
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}{sslmode}"


def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL")
    if not url:
        url = _build_db_url_from_parts()

    if not url:
        raise RuntimeError(
            "DATABASE_URL not set and required parts missing. "
            "Set DATABASE_URL or user/password/host/port/dbname in .env."
        )

    return create_engine(url, future=True)


def ensure_schema(engine: Engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS nba_games (
        game_id TEXT PRIMARY KEY,
        season TEXT NOT NULL,
        game_date DATE NOT NULL,
        home_team_id INTEGER NOT NULL,
        away_team_id INTEGER NOT NULL,
        home_score INTEGER NOT NULL,
        away_score INTEGER NOT NULL,
        home_win BOOLEAN NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_nba_games_date ON nba_games (game_date);
    CREATE INDEX IF NOT EXISTS idx_nba_games_season ON nba_games (season);
    CREATE INDEX IF NOT EXISTS idx_nba_games_home ON nba_games (home_team_id);
    CREATE INDEX IF NOT EXISTS idx_nba_games_away ON nba_games (away_team_id);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
