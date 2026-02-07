from __future__ import annotations

import os
from sqlalchemy import text
from dotenv import load_dotenv

from src.db import get_engine, ensure_schema
from src.nba_stats import fetch_league_games, parse_game_log

load_dotenv()


def upsert_games(engine, games):
    if not games:
        return 0

    sql = """
    INSERT INTO nba_games (
        game_id, season, game_date, home_team_id, away_team_id,
        home_score, away_score, home_win
    ) VALUES (
        :game_id, :season, :game_date, :home_team_id, :away_team_id,
        :home_score, :away_score, :home_win
    )
    ON CONFLICT (game_id) DO UPDATE SET
        season = EXCLUDED.season,
        game_date = EXCLUDED.game_date,
        home_team_id = EXCLUDED.home_team_id,
        away_team_id = EXCLUDED.away_team_id,
        home_score = EXCLUDED.home_score,
        away_score = EXCLUDED.away_score,
        home_win = EXCLUDED.home_win;
    """

    with engine.begin() as conn:
        conn.execute(text(sql), games)
    return len(games)


def main():
    season = os.getenv("NBA_SEASON", "2024-25")

    engine = get_engine()
    ensure_schema(engine)

    payload = fetch_league_games(season)
    games = parse_game_log(payload)

    count = upsert_games(engine, games)
    print(f"Upserted {count} games for season {season}.")


if __name__ == "__main__":
    main()
