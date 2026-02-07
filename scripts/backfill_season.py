from __future__ import annotations

import datetime as dt
from sqlalchemy import text

from src.db import get_engine
from src.season_utils import season_from_date


def main():
    engine = get_engine()
    sql = """
        SELECT game_id, game_date
        FROM nba_games
        WHERE season IS NULL OR season = '' OR season LIKE '2%'
    """

    with engine.begin() as conn:
        rows = conn.execute(text(sql)).fetchall()

    if not rows:
        print("No rows to backfill.")
        return

    updates = []
    for game_id, game_date in rows:
        if isinstance(game_date, str):
            game_date = dt.date.fromisoformat(game_date)
        season = season_from_date(game_date)
        updates.append({"game_id": game_id, "season": season})

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE nba_games SET season = :season WHERE game_id = :game_id"),
            updates,
        )

    print(f"Backfilled {len(updates)} rows.")


if __name__ == "__main__":
    main()
