from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sqlalchemy import text

from src.db import get_engine


def load_games(seasons: list[str] | None = None) -> pd.DataFrame:
    engine = get_engine()

    if seasons:
        placeholders = ",".join([":s" + str(i) for i in range(len(seasons))])
        params = {"s" + str(i): seasons[i] for i in range(len(seasons))}
        sql = f"""
            SELECT game_id, season, game_date, home_team_id, away_team_id,
                   home_score, away_score, home_win
            FROM nba_games
            WHERE season IN ({placeholders})
            ORDER BY game_date ASC
        """
        return pd.read_sql(text(sql), con=engine, params=params)

    sql = """
        SELECT game_id, season, game_date, home_team_id, away_team_id,
               home_score, away_score, home_win
        FROM nba_games
        ORDER BY game_date ASC
    """
    return pd.read_sql(text(sql), con=engine)


def build_team_index(df: pd.DataFrame) -> Tuple[dict[int, int], list[int]]:
    team_ids = pd.unique(pd.concat([df["home_team_id"], df["away_team_id"]]))
    team_ids = sorted(int(t) for t in team_ids)
    index = {tid: i for i, tid in enumerate(team_ids)}
    return index, team_ids


def parse_seasons_from_env() -> list[str] | None:
    raw = os.getenv("NBA_SEASONS")
    if not raw:
        return None
    val = [s.strip() for s in raw.split(",") if s.strip()]
    return val
