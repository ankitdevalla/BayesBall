from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any, Dict

from nba_api.stats.endpoints import leaguegamelog

from src.season_utils import season_from_date


def fetch_league_games(season: str) -> Dict[str, Any]:
    """
    Fetches league game log data for a season using nba_api LeagueGameLog.
    Season format: "2024-25".
    """
    timeout = int(os.getenv("NBA_TIMEOUT", "60"))
    retries = int(os.getenv("NBA_RETRIES", "3"))
    backoff = float(os.getenv("NBA_BACKOFF", "1.5"))

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            lg = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="T",
                timeout=timeout,
            )
            return lg.get_dict()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
                continue
            raise

    if last_err:
        raise last_err

    raise RuntimeError("Failed to fetch league game log")


def parse_game_log(payload: Dict[str, Any]) -> list[dict[str, Any]]:
    """
    Parses NBA LeagueGameLog payload into canonical game rows.
    Returns one row per game (home/away merged).
    """
    result_sets = payload.get("resultSets") or payload.get("resultSet")
    if isinstance(result_sets, list):
        result_set = result_sets[0]
    elif result_sets is not None:
        result_set = result_sets
    else:
        raise ValueError("LeagueGameLog payload missing resultSets")

    headers = result_set["headers"]
    rows = result_set["rowSet"]

    # Build per-team rows
    team_rows = [dict(zip(headers, r)) for r in rows]

    # Key by GAME_ID; each game has two team rows
    by_game: dict[str, list[dict[str, Any]]] = {}
    for r in team_rows:
        gid = str(r["GAME_ID"])
        by_game.setdefault(gid, []).append(r)

    games: list[dict[str, Any]] = []
    for gid, entries in by_game.items():
        if len(entries) != 2:
            # Skip incomplete entries
            continue

        # Identify home/away via MATCHUP string
        home_row = None
        away_row = None
        for r in entries:
            matchup = r.get("MATCHUP", "")
            if "vs." in matchup:
                home_row = r
            elif "@" in matchup:
                away_row = r

        if not home_row or not away_row:
            continue

        game_date_raw = str(home_row["GAME_DATE"])
        if "-" in game_date_raw:
            game_date = dt.datetime.strptime(game_date_raw, "%Y-%m-%d").date()
        else:
            game_date = dt.datetime.strptime(game_date_raw, "%b %d, %Y").date()
        season_str = season_from_date(game_date)
        home_score = int(home_row["PTS"])
        away_score = int(away_row["PTS"])

        games.append(
            {
                "game_id": gid,
                "season": season_str,
                "game_date": game_date,
                "home_team_id": int(home_row["TEAM_ID"]),
                "away_team_id": int(away_row["TEAM_ID"]),
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_score > away_score,
            }
        )

    return games
