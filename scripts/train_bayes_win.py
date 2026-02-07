from __future__ import annotations

import os
from pathlib import Path

from src.data import load_games, build_team_index, parse_seasons_from_env
from src.bayes_win_model import fit_model, save_artifacts
from src.weights import compute_recency_weights


def main():
    seasons = parse_seasons_from_env()
    df = load_games(seasons=seasons)

    if df.empty:
        raise RuntimeError("No games found for the selected seasons.")

    team_index, team_ids = build_team_index(df)

    home_idx = df["home_team_id"].map(team_index).to_numpy(dtype=int)
    away_idx = df["away_team_id"].map(team_index).to_numpy(dtype=int)
    y = df["home_win"].to_numpy(dtype=int)

    use_weights = os.getenv("NBA_USE_RECENCY_WEIGHTS", "1") == "1"
    weights = compute_recency_weights(df["game_date"]) if use_weights else None

    draws = int(os.getenv("BAYES_DRAWS", "1000"))
    tune = int(os.getenv("BAYES_TUNE", "1000"))
    target_accept = float(os.getenv("BAYES_TARGET_ACCEPT", "0.9"))

    trace = fit_model(
        home_idx=home_idx,
        away_idx=away_idx,
        y=y,
        n_teams=len(team_ids),
        weights=weights,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
    )

    out_dir = Path(os.getenv("MODEL_DIR", "models"))
    save_artifacts(trace, team_ids, out_dir)

    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    main()
