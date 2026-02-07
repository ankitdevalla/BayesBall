from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.bayes_win_model import load_artifacts, predict_home_win


def main():
    parser = argparse.ArgumentParser(description="Predict home win probability")
    parser.add_argument("--home", type=int, required=True, help="Home team ID")
    parser.add_argument("--away", type=int, required=True, help="Away team ID")
    args = parser.parse_args()

    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    trace, team_ids = load_artifacts(model_dir)

    result = predict_home_win(
        trace=trace,
        team_ids=team_ids,
        home_team_id=args.home,
        away_team_id=args.away,
    )

    print("Home win probability")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
