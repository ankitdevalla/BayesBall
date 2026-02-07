from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd


def compute_recency_weights(dates: pd.Series) -> np.ndarray:
    """
    Exponential decay weights based on recency.
    Uses NBA_RECENCY_HALF_LIFE_DAYS (default 365).
    Weight = 0.5 ** (age_days / half_life)
    """
    half_life = float(os.getenv("NBA_RECENCY_HALF_LIFE_DAYS", "365"))

    # Convert to datetime.date
    d = pd.to_datetime(dates).dt.date
    most_recent = max(d)

    ages = np.array([(most_recent - x).days for x in d], dtype=float)
    weights = 0.5 ** (ages / half_life)
    return weights
