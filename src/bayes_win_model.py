from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import arviz as az
import numpy as np
import pymc as pm


def fit_model(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    y: np.ndarray,
    n_teams: int,
    weights: np.ndarray | None = None,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    seed: int = 42,
) -> az.InferenceData:
    with pm.Model() as model:
        # Team strengths centered at 0 for identifiability
        raw_strength = pm.Normal("raw_strength", mu=0.0, sigma=1.0, shape=n_teams)
        strength = pm.Deterministic("strength", raw_strength - pm.math.mean(raw_strength))

        home_adv = pm.Normal("home_adv", mu=0.0, sigma=1.0)

        logit_p = home_adv + strength[home_idx] - strength[away_idx]

        if weights is None:
            pm.Bernoulli("home_win", logit_p=logit_p, observed=y)
        else:
            logp = pm.logp(pm.Bernoulli.dist(logit_p=logit_p), y)
            pm.Potential("weighted_logp", weights * logp)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=seed,
            chains=2,
            cores=2,
        )

    return trace


def save_artifacts(
    trace: az.InferenceData,
    team_ids: list[int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(trace, out_dir / "bayes_win_model.nc")
    with (out_dir / "team_index.json").open("w") as f:
        json.dump({"team_ids": team_ids}, f)


def load_artifacts(out_dir: Path) -> tuple[az.InferenceData, list[int]]:
    trace = az.from_netcdf(out_dir / "bayes_win_model.nc")
    with (out_dir / "team_index.json").open("r") as f:
        team_ids = json.load(f)["team_ids"]
    return trace, team_ids


def predict_home_win(
    trace: az.InferenceData,
    team_ids: list[int],
    home_team_id: int,
    away_team_id: int,
) -> Dict[str, Any]:
    if home_team_id not in team_ids or away_team_id not in team_ids:
        raise ValueError("Team id not found in trained model.")

    idx = {tid: i for i, tid in enumerate(team_ids)}
    h = idx[home_team_id]
    a = idx[away_team_id]

    strength = trace.posterior["strength"].values
    home_adv = trace.posterior["home_adv"].values

    # Flatten chains, draws
    strength = strength.reshape(-1, strength.shape[-1])
    home_adv = home_adv.reshape(-1)

    logit_p = home_adv + strength[:, h] - strength[:, a]
    p = 1 / (1 + np.exp(-logit_p))

    return {
        "p_mean": float(np.mean(p)),
        "p_p10": float(np.quantile(p, 0.10)),
        "p_p90": float(np.quantile(p, 0.90)),
        "p_median": float(np.median(p)),
    }
