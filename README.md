# Bayesian Market–Sports Forecasting System for the NBA

**Core research question**
When should we trust the market, when should we trust a statistical sports model, and how does that change over time?

Markets come from Kalshi / Polymarket, treated as noisy belief aggregators (not ground truth).

## System Overview

```
NBA Data ──▶ Bayesian Sports Model ──▶ P_model ± σ_model
Market Odds ─▶ Belief Dynamics Model ─▶ P_market ± σ_market
                                      │
                                      ▼
                         Bayesian Gating / Meta-Inference
                                      │
                                      ▼
                             P_final ± σ_final
```

## ML Components (brief)

**1) Bayesian NBA Win Model**
- Hierarchical Bayesian logistic regression
- Inputs: team strength, home court, rest, back-to-back, injuries, travel, recent form
- Team strength evolves over time with priors to prevent early-season overfit

**2) Market Belief Dynamics**
- Models belief movement, not just price
- Features: price velocity, volume, liquidity, time-to-game, order dispersion
- Options: Bayesian state-space / Kalman / neural seq encoder → Bayesian head

**3) Bayesian Gating (learned trust)**
- Learns when to trust model vs market
- Inputs: |P_model − P_market|, σ_model, σ_market, liquidity, time remaining, calibration error
- Mixture of experts:
  `P_final = w * P_model + (1 − w) * P_market`, `w ~ Beta(α(x), β(x))`

**4) Neural Enhancements (optional)**
- Team embeddings, injury impact, news/sentiment encodings
- Bayesian inference stays at the top level

## Evaluation
- Brier score, log loss
- Calibration / reliability diagrams
- Sharpness vs accuracy

## Web App (minimal but serious)

**Pages**
- Dashboard: games ranked by market–model disagreement
- Game Detail: probability trajectories, trust weight evolution, uncertainty bands
- Research: calibration plots, model vs market breakdown

**APIs**
- `GET /games`
- `GET /games/{id}`
- `POST /forecast`
- `GET /evaluation/calibration`

## Recommended Build Order
1. Bayesian NBA win model (offline)
2. Market odds ingestion + alignment
3. Gating model (static)
4. Time evolution
5. Web API
6. Frontend
7. Optional neural enhancements

## Data Ingestion (NBA official stats API)

Primary source: `stats.nba.com` via `nba_api` (LeagueGameLog endpoint)

**Setup**
1. Create `.env` from `.env.example` and fill in either:
   - `DATABASE_URL` (SQLAlchemy URL), or
   - `user`, `password`, `host`, `port`, `dbname` (parts)
2. Install deps: `pip install -r requirements.txt`

**Ingest games into Postgres**
```
python -m scripts.ingest_games
```

This will create a `nba_games` table and upsert game outcomes.

If the NBA stats endpoint times out, you can tune retries/timeouts via:
`NBA_TIMEOUT`, `NBA_RETRIES`, `NBA_BACKOFF`.

**Backfill season values (if missing/incorrect)**
```
python -m scripts.backfill_season
```

## Bayesian Win Model (PyMC)

**Train**
```
python -m scripts.train_bayes_win
```

By default, this trains on all seasons in `nba_games`. To specify seasons:
```
NBA_SEASONS=2023-24,2024-25
```

Recency weighting is enabled by default. Configure it with:
```
NBA_USE_RECENCY_WEIGHTS=1
NBA_RECENCY_HALF_LIFE_DAYS=365
```

Model artifacts are saved to `MODEL_DIR` (default `models/`).

**Predict**
```
python -m scripts.predict_bayes_win --home 1610612747 --away 1610612744
```

Outputs posterior mean and 10–90% interval for home win probability.
