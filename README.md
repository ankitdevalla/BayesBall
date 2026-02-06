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

