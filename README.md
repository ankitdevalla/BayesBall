# Bayesian Market–Sports Forecasting System for the NBA

## 🎯 Final Project Definition
**Core research question**

When should we trust the market, when should we trust a statistical sports model, and how does that change over time?

Markets come from Kalshi / Polymarket, but you treat them as noisy belief aggregators, not ground truth.

## System Overview (mental model)

You’re building a probabilistic ensemble with learned trust:

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

## ML Components (deep dive)

### 1️⃣ Bayesian NBA Win Probability Model (core ML)

This is your anchor model.

**Inputs**
- Team latent strength (learned)
- Home court
- Rest days
- Back-to-back
- Injuries (binary or severity-weighted)
- Travel distance
- Recent form (rolling window)

**Model**
- Hierarchical Bayesian logistic regression

**Key idea**
- Each team has a latent strength parameter
- Strength evolves slowly over time
- Priors prevent overfitting early in season

**Mathematically**
```
logit(P(home_win)) =
  (θ_home − θ_away)
  + β_rest
  + β_home
  + β_injuries

θ_team ~ Normal(μ_league, σ_league)
Time evolution: random walk or Kalman-style update
```

**Why AI labs like this**
- You’re modeling structure
- You’re explicit about uncertainty
- You can explain failures

### 2️⃣ Market Belief Dynamics Model

Instead of “the market price”, model how beliefs move.

**Features**
- Price velocity
- Volume spikes
- Liquidity
- Time-to-game
- Entropy / dispersion of orders

**Model options**
- Bayesian state-space model
- Kalman filter over implied probability
- Neural sequence encoder → Bayesian output head

This answers: **“Is this move informative or noise?”**

### 3️⃣ Bayesian Gating Model (this is the money)

This is the most AI-lab-coded part.

You’re not just averaging probabilities — you’re learning who to trust and when.

**Inputs**
- |P_model − P_market|
- σ_model
- σ_market
- Liquidity
- Time remaining
- Historical calibration error

**Model**
- Bayesian mixture of experts

```
P_final = w * P_model + (1 − w) * P_market
w ~ Beta(α(x), β(x))
```

Where `w` is learned, not fixed.

This gives you:
- Interpretable trust weights
- Uncertainty-aware blending
- Graceful degradation

### 4️⃣ Neural Enhancement (used correctly)

Neural nets are tools, not the thesis.

Use them for:
- Team embeddings (learn latent team vectors)
- Injury impact encoding
- News/sentiment embeddings (optional)

But keep Bayesian inference at the top level.

## Evaluation (important)

You don’t evaluate on PnL.

You evaluate on:
- Brier score
- Log loss
- Calibration curves
- Reliability diagrams
- Sharpness vs accuracy

That’s exactly how real forecasting systems are evaluated.

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

This shows you can build infra around ML, not just train models.

## What this signals to AI labs

They’ll see:
- Probabilistic modeling maturity
- Uncertainty reasoning
- Time-series inference
- System design
- Research framing
- Clean abstraction boundaries

This reads much closer to:

“I could work on forecasting / inference / evaluation infra”

than:

“I built a sports betting model”

## Recommended build order

1. Bayesian NBA win model (offline)
2. Market odds ingestion + alignment
3. Gating model (static)
4. Time evolution
5. Web API
6. Frontend
7. Optional neural enhancements
