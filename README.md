# MLPicks – Basketball Player Prop Platform

MLPicks combines a FastAPI backend, scikit-learn regression models, a React dashboard, and backtesting utilities to generate player prop picks (points, rebounds, assists) from free NBA data sources (balldontlie-style APIs). The stack delivers model predictions, probability-based confidence, edge/risk scoring, lightweight explainability, caching, and model metadata.

## Architecture Overview

- **Backend (FastAPI)** – `app/main.py` exposes `/health`, `/player_props`, and `/models/info`. The prediction endpoint calls feature engineering, regression models, probability helpers, and explainability to return picks with confidence, edge metrics, risk scoring, and natural-language reasons.
- **Data providers + caching** – `app/data_providers.py` defines `StatsProvider` with a concrete implementation that pulls from a free NBA API, enriched by a CSV cache (`app/storage.py`) to minimize network calls.
- **Feature engineering** – `app/feature_engineering.py` builds single-row player-game features capturing recent form, opponent context, pace, rest, and usage proxies.
- **Models & registry** – `app/models.py` handles training helpers, model/σ loading, feature-importance persistence, and metadata registry management (`models/metadata.json`, served via `/models/info`).
- **Frontend (React + Vite)** – `frontend/` hosts a TypeScript dashboard where users enter matchup data/prop lines and visualize picks with confidence bars, edge values, risk labels, and explanations.
- **Backtesting** – `backtesting/backtest_player_props.py` evaluates historical performance and logs results, while `backtesting/plot_confidence_calibration.py` charts hit rate vs. confidence buckets.
- **Tests & CI** – `tests/` contains pytest suites for utilities and feature engineering; `.github/workflows/ci.yml` runs them on push/PR.

## Backend Setup & Usage

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Training Models

Prepare CSVs with one row per player-game containing identifier columns, target stat columns (`points_actual`, `rebounds_actual`, `assists_actual`), and all feature columns produced by `build_player_features_for_game`. Train individual models:

```bash
python train_points_model.py --data-path data/player_games_points.csv --version v1.0.0
python train_rebounds_model.py --data-path data/player_games_rebounds.csv --version v1.0.0
python train_assists_model.py --data-path data/player_games_assists.csv --version v1.0.0
```

Each script:

- Performs a chronological split, trains a Gradient Boosting regressor, and calculates MAE/RMSE/sigma.
- Saves artifacts to `models/<stat>_model.joblib` and `models/<stat>_sigma.json`.
- Stores feature importances (`models/<stat>_feature_importance.json`) for explainability.
- Updates `models/metadata.json` with `{version, trained_on, data_range, mae, rmse}`.

### Running the API

```bash
uvicorn app.main:app --reload
```

Endpoints:

- `GET /health` – basic status.
- `GET /models/info` – returns the metadata registry (sample entry: `{"points_model": {"version": "v1.0.0", "trained_on": "2025-11-16", ...}}`).
- `POST /player_props` – body:

```json
{
  "player_id": "237",
  "team_id": "14",
  "opponent_team_id": "6",
  "game_date": "2024-03-14",
  "prop_lines": [
    {"stat_type": "points", "line": 27.5},
    {"stat_type": "rebounds", "line": 7.5},
    {"stat_type": "assists", "line": 6.5}
  ]
}
```

Response per stat includes projection, probabilities, pick side, confidence score/label, edge value (`model_projection - line` in the chosen direction), edge probability (`max(prob) - 0.5`), risk score/label (scaled by σ), and a short explanation built from the top feature importances.

### Confidence / Edge / Risk Definitions

- `prob_over`, `prob_under`: Normal approximation using model mean + σ.
- `confidence_score`: `max(prob_over, prob_under) * 100`.
- `confidence_label`: `low` (<60), `medium` (60–70), `high` (70–80), `very_high` (80+).
- `edge_value`: Distance between projection and line in the pick direction.
- `edge_prob`: Distance from a coin flip (`max(prob) - 0.5`).
- `risk_score`: `min(1, σ / 15)` → `risk_label` (`low`, `medium`, `high`).

## Frontend Dashboard

The React dashboard lives in `frontend/` (Vite + TypeScript).

```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

Features:

- Form inputs for `player_id`, `team_id`, `opponent_team_id`, `game_date`, and prop lines for points/rebounds/assists.
- Calls the backend `/player_props` endpoint and renders cards with projection, line, pick, confidence bar (color-coded by label), edge, risk label (green/yellow/red), and explanation text.
- Basic loading/error states. Configure backend URL via `VITE_API_URL`.

## Backtesting & Evaluation

Assuming a historical CSV with columns:

```
player_id,team_id,opponent_team_id,game_date,
points_line,rebounds_line,assists_line,
points_actual,rebounds_actual,assists_actual
```

Run the backtest:

```bash
python backtesting/backtest_player_props.py \
  --data-path data/backtesting_player_props.csv \
  --output-path backtesting/results_player_props.csv
```

The script:

- Builds features via the live `StatsProvider` (leveraging cache), runs all three models, and records hit/miss plus metadata per prop.
- Prints hit rates by stat and by confidence bucket (50–60, 60–70, 70–80, 80+).
- Writes detailed rows to `backtesting/results_player_props.csv`.

Plot calibration:

```bash
python backtesting/plot_confidence_calibration.py \
  --results-path backtesting/results_player_props.csv \
  --output-path backtesting/confidence_calibration.png
```

## Data Provider & Caching

- `FreeNBABasicStatsProvider` uses balldontlie-style endpoints with pagination and retries.
- `app/storage.py` caches player and team game histories in `cache/players/` and `cache/teams/`. Every fetch merges fresh API data with cached CSVs to reduce redundant calls.

## Testing & CI

```bash
python -m pytest
```

- `tests/test_utils.py` – validates probability math, edge/risk outputs, and deterministic cases.
- `tests/test_feature_engineering.py` – uses a fake provider to ensure feature generation produces filled, single-row frames.
- GitHub Actions workflow (`.github/workflows/ci.yml`) installs dependencies and runs pytest on push/pull requests.

## Additional Notes

- Requirements (`requirements.txt`) cover FastAPI, pandas, scikit-learn, scipy, matplotlib, and pytest.
- The `cache/` directory is created automatically when the provider stores responses.
- Feature-importance JSONs drive both the explanation text and transparency for each model.
- The frontend/backend communicate over simple JSON (no auth); extend as needed for production (auth, rate limiting, persistent storage, etc.).
