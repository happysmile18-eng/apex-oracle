# Apex Oracle — F1 race predictor

A simple web app that:

- detects the next Formula 1 race automatically;
- loads historical results, current standings and qualifying from the public Jolpica F1 API;
- trains a random-forest + logistic-regression ensemble;
- lets you add weekend pace, upgrades, reliability and wet-weather knowledge;
- runs 5,000–50,000 Monte Carlo simulations;
- produces a winner, top-six probabilities and a biggest-surprise selection;
- displays a forward, time-based winner backtest.

## Fastest way to run it

1. Install Python 3.11 or newer.
2. Open a terminal in this folder.
3. Run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The browser opens automatically.

## Free public hosting — Streamlit Community Cloud

1. Create a GitHub repository and upload this folder.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Choose **New app**.
4. Select the repository and set the main file to `app.py`.
5. Deploy. No API key is required.

Netlify cannot run this Python model directly. Streamlit Community Cloud, Render or Railway can.

## Using it well

- Before qualifying, the app uses championship order as a provisional grid estimate.
- After qualifying, reload the page; the official qualifying positions should populate automatically.
- Correct the grid for penalties.
- Use the manual adjustments only for confirmed information. `+1` should mean a meaningful but not enormous advantage; reserve `+3` for an exceptional signal.
- The prediction is strongest after qualifying and before the race.

## Model design

The training set is created sequentially, so every row uses information available before that race. Features include grid, championship rank/share, recent driver and constructor results, points, win rate, reliability, circuit history and season progress.

The app blends:

- 62% balanced random forest;
- 38% regularised logistic regression.

The final race forecast applies optional weekend adjustments and simulates complete finishing orders with Gumbel/Plackett–Luce sampling. Rain and safety-car settings increase uncertainty.

## Important limitation

No public model can see private team simulations, exact fuel loads, setup compromises or last-minute technical failures. Treat probabilities as decision support, not guarantees.

## Data and trademarks

Race data comes from the open-source Jolpica F1 API. F1, Formula 1 and team/driver marks belong to their respective owners. This project is unofficial.
