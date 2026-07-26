# Apex Oracle Pro 2.0

A complete Formula 1 race-ranking website built around **one learned model**: XGBoost LambdaMART (`rank:ndcg`). It ranks every driver from first to last and explains the strongest reasons behind each position.

## What is improved

- One true learning-to-rank model instead of mixing unrelated classifiers.
- Sequential, leakage-safe features built before every historical race.
- Driver and constructor Elo ratings.
- Rolling form, qualifying, reliability, teammate and circuit features.
- Automatic final starting grid from OpenF1 when available.
- Automatic practice analysis: robust pace, long runs, tyre degradation and lap count.
- Forward holdout test on the newest races.
- Calibrated Monte Carlo probabilities.
- Full ranking from P1 to last, plus winner, podium, top-six, DNF and expected finish.
- XGBoost contribution explanations for every driver.
- Timestamped JSON prediction receipt with SHA-256 fingerprint.

## Deploy on Streamlit Community Cloud

Upload these items directly to the root of your GitHub repository:

- `app.py`
- `requirements.txt`
- `README.md`
- `.streamlit/config.toml` (optional styling)

Then use:

- Branch: `main`
- Main file path: `app.py`

## Local run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## How to use it

1. Select the next race.
2. Wait for history/model loading.
3. Verify the final grid after penalties.
4. Leave manual adjustments at zero unless you have confirmed evidence.
5. Set rain and safety-car probability.
6. Generate the complete ranking.
7. Download the timestamped prediction receipt before the race.

## Honest accuracy statement

This is designed as a strong public-data model, but it is not honestly possible to call any new model “the world's most accurate” before it has accumulated a long, locked, live record. The app displays a forward historical holdout, and the receipt feature allows future predictions to be audited without editing them after the result.

## Data

- Jolpica: historical results, schedules, qualifying and standings.
- OpenF1: starting grid, practice laps, stints and weather.

This project is unofficial and is not associated with Formula 1, the FIA, Jolpica or OpenF1. Formula 1 names and marks belong to their owners.
