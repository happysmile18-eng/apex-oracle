# Apex Oracle Pro 3.0

A Formula 1 race-ranking website built around **one learned model**: XGBoost LambdaMART (`rank:ndcg`). It ranks every driver from first to last and explains the strongest model factors behind each position.

## Guaranteed-start fix in Version 3.0

The complete training archive is now compressed and bundled directly inside `app.py`.

The model therefore **does not download historical seasons when Streamlit starts**. It immediately has 141 completed races from 2020 through the 2026 Belgian Grand Prix available for training. Jolpica is used only as a best-effort refresh for newly completed races and for live schedule/standings/qualifying information. If that refresh fails, the model still trains and the site still opens.

Version 3.0 also:

- Loads completed live races one round at a time, avoiding split pagination records.
- Falls back to the last bundled race roster if current standings are temporarily unavailable.
- Selects the active/next race from its UTC start time rather than requiring race-results data.
- Uses lowercase official Jolpica standings routes.
- Shows `Apex Oracle Pro · Version 3.0.0` at the top so deployment can be verified.

## Model

- One XGBoost LambdaMART learning-to-rank model.
- Leakage-safe pre-race features.
- Driver and constructor Elo.
- Rolling form, qualifying, reliability, teammate and circuit features.
- OpenF1 practice pace, long runs, degradation, starting grid and weather when available.
- Forward holdout validation on the newest historical races.
- Monte Carlo race-order simulation.
- Full P1-to-last ranking with win, podium, top-six, expected finish and DNF estimates.
- XGBoost contribution explanations for each driver.
- Timestamped JSON prediction receipt with SHA-256 fingerprint.

## Deploy

Upload these files directly to the root of the GitHub repository:

- `app.py`
- `requirements.txt`
- `README.md`

Streamlit settings:

- Branch: `main`
- Main file path: `app.py`

After committing, Streamlit normally rebuilds automatically. Reboot the app once only if the header does not change to **Version 3.0.0**.

## Local run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Accuracy

The app reports a forward historical holdout rather than a random train/test split. That is a more realistic test, but no newly released model can honestly be declared the world's most accurate until it has accumulated a long, locked record of live pre-race predictions.

This project is unofficial and is not associated with Formula 1, the FIA, Jolpica or OpenF1.
