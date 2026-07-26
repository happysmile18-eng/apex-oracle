from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


APP_VERSION = "1.0.0"
API_BASE = "https://api.jolpi.ca/ergast/f1"
FEATURES = [
    "grid",
    "championship_rank",
    "championship_share",
    "recent_finish",
    "recent_points",
    "recent_win_rate",
    "reliability",
    "constructor_recent_finish",
    "constructor_recent_points",
    "constructor_win_rate",
    "circuit_driver_finish",
    "circuit_constructor_finish",
    "season_progress",
]


st.set_page_config(
    page_title="Apex Oracle — F1 Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
  --bg: #090a0d;
  --panel: #111319;
  --panel2: #171a21;
  --line: rgba(255,255,255,.09);
  --text: #f4f5f7;
  --muted: #a4a8b3;
  --red: #ff2b37;
  --red2: #be101a;
  --green: #42d392;
}
.stApp { background: radial-gradient(circle at 8% -10%, #331016 0, transparent 28%), var(--bg); }
.block-container { max-width: 1220px; padding-top: 1.4rem; padding-bottom: 4rem; }
[data-testid="stHeader"] { background: transparent; }
.hero { border: 1px solid var(--line); border-radius: 24px; padding: 28px 30px; background: linear-gradient(135deg, rgba(255,43,55,.13), rgba(17,19,25,.93) 38%, rgba(17,19,25,.98)); box-shadow: 0 18px 70px rgba(0,0,0,.25); }
.eyebrow { color: var(--red); font-size: .78rem; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; }
.hero h1 { margin: .35rem 0 .45rem; font-size: clamp(2.1rem,5vw,4.4rem); letter-spacing: -.055em; line-height: .98; }
.hero p { color: var(--muted); max-width: 760px; font-size: 1.02rem; }
.pill { display:inline-block; margin:.35rem .35rem 0 0; padding:.38rem .7rem; border-radius:999px; border:1px solid var(--line); color:#dfe1e6; background:rgba(255,255,255,.035); font-size:.78rem; }
.card { border: 1px solid var(--line); border-radius: 20px; padding: 20px; background: linear-gradient(180deg, rgba(255,255,255,.026), rgba(255,255,255,.012)); height:100%; }
.card-title { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.13em; font-weight:800; }
.big { font-size:2rem; font-weight:850; letter-spacing:-.04em; margin-top:.25rem; }
.sub { color:var(--muted); font-size:.88rem; }
.winner-card { border:1px solid rgba(255,43,55,.4); border-radius:22px; padding:24px; background:linear-gradient(135deg, rgba(255,43,55,.16), rgba(17,19,25,.96) 48%); }
.winner-name { font-size:clamp(2.2rem,6vw,4.8rem); font-weight:900; letter-spacing:-.06em; line-height:.95; margin:.45rem 0; }
.prob { font-size:1.6rem; font-weight:850; color:var(--red); }
.rank-row { display:grid; grid-template-columns:46px 1fr 90px; gap:12px; align-items:center; padding:12px 2px; border-bottom:1px solid var(--line); }
.rank-no { font-size:1.08rem; color:var(--muted); font-weight:800; }
.rank-name { font-weight:800; }
.rank-team { color:var(--muted); font-size:.78rem; }
.rank-p { text-align:right; font-variant-numeric:tabular-nums; font-weight:800; }
.bar { height:5px; background:rgba(255,255,255,.08); border-radius:8px; overflow:hidden; margin-top:7px; }
.bar > span { display:block; height:100%; background:linear-gradient(90deg,var(--red2),var(--red)); border-radius:8px; }
.good { color:var(--green); }
.warning { border-left:3px solid var(--red); background:rgba(255,43,55,.07); padding:12px 14px; border-radius:8px; color:#e9e9ec; }
.small-note { color:var(--muted); font-size:.78rem; }
div.stButton > button { border-radius:14px; min-height:48px; font-weight:800; border:1px solid rgba(255,43,55,.45); background:linear-gradient(180deg,#ff3742,#d91420); color:white; }
div.stButton > button:hover { border-color:#ff6670; color:white; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
hr { border-color:var(--line); }
</style>
""",
    unsafe_allow_html=True,
)


@dataclass
class PredictionModels:
    forest: RandomForestClassifier
    logistic: Pipeline
    backtest_accuracy: float
    backtest_races: int
    training_races: int


class APIError(RuntimeError):
    pass


@st.cache_data(ttl=60 * 60, show_spinner=False)
def api_get(path: str) -> dict[str, Any]:
    url = f"{API_BASE}/{path.lstrip('/')}"
    try:
        response = requests.get(
            url,
            timeout=25,
            headers={"User-Agent": f"ApexOracle/{APP_VERSION}"},
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise APIError(f"Could not load F1 data from Jolpica: {exc}") from exc


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_schedule(season: int) -> list[dict[str, Any]]:
    payload = api_get(f"{season}.json?limit=100")
    return payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_results(season: int) -> list[dict[str, Any]]:
    payload = api_get(f"{season}/results.json?limit=2000")
    return payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_qualifying(season: int, round_number: int) -> list[dict[str, Any]]:
    payload = api_get(f"{season}/{round_number}/qualifying.json?limit=100")
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0].get("QualifyingResults", []) if races else []


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_driver_standings(season: int) -> list[dict[str, Any]]:
    payload = api_get(f"{season}/driverStandings.json?limit=100")
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    return lists[0].get("DriverStandings", []) if lists else []


def parse_race_datetime(race: dict[str, Any]) -> datetime:
    date_text = race.get("date", "")
    time_text = race.get("time", "12:00:00Z")
    try:
        return datetime.fromisoformat(f"{date_text}T{time_text.replace('Z', '+00:00')}")
    except ValueError:
        return datetime.fromisoformat(f"{date_text}T12:00:00+00:00")


def mean_or(values: deque[float] | list[float], default: float) -> float:
    return float(np.mean(values)) if len(values) else float(default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def result_is_finish(status: str) -> bool:
    status_lower = status.lower()
    return status_lower == "finished" or status_lower.startswith("+")


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_history(start_season: int, current_season: int) -> list[dict[str, Any]]:
    races: list[dict[str, Any]] = []
    for year in range(start_season, current_season + 1):
        for race in fetch_results(year):
            race_copy = dict(race)
            race_copy["season"] = str(year)
            races.append(race_copy)
    races.sort(key=lambda r: (int(r.get("season", 0)), int(r.get("round", 0))))
    return races


def build_feature_dataset(races: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    driver_finish: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5))
    driver_points: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5))
    driver_wins: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10))
    driver_finishes: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10))
    constructor_finish: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10))
    constructor_points: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10))
    constructor_wins: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=15))
    track_driver: dict[tuple[str, str], deque[float]] = defaultdict(lambda: deque(maxlen=6))
    track_constructor: dict[tuple[str, str], deque[float]] = defaultdict(lambda: deque(maxlen=10))

    rows: list[dict[str, Any]] = []
    state_by_season: dict[int, dict[str, dict[str, float]]] = {}

    for race in races:
        season = int(race.get("season", 0))
        round_no = int(race.get("round", 0))
        results = race.get("Results", [])
        if not results:
            continue

        if season not in state_by_season:
            state_by_season[season] = {
                "driver_points": defaultdict(float),
                "constructor_points": defaultdict(float),
                "driver_races": defaultdict(float),
                "constructor_races": defaultdict(float),
            }
        season_state = state_by_season[season]
        season_total_points = sum(season_state["driver_points"].values())
        constructor_total_points = sum(season_state["constructor_points"].values())
        ordered_drivers = sorted(
            season_state["driver_points"],
            key=season_state["driver_points"].get,
            reverse=True,
        )
        current_ranks = {driver: idx + 1 for idx, driver in enumerate(ordered_drivers)}
        circuit_id = race.get("Circuit", {}).get("circuitId", "unknown")
        race_id = f"{season}-{round_no:02d}"

        for result in results:
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            driver_id = driver.get("driverId", "unknown")
            constructor_id = constructor.get("constructorId", "unknown")
            grid = int(safe_float(result.get("grid"), 20))
            if grid <= 0:
                grid = 20

            rows.append(
                {
                    "race_id": race_id,
                    "season": season,
                    "round": round_no,
                    "driver_id": driver_id,
                    "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                    "constructor_id": constructor_id,
                    "constructor_name": constructor.get("name", constructor_id),
                    "grid": grid,
                    "championship_rank": current_ranks.get(driver_id, 12),
                    "championship_share": season_state["driver_points"][driver_id] / max(season_total_points, 1.0),
                    "recent_finish": mean_or(driver_finish[driver_id], 12.0),
                    "recent_points": mean_or(driver_points[driver_id], 0.0),
                    "recent_win_rate": mean_or(driver_wins[driver_id], 0.0),
                    "reliability": mean_or(driver_finishes[driver_id], 0.78),
                    "constructor_recent_finish": mean_or(constructor_finish[constructor_id], 12.0),
                    "constructor_recent_points": mean_or(constructor_points[constructor_id], 0.0),
                    "constructor_win_rate": mean_or(constructor_wins[constructor_id], 0.0),
                    "circuit_driver_finish": mean_or(track_driver[(driver_id, circuit_id)], mean_or(driver_finish[driver_id], 12.0)),
                    "circuit_constructor_finish": mean_or(track_constructor[(constructor_id, circuit_id)], mean_or(constructor_finish[constructor_id], 12.0)),
                    "season_progress": min(round_no / 24.0, 1.0),
                    "winner": int(str(result.get("position")) == "1"),
                }
            )

        for result in results:
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            driver_id = driver.get("driverId", "unknown")
            constructor_id = constructor.get("constructorId", "unknown")
            position = int(safe_float(result.get("position"), 20))
            points = safe_float(result.get("points"), 0.0)
            won = float(position == 1)
            finished = float(result_is_finish(str(result.get("status", ""))))

            driver_finish[driver_id].append(position)
            driver_points[driver_id].append(points)
            driver_wins[driver_id].append(won)
            driver_finishes[driver_id].append(finished)
            constructor_finish[constructor_id].append(position)
            constructor_points[constructor_id].append(points)
            constructor_wins[constructor_id].append(won)
            track_driver[(driver_id, circuit_id)].append(position)
            track_constructor[(constructor_id, circuit_id)].append(position)
            season_state["driver_points"][driver_id] += points
            season_state["constructor_points"][constructor_id] += points
            season_state["driver_races"][driver_id] += 1
            season_state["constructor_races"][constructor_id] += 1

    context = {
        "driver_finish": driver_finish,
        "driver_points": driver_points,
        "driver_wins": driver_wins,
        "driver_finishes": driver_finishes,
        "constructor_finish": constructor_finish,
        "constructor_points": constructor_points,
        "constructor_wins": constructor_wins,
        "track_driver": track_driver,
        "track_constructor": track_constructor,
        "season_state": state_by_season,
    }
    return pd.DataFrame(rows), context


def fit_models(train: pd.DataFrame) -> tuple[RandomForestClassifier, Pipeline]:
    x = train[FEATURES].astype(float)
    y = train["winner"].astype(int)

    forest = RandomForestClassifier(
        n_estimators=480,
        max_depth=8,
        min_samples_leaf=3,
        max_features=0.8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(x, y)

    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.55,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    logistic.fit(x, y)
    return forest, logistic


def ensemble_raw(forest: RandomForestClassifier, logistic: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    x = frame[FEATURES].astype(float)
    p_forest = forest.predict_proba(x)[:, 1]
    p_logit = logistic.predict_proba(x)[:, 1]
    return 0.62 * p_forest + 0.38 * p_logit


@st.cache_resource(show_spinner=False)
def train_models(dataset: pd.DataFrame) -> PredictionModels:
    race_ids = list(dataset["race_id"].drop_duplicates())
    test_count = min(28, max(10, int(len(race_ids) * 0.18)))
    split_at = max(20, len(race_ids) - test_count)
    train_ids = set(race_ids[:split_at])
    test_ids = race_ids[split_at:]

    train = dataset[dataset["race_id"].isin(train_ids)]
    test = dataset[dataset["race_id"].isin(test_ids)]
    forest_bt, logistic_bt = fit_models(train)
    test = test.copy()
    test["raw"] = ensemble_raw(forest_bt, logistic_bt, test)
    selected = test.loc[test.groupby("race_id")["raw"].idxmax()]
    accuracy = float(selected["winner"].mean()) if len(selected) else 0.0

    forest, logistic = fit_models(dataset)
    return PredictionModels(
        forest=forest,
        logistic=logistic,
        backtest_accuracy=accuracy,
        backtest_races=len(test_ids),
        training_races=len(race_ids),
    )


def build_candidates(
    season: int,
    race: dict[str, Any],
    context: dict[str, Any],
) -> tuple[pd.DataFrame, bool]:
    round_no = int(race.get("round", 0))
    circuit_id = race.get("Circuit", {}).get("circuitId", "unknown")
    qualifying = fetch_qualifying(season, round_no)
    standings = fetch_driver_standings(season)

    qualifying_map = {
        item.get("Driver", {}).get("driverId"): int(safe_float(item.get("position"), 20))
        for item in qualifying
    }
    grid_known = bool(qualifying_map)

    candidates: list[dict[str, Any]] = []
    season_state = context["season_state"].get(
        season,
        {
            "driver_points": defaultdict(float),
            "constructor_points": defaultdict(float),
        },
    )
    season_total = sum(season_state["driver_points"].values())

    if standings:
        source = standings
    else:
        completed = fetch_results(season)
        last_results = completed[-1].get("Results", []) if completed else []
        source = [
            {
                "position": idx + 1,
                "points": result.get("points", 0),
                "Driver": result.get("Driver", {}),
                "Constructors": [result.get("Constructor", {})],
            }
            for idx, result in enumerate(last_results)
        ]

    for idx, item in enumerate(source):
        driver = item.get("Driver", {})
        constructors = item.get("Constructors", [])
        constructor = constructors[0] if constructors else {}
        driver_id = driver.get("driverId", "unknown")
        constructor_id = constructor.get("constructorId", "unknown")
        rank = int(safe_float(item.get("position"), idx + 1))
        estimated_grid = max(1, min(20, rank))
        grid = qualifying_map.get(driver_id, estimated_grid)

        candidates.append(
            {
                "driver_id": driver_id,
                "Driver": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                "Team": constructor.get("name", constructor_id),
                "grid": grid,
                "championship_rank": rank,
                "championship_share": season_state["driver_points"].get(driver_id, safe_float(item.get("points"), 0.0)) / max(season_total, 1.0),
                "recent_finish": mean_or(context["driver_finish"][driver_id], 12.0),
                "recent_points": mean_or(context["driver_points"][driver_id], 0.0),
                "recent_win_rate": mean_or(context["driver_wins"][driver_id], 0.0),
                "reliability": mean_or(context["driver_finishes"][driver_id], 0.78),
                "constructor_recent_finish": mean_or(context["constructor_finish"][constructor_id], 12.0),
                "constructor_recent_points": mean_or(context["constructor_points"][constructor_id], 0.0),
                "constructor_win_rate": mean_or(context["constructor_wins"][constructor_id], 0.0),
                "circuit_driver_finish": mean_or(context["track_driver"][(driver_id, circuit_id)], mean_or(context["driver_finish"][driver_id], 12.0)),
                "circuit_constructor_finish": mean_or(context["track_constructor"][(constructor_id, circuit_id)], mean_or(context["constructor_finish"][constructor_id], 12.0)),
                "season_progress": min(round_no / 24.0, 1.0),
                "Weekend pace": 0.0,
                "Upgrade": 0.0,
                "Reliability adj.": 0.0,
                "Wet skill": 0.0,
            }
        )

    frame = pd.DataFrame(candidates)
    return frame, grid_known


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -60, 60))
    return exp / exp.sum()


def run_simulation(
    candidates: pd.DataFrame,
    models: PredictionModels,
    rain_probability: float,
    safety_car_probability: float,
    simulations: int,
) -> pd.DataFrame:
    frame = candidates.copy()
    raw = ensemble_raw(models.forest, models.logistic, frame)
    raw = np.clip(raw, 1e-7, None)

    manual = (
        0.30 * frame["Weekend pace"].astype(float).to_numpy()
        + 0.20 * frame["Upgrade"].astype(float).to_numpy()
        + 0.18 * frame["Reliability adj."].astype(float).to_numpy()
        + 0.16 * rain_probability * frame["Wet skill"].astype(float).to_numpy()
    )
    logits = np.log(raw) + manual
    chaos_temperature = 1.0 + 0.42 * rain_probability + 0.25 * safety_car_probability
    logits = logits / chaos_temperature
    winner_probability = softmax(logits)

    rng = np.random.default_rng(20260726)
    noise = rng.gumbel(size=(simulations, len(frame)))
    rankings = np.argsort(-(logits[None, :] + noise), axis=1)
    positions = np.empty_like(rankings)
    positions[np.arange(simulations)[:, None], rankings] = np.arange(1, len(frame) + 1)

    frame["Win %"] = winner_probability * 100
    frame["Podium %"] = (positions <= 3).mean(axis=0) * 100
    frame["Top 6 %"] = (positions <= 6).mean(axis=0) * 100
    frame["Expected finish"] = positions.mean(axis=0)
    frame["Positions gained"] = frame["grid"].astype(float) - frame["Expected finish"]
    return frame.sort_values("Win %", ascending=False).reset_index(drop=True)


def explain_winner(row: pd.Series, field: pd.DataFrame, grid_known: bool) -> str:
    reasons: list[str] = []
    if row["grid"] <= 3:
        reasons.append(f"starts from P{int(row['grid'])}, the strongest track-position base")
    elif not grid_known:
        reasons.append("has one of the model's strongest projected starting positions before qualifying")

    if row["recent_finish"] <= field["recent_finish"].quantile(0.25):
        reasons.append(f"elite recent race form (average finish {row['recent_finish']:.1f})")
    if row["constructor_recent_finish"] <= field["constructor_recent_finish"].quantile(0.25):
        reasons.append("one of the grid's strongest recent cars")
    if row["circuit_driver_finish"] <= field["circuit_driver_finish"].quantile(0.25):
        reasons.append(f"strong circuit history (average finish {row['circuit_driver_finish']:.1f})")
    if row["reliability"] >= field["reliability"].quantile(0.75):
        reasons.append("high recent finishing reliability")
    if row["Weekend pace"] >= 1:
        reasons.append("the positive weekend-pace adjustment")
    if row["Upgrade"] >= 1:
        reasons.append("the declared upgrade-impact adjustment")

    if not reasons:
        reasons.append("the best combined score across grid position, driver form, car form and circuit history")
    return "; ".join(reasons[:4]).capitalize() + "."


def race_location(race: dict[str, Any]) -> str:
    location = race.get("Circuit", {}).get("Location", {})
    return ", ".join(filter(None, [location.get("locality"), location.get("country")]))


def format_race_time(race: dict[str, Any]) -> str:
    dt = parse_race_datetime(race)
    return dt.strftime("%A %d %B %Y · %H:%M UTC")


def render_rankings(prediction: pd.DataFrame, count: int = 6) -> None:
    html = []
    for idx, row in prediction.head(count).iterrows():
        p = float(row["Win %"])
        html.append(
            f"""
<div class="rank-row">
  <div class="rank-no">{idx + 1}</div>
  <div>
    <div class="rank-name">{row['Driver']}</div>
    <div class="rank-team">{row['Team']} · projected P{row['Expected finish']:.1f}</div>
    <div class="bar"><span style="width:{min(100, p * 2.4):.1f}%"></span></div>
  </div>
  <div class="rank-p">{p:.1f}%</div>
</div>
"""
        )
    st.markdown("".join(html), unsafe_allow_html=True)


st.markdown(
    """
<div class="hero">
  <div class="eyebrow">F1 race intelligence</div>
  <h1>APEX ORACLE</h1>
  <p>A live, explainable Formula 1 predictor that learns from historical races, detects the next Grand Prix, adds qualifying and your weekend intelligence, then runs thousands of race simulations.</p>
  <span class="pill">Random forest</span><span class="pill">Logistic model</span><span class="pill">Monte Carlo</span><span class="pill">Live Jolpica data</span>
</div>
""",
    unsafe_allow_html=True,
)

try:
    now = datetime.now(timezone.utc)
    season = now.year
    schedule = fetch_schedule(season)
    upcoming = [race for race in schedule if parse_race_datetime(race) > now]
    if not upcoming and season < now.year + 1:
        next_schedule = fetch_schedule(season + 1)
        if next_schedule:
            season += 1
            upcoming = next_schedule
    if not upcoming:
        raise APIError("No future Grand Prix was found in the current schedule.")

    race_options = {
        f"Round {race.get('round')} · {race.get('raceName')}": race for race in upcoming[:6]
    }
    selected_label = st.selectbox("Prediction race", list(race_options), label_visibility="collapsed")
    selected_race = race_options[selected_label]

    race_dt = parse_race_datetime(selected_race)
    days_left = max(0, math.ceil((race_dt - now).total_seconds() / 86400))

    st.write("")
    c1, c2, c3 = st.columns([1.45, 1, 1])
    with c1:
        st.markdown(
            f"""
<div class="card"><div class="card-title">Next selected race</div><div class="big">{selected_race.get('raceName')}</div><div class="sub">{race_location(selected_race)} · Round {selected_race.get('round')}</div></div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
<div class="card"><div class="card-title">Race start</div><div class="big">{days_left} days</div><div class="sub">{format_race_time(selected_race)}</div></div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="card"><div class="card-title">Data mode</div><div class="big">Live</div><div class="sub">Schedule, results, standings and qualifying</div></div>
""",
            unsafe_allow_html=True,
        )

    with st.spinner("Loading race history and training the model…"):
        history = load_history(max(2018, season - 8), season)
        dataset, context = build_feature_dataset(history)
        if dataset.empty or dataset["winner"].sum() < 30:
            raise APIError("Not enough historical race data was available to train the model.")
        models = train_models(dataset)
        candidates, grid_known = build_candidates(season, selected_race, context)

    st.write("")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Historical races learned", models.training_races)
    with m2:
        st.metric("Forward backtest", f"{models.backtest_accuracy * 100:.0f}% winner accuracy")
    with m3:
        st.metric("Qualifying detected", "Yes" if grid_known else "Not yet")

    if not grid_known:
        st.markdown(
            '<div class="warning">Qualifying is not published yet. Grid positions are estimated from the championship order. Return after qualifying—or edit the grid below—for a much stronger prediction.</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Weekend intelligence")
    st.caption("The app already uses form, reliability, team performance, circuit history and the grid. Edit only what the historical data cannot know yet. Use −3 to +3; leave zero when unsure.")

    editor_columns = ["Driver", "Team", "grid", "Weekend pace", "Upgrade", "Reliability adj.", "Wet skill"]
    edited = st.data_editor(
        candidates[editor_columns],
        hide_index=True,
        use_container_width=True,
        disabled=["Driver", "Team"],
        column_config={
            "grid": st.column_config.NumberColumn("Final grid", min_value=1, max_value=20, step=1),
            "Weekend pace": st.column_config.NumberColumn("Weekend pace", min_value=-3.0, max_value=3.0, step=0.5),
            "Upgrade": st.column_config.NumberColumn("Upgrade impact", min_value=-3.0, max_value=3.0, step=0.5),
            "Reliability adj.": st.column_config.NumberColumn("Reliability", min_value=-3.0, max_value=3.0, step=0.5),
            "Wet skill": st.column_config.NumberColumn("Wet skill", min_value=-3.0, max_value=3.0, step=0.5),
        },
        key=f"editor-{season}-{selected_race.get('round')}",
    )

    candidates = candidates.drop(columns=["grid", "Weekend pace", "Upgrade", "Reliability adj.", "Wet skill"]).merge(
        edited[["Driver", "grid", "Weekend pace", "Upgrade", "Reliability adj.", "Wet skill"]],
        on="Driver",
        how="left",
    )

    a1, a2, a3 = st.columns([1, 1, 1])
    with a1:
        rain = st.slider("Rain probability", 0, 100, 10, 5) / 100
    with a2:
        safety_car = st.slider("Safety-car probability", 0, 100, 35, 5) / 100
    with a3:
        simulations = st.select_slider("Simulations", options=[5000, 10000, 25000, 50000], value=25000)

    if st.button("Generate race prediction", use_container_width=True, type="primary"):
        prediction = run_simulation(candidates, models, rain, safety_car, simulations)
        st.session_state["prediction"] = prediction
        st.session_state["prediction_meta"] = {
            "race": selected_race.get("raceName"),
            "grid_known": grid_known,
            "rain": rain,
            "safety_car": safety_car,
        }

    prediction = st.session_state.get("prediction")
    meta = st.session_state.get("prediction_meta", {})
    if prediction is not None and meta.get("race") == selected_race.get("raceName"):
        winner = prediction.iloc[0]
        surprise_pool = prediction[(prediction["grid"] >= 7) & (prediction["Driver"] != winner["Driver"])]
        if surprise_pool.empty:
            surprise_pool = prediction.iloc[1:]
        surprise = surprise_pool.sort_values(["Top 6 %", "Positions gained"], ascending=False).iloc[0]

        st.divider()
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown(
                f"""
<div class="winner-card">
  <div class="eyebrow">Model winner</div>
  <div class="winner-name">{winner['Driver']}</div>
  <div class="prob">{winner['Win %']:.1f}% win probability</div>
  <div class="sub">{winner['Team']} · starts P{int(winner['grid'])} · podium {winner['Podium %']:.1f}%</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(f"**Why:** {explain_winner(winner, prediction, grid_known)}")
            st.markdown(
                f"**Biggest surprise:** **{surprise['Driver']}** — {surprise['Top 6 %']:.1f}% chance of a top-six finish from P{int(surprise['grid'])}, with an expected finish of P{surprise['Expected finish']:.1f}."
            )
        with right:
            st.markdown('<div class="card-title">Winner ranking</div>', unsafe_allow_html=True)
            render_rankings(prediction, 6)

        st.subheader("Full prediction")
        output = prediction[
            ["Driver", "Team", "grid", "Win %", "Podium %", "Top 6 %", "Expected finish", "Positions gained"]
        ].copy()
        for col in ["Win %", "Podium %", "Top 6 %", "Expected finish", "Positions gained"]:
            output[col] = output[col].round(1)
        output = output.rename(columns={"grid": "Grid"})
        st.dataframe(output, hide_index=True, use_container_width=True)

        csv = output.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction CSV",
            data=csv,
            file_name=f"{season}_{selected_race.get('round')}_{selected_race.get('raceName','race').replace(' ', '_')}_prediction.csv",
            mime="text/csv",
        )

    with st.expander("How the model works"):
        st.markdown(
            """
**1. Historical learning.** It builds one pre-race record per driver from recent finishing form, points, wins, reliability, constructor strength, circuit history, championship position and grid position.

**2. Two-model ensemble.** A balanced random forest captures nonlinear patterns; a regularised logistic model keeps probabilities stable. Their outputs are blended.

**3. Honest time-based test.** The displayed backtest trains on older races and tests on the newest held-out races—not random rows from the same events.

**4. Weekend layer.** Qualifying is loaded automatically. You may add information the historic dataset cannot know, such as a major upgrade, long-run pace or reliability concern.

**5. Monte Carlo.** Thousands of complete finishing orders are simulated. Rain and safety-car probability increase uncertainty instead of automatically favouring the favourite.
"""
        )

    st.caption(
        "Probabilities are estimates, not certainties. The public data does not include every fuel load, setup choice, private simulation or technical issue known to the teams. Data: Jolpica F1 API. Formula 1 marks belong to their respective owners."
    )

except APIError as exc:
    st.error(str(exc))
    st.info("Check the internet connection, then press R to rerun the app. The website deliberately avoids invented fallback predictions when live data is unavailable.")
except Exception as exc:
    st.error(f"The predictor stopped safely: {exc}")
    st.info("Rerun once. If the issue remains, see README.md for troubleshooting.")
