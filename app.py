from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
from xgboost import XGBRanker


APP_VERSION = "2.0.0"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE = "https://api.openf1.org/v1"
DEFAULT_ELO = 1500.0
RNG_SEED = 20260726

FEATURES = [
    "field_size",
    "season_progress",
    "grid",
    "grid_norm",
    "grid_delta_teammate",
    "driver_champ_rank_norm",
    "driver_champ_share",
    "constructor_champ_rank_norm",
    "constructor_champ_share",
    "driver_elo_z",
    "constructor_elo_z",
    "elo_delta_teammate",
    "recent_finish_mean_3",
    "recent_finish_mean_5",
    "recent_finish_mean_10",
    "recent_finish_median_5",
    "recent_finish_std_5",
    "recent_finish_trend_5",
    "recent_points_mean_3",
    "recent_points_mean_5",
    "recent_points_mean_10",
    "recent_grid_mean_5",
    "recent_grid_gain_mean_5",
    "recent_dnf_rate_5",
    "recent_dnf_rate_10",
    "recent_win_rate_10",
    "recent_podium_rate_10",
    "recent_top6_rate_10",
    "constructor_finish_mean_5",
    "constructor_finish_mean_10",
    "constructor_best_finish_5",
    "constructor_points_mean_5",
    "constructor_dnf_rate_10",
    "constructor_win_rate_10",
    "constructor_podium_rate_10",
    "circuit_driver_finish",
    "circuit_driver_gain",
    "circuit_driver_starts_log",
    "circuit_constructor_finish",
    "circuit_constructor_starts_log",
    "circuit_grid_corr",
    "circuit_avg_abs_gain",
    "circuit_dnf_rate",
    "circuit_pole_win_rate",
    "experience_log",
    "team_tenure_log",
]

FEATURE_LABELS = {
    "field_size": "field size",
    "season_progress": "season stage",
    "grid": "starting grid",
    "grid_norm": "grid position relative to the field",
    "grid_delta_teammate": "qualifying position versus teammate",
    "driver_champ_rank_norm": "championship position",
    "driver_champ_share": "share of championship points",
    "constructor_champ_rank_norm": "constructor championship position",
    "constructor_champ_share": "constructor share of points",
    "driver_elo_z": "driver strength rating",
    "constructor_elo_z": "car/team strength rating",
    "elo_delta_teammate": "driver rating versus teammate",
    "recent_finish_mean_3": "last-three-race finishing form",
    "recent_finish_mean_5": "last-five-race finishing form",
    "recent_finish_mean_10": "longer-term finishing form",
    "recent_finish_median_5": "median recent finish",
    "recent_finish_std_5": "recent consistency",
    "recent_finish_trend_5": "recent form direction",
    "recent_points_mean_3": "last-three-race scoring",
    "recent_points_mean_5": "last-five-race scoring",
    "recent_points_mean_10": "longer-term scoring",
    "recent_grid_mean_5": "recent qualifying form",
    "recent_grid_gain_mean_5": "recent race-day position gain",
    "recent_dnf_rate_5": "short-term reliability",
    "recent_dnf_rate_10": "long-term reliability",
    "recent_win_rate_10": "recent win rate",
    "recent_podium_rate_10": "recent podium rate",
    "recent_top6_rate_10": "recent top-six rate",
    "constructor_finish_mean_5": "team's current race pace",
    "constructor_finish_mean_10": "team's longer-term race pace",
    "constructor_best_finish_5": "team's recent ceiling",
    "constructor_points_mean_5": "team's recent points production",
    "constructor_dnf_rate_10": "team reliability",
    "constructor_win_rate_10": "team win rate",
    "constructor_podium_rate_10": "team podium rate",
    "circuit_driver_finish": "driver history at this circuit",
    "circuit_driver_gain": "driver's race-day progress at this circuit",
    "circuit_driver_starts_log": "driver experience at this circuit",
    "circuit_constructor_finish": "team history at this circuit",
    "circuit_constructor_starts_log": "team experience at this circuit",
    "circuit_grid_corr": "importance of grid position at this circuit",
    "circuit_avg_abs_gain": "historical position movement at this circuit",
    "circuit_dnf_rate": "historical retirement rate at this circuit",
    "circuit_pole_win_rate": "historical pole conversion at this circuit",
    "experience_log": "career experience",
    "team_tenure_log": "experience with the current team",
}


st.set_page_config(
    page_title="Apex Oracle Pro — F1 Race Ranker",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
  --bg:#08090c; --panel:#11141a; --panel2:#171b23; --line:rgba(255,255,255,.09);
  --text:#f5f6f8; --muted:#a7acb8; --red:#ff3341; --red2:#c5111d; --green:#4dde9b;
}
.stApp { background:radial-gradient(circle at 8% -8%,#351017 0,transparent 28%),var(--bg); }
.block-container { max-width:1280px; padding-top:1.3rem; padding-bottom:4rem; }
[data-testid="stHeader"] { background:transparent; }
.hero { border:1px solid var(--line); border-radius:25px; padding:28px 30px; background:linear-gradient(135deg,rgba(255,51,65,.14),rgba(17,20,26,.96) 42%); box-shadow:0 20px 80px rgba(0,0,0,.28); }
.eyebrow { color:var(--red); font-size:.75rem; font-weight:850; letter-spacing:.18em; text-transform:uppercase; }
.hero h1 { margin:.35rem 0 .4rem; font-size:clamp(2.15rem,5vw,4.6rem); letter-spacing:-.06em; line-height:.96; }
.hero p { color:var(--muted); max-width:800px; font-size:1rem; }
.pill { display:inline-block; margin:.38rem .34rem 0 0; padding:.38rem .68rem; border-radius:999px; border:1px solid var(--line); color:#e1e4ea; background:rgba(255,255,255,.035); font-size:.76rem; }
.card { border:1px solid var(--line); border-radius:20px; padding:20px; background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.012)); height:100%; }
.card-title { color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.13em; font-weight:850; }
.big { font-size:1.9rem; font-weight:900; letter-spacing:-.04em; margin-top:.28rem; }
.sub { color:var(--muted); font-size:.86rem; }
.winner-card { border:1px solid rgba(255,51,65,.46); border-radius:22px; padding:24px; background:linear-gradient(135deg,rgba(255,51,65,.18),rgba(17,20,26,.97) 50%); }
.winner-name { font-size:clamp(2.25rem,6vw,4.9rem); font-weight:950; letter-spacing:-.065em; line-height:.94; margin:.45rem 0; }
.prob { font-size:1.55rem; font-weight:900; color:var(--red); }
.warning { border-left:3px solid var(--red); background:rgba(255,51,65,.075); padding:12px 14px; border-radius:8px; }
.good { color:var(--green); }
.small { color:var(--muted); font-size:.78rem; }
div.stButton > button { border-radius:14px; min-height:49px; font-weight:850; border:1px solid rgba(255,51,65,.5); background:linear-gradient(180deg,#ff3c49,#d91422); color:white; }
div.stButton > button:hover { color:white; border-color:#ff7580; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
hr { border-color:var(--line); }
</style>
""",
    unsafe_allow_html=True,
)


@dataclass
class ModelBundle:
    model: XGBRanker
    temperature: float
    winner_accuracy: float
    podium_capture: float
    mean_spearman: float
    mean_position_error: float
    test_races: int
    training_races: int


class DataError(RuntimeError):
    pass


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, default)))


def normalise_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_dt(text: str | None, default: datetime | None = None) -> datetime:
    if not text:
        return default or datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return default or datetime.now(timezone.utc)


def race_datetime(race: dict[str, Any]) -> datetime:
    return parse_dt(f"{race.get('date', '')}T{race.get('time', '12:00:00Z')}")


def mean(values: Iterable[float], default: float) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float(default)


def median(values: Iterable[float], default: float) -> float:
    values = list(values)
    return float(np.median(values)) if values else float(default)


def std(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return float(np.std(values)) if len(values) >= 2 else float(default)


def trend(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 3:
        return 0.0
    return float(np.polyfit(np.arange(len(values)), np.asarray(values, dtype=float), 1)[0])


def rate(values: Iterable[float], default: float = 0.0) -> float:
    return mean(values, default)


def finished_status(status: str) -> bool:
    s = (status or "").lower()
    return s == "finished" or s.startswith("+")


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-5)
    z = np.asarray(values, dtype=float) / t
    z -= np.max(z)
    exp = np.exp(np.clip(z, -60, 60))
    return exp / max(exp.sum(), 1e-12)


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


@st.cache_data(ttl=60 * 60, show_spinner=False)
def jolpica_get(path: str) -> dict[str, Any]:
    url = f"{JOLPICA_BASE}/{path.lstrip('/')}"
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": f"ApexOraclePro/{APP_VERSION}"})
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataError(f"Jolpica data could not be loaded: {exc}") from exc


@st.cache_data(ttl=60 * 20, show_spinner=False)
def openf1_get(endpoint: str, params: tuple[tuple[str, Any], ...]) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            f"{OPENF1_BASE}/{endpoint}",
            params=dict(params),
            timeout=35,
            headers={"User-Agent": f"ApexOraclePro/{APP_VERSION}"},
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []
    except (requests.RequestException, ValueError):
        return []


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_schedule(season: int) -> list[dict[str, Any]]:
    payload = jolpica_get(f"{season}.json?limit=100")
    return payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def fetch_results(season: int) -> list[dict[str, Any]]:
    payload = jolpica_get(f"{season}/results.json?limit=3000")
    return payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_qualifying(season: int, round_no: int) -> list[dict[str, Any]]:
    payload = jolpica_get(f"{season}/{round_no}/qualifying.json?limit=100")
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0].get("QualifyingResults", []) if races else []


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_driver_standings(season: int) -> list[dict[str, Any]]:
    payload = jolpica_get(f"{season}/driverStandings.json?limit=100")
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    return lists[0].get("DriverStandings", []) if lists else []


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_constructor_standings(season: int) -> list[dict[str, Any]]:
    payload = jolpica_get(f"{season}/constructorStandings.json?limit=100")
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    return lists[0].get("ConstructorStandings", []) if lists else []


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_history(start_season: int, current_season: int) -> list[dict[str, Any]]:
    races: list[dict[str, Any]] = []
    for year in range(start_season, current_season + 1):
        for race in fetch_results(year):
            copy = dict(race)
            copy["season"] = str(year)
            races.append(copy)
    races.sort(key=lambda r: (safe_int(r.get("season")), safe_int(r.get("round"))))
    return races


def new_state() -> dict[str, Any]:
    return {
        "driver_elo": defaultdict(lambda: DEFAULT_ELO),
        "constructor_elo": defaultdict(lambda: DEFAULT_ELO),
        "driver_finish": defaultdict(lambda: deque(maxlen=10)),
        "driver_points": defaultdict(lambda: deque(maxlen=10)),
        "driver_grid": defaultdict(lambda: deque(maxlen=10)),
        "driver_gain": defaultdict(lambda: deque(maxlen=10)),
        "driver_dnf": defaultdict(lambda: deque(maxlen=10)),
        "driver_win": defaultdict(lambda: deque(maxlen=10)),
        "driver_podium": defaultdict(lambda: deque(maxlen=10)),
        "driver_top6": defaultdict(lambda: deque(maxlen=10)),
        "constructor_finish": defaultdict(lambda: deque(maxlen=10)),
        "constructor_best": defaultdict(lambda: deque(maxlen=10)),
        "constructor_points": defaultdict(lambda: deque(maxlen=10)),
        "constructor_dnf": defaultdict(lambda: deque(maxlen=10)),
        "constructor_win": defaultdict(lambda: deque(maxlen=10)),
        "constructor_podium": defaultdict(lambda: deque(maxlen=10)),
        "circuit_driver_finish": defaultdict(lambda: deque(maxlen=8)),
        "circuit_driver_gain": defaultdict(lambda: deque(maxlen=8)),
        "circuit_constructor_finish": defaultdict(lambda: deque(maxlen=12)),
        "circuit_grids": defaultdict(lambda: deque(maxlen=250)),
        "circuit_finishes": defaultdict(lambda: deque(maxlen=250)),
        "circuit_dnf": defaultdict(lambda: deque(maxlen=250)),
        "circuit_pole_win": defaultdict(lambda: deque(maxlen=20)),
        "experience": defaultdict(int),
        "team_tenure": defaultdict(int),
        "season_driver_points": defaultdict(float),
        "season_constructor_points": defaultdict(float),
        "current_season": None,
    }


def regress_elos(state: dict[str, Any]) -> None:
    for key in list(state["driver_elo"].keys()):
        state["driver_elo"][key] = 0.76 * state["driver_elo"][key] + 0.24 * DEFAULT_ELO
    for key in list(state["constructor_elo"].keys()):
        state["constructor_elo"][key] = 0.68 * state["constructor_elo"][key] + 0.32 * DEFAULT_ELO


def update_pairwise_elo(ratings: dict[str, float], ids: list[str], positions: list[float], k: float) -> None:
    if len(ids) < 2:
        return
    deltas = {item: 0.0 for item in ids}
    scale = max(len(ids) - 1, 1)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            expected_a = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            if positions[i] < positions[j]:
                actual_a = 1.0
            elif positions[i] > positions[j]:
                actual_a = 0.0
            else:
                actual_a = 0.5
            change = k * (actual_a - expected_a) / scale
            deltas[a] += change
            deltas[b] -= change
    for item, change in deltas.items():
        ratings[item] += change


def rank_maps(points: dict[str, float]) -> tuple[dict[str, int], float]:
    ordered = sorted(points, key=lambda key: points[key], reverse=True)
    return {key: index + 1 for index, key in enumerate(ordered)}, float(sum(points.values()))


def driver_snapshot(state: dict[str, Any], driver_id: str) -> dict[str, float]:
    finishes = state["driver_finish"][driver_id]
    points = state["driver_points"][driver_id]
    grids = state["driver_grid"][driver_id]
    gains = state["driver_gain"][driver_id]
    dnfs = state["driver_dnf"][driver_id]
    return {
        "recent_finish_mean_3": mean(list(finishes)[-3:], 12.0),
        "recent_finish_mean_5": mean(list(finishes)[-5:], 12.0),
        "recent_finish_mean_10": mean(finishes, 12.0),
        "recent_finish_median_5": median(list(finishes)[-5:], 12.0),
        "recent_finish_std_5": std(list(finishes)[-5:], 4.5),
        "recent_finish_trend_5": trend(list(finishes)[-5:]),
        "recent_points_mean_3": mean(list(points)[-3:], 0.0),
        "recent_points_mean_5": mean(list(points)[-5:], 0.0),
        "recent_points_mean_10": mean(points, 0.0),
        "recent_grid_mean_5": mean(list(grids)[-5:], 12.0),
        "recent_grid_gain_mean_5": mean(list(gains)[-5:], 0.0),
        "recent_dnf_rate_5": rate(list(dnfs)[-5:], 0.18),
        "recent_dnf_rate_10": rate(dnfs, 0.18),
        "recent_win_rate_10": rate(state["driver_win"][driver_id], 0.0),
        "recent_podium_rate_10": rate(state["driver_podium"][driver_id], 0.0),
        "recent_top6_rate_10": rate(state["driver_top6"][driver_id], 0.0),
    }


def constructor_snapshot(state: dict[str, Any], constructor_id: str) -> dict[str, float]:
    finishes = state["constructor_finish"][constructor_id]
    return {
        "constructor_finish_mean_5": mean(list(finishes)[-5:], 12.0),
        "constructor_finish_mean_10": mean(finishes, 12.0),
        "constructor_best_finish_5": mean(list(state["constructor_best"][constructor_id])[-5:], 10.0),
        "constructor_points_mean_5": mean(list(state["constructor_points"][constructor_id])[-5:], 0.0),
        "constructor_dnf_rate_10": rate(state["constructor_dnf"][constructor_id], 0.16),
        "constructor_win_rate_10": rate(state["constructor_win"][constructor_id], 0.0),
        "constructor_podium_rate_10": rate(state["constructor_podium"][constructor_id], 0.0),
    }


def circuit_snapshot(state: dict[str, Any], circuit_id: str, driver_id: str, constructor_id: str) -> dict[str, float]:
    grids = list(state["circuit_grids"][circuit_id])
    finishes = list(state["circuit_finishes"][circuit_id])
    corr = 0.68
    if len(grids) >= 12 and np.std(grids) > 0 and np.std(finishes) > 0:
        corr = float(np.corrcoef(grids, finishes)[0, 1])
        if not np.isfinite(corr):
            corr = 0.68
    driver_hist = state["circuit_driver_finish"][(driver_id, circuit_id)]
    driver_gain = state["circuit_driver_gain"][(driver_id, circuit_id)]
    team_hist = state["circuit_constructor_finish"][(constructor_id, circuit_id)]
    return {
        "circuit_driver_finish": mean(driver_hist, mean(state["driver_finish"][driver_id], 12.0)),
        "circuit_driver_gain": mean(driver_gain, mean(state["driver_gain"][driver_id], 0.0)),
        "circuit_driver_starts_log": math.log1p(len(driver_hist)),
        "circuit_constructor_finish": mean(team_hist, mean(state["constructor_finish"][constructor_id], 12.0)),
        "circuit_constructor_starts_log": math.log1p(len(team_hist)),
        "circuit_grid_corr": corr,
        "circuit_avg_abs_gain": mean([abs(g - f) for g, f in zip(grids, finishes)], 3.0),
        "circuit_dnf_rate": rate(state["circuit_dnf"][circuit_id], 0.18),
        "circuit_pole_win_rate": rate(state["circuit_pole_win"][circuit_id], 0.5),
    }


def make_feature_row(
    state: dict[str, Any],
    *,
    driver_id: str,
    constructor_id: str,
    circuit_id: str,
    grid: int,
    field_size: int,
    round_no: int,
    season_rounds: int,
    driver_rank: int,
    driver_points: float,
    driver_total_points: float,
    constructor_rank: int,
    constructor_points: float,
    constructor_total_points: float,
    teammate_grid: float,
    teammate_form: float,
    teammate_elo: float,
) -> dict[str, float]:
    row: dict[str, float] = {
        "field_size": float(field_size),
        "season_progress": min(round_no / max(season_rounds, 1), 1.0),
        "grid": float(grid),
        "grid_norm": (grid - 1) / max(field_size - 1, 1),
        "grid_delta_teammate": float(grid - teammate_grid),
        "driver_champ_rank_norm": (driver_rank - 1) / max(field_size - 1, 1),
        "driver_champ_share": driver_points / max(driver_total_points, 1.0),
        "constructor_champ_rank_norm": (constructor_rank - 1) / max(max(field_size // 2, 10) - 1, 1),
        "constructor_champ_share": constructor_points / max(constructor_total_points, 1.0),
        "driver_elo_z": (state["driver_elo"][driver_id] - DEFAULT_ELO) / 200.0,
        "constructor_elo_z": (state["constructor_elo"][constructor_id] - DEFAULT_ELO) / 200.0,
        "elo_delta_teammate": (state["driver_elo"][driver_id] - teammate_elo) / 200.0,
        "experience_log": math.log1p(state["experience"][driver_id]),
        "team_tenure_log": math.log1p(state["team_tenure"][(driver_id, constructor_id)]),
    }
    row.update(driver_snapshot(state, driver_id))
    row.update(constructor_snapshot(state, constructor_id))
    row.update(circuit_snapshot(state, circuit_id, driver_id, constructor_id))
    # Teammate-relative form is embedded in the existing trend feature without adding a redundant feature.
    row["recent_finish_trend_5"] += 0.12 * (row["recent_finish_mean_5"] - teammate_form)
    return row


def build_dataset(races: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    state = new_state()
    rows: list[dict[str, Any]] = []
    race_index = 0

    rounds_by_season: dict[int, int] = defaultdict(lambda: 24)
    for race in races:
        year = safe_int(race.get("season"))
        rounds_by_season[year] = max(rounds_by_season[year], safe_int(race.get("round")))

    for race in races:
        season = safe_int(race.get("season"))
        round_no = safe_int(race.get("round"))
        results = race.get("Results", [])
        if not results:
            continue
        if state["current_season"] != season:
            if state["current_season"] is not None:
                regress_elos(state)
            state["season_driver_points"] = defaultdict(float)
            state["season_constructor_points"] = defaultdict(float)
            state["current_season"] = season

        driver_ranks, driver_total = rank_maps(state["season_driver_points"])
        constructor_ranks, constructor_total = rank_maps(state["season_constructor_points"])
        field_size = len(results)
        circuit_id = race.get("Circuit", {}).get("circuitId", "unknown")
        race_id = f"{season}-{round_no:02d}"
        race_index += 1

        team_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            team_members[result.get("Constructor", {}).get("constructorId", "unknown")].append(result)

        pre_form = {
            result.get("Driver", {}).get("driverId", "unknown"): driver_snapshot(
                state, result.get("Driver", {}).get("driverId", "unknown")
            )["recent_finish_mean_5"]
            for result in results
        }

        for result in results:
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            driver_id = driver.get("driverId", "unknown")
            constructor_id = constructor.get("constructorId", "unknown")
            grid = safe_int(result.get("grid"), field_size)
            if grid <= 0:
                grid = field_size
            mates = [r for r in team_members[constructor_id] if r.get("Driver", {}).get("driverId") != driver_id]
            if mates:
                mate = mates[0]
                mate_id = mate.get("Driver", {}).get("driverId", "unknown")
                teammate_grid = safe_float(mate.get("grid"), grid)
                teammate_form = pre_form.get(mate_id, 12.0)
                teammate_elo = state["driver_elo"][mate_id]
            else:
                teammate_grid = float(grid)
                teammate_form = pre_form.get(driver_id, 12.0)
                teammate_elo = state["driver_elo"][driver_id]

            row = make_feature_row(
                state,
                driver_id=driver_id,
                constructor_id=constructor_id,
                circuit_id=circuit_id,
                grid=grid,
                field_size=field_size,
                round_no=round_no,
                season_rounds=rounds_by_season[season],
                driver_rank=driver_ranks.get(driver_id, max(len(driver_ranks) + 1, 10)),
                driver_points=state["season_driver_points"][driver_id],
                driver_total_points=driver_total,
                constructor_rank=constructor_ranks.get(constructor_id, max(len(constructor_ranks) + 1, 5)),
                constructor_points=state["season_constructor_points"][constructor_id],
                constructor_total_points=constructor_total,
                teammate_grid=teammate_grid,
                teammate_form=teammate_form,
                teammate_elo=teammate_elo,
            )
            finish = safe_int(result.get("position"), field_size)
            row.update(
                {
                    "race_id": race_id,
                    "race_index": race_index,
                    "season": season,
                    "round": round_no,
                    "driver_id": driver_id,
                    "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                    "constructor_id": constructor_id,
                    "constructor_name": constructor.get("name", constructor_id),
                    "finish_position": finish,
                    "relevance": max(field_size - finish, 0),
                    "winner": int(finish == 1),
                }
            )
            rows.append(row)

        # Update rolling state only after every pre-race row has been created.
        ids: list[str] = []
        positions: list[float] = []
        team_aggregates: dict[str, dict[str, Any]] = defaultdict(lambda: {"positions": [], "points": 0.0, "dnfs": []})
        for result in results:
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            driver_id = driver.get("driverId", "unknown")
            constructor_id = constructor.get("constructorId", "unknown")
            finish = safe_int(result.get("position"), field_size)
            grid = safe_int(result.get("grid"), field_size)
            if grid <= 0:
                grid = field_size
            points = safe_float(result.get("points"), 0.0)
            dnf = float(not finished_status(str(result.get("status", ""))))

            state["driver_finish"][driver_id].append(finish)
            state["driver_points"][driver_id].append(points)
            state["driver_grid"][driver_id].append(grid)
            state["driver_gain"][driver_id].append(grid - finish)
            state["driver_dnf"][driver_id].append(dnf)
            state["driver_win"][driver_id].append(float(finish == 1))
            state["driver_podium"][driver_id].append(float(finish <= 3))
            state["driver_top6"][driver_id].append(float(finish <= 6))
            state["circuit_driver_finish"][(driver_id, circuit_id)].append(finish)
            state["circuit_driver_gain"][(driver_id, circuit_id)].append(grid - finish)
            state["circuit_grids"][circuit_id].append(grid)
            state["circuit_finishes"][circuit_id].append(finish)
            state["circuit_dnf"][circuit_id].append(dnf)
            state["experience"][driver_id] += 1
            state["team_tenure"][(driver_id, constructor_id)] += 1
            state["season_driver_points"][driver_id] += points
            state["season_constructor_points"][constructor_id] += points
            ids.append(driver_id)
            positions.append(float(finish))

            team_aggregates[constructor_id]["positions"].append(finish)
            team_aggregates[constructor_id]["points"] += points
            team_aggregates[constructor_id]["dnfs"].append(dnf)

        state["circuit_pole_win"][circuit_id].append(
            float(any(safe_int(r.get("grid")) == 1 and safe_int(r.get("position")) == 1 for r in results))
        )
        update_pairwise_elo(state["driver_elo"], ids, positions, k=26.0)

        team_ids = list(team_aggregates)
        team_positions = [mean(team_aggregates[t]["positions"], field_size) for t in team_ids]
        update_pairwise_elo(state["constructor_elo"], team_ids, team_positions, k=18.0)
        for constructor_id, agg in team_aggregates.items():
            average_finish = mean(agg["positions"], 12.0)
            best_finish = min(agg["positions"])
            state["constructor_finish"][constructor_id].append(average_finish)
            state["constructor_best"][constructor_id].append(best_finish)
            state["constructor_points"][constructor_id].append(agg["points"])
            state["constructor_dnf"][constructor_id].append(mean(agg["dnfs"], 0.0))
            state["constructor_win"][constructor_id].append(float(best_finish == 1))
            state["constructor_podium"][constructor_id].append(float(best_finish <= 3))
            state["circuit_constructor_finish"][(constructor_id, circuit_id)].append(average_finish)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, state
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame, state


def create_ranker() -> XGBRanker:
    return XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=620,
        max_depth=4,
        learning_rate=0.025,
        min_child_weight=3.0,
        subsample=0.88,
        colsample_bytree=0.90,
        reg_alpha=0.08,
        reg_lambda=2.4,
        gamma=0.02,
        tree_method="hist",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=12,
        random_state=RNG_SEED,
        n_jobs=-1,
    )


def fit_ranker(frame: pd.DataFrame) -> XGBRanker:
    ordered = frame.sort_values(["race_index", "driver_id"]).reset_index(drop=True)
    model = create_ranker()
    model.fit(
        ordered[FEATURES].astype(float),
        ordered["relevance"].astype(int),
        qid=ordered["race_index"].astype(int).to_numpy(),
        verbose=False,
    )
    return model


def calibrate_temperature(scores_by_race: list[tuple[np.ndarray, int]]) -> float:
    if not scores_by_race:
        return 1.0
    candidates = np.linspace(0.12, 3.8, 100)
    best_temp, best_loss = 1.0, float("inf")
    for temp in candidates:
        loss = 0.0
        for scores, winner_index in scores_by_race:
            probs = softmax(scores, temp)
            loss -= math.log(max(probs[winner_index], 1e-12))
        loss /= len(scores_by_race)
        if loss < best_loss:
            best_loss, best_temp = loss, float(temp)
    return best_temp


@st.cache_resource(show_spinner=False)
def train_model(dataset: pd.DataFrame) -> ModelBundle:
    race_ids = list(dataset.sort_values("race_index")["race_id"].drop_duplicates())
    test_count = min(36, max(20, int(len(race_ids) * 0.16)))
    split = max(45, len(race_ids) - test_count)
    train_ids = set(race_ids[:split])
    test_ids = race_ids[split:]

    train = dataset[dataset["race_id"].isin(train_ids)].copy()
    test = dataset[dataset["race_id"].isin(test_ids)].copy()
    validation_model = fit_ranker(train)
    test["score"] = validation_model.predict(test[FEATURES].astype(float))

    winner_hits: list[float] = []
    podium_hits: list[float] = []
    correlations: list[float] = []
    position_errors: list[float] = []
    scores_for_calibration: list[tuple[np.ndarray, int]] = []

    for _, group in test.groupby("race_id", sort=False):
        group = group.copy().reset_index(drop=True)
        predicted = group.sort_values("score", ascending=False).reset_index(drop=True)
        actual_winner = group.loc[group["finish_position"].idxmin(), "driver_id"]
        winner_hits.append(float(predicted.iloc[0]["driver_id"] == actual_winner))
        podium_hits.append(float(actual_winner in set(predicted.head(3)["driver_id"])))
        predicted_rank_map = {driver: idx + 1 for idx, driver in enumerate(predicted["driver_id"])}
        group["predicted_rank"] = group["driver_id"].map(predicted_rank_map)
        corr = group[["predicted_rank", "finish_position"]].corr(method="spearman").iloc[0, 1]
        correlations.append(float(corr) if np.isfinite(corr) else 0.0)
        position_errors.extend(abs(group["predicted_rank"] - group["finish_position"]).tolist())
        winner_index = int(group["finish_position"].argmin())
        scores_for_calibration.append((group["score"].to_numpy(dtype=float), winner_index))

    temperature = calibrate_temperature(scores_for_calibration)
    final_model = fit_ranker(dataset)
    return ModelBundle(
        model=final_model,
        temperature=temperature,
        winner_accuracy=mean(winner_hits, 0.0),
        podium_capture=mean(podium_hits, 0.0),
        mean_spearman=mean(correlations, 0.0),
        mean_position_error=mean(position_errors, 0.0),
        test_races=len(test_ids),
        training_races=len(race_ids),
    )


def find_active_race(season: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schedule = fetch_schedule(season)
    if not schedule:
        raise DataError(f"No {season} schedule was returned.")
    completed_rounds = {safe_int(r.get("round")) for r in fetch_results(season) if r.get("Results")}
    candidates = [r for r in schedule if safe_int(r.get("round")) not in completed_rounds]
    if candidates:
        selected = min(candidates, key=lambda r: safe_int(r.get("round")))
    else:
        selected = schedule[-1]
    return schedule, selected


def openf1_meeting_for_race(season: int, race: dict[str, Any]) -> dict[str, Any] | None:
    meetings = openf1_get("meetings", (("year", season),))
    if not meetings:
        return None
    target = race_datetime(race)
    circuit_name = normalise_name(race.get("Circuit", {}).get("circuitName", ""))
    country = normalise_name(race.get("Circuit", {}).get("Location", {}).get("country", ""))

    def distance(item: dict[str, Any]) -> tuple[float, int]:
        date_distance = abs((parse_dt(item.get("date_start")) - target).total_seconds())
        names = normalise_name(
            " ".join(
                [
                    str(item.get("meeting_name", "")),
                    str(item.get("circuit_short_name", "")),
                    str(item.get("country_name", "")),
                ]
            )
        )
        name_penalty = 0 if (circuit_name and circuit_name[:7] in names) or (country and country in names) else 1
        return date_distance, name_penalty

    return min(meetings, key=distance)


def zscore_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = float(values.std())
    if not np.isfinite(sd) or sd < 1e-9:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - values.mean()) / sd


def analyse_practice_session(session: dict[str, Any]) -> pd.DataFrame:
    session_key = session.get("session_key")
    if session_key is None:
        return pd.DataFrame()
    laps = pd.DataFrame(openf1_get("laps", (("session_key", session_key),)))
    drivers = pd.DataFrame(openf1_get("drivers", (("session_key", session_key),)))
    stints = pd.DataFrame(openf1_get("stints", (("session_key", session_key),)))
    if laps.empty or drivers.empty or "driver_number" not in laps:
        return pd.DataFrame()

    laps["lap_duration"] = pd.to_numeric(laps.get("lap_duration"), errors="coerce")
    laps["lap_number"] = pd.to_numeric(laps.get("lap_number"), errors="coerce")
    if "is_pit_out_lap" in laps.columns:
        pit_out = laps["is_pit_out_lap"].fillna(False).astype(bool)
    else:
        pit_out = pd.Series(False, index=laps.index)
    laps = laps[(laps["lap_duration"].between(40, 180)) & (~pit_out)]
    if laps.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for number, group in laps.groupby("driver_number"):
        group = group.sort_values("lap_number").dropna(subset=["lap_duration"])
        if len(group) < 2:
            continue
        fastest = float(group["lap_duration"].min())
        quick = group.nsmallest(min(5, len(group)), "lap_duration")
        robust_pace = float(quick["lap_duration"].median())
        long_run = np.nan
        degradation = np.nan
        if not stints.empty and "driver_number" in stints:
            ds = stints[stints["driver_number"] == number].copy()
            if not ds.empty:
                ds["length"] = pd.to_numeric(ds.get("lap_end"), errors="coerce") - pd.to_numeric(
                    ds.get("lap_start"), errors="coerce"
                ) + 1
                ds = ds.sort_values("length", ascending=False)
                for _, stint in ds.iterrows():
                    start = safe_int(stint.get("lap_start"))
                    end = safe_int(stint.get("lap_end"))
                    stint_laps = group[group["lap_number"].between(start + 1, end - 1)].copy()
                    if len(stint_laps) < 4:
                        continue
                    med = float(stint_laps["lap_duration"].median())
                    stint_laps = stint_laps[stint_laps["lap_duration"] <= med + 4.0]
                    if len(stint_laps) < 4:
                        continue
                    long_run = float(stint_laps["lap_duration"].median())
                    x = stint_laps["lap_number"].to_numpy(dtype=float)
                    y = stint_laps["lap_duration"].to_numpy(dtype=float)
                    degradation = float(np.polyfit(x - x.min(), y, 1)[0])
                    break
        rows.append(
            {
                "driver_number": safe_int(number),
                "fastest": fastest,
                "robust_pace": robust_pace,
                "long_run": long_run,
                "degradation": degradation,
                "lap_count": len(group),
            }
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    metrics["practice_one_lap"] = -zscore_series(metrics["robust_pace"])
    metrics["practice_long_run"] = -zscore_series(metrics["long_run"]).fillna(0.0)
    metrics["practice_deg"] = -zscore_series(metrics["degradation"]).fillna(0.0)
    metrics["practice_laps"] = zscore_series(metrics["lap_count"]).fillna(0.0)
    metrics["session_index"] = (
        0.46 * metrics["practice_one_lap"]
        + 0.34 * metrics["practice_long_run"]
        + 0.12 * metrics["practice_deg"]
        + 0.08 * metrics["practice_laps"]
    ).clip(-3.0, 3.0)
    driver_names = drivers.drop_duplicates("driver_number").set_index("driver_number")["full_name"].to_dict()
    metrics["openf1_name"] = metrics["driver_number"].map(driver_names).fillna("")
    return metrics


@st.cache_data(ttl=60 * 20, show_spinner=False)
def load_openf1_context(season: int, race: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    meeting = openf1_meeting_for_race(season, race)
    if not meeting:
        return pd.DataFrame(), {}, {}
    meeting_key = meeting.get("meeting_key")
    sessions = openf1_get("sessions", (("meeting_key", meeting_key),))
    now = datetime.now(timezone.utc)
    completed = [s for s in sessions if parse_dt(s.get("date_end")) <= now and not s.get("is_cancelled")]
    practices = [s for s in completed if str(s.get("session_type", "")).lower() == "practice"][-3:]

    practice_frames: list[pd.DataFrame] = []
    weights: list[float] = []
    for session in practices:
        frame = analyse_practice_session(session)
        if not frame.empty:
            name = str(session.get("session_name", ""))
            weight = 0.22 if "1" in name else 0.38 if "2" in name else 0.40
            frame = frame.copy()
            frame["session_weight"] = weight
            practice_frames.append(frame)
            weights.append(weight)

    practice = pd.DataFrame()
    if practice_frames:
        combined = pd.concat(practice_frames, ignore_index=True)
        for col in ["session_index", "practice_one_lap", "practice_long_run", "practice_deg", "practice_laps"]:
            combined[f"weighted_{col}"] = combined[col] * combined["session_weight"]
        grouped = combined.groupby("openf1_name", as_index=False).agg(
            weight=("session_weight", "sum"),
            Auto_weekend=("weighted_session_index", "sum"),
            Practice_pace=("weighted_practice_one_lap", "sum"),
            Long_run=("weighted_practice_long_run", "sum"),
            Tyre_deg=("weighted_practice_deg", "sum"),
            Practice_laps=("weighted_practice_laps", "sum"),
        )
        for col in ["Auto_weekend", "Practice_pace", "Long_run", "Tyre_deg", "Practice_laps"]:
            grouped[col] = (grouped[col] / grouped["weight"].replace(0, np.nan)).fillna(0.0).clip(-3, 3)
        practice = grouped.drop(columns="weight")

    race_sessions = [s for s in sessions if str(s.get("session_type", "")).lower() == "race"]
    grid_map: dict[str, int] = {}
    if race_sessions:
        race_session = race_sessions[-1]
        race_key = race_session.get("session_key")
        grid = openf1_get("starting_grid", (("session_key", race_key),))
        driver_source = completed[-1] if completed else race_session
        drivers = openf1_get("drivers", (("session_key", driver_source.get("session_key")),))
        names = {safe_int(d.get("driver_number")): str(d.get("full_name", "")) for d in drivers}
        for item in grid:
            name = names.get(safe_int(item.get("driver_number")), "")
            if name:
                grid_map[normalise_name(name)] = safe_int(item.get("position"), 20)

    weather_rows = openf1_get("weather", (("meeting_key", meeting_key),))
    weather = weather_rows[-1] if weather_rows else {}
    return practice, grid_map, weather


def name_match_key(full_name: str) -> tuple[str, str]:
    norm = normalise_name(full_name)
    parts = re.findall(r"[A-Za-zÀ-ÿ'-]+", full_name or "")
    last = normalise_name(parts[-1]) if parts else norm
    return norm, last


def match_external_name(candidate_name: str, available: Iterable[str]) -> str | None:
    norm, last = name_match_key(candidate_name)
    candidates = list(available)
    exact = {normalise_name(item): item for item in candidates}
    if norm in exact:
        return exact[norm]
    for item in candidates:
        item_norm = normalise_name(item)
        if last and (item_norm.endswith(last) or last in item_norm):
            return item
    return None


def build_current_candidates(
    season: int,
    race: dict[str, Any],
    state: dict[str, Any],
    practice: pd.DataFrame,
    openf1_grid: dict[str, int],
) -> tuple[pd.DataFrame, str]:
    round_no = safe_int(race.get("round"))
    circuit_id = race.get("Circuit", {}).get("circuitId", "unknown")
    standings = fetch_driver_standings(season)
    constructors = fetch_constructor_standings(season)
    qualifying = fetch_qualifying(season, round_no)

    qualifying_map = {
        item.get("Driver", {}).get("driverId", "unknown"): safe_int(item.get("position"), 20) for item in qualifying
    }
    constructor_rank = {
        item.get("Constructor", {}).get("constructorId", "unknown"): safe_int(item.get("position"), index + 1)
        for index, item in enumerate(constructors)
    }
    constructor_points = {
        item.get("Constructor", {}).get("constructorId", "unknown"): safe_float(item.get("points"), 0.0)
        for item in constructors
    }
    total_constructor_points = sum(constructor_points.values())

    source: list[dict[str, Any]] = standings
    if not source:
        latest = fetch_results(season)
        last_results = latest[-1].get("Results", []) if latest else []
        source = [
            {
                "position": index + 1,
                "points": result.get("points", 0),
                "Driver": result.get("Driver", {}),
                "Constructors": [result.get("Constructor", {})],
            }
            for index, result in enumerate(last_results)
        ]
    if not source and qualifying:
        source = [
            {
                "position": index + 1,
                "points": 0,
                "Driver": item.get("Driver", {}),
                "Constructors": [item.get("Constructor", {})],
            }
            for index, item in enumerate(qualifying)
        ]
    if not source:
        raise DataError("No current driver list was available.")

    driver_points_map = {
        item.get("Driver", {}).get("driverId", "unknown"): safe_float(item.get("points"), 0.0) for item in source
    }
    total_driver_points = sum(driver_points_map.values())
    field_size = len(source)
    rounds = max([safe_int(r.get("round")) for r in fetch_schedule(season)] or [24])

    prelim: list[dict[str, Any]] = []
    practice_lookup = practice.set_index("openf1_name").to_dict("index") if not practice.empty else {}
    for index, item in enumerate(source):
        driver = item.get("Driver", {})
        constructor_list = item.get("Constructors", [])
        constructor = constructor_list[-1] if constructor_list else {}
        driver_id = driver.get("driverId", "unknown")
        constructor_id = constructor.get("constructorId", "unknown")
        name = f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip()
        external_grid_name = match_external_name(name, openf1_grid.keys())
        if external_grid_name:
            grid = openf1_grid[external_grid_name]
            grid_source = "Official OpenF1 starting grid"
        elif driver_id in qualifying_map:
            grid = qualifying_map[driver_id]
            grid_source = "Qualifying classification"
        else:
            grid = min(safe_int(item.get("position"), index + 1), field_size)
            grid_source = "Championship estimate"
        practice_name = match_external_name(name, practice_lookup.keys())
        p = practice_lookup.get(practice_name, {}) if practice_name else {}
        prelim.append(
            {
                "driver_id": driver_id,
                "Driver": name,
                "family_name": driver.get("familyName", ""),
                "constructor_id": constructor_id,
                "Team": constructor.get("name", constructor_id),
                "grid": max(1, min(grid, field_size)),
                "driver_rank": safe_int(item.get("position"), index + 1),
                "driver_points": driver_points_map[driver_id],
                "constructor_rank": constructor_rank.get(constructor_id, max(len(constructor_rank) + 1, 5)),
                "constructor_points": constructor_points.get(constructor_id, state["season_constructor_points"][constructor_id]),
                "Auto weekend": safe_float(p.get("Auto_weekend"), 0.0),
                "Practice pace": safe_float(p.get("Practice_pace"), 0.0),
                "Long-run pace": safe_float(p.get("Long_run"), 0.0),
                "Tyre management": safe_float(p.get("Tyre_deg"), 0.0),
                "Practice laps": safe_float(p.get("Practice_laps"), 0.0),
                "grid_source": grid_source,
            }
        )

    by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prelim:
        by_team[item["constructor_id"]].append(item)

    rows: list[dict[str, Any]] = []
    for item in prelim:
        mates = [m for m in by_team[item["constructor_id"]] if m["driver_id"] != item["driver_id"]]
        if mates:
            mate = mates[0]
            teammate_grid = mate["grid"]
            teammate_form = driver_snapshot(state, mate["driver_id"])["recent_finish_mean_5"]
            teammate_elo = state["driver_elo"][mate["driver_id"]]
        else:
            teammate_grid = item["grid"]
            teammate_form = driver_snapshot(state, item["driver_id"])["recent_finish_mean_5"]
            teammate_elo = state["driver_elo"][item["driver_id"]]
        features = make_feature_row(
            state,
            driver_id=item["driver_id"],
            constructor_id=item["constructor_id"],
            circuit_id=circuit_id,
            grid=item["grid"],
            field_size=field_size,
            round_no=round_no,
            season_rounds=rounds,
            driver_rank=item["driver_rank"],
            driver_points=item["driver_points"],
            driver_total_points=total_driver_points,
            constructor_rank=item["constructor_rank"],
            constructor_points=item["constructor_points"],
            constructor_total_points=total_constructor_points,
            teammate_grid=teammate_grid,
            teammate_form=teammate_form,
            teammate_elo=teammate_elo,
        )
        features.update(item)
        features.update(
            {
                "Pace adjustment": 0.0,
                "Upgrade": 0.0,
                "Reliability adjustment": 0.0,
                "Wet skill": 0.0,
                "Start/incident risk": 0.0,
            }
        )
        rows.append(features)

    frame = pd.DataFrame(rows)
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sources = set(frame["grid_source"])
    source_label = ", ".join(sorted(sources))
    return frame, source_label


def feature_value_text(feature: str, value: float) -> str:
    if feature == "grid":
        return f"P{int(round(value))}"
    if feature.endswith("rate_5") or feature.endswith("rate_10") or "share" in feature or "pole_win_rate" in feature:
        return f"{value * 100:.0f}%"
    if "finish" in feature or "grid_mean" in feature:
        return f"P{value:.1f}"
    if "elo" in feature:
        return f"{value:+.2f} rating units"
    if "gain" in feature:
        return f"{value:+.1f} positions"
    return f"{value:.2f}"


def shap_explanations(model: XGBRanker, candidates: pd.DataFrame) -> dict[str, dict[str, Any]]:
    matrix = xgb.DMatrix(candidates[FEATURES].astype(float), feature_names=FEATURES)
    contributions = model.get_booster().predict(matrix, pred_contribs=True)
    output: dict[str, dict[str, Any]] = {}
    for row_index, (_, row) in enumerate(candidates.iterrows()):
        contrib = contributions[row_index][:-1]
        pairs = sorted(zip(FEATURES, contrib), key=lambda item: abs(item[1]), reverse=True)
        positives = [item for item in pairs if item[1] > 0][:3]
        negatives = [item for item in pairs if item[1] < 0][:2]
        positive_text = [
            f"{FEATURE_LABELS.get(feature, feature)} ({feature_value_text(feature, safe_float(row[feature]))})"
            for feature, _ in positives
        ]
        negative_text = [
            f"{FEATURE_LABELS.get(feature, feature)} ({feature_value_text(feature, safe_float(row[feature]))})"
            for feature, _ in negatives
        ]
        output[row["driver_id"]] = {
            "positive": positive_text,
            "negative": negative_text,
            "contributions": {feature: float(value) for feature, value in zip(FEATURES, contrib)},
        }
    return output


def simulate_race(
    candidates: pd.DataFrame,
    bundle: ModelBundle,
    *,
    rain_probability: float,
    safety_car_probability: float,
    simulations: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frame = candidates.copy().reset_index(drop=True)
    base_score = bundle.model.predict(frame[FEATURES].astype(float)).astype(float)
    score_scale = max(float(np.std(base_score)), 0.18)

    context_index = (
        0.20 * frame["Auto weekend"].astype(float).to_numpy()
        + 0.10 * frame["Pace adjustment"].astype(float).to_numpy()
        + 0.06 * frame["Upgrade"].astype(float).to_numpy()
        + 0.08 * frame["Reliability adjustment"].astype(float).to_numpy()
        + 0.06 * rain_probability * frame["Wet skill"].astype(float).to_numpy()
        - 0.05 * frame["Start/incident risk"].astype(float).to_numpy()
    )
    adjusted_score = base_score + score_scale * context_index

    dnf_base = np.clip(
        0.025
        + 0.62 * frame["recent_dnf_rate_10"].astype(float).to_numpy()
        + 0.28 * frame["constructor_dnf_rate_10"].astype(float).to_numpy(),
        0.02,
        0.42,
    )
    dnf_logit = np.log(dnf_base / (1 - dnf_base))
    dnf_logit += (
        -0.48 * frame["Reliability adjustment"].astype(float).to_numpy()
        + 0.22 * frame["Start/incident risk"].astype(float).to_numpy()
        + 0.20 * rain_probability
        + 0.08 * safety_car_probability
    )
    dnf_probability = np.clip(sigmoid(dnf_logit), 0.01, 0.58)

    rng = np.random.default_rng(RNG_SEED)
    temperature = bundle.temperature * (1.0 + 0.34 * rain_probability + 0.18 * safety_car_probability)
    utilities = adjusted_score[None, :] + rng.gumbel(
        loc=0.0, scale=max(temperature, 0.05), size=(simulations, len(frame))
    )
    dnf_mask = rng.random((simulations, len(frame))) < dnf_probability[None, :]
    if dnf_mask.any():
        retirement_penalty = 5.2 * score_scale + rng.exponential(1.7 * score_scale, size=utilities.shape)
        utilities = np.where(dnf_mask, utilities - retirement_penalty, utilities)

    order = np.argsort(-utilities, axis=1)
    positions = np.empty_like(order)
    positions[np.arange(simulations)[:, None], order] = np.arange(1, len(frame) + 1)

    frame["Base score"] = base_score
    frame["Context index"] = context_index
    frame["Adjusted score"] = adjusted_score
    frame["Win %"] = (positions == 1).mean(axis=0) * 100
    frame["Podium %"] = (positions <= 3).mean(axis=0) * 100
    frame["Top 6 %"] = (positions <= 6).mean(axis=0) * 100
    frame["Expected finish"] = positions.mean(axis=0)
    frame["DNF %"] = dnf_probability * 100
    frame["Expected gain"] = frame["grid"].astype(float) - frame["Expected finish"]

    explanations = shap_explanations(bundle.model, frame)
    why: list[str] = []
    risks: list[str] = []
    for _, row in frame.iterrows():
        expl = explanations[row["driver_id"]]
        reason_parts = expl["positive"][:2]
        if row["Auto weekend"] >= 0.55:
            reason_parts.append(f"strong practice index ({row['Auto weekend']:+.2f})")
        if row["Upgrade"] >= 0.5:
            reason_parts.append(f"positive upgrade input ({row['Upgrade']:+.1f})")
        why.append("; ".join(reason_parts) if reason_parts else "balanced profile without one dominant advantage")

        risk_parts = expl["negative"][:1]
        if row["DNF %"] >= 18:
            risk_parts.append(f"{row['DNF %']:.0f}% estimated retirement risk")
        if row["grid"] >= 10:
            risk_parts.append(f"starts P{int(row['grid'])}")
        risks.append("; ".join(risk_parts) if risk_parts else "no major model-level weakness")
    frame["Why"] = why
    frame["Main risk"] = risks

    frame = frame.sort_values(["Expected finish", "Win %"], ascending=[True, False]).reset_index(drop=True)
    frame.insert(0, "Rank", np.arange(1, len(frame) + 1))
    return frame, explanations


def prediction_receipt(race: dict[str, Any], prediction: pd.DataFrame, metadata: dict[str, Any]) -> bytes:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "app_version": APP_VERSION,
        "model": "XGBoost LambdaMART rank:ndcg",
        "race": race.get("raceName"),
        "season": race.get("season"),
        "round": race.get("round"),
        "metadata": metadata,
        "ranking": prediction[
            ["Rank", "Driver", "Team", "grid", "Win %", "Podium %", "Top 6 %", "Expected finish", "DNF %"]
        ].round(4).to_dict("records"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return json.dumps(payload, indent=2).encode("utf-8")


def render_driver_detail(row: pd.Series, explanations: dict[str, dict[str, Any]]) -> None:
    detail = explanations.get(row["driver_id"], {})
    st.markdown(f"### {row['Driver']} — predicted P{int(row['Rank'])}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win", f"{row['Win %']:.1f}%")
    c2.metric("Podium", f"{row['Podium %']:.1f}%")
    c3.metric("Expected finish", f"P{row['Expected finish']:.1f}")
    c4.metric("DNF estimate", f"{row['DNF %']:.1f}%")
    st.markdown(f"**Why the model ranks them here:** {row['Why']}.")
    st.markdown(f"**Main risk:** {row['Main risk']}.")

    contributions = detail.get("contributions", {})
    if contributions:
        table = pd.DataFrame(
            [
                {
                    "Factor": FEATURE_LABELS.get(feature, feature),
                    "Model contribution": value,
                    "Direction": "Helps" if value > 0 else "Hurts",
                }
                for feature, value in sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)


# ------------------------------- UI ---------------------------------

st.markdown(
    f"""
<div class="hero">
  <div class="eyebrow">Apex Oracle Pro · Version {APP_VERSION}</div>
  <h1>One model. Every driver ranked.</h1>
  <p>A single LambdaMART learning-to-rank model evaluates the entire Formula 1 field together, then a transparent race simulation converts its scores into winner, podium, top-six and expected-finish probabilities.</p>
  <span class="pill">Single XGBoost ranker</span><span class="pill">No betting odds</span><span class="pill">Time-safe features</span><span class="pill">OpenF1 practice layer</span><span class="pill">SHAP explanations</span>
</div>
""",
    unsafe_allow_html=True,
)

try:
    now = datetime.now(timezone.utc)
    season = now.year
    schedule, default_race = find_active_race(season)
    race_names = [f"R{safe_int(r.get('round'))} · {r.get('raceName')}" for r in schedule]
    default_index = schedule.index(default_race)
    selected_label = st.selectbox("Race", race_names, index=default_index)
    selected_race = schedule[race_names.index(selected_label)]

    location = selected_race.get("Circuit", {}).get("Location", {})
    race_date = race_datetime(selected_race)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(
            f'<div class="card"><div class="card-title">Selected race</div><div class="big">{selected_race.get("raceName")}</div><div class="sub">{location.get("locality")}, {location.get("country")}</div></div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f'<div class="card"><div class="card-title">Race start</div><div class="big">{race_date.strftime("%d %b")}</div><div class="sub">{race_date.strftime("%H:%M UTC")}</div></div>',
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            '<div class="card"><div class="card-title">Prediction philosophy</div><div class="big">Rank the field</div><div class="sub">Optimises complete order—not only the winner</div></div>',
            unsafe_allow_html=True,
        )

    with st.spinner("Loading race history, building leakage-safe features and training LambdaMART…"):
        history = load_history(max(2014, season - 12), season)
        dataset, state = build_dataset(history)
        if dataset.empty or dataset["race_id"].nunique() < 80:
            raise DataError("Not enough historical races were available to train the ranking model.")
        bundle = train_model(dataset)
        practice, official_grid, weather = load_openf1_context(season, selected_race)
        candidates, grid_source = build_current_candidates(season, selected_race, state, practice, official_grid)

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Races learned", bundle.training_races)
    m2.metric("Held-out winner accuracy", f"{bundle.winner_accuracy * 100:.1f}%")
    m3.metric("Winner inside predicted top 3", f"{bundle.podium_capture * 100:.1f}%")
    m4.metric("Mean rank correlation", f"{bundle.mean_spearman:.2f}")
    st.caption(
        f"Validation uses the newest {bundle.test_races} races as a forward holdout. Mean absolute finishing-position error: {bundle.mean_position_error:.2f}. Grid source: {grid_source}."
    )

    if "Championship estimate" in grid_source:
        st.markdown(
            '<div class="warning">A final grid was not available for every driver. Correct the grid below after penalties are confirmed; grid position is one of the strongest race predictors.</div>',
            unsafe_allow_html=True,
        )

    if weather:
        st.caption(
            f"Latest track reading from OpenF1: air {safe_float(weather.get('air_temperature')):.1f}°C · track {safe_float(weather.get('track_temperature')):.1f}°C · humidity {safe_float(weather.get('humidity')):.0f}% · rainfall {'yes' if weather.get('rainfall') else 'no'}."
        )

    st.subheader("Final inputs")
    st.caption(
        "The model already knows the grid, championship, Elo strength, recent form, reliability, teammate comparison and circuit history. OpenF1 practice analysis is added automatically when available. Only change manual columns for confirmed information."
    )

    editor_cols = [
        "Driver",
        "Team",
        "grid",
        "Auto weekend",
        "Pace adjustment",
        "Upgrade",
        "Reliability adjustment",
        "Wet skill",
        "Start/incident risk",
    ]
    edited = st.data_editor(
        candidates[editor_cols],
        hide_index=True,
        use_container_width=True,
        disabled=["Driver", "Team", "Auto weekend"],
        column_config={
            "grid": st.column_config.NumberColumn("Final grid", min_value=1, max_value=len(candidates), step=1),
            "Auto weekend": st.column_config.NumberColumn("Auto practice", format="%.2f"),
            "Pace adjustment": st.column_config.NumberColumn("Manual pace", min_value=-3.0, max_value=3.0, step=0.5),
            "Upgrade": st.column_config.NumberColumn("Upgrade", min_value=-3.0, max_value=3.0, step=0.5),
            "Reliability adjustment": st.column_config.NumberColumn("Reliability", min_value=-3.0, max_value=3.0, step=0.5),
            "Wet skill": st.column_config.NumberColumn("Wet skill", min_value=-3.0, max_value=3.0, step=0.5),
            "Start/incident risk": st.column_config.NumberColumn("Incident risk", min_value=-3.0, max_value=3.0, step=0.5),
        },
        key=f"inputs-{season}-{selected_race.get('round')}",
    )
    update_cols = [
        "Driver",
        "grid",
        "Pace adjustment",
        "Upgrade",
        "Reliability adjustment",
        "Wet skill",
        "Start/incident risk",
    ]
    candidates = candidates.drop(columns=[c for c in update_cols if c != "Driver"]).merge(
        edited[update_cols], on="Driver", how="left"
    )
    # Keep all grid-derived model features consistent when the user corrects a penalty or pit-lane start.
    candidates["grid"] = pd.to_numeric(candidates["grid"], errors="coerce").fillna(len(candidates)).clip(1, len(candidates))
    candidates["grid_norm"] = (candidates["grid"] - 1) / max(len(candidates) - 1, 1)
    candidates["grid_delta_teammate"] = 0.0
    for _, team_group in candidates.groupby("constructor_id"):
        indices = team_group.index.tolist()
        for idx in indices:
            others = [other for other in indices if other != idx]
            if others:
                candidates.loc[idx, "grid_delta_teammate"] = float(
                    candidates.loc[idx, "grid"] - candidates.loc[others, "grid"].mean()
                )

    s1, s2, s3 = st.columns(3)
    with s1:
        rain = st.slider("Race rain probability", 0, 100, 10, 5) / 100.0
    with s2:
        safety_car = st.slider("Safety-car probability", 0, 100, 35, 5) / 100.0
    with s3:
        simulations = st.select_slider("Race simulations", [10000, 25000, 50000, 100000], value=50000)

    if st.button("Generate complete race ranking", type="primary", use_container_width=True):
        prediction, explanations = simulate_race(
            candidates,
            bundle,
            rain_probability=rain,
            safety_car_probability=safety_car,
            simulations=simulations,
        )
        st.session_state["prediction"] = prediction
        st.session_state["explanations"] = explanations
        st.session_state["prediction_race"] = selected_race.get("raceName")
        st.session_state["prediction_meta"] = {
            "rain_probability": rain,
            "safety_car_probability": safety_car,
            "simulations": simulations,
            "grid_source": grid_source,
            "temperature": bundle.temperature,
        }

    prediction = st.session_state.get("prediction")
    explanations = st.session_state.get("explanations", {})
    if prediction is not None and st.session_state.get("prediction_race") == selected_race.get("raceName"):
        winner = prediction.iloc[0]
        surprise_pool = prediction[(prediction["grid"] >= 8) & (prediction["Expected gain"] >= 1.0)]
        if surprise_pool.empty:
            surprise_pool = prediction.iloc[1:].copy()
        surprise = surprise_pool.sort_values(["Expected gain", "Top 6 %"], ascending=False).iloc[0]

        st.divider()
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown(
                f"""
<div class="winner-card">
  <div class="eyebrow">Model's predicted winner</div>
  <div class="winner-name">{winner['Driver']}</div>
  <div class="prob">{winner['Win %']:.1f}% win probability</div>
  <div class="sub">{winner['Team']} · starts P{int(winner['grid'])} · expected finish P{winner['Expected finish']:.1f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(f"**Why:** {winner['Why']}.")
            st.markdown(
                f"**Biggest surprise:** **{surprise['Driver']}**—predicted P{int(surprise['Rank'])} from grid P{int(surprise['grid'])}, with {surprise['Top 6 %']:.1f}% top-six probability."
            )
        with right:
            st.markdown('<div class="card-title">Top six prediction</div>', unsafe_allow_html=True)
            top = prediction.head(6)[["Rank", "Driver", "Win %", "Expected finish"]].copy()
            top["Win %"] = top["Win %"].map(lambda x: f"{x:.1f}%")
            top["Expected finish"] = top["Expected finish"].map(lambda x: f"P{x:.1f}")
            st.dataframe(top, hide_index=True, use_container_width=True)

        st.subheader("Complete predicted order: first to last")
        output = prediction[
            [
                "Rank",
                "Driver",
                "Team",
                "grid",
                "Win %",
                "Podium %",
                "Top 6 %",
                "Expected finish",
                "Expected gain",
                "DNF %",
                "Why",
                "Main risk",
            ]
        ].copy()
        output = output.rename(columns={"grid": "Grid"})
        for col in ["Win %", "Podium %", "Top 6 %", "Expected finish", "Expected gain", "DNF %"]:
            output[col] = output[col].round(1)
        st.dataframe(output, hide_index=True, use_container_width=True, height=min(760, 40 + 35 * len(output)))

        selected_driver = st.selectbox("Explain one driver in full", prediction["Driver"].tolist())
        render_driver_detail(prediction[prediction["Driver"] == selected_driver].iloc[0], explanations)

        metadata = st.session_state.get("prediction_meta", {})
        receipt = prediction_receipt(selected_race, prediction, metadata)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download full prediction CSV",
                prediction.to_csv(index=False).encode("utf-8"),
                file_name=f"{season}_R{selected_race.get('round')}_apex_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Download timestamped prediction receipt",
                receipt,
                file_name=f"{season}_R{selected_race.get('round')}_prediction_receipt.json",
                mime="application/json",
                use_container_width=True,
            )

    with st.expander("Exactly how the one-model calculation works"):
        st.markdown(
            """
### The only learned model: XGBoost LambdaMART

Every race is treated as one ranking group. The model compares drivers **within the same race** and learns to order the complete field. Its `rank:ndcg` objective rewards getting the top of the order right while still learning from every finishing position.

**Historical inputs are frozen before each race**, so the training rows cannot see the result they are trying to predict. The model uses final grid, championship position, driver and constructor Elo, recent race and qualifying form, DNF history, teammate comparison, circuit history and experience.

### Race-weekend calculation

```text
Final score = LambdaMART base score
            + 0.20 × automatic OpenF1 practice index
            + 0.10 × manual pace input
            + 0.06 × upgrade input
            + 0.08 × reliability input
            + 0.06 × rain probability × wet-skill input
            − 0.05 × incident-risk input
```

The OpenF1 practice index combines robust one-lap pace (46%), longest-stint pace (34%), degradation (12%) and lap count (8%). It is deliberately smaller than the learned model because public practice data cannot reveal exact fuel loads or engine modes.

### Probabilities

The newest historical races are held out in time order. Their model scores calibrate the simulation temperature. Monte Carlo then produces complete race orders, while empirical driver/team reliability creates separate retirement risk. Rain and safety-car probabilities increase uncertainty rather than automatically selecting a different winner.

### Why each driver is ranked there

The site uses XGBoost contribution values to show which historical factors raised or lowered each driver's model score. These are model explanations—not invented prose.
"""
        )

    st.caption(
        "This is a high-end public-data forecasting system, not a guarantee. No public model can access private fuel loads, setup targets, team simulators or last-minute failures. World-best status must be earned through a long, timestamped live prediction record. Data: Jolpica and OpenF1."
    )

except DataError as exc:
    st.error(str(exc))
    st.info("Check the connection and rerun. The app deliberately refuses to invent fallback race data.")
except Exception as exc:
    st.error(f"The predictor stopped safely: {exc}")
    st.info("Rerun once. If it repeats, copy the full error from Streamlit's Manage app → Logs.")
