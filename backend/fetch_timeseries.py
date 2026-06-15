"""
fetch_timeseries.py
===================
Télécharge les séismes de la semaine écoulée (lundi → dimanche),
calcule les features hebdomadaires (lags, rolling, energy...),
prédit le nb de séismes pour la semaine prochaine,
et sauvegarde → data/output_forecast.json
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── CHEMINS ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

XGB_PATH = BASE_DIR / "models" / "model_xgb_global.pkl"
XGB_LOW_PATH = BASE_DIR / "models" / "model_xgb_global_low.pkl"
XGB_HIGH_PATH = BASE_DIR / "models" / "model_xgb_global_high.pkl"

HISTORY_CSV = BASE_DIR / "data" / "weekly_history.csv"
OUTPUT = BASE_DIR / "data" / "output_forecast.json"

FEATURES = [
    "nb_zones",
    "energy_per_seisme", "log_energy",
    "energy_lag_1", "energy_rolling_4",
    "mag_max", "mag_spread",
    "depth_mean",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_4", "rolling_mean_8", "rolling_mean_26",
    "rolling_std_8", "rolling_max_4",
    "zones_lag_1", "zones_lag_2",
    "week_sin",
]

def get_week_bounds(ref_date=None):
    """Retourne (lundi_start, dimanche_end) pour la semaine contenant ref_date (par défaut aujourd'hui)."""
    if ref_date is None:
        ref_date = datetime.now(timezone.utc)
    # Lundi de la semaine
    start = ref_date - timedelta(days=ref_date.weekday())
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return start, end

def fetch_week_events(start_date, end_date):
    """Télécharge tous les séismes entre start_date et end_date (inclus)."""
    params = {
        "format": "geojson",
        "starttime": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 2.0,
        "orderby": "time",
    }
    resp = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params=params, timeout=30
    )
    resp.raise_for_status()
    return resp.json()

def aggregate_week(events, week_end_date):
    """Agrège les événements d'une semaine en une ligne de features."""
    df = pd.DataFrame([{
        "time_utc": pd.to_datetime(f["properties"]["time"], unit="ms", utc=True),
        "latitude": f["geometry"]["coordinates"][1],
        "longitude": f["geometry"]["coordinates"][0],
        "depth_km": f["geometry"]["coordinates"][2],
        "mag": f["properties"].get("mag"),
        "place": f["properties"].get("place"),
        "tsunami": f["properties"].get("tsunami", 0),
    } for f in events["features"] if f["properties"].get("type") == "earthquake"])
    
    if len(df) == 0:
        return {
            "date": week_end_date,
            "nb_seismes": 0,
            "mag_max": 0.0,
            "mag_mean": 0.0,
            "mag_median": 0.0,
            "depth_mean": 0.0,
            "tsunami_sum": 0,
            "nb_zones": 0,
            "energy_sum": 0.0,
        }
    
    df["energy"] = 10 ** (1.5 * df["mag"].fillna(0))
    return {
        "date": week_end_date,
        "nb_seismes": len(df),
        "mag_max": df["mag"].max() if len(df) else 0.0,
        "mag_mean": df["mag"].mean() if len(df) else 0.0,
        "mag_median": df["mag"].median() if len(df) else 0.0,
        "depth_mean": df["depth_km"].mean() if len(df) else 0.0,
        "tsunami_sum": df["tsunami"].sum(),
        "nb_zones": df["place"].nunique(),
        "energy_sum": df["energy"].sum(),
    }

def run():
    print("=" * 55)
    print("  TIMESERIES GLOBAL — Semaines fixes (lundi → dimanche)")
    print("=" * 55)

    # Chargement modèles
    if not XGB_PATH.exists():
        print(f"❌ Modèle central introuvable : {XGB_PATH}")
        return
    xgb_model = joblib.load(XGB_PATH)
    print(f"✅ Modèle central chargé")
    
    xgb_low = joblib.load(XGB_LOW_PATH) if XGB_LOW_PATH.exists() else None
    xgb_high = joblib.load(XGB_HIGH_PATH) if XGB_HIGH_PATH.exists() else None

    # Semaine en cours (lundi → dimanche)
    now = datetime.now(timezone.utc)
    week_start, week_end = get_week_bounds(now)
    print(f"\n📅 Semaine analysée : {week_start.date()} → {week_end.date()}")

    # Récupération des événements de cette semaine
    events = fetch_week_events(week_start, week_end)
    print(f"📊 {len(events['features'])} événements récupérés")

    new_row = aggregate_week(events, week_end.date())

    # Chargement / mise à jour de l'historique
    if HISTORY_CSV.exists():
        weekly = pd.read_csv(HISTORY_CSV, parse_dates=["date"])
        # Supprimer l'ancienne ligne de la même semaine si elle existe
        weekly = weekly[weekly["date"] != pd.Timestamp(week_end.date())].copy()
    else:
        weekly = pd.DataFrame()

    new_df = pd.DataFrame([new_row])
    weekly = pd.concat([weekly, new_df], ignore_index=True)
    weekly = weekly.sort_values("date").reset_index(drop=True)

    # Recalcul des features sur tout l'historique
    weekly["energy_per_seisme"] = weekly["energy_sum"] / (weekly["nb_seismes"] + 1)
    weekly["log_energy"] = np.log1p(weekly["energy_sum"])
    weekly["mag_spread"] = weekly["mag_max"] - weekly["mag_mean"]

    for lag in [1, 2, 3, 4, 8, 12, 52]:
        weekly[f"lag_{lag}"] = weekly["nb_seismes"].shift(lag)
    for lag in [1, 2, 4]:
        weekly[f"energy_lag_{lag}"] = weekly["energy_sum"].shift(lag)
    for lag in [1, 2]:
        weekly[f"zones_lag_{lag}"] = weekly["nb_zones"].shift(lag)
    for w in [4, 8, 12, 26]:
        weekly[f"rolling_mean_{w}"] = weekly["nb_seismes"].shift(1).rolling(w).mean()
    for w in [4, 8, 12]:
        weekly[f"rolling_std_{w}"] = weekly["nb_seismes"].shift(1).rolling(w).std()
    for w in [4, 8]:
        weekly[f"rolling_max_{w}"] = weekly["nb_seismes"].shift(1).rolling(w).max()
    weekly["energy_rolling_4"] = weekly["energy_sum"].shift(1).rolling(4).mean()
    weekly["energy_rolling_8"] = weekly["energy_sum"].shift(1).rolling(8).mean()
    weekly["week_of_year"] = weekly["date"].dt.isocalendar().week.astype(int)
    weekly["week_sin"] = np.sin(2 * np.pi * weekly["week_of_year"] / 52)
    weekly["week_cos"] = np.cos(2 * np.pi * weekly["week_of_year"] / 52)

    weekly.to_csv(HISTORY_CSV, index=False)
    print(f"💾 Historique mis à jour → {len(weekly)} semaines")

    # Préparation de la prédiction pour la semaine suivante
    last = weekly.iloc[-1]
    next_date = last["date"] + timedelta(days=7)
    next_week_num = next_date.isocalendar()[1]
    
    X_next_dict = {
        "nb_zones": last["nb_zones"],
        "energy_per_seisme": last["energy_per_seisme"],
        "log_energy": last["log_energy"],
        "energy_lag_1": last["energy_sum"],
        "energy_rolling_4": last["energy_rolling_4"],
        "mag_max": last["mag_max"],
        "mag_spread": last["mag_spread"],
        "depth_mean": last["depth_mean"],
        "lag_1": last["nb_seismes"],
        "lag_2": last["lag_1"],
        "lag_3": last["lag_2"],
        "rolling_mean_4": last["rolling_mean_4"],
        "rolling_mean_8": last["rolling_mean_8"],
        "rolling_mean_26": last["rolling_mean_26"],
        "rolling_std_8": last["rolling_std_8"],
        "rolling_max_4": last["rolling_max_4"],
        "zones_lag_1": last["nb_zones"],
        "zones_lag_2": last["zones_lag_1"],
        "week_sin": np.sin(2 * np.pi * next_week_num / 52),
    }
    X_next = np.array([[X_next_dict[f] for f in FEATURES]])

    pred_central = int(np.clip(xgb_model.predict(X_next)[0], 0, None))
    pred_low = int(np.clip(xgb_low.predict(X_next)[0], 0, None)) if xgb_low else max(0, pred_central - 50)
    pred_high = int(np.clip(xgb_high.predict(X_next)[0], 0, None)) if xgb_high else pred_central + 50

    print(f"\n🔮 Prédiction semaine du {next_date.date()}")
    print(f"   Centrale  : {pred_central} séismes")
    print(f"   Intervalle: [{pred_low} – {pred_high}]")
    print(f"   Semaine actuelle : {int(last['nb_seismes'])} séismes")

    # Dernières 12 semaines pour le graphique
    last_12 = weekly.tail(12)[["date", "nb_seismes"]].copy()
    last_12["date"] = last_12["date"].dt.strftime("%Y-%m-%d")
    history_list = last_12.to_dict(orient="records")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "next_week_start": str(next_date.date()),
        "model_used": XGB_PATH.name,
        "prediction": {
            "central": pred_central,
            "low": pred_low,
            "high": pred_high,
        },
        "current_week": {
            "start": str(last["date"].date()),
            "nb_seismes": int(last["nb_seismes"]),
            "mag_max": round(float(last["mag_max"]), 2),
            "nb_zones": int(last["nb_zones"]),
        },
        "history_12w": history_list,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Sauvegardé → {OUTPUT}")
    print("=" * 55)

if __name__ == "__main__":
    run()