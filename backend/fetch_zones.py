"""
fetch_zones.py
==============
Version corrigée pour Docker - Utilise uniquement les modèles fixes,
sans recherche de versions ni chargement de modèles par zone.
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

BASE_DIR = Path(__file__).parent.parent
KMEANS_PATH = BASE_DIR / "models" / "kmeans_zones.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler_geo.pkl"
ZONE_HIST_CSV = BASE_DIR / "data" / "zone_history.csv"
OUTPUT = BASE_DIR / "data" / "output_zones.json"

FEATURES_ZONE = ["lag_1", "lag_2", "lag_3", "lag_4", "rolling_mean_4", "rolling_mean_8"]

def run():
    print("=" * 55)
    print("  ZONES DE RISQUE — Démarrage")
    print("=" * 55)

    # Charger les modèles fixes
    print("\n📦 Chargement des modèles...")
    try:
        kmeans = joblib.load(KMEANS_PATH)
        scaler_geo = joblib.load(SCALER_PATH)
        print("   ✅ Modèles KMeans + Scaler chargés")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return

    # Charger l'historique
    if not ZONE_HIST_CSV.exists():
        print("❌ zone_history.csv introuvable")
        return

    zone_hist = pd.read_csv(ZONE_HIST_CSV, parse_dates=["date"])
    zone_hist = zone_hist.sort_values(["zone", "date"]).reset_index(drop=True)
    print(f"\n📊 Historique: {len(zone_hist)} lignes")

    # Fetch USGS (7 derniers jours)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)

    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 2.0,
        "orderby": "time",
    }

    print(f"\n🌍 Fetch USGS: {start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')}")
    resp = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"   📊 {len(data['features'])} événements")

    # Parser les données
    rows = []
    for feat in data["features"]:
        p = feat["properties"]
        g = feat["geometry"]["coordinates"]
        if p.get("type") != "earthquake":
            continue
        rows.append({
            "latitude": g[1],
            "longitude": g[0],
        })

    df_week = pd.DataFrame(rows).dropna()
    print(f"   Séismes avec coordonnées: {len(df_week)}")

    if len(df_week) == 0:
        print("   ⚠️ Aucun séisme")
        # On ne sauvegarde pas vide, on garde l'ancienne prédiction ? Ou on sauvegarde vide.
        # Ici on va sauvegarder vide mais avec le timestamp.
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "next_week_start": str((datetime.now(timezone.utc) + timedelta(days=7)).date()),
            "nb_zones": 0,
            "zones": [],
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print("💾 Sauvegardé (vide)")
        return

    # Assigner les zones
    coords_scaled = scaler_geo.transform(df_week[["latitude", "longitude"]])
    df_week["zone"] = kmeans.predict(coords_scaled).astype(int)

    # Agrégation
    week_start = pd.Timestamp(end_time.date()) - pd.Timedelta(days=end_time.weekday())
    
    # ✅ Calcul des coordonnées moyennes pour chaque zone
    zone_agg = df_week.groupby("zone").size().reset_index(name="nb_seismes")
    zone_coords = df_week.groupby("zone")[["latitude", "longitude"]].mean().reset_index()
    zone_agg = zone_agg.merge(zone_coords, on="zone")
    zone_agg["date"] = week_start

    # Mettre à jour l'historique
    zone_hist = zone_hist[zone_hist["date"] != week_start].copy()
    zone_hist = pd.concat([zone_hist, zone_agg], ignore_index=True)
    zone_hist = zone_hist.sort_values(["zone", "date"]).reset_index(drop=True)

    # Calculer les features (pour usage futur)
    for lag in [1, 2, 3, 4]:
        zone_hist[f"lag_{lag}"] = zone_hist.groupby("zone")["nb_seismes"].shift(lag)

    for w in [4, 8]:
        zone_hist[f"rolling_mean_{w}"] = zone_hist.groupby("zone")["nb_seismes"].shift(1).rolling(w).mean().reset_index(level=0, drop=True)

    zone_hist.to_csv(ZONE_HIST_CSV, index=False)
    print(f"\n💾 Historique mis à jour")

    # ✅ PRÉDICTION : On enlève le dropna() pour ne pas perdre les zones qui ont seulement 1 semaine d'historique
    last_per_zone = zone_hist.sort_values("date").groupby("zone").tail(1).reset_index(drop=True)
    
    results = []
    for _, row in last_per_zone.iterrows():
        zone_id = int(row["zone"])
        # Utiliser nb_seismes comme prédiction si les lags sont vides
        pred = int(row.get("nb_seismes", 0))
        results.append({
            "zone": zone_id, 
            "pred_seismes": pred,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"])
        })

    results_df = pd.DataFrame(results)
    
    # Niveaux de risque basés sur les percentiles
    if len(results_df) > 0:
        p33 = np.percentile(results_df["pred_seismes"], 33)
        p66 = np.percentile(results_df["pred_seismes"], 66)
    else:
        p33, p66 = 0, 0

    def risk_level(val):
        if val <= p33:
            return "faible"
        elif val <= p66:
            return "modere"
        return "eleve"

    results_df["risk_level"] = results_df["pred_seismes"].apply(risk_level)
    results_df = results_df.sort_values("pred_seismes", ascending=False)

    # Construire le JSON
    zones_list = []
    for _, row in results_df.iterrows():
        zones_list.append({
            "zone": int(row["zone"]),
            "pred_seismes": int(row["pred_seismes"]),
            "risk_level": row["risk_level"],
            "lat": float(row["latitude"]),   # ✅ Maintenant rempli
            "lon": float(row["longitude"]),  # ✅ Maintenant rempli
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "next_week_start": str((week_start + pd.Timedelta(weeks=1)).date()),
        "nb_zones": len(zones_list),
        "zones": zones_list,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Sauvegardé → {OUTPUT}")
    print("=" * 55)

if __name__ == "__main__":
    run()