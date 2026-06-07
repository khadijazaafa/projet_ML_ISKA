"""
fetch_zones.py
==============
Télécharge les séismes de la semaine écoulée depuis USGS,
assigne chaque séisme à une zone (via kmeans),
agrège par zone, calcule les features (lags, rolling),
prédit le nb de séismes par zone la semaine prochaine,
et sauvegarde → data/output_zones.json

🔄 Version corrigée - Approche simplifiée sans conflit de colonnes
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
KMEANS_PATH = BASE_DIR / "models" / "kmeans_zones.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler_geo.pkl"
ZONE_HIST_CSV = BASE_DIR / "data" / "zone_history.csv"
OUTPUT = BASE_DIR / "data" / "output_zones.json"
VERSIONS_DIR = BASE_DIR / "models" / "versions"

# Features pour les modèles de zones
FEATURES_ZONE = [
    "lag_1", "lag_2", "lag_3", "lag_4",
    "rolling_mean_4", "rolling_mean_8",
]


def get_all_zone_models():
    """Charge TOUS les modèles disponibles pour chaque zone."""
    print(" 🔍 Recherche des modèles par zone...")
    zone_models = {}
    
    if VERSIONS_DIR.exists():
        print("   📂 Recherche dans models/versions/...")
        
        for model_file in VERSIONS_DIR.glob("xgboost_zone_zone_*.pkl"):
            try:
                # Extraire le numéro de zone
                parts = model_file.stem.split('_')
                zone_id = None
                for i, part in enumerate(parts):
                    if part == 'zone' and i+1 < len(parts):
                        zone_str = parts[i+1]
                        zone_str = zone_str.split('.')[0] if '.' in zone_str else zone_str
                        if zone_str.isdigit():
                            zone_id = int(zone_str)
                            break
                
                if zone_id is not None:
                    zone_models[zone_id] = joblib.load(model_file)
                    print(f"   ✓ Zone {zone_id}: {model_file.name}")
            except Exception as e:
                print(f"   ⚠️ Erreur chargement {model_file.name}: {e}")
        
        if zone_models:
            print(f"   ✅ {len(zone_models)} modèles chargés")
            return zone_models
    
    print("   ⚠️ Aucun modèle trouvé!")
    return {}


def predict_zone(model, row, feature_list):
    """Prédiction pour une zone avec gestion des features."""
    try:
        feature_vector = [row.get(feat, 0) for feat in feature_list]
        X = np.array([feature_vector])
        pred = model.predict(X)[0]
        return max(0, int(np.clip(pred, 0, None)))
    except Exception as e:
        print(f"      ⚠️ Erreur prédiction: {e}")
        return None


def run():
    print("=" * 55)
    print("  ZONES DE RISQUE — Démarrage")
    print("=" * 55)

    # ── 1. CHARGER LES MODÈLES ────────────────────────────────
    print("\n📦 Chargement des modèles...")
    kmeans = joblib.load(KMEANS_PATH)
    scaler_geo = joblib.load(SCALER_PATH)
    zone_models = get_all_zone_models()
    print(f"   ✅ Modèles chargés (KMeans, Scaler, {len(zone_models)} zones)")

    # ── 2. CHARGER L'HISTORIQUE ───────────────────────────────
    if not ZONE_HIST_CSV.exists():
        raise FileNotFoundError(f"zone_history.csv introuvable")

    zone_hist = pd.read_csv(ZONE_HIST_CSV, parse_dates=["date"])
    zone_hist = zone_hist.sort_values(["zone", "date"]).reset_index(drop=True)
    print(f"\n📊 Historique: {len(zone_hist)} lignes, {zone_hist['zone'].nunique()} zones")

    # ── 3. FETCH USGS ─────────────────────────────────────────
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
    resp = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params=params, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"   📊 {len(data['features'])} événements")

    # ── 4. PARSING ────────────────────────────────────────────
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
        print("   ⚠️ Aucun séisme avec coordonnées!")
        # Créer un output vide
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "next_week_start": str((pd.Timestamp(end_time.date()) - pd.Timedelta(days=end_time.weekday()) + pd.Timedelta(weeks=1)).date()),
            "models_loaded": len(zone_models),
            "nb_zones": 0,
            "zones": [],
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print("=" * 55)
        return

    # ── 5. ASSIGNER LES ZONES ─────────────────────────────────
    coords_scaled = scaler_geo.transform(df_week[["latitude", "longitude"]])
    df_week["zone"] = kmeans.predict(coords_scaled).astype(int)

    # ── 6. AGRÉGER PAR ZONE ───────────────────────────────────
    week_start = pd.Timestamp(end_time.date()) - pd.Timedelta(days=end_time.weekday())
    
    zone_agg = df_week.groupby("zone").size().reset_index(name="nb_seismes")
    zone_agg["date"] = week_start

    # Toutes les zones existantes
    all_zones = sorted(zone_hist["zone"].unique())
    full_week = pd.DataFrame({"zone": all_zones, "date": week_start})
    zone_agg = full_week.merge(zone_agg, on=["zone", "date"], how="left")
    zone_agg["nb_seismes"] = zone_agg["nb_seismes"].fillna(0).astype(int)

    # ── 7. METTRE À JOUR L'HISTORIQUE ─────────────────────────
    zone_hist = zone_hist[zone_hist["date"] != week_start].copy()
    zone_hist = pd.concat([zone_hist, zone_agg], ignore_index=True)
    zone_hist = zone_hist.sort_values(["zone", "date"]).reset_index(drop=True)

    # ── 8. CALCULER LES FEATURES ──────────────────────────────
    for lag in [1, 2, 3, 4]:
        zone_hist[f"lag_{lag}"] = zone_hist.groupby("zone")["nb_seismes"].shift(lag)

    for w in [4, 8]:
        zone_hist[f"rolling_mean_{w}"] = (
            zone_hist.groupby("zone")["nb_seismes"]
            .shift(1)
            .rolling(w)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Sauvegarder l'historique
    zone_hist.to_csv(ZONE_HIST_CSV, index=False)
    print(f"\n💾 Historique mis à jour: {len(zone_hist)} lignes")

    # ── 9. PRÉDIRE POUR CHAQUE ZONE ───────────────────────────
    last_per_zone = (
        zone_hist.dropna()
        .sort_values("date")
        .groupby("zone")
        .tail(1)
        .reset_index(drop=True)
    )
    
    results = []
    
    print("\n🎯 Prédictions par zone:")
    
    for _, row in last_per_zone.iterrows():
        zone_id = int(row["zone"])
        
        # Prédiction avec le modèle de la zone
        pred = None
        if zone_id in zone_models:
            pred = predict_zone(zone_models[zone_id], row, FEATURES_ZONE)
        
        # Fallback: moyenne historique
        if pred is None:
            hist_vals = zone_hist[zone_hist["zone"] == zone_id]["nb_seismes"]
            pred = int(hist_vals.mean()) if len(hist_vals) > 0 else 0
        
        results.append({
            "zone": zone_id,
            "pred_seismes": pred,
            "lat": row.get("lat", None),
            "lon": row.get("lon", None),
        })
        
        print(f"   Zone {zone_id}: {pred} séismes")

    # ── 10. CENTROIDES ────────────────────────────────────────
    centroids = df_week.groupby("zone")[["latitude", "longitude"]].mean().reset_index()
    centroids.columns = ["zone", "lat", "lon"]

    # Fusionner avec les résultats
    results_df = pd.DataFrame(results)
    results_df = results_df.merge(centroids, on="zone", how="left")

    # ── 11. NIVEAU DE RISQUE ──────────────────────────────────
    pred_values = results_df["pred_seismes"].values
    if len(pred_values) > 0 and np.std(pred_values) > 0:
        p33 = np.percentile(pred_values, 33)
        p66 = np.percentile(pred_values, 66)
    else:
        p33, p66 = 0, 0

    def risk_level(val):
        if val <= p33:
            return "faible"
        elif val <= p66:
            return "modere"
        return "eleve"

    results_df["risk_level"] = results_df["pred_seismes"].apply(risk_level)
    results_df = results_df.sort_values("pred_seismes", ascending=False).reset_index(drop=True)

    # ── 12. CONSTRUIRE LE JSON ────────────────────────────────
    zones_list = []
    for _, row in results_df.iterrows():
        zones_list.append({
            "zone": int(row["zone"]),
            "pred_seismes": int(row["pred_seismes"]),
            "risk_level": row["risk_level"],
            "lat": round(float(row["lat"]), 4) if pd.notna(row.get("lat")) else None,
            "lon": round(float(row["lon"]), 4) if pd.notna(row.get("lon")) else None,
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "next_week_start": str((week_start + pd.Timedelta(weeks=1)).date()),
        "models_loaded": len(zone_models),
        "nb_zones": len(zones_list),
        "zones": zones_list,
    }

    # ── 13. SAUVEGARDER ───────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Sauvegardé → {OUTPUT}")
    print(f"\n📊 Top 3 zones à risque :")
    for z in zones_list[:3]:
        print(f"   Zone {z['zone']} : {z['pred_seismes']} séismes ({z['risk_level']})")
    print("=" * 55)


if __name__ == "__main__":
    run()