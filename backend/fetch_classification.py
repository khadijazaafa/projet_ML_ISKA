"""
fetch_classification.py
=======================
Télécharge les séismes des dernières 24h depuis USGS,
applique tout le preprocessing, prédit la classe (Faible/Modéré/Strong)
et sauvegarde → data/output_classification.json

🔧 Modification : Utilise uniquement le modèle rf_pipeline.pkl,
sans chercher de version plus récente.
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
from sklearn.impute import KNNImputer, SimpleImputer

# ── CHEMINS ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "rf_pipeline.pkl"
OUTPUT = BASE_DIR / "data" / "output_classification.json"


def run():
    print("=" * 55)
    print("  CLASSIFICATION — Démarrage")
    print("=" * 55)

    # ── 1. CHARGEMENT DIRECT DU MODÈLE ─────────────────────────
    print("\n📦 Chargement du modèle...")
    if not MODEL_PATH.exists():
        print(f" ❌ Modèle introuvable : {MODEL_PATH}")
        return
    try:
        rf_model = joblib.load(MODEL_PATH)
        print(f" ✅ Modèle chargé : {MODEL_PATH.name}")
    except Exception as e:
        print(f" ❌ Erreur chargement modèle : {e}")
        return

    # ── 2. TÉLÉCHARGEMENT USGS (24h) ──────────────────────────
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)

    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 2.0,
        "orderby": "time",
    }

    print(f"\n🌍 Fetch USGS : {start_time.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
    resp = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params=params, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    print(f" 📊 {len(data['features'])} événements récupérés")

    # ── 3. PARSING GeoJSON → DataFrame ───────────────────────
    rows = []
    for feat in data["features"]:
        p = feat["properties"]
        g = feat["geometry"]["coordinates"]
        rows.append({
            "id": feat["id"],
            "longitude": g[0],
            "latitude": g[1],
            "depth_km": g[2],
            "time_utc": pd.to_datetime(p["time"], unit="ms", utc=True),
            "mag": p.get("mag"),
            "magType": p.get("magType"),
            "nst": p.get("nst"),
            "gap": p.get("gap"),
            "dmin": p.get("dmin"),
            "rms": p.get("rms"),
            "net": p.get("net"),
            "sig": p.get("sig"),
            "tsunami": p.get("tsunami", 0),
            "place": p.get("place"),
            "type": p.get("type"),
            "status": p.get("status"),
        })

    df_raw = pd.DataFrame(rows)

    # ── 4. FILTRE type='earthquake' ───────────────────────────
    df_raw = df_raw[df_raw["type"] == "earthquake"].copy().reset_index(drop=True)
    print(f"   Après filtre type='earthquake' : {len(df_raw)}")

    # Copie pour affichage final
    df_info = df_raw[["id", "time_utc", "place", "mag", "latitude", "longitude", "depth_km"]].copy()

    if len(df_raw) == 0:
        print("⚠️ Aucun séisme dans les 24h — output vide")
        _save_empty(OUTPUT, start_time, end_time)
        return

    # ── 5. OUTLIERS ───────────────────────────────────────────
    conditions = (
        (df_raw["depth_km"] >= 0) & (df_raw["depth_km"] <= 700) &
        (df_raw["nst"] >= 3) & (df_raw["nst"] <= 1000) &
        (df_raw["gap"] >= 0) & (df_raw["gap"] <= 360) &
        (df_raw["dmin"] >= 0) & (df_raw["dmin"] <= 100) &
        (df_raw["rms"] >= 0) & (df_raw["rms"] <= 10) &
        (df_raw["sig"] >= 0) & (df_raw["sig"] <= 5000) &
        (df_raw["mag"] >= 2) & (df_raw["mag"] <= 9)
    )
    mask = conditions.values
    df_raw = df_raw[mask].reset_index(drop=True)
    df_info = df_info[mask].reset_index(drop=True)
    print(f"   Après suppression outliers : {len(df_raw)}")

    if len(df_raw) == 0:
        print("⚠️ Tous les séismes filtrés comme outliers")
        _save_empty(OUTPUT, start_time, end_time)
        return

    # ── 6. VARIABLES TEMPORELLES ──────────────────────────────
    df_raw["time_utc"] = pd.to_datetime(df_raw["time_utc"])
    df_raw["hour"] = df_raw["time_utc"].dt.hour
    df_raw["dayofweek"] = df_raw["time_utc"].dt.dayofweek
    df_raw["month"] = df_raw["time_utc"].dt.month

    # ── 7. LOG TRANSFORMS ─────────────────────────────────────
    df_raw["depth_km_log"] = np.log1p(df_raw["depth_km"])
    df_raw["nst_log"] = np.log1p(df_raw["nst"])
    df_raw["dmin_log"] = np.log1p(df_raw["dmin"])
    df_raw["sig_log"] = np.log1p(df_raw["sig"])

    # ── 8. IMPUTATION KNN ─────────────────────────────────────
    for target_col, cols in [
        ("gap", ["latitude", "longitude", "hour", "gap"]),
        ("rms", ["latitude", "longitude", "hour", "rms"]),
        ("dmin_log", ["latitude", "longitude", "hour", "dmin_log"]),
        ("nst_log", ["latitude", "longitude", "hour", "nst_log"]),
        ("sig_log", ["latitude", "longitude", "hour", "sig_log"]),
    ]:
        imp = KNNImputer(n_neighbors=5)
        df_raw[target_col] = imp.fit_transform(df_raw[cols])[:, -1]

    # ── 9. ONE-HOT ENCODE 'net' ───────────────────────────────
    # Récupérer les colonnes net_ attendues par le modèle
    net_cols_train = [c for c in rf_model.feature_names_in_ if c.startswith("net_")]
    net_dummies = pd.get_dummies(df_raw["net"], prefix="net")

    for col in net_cols_train:
        if col not in net_dummies.columns:
            net_dummies[col] = 0
    net_dummies = net_dummies[net_cols_train]

    # ── 10. ASSEMBLER X_new ───────────────────────────────────
    FEATURES_NUM = [
        "latitude", "longitude", "gap", "rms", "tsunami",
        "depth_km_log", "nst_log", "dmin_log", "sig_log",
        "hour", "dayofweek", "month",
    ]

    X_new = pd.concat([
        df_raw[FEATURES_NUM].reset_index(drop=True),
        net_dummies.reset_index(drop=True)
    ], axis=1)

    si = SimpleImputer(strategy="median")
    X_new = pd.DataFrame(si.fit_transform(X_new), columns=X_new.columns)

    # ── 11. ALIGNEMENT FEATURES ───────────────────────────────
    expected = list(rf_model.feature_names_in_)
    for col in expected:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[expected]

    # ── 12. PRÉDICTION ────────────────────────────────────────
    y_pred = rf_model.predict(X_new)
    y_pred_proba = rf_model.predict_proba(X_new)
    print("   ✅ Prédiction effectuée")

    CLASS_MAP = {0: "Faible", 1: "Modéré", 2: "Strong"}
    EMOJI_MAP = {0: "🟢", 1: "🟡", 2: "🔴"}

    # ── 13. CONSTRUIRE LE RÉSULTAT ────────────────────────────
    records = []
    for i in range(len(df_info)):
        records.append({
            "id": str(df_info.loc[i, "id"]),
            "time_utc": str(df_info.loc[i, "time_utc"]),
            "place": str(df_info.loc[i, "place"]),
            "mag": float(df_info.loc[i, "mag"]),
            "latitude": float(df_info.loc[i, "latitude"]),
            "longitude": float(df_info.loc[i, "longitude"]),
            "depth_km": float(df_info.loc[i, "depth_km"]),
            "emoji": EMOJI_MAP[int(y_pred[i])],
            "mag_class": CLASS_MAP[int(y_pred[i])],
            "prob_Faible": round(float(y_pred_proba[i, 0]) * 100, 1),
            "prob_Modere": round(float(y_pred_proba[i, 1]) * 100, 1),
            "prob_Strong": round(float(y_pred_proba[i, 2]) * 100, 1),
            "confiance": round(float(y_pred_proba[i].max()) * 100, 1),
        })

    # Tri par temps décroissant
    records = sorted(records, key=lambda x: x["time_utc"], reverse=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_start": start_time.isoformat(),
        "period_end": end_time.isoformat(),
        "total": len(records),
        "count_faible": int((y_pred == 0).sum()),
        "count_modere": int((y_pred == 1).sum()),
        "count_strong": int((y_pred == 2).sum()),
        "model_used": MODEL_PATH.name,
        "earthquakes": records,
    }

    # ── 14. SAUVEGARDE ────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Sauvegardé → {OUTPUT}")
    print(f"   🟢 Faible : {summary['count_faible']}")
    print(f"   🟡 Modéré : {summary['count_modere']}")
    print(f"   🔴 Strong : {summary['count_strong']}")
    print(f"   🤖 Modèle utilisé: {summary['model_used']}")
    print("=" * 55)


def _save_empty(path, start, end):
    """Sauvegarde un JSON vide si pas de données."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "total": 0,
            "count_faible": 0,
            "count_modere": 0,
            "count_strong": 0,
            "model_used": "none",
            "earthquakes": [],
        }, f, indent=2)


if __name__ == "__main__":
    run()