"""
fetch_timeseries.py
===================
Télécharge les séismes de la semaine écoulée depuis USGS,
calcule les features hebdomadaires (lags, rolling, energy...),
prédit le nb de séismes pour la semaine prochaine (global),
et sauvegarde → data/output_forecast.json

🔄 Amélioration : Charge automatiquement le meilleur modèle disponible
(versionné ou modèle par défaut)
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

# Modèles par défaut (fallback)
XGB_PATH_DEFAULT = BASE_DIR / "models" / "model_xgb_global.pkl"
XGB_LOW_PATH_DEFAULT = BASE_DIR / "models" / "model_xgb_global_low.pkl"
XGB_HGH_PATH_DEFAULT = BASE_DIR / "models" / "model_xgb_global_high.pkl"

# Dossiers versionnés
VERSIONS_DIR = BASE_DIR / "models" / "versions"
METADATA_PATH = BASE_DIR / "data" / "current_best_model.json"

HISTORY_CSV = BASE_DIR / "data" / "weekly_history.csv"
OUTPUT = BASE_DIR / "data" / "output_forecast.json"

# Features exactement comme dans le notebook
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


def get_best_global_model():
    """
    Charge le meilleur modèle global disponible.
    Priorité :
    1. Modèle spécifié dans current_best_model.json (type 'global')
    2. Dernier modèle versionné (xgboost_global_*.pkl)
    3. Modèle par défaut model_xgb_global.pkl
    
    Retourne : (model_central, model_low, model_high, model_path, model_name)
    """
    print(" 🔍 Recherche du meilleur modèle global...")
    
    model_central = None
    model_low = None
    model_high = None
    model_path = None
    model_name = None
    
    # 1. Vérifier si un modèle est spécifié dans les métadonnées
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            
            prod_model = metadata.get('production_model', {})
            prod_model_name = prod_model.get('model_name', '')
            prod_model_type = prod_model.get('model_type', '')
            
            if prod_model_name and 'global' in prod_model_type.lower():
                # Chercher le modèle central dans versions/
                model_path_central = VERSIONS_DIR / f"{prod_model_name}.pkl"
                
                # Chercher les modèles low et high correspondants
                base_name = prod_model_name.replace('xgboost_global', '')
                model_path_low = VERSIONS_DIR / f"xgboost_global_low{base_name}.pkl"
                model_path_high = VERSIONS_DIR / f"xgboost_global_high{base_name}.pkl"
                
                if model_path_central.exists():
                    model_central = joblib.load(model_path_central)
                    model_path = str(model_path_central)
                    model_name = prod_model_name
                    
                    # Charger low et high s'ils existent
                    if model_path_low.exists():
                        model_low = joblib.load(model_path_low)
                    if model_path_high.exists():
                        model_high = joblib.load(model_path_high)
                    
                    print(f"   ✓ Modèle central chargé depuis métadonnées: {prod_model_name}")
                    return model_central, model_low, model_high, model_path, model_name
        except Exception as e:
            print(f"   ⚠️ Erreur lecture métadonnées: {e}")
    
    # 2. Chercher le dernier modèle global versionné
    if VERSIONS_DIR.exists():
        global_models = []
        for model_file in VERSIONS_DIR.glob("xgboost_global_*.pkl"):
            # Ignorer les modèles low et high
            if 'low' not in model_file.stem and 'high' not in model_file.stem:
                mtime = model_file.stat().st_mtime
                global_models.append((mtime, model_file))
        
        if global_models:
            # Prendre le plus récent
            global_models.sort(key=lambda x: x[0], reverse=True)
            latest_model_path = global_models[0][1]
            model_central = joblib.load(latest_model_path)
            model_path = str(latest_model_path)
            model_name = latest_model_path.stem
            print(f"   ✓ Dernier modèle versionné chargé: {latest_model_path.name}")
            
            # Chercher les modèles low et high correspondants
            base_name = latest_model_path.stem.replace('xgboost_global', '')
            low_path = VERSIONS_DIR / f"xgboost_global_low{base_name}.pkl"
            high_path = VERSIONS_DIR / f"xgboost_global_high{base_name}.pkl"
            
            if low_path.exists():
                model_low = joblib.load(low_path)
                print(f"   ✓ Modèle low chargé")
            if high_path.exists():
                model_high = joblib.load(high_path)
                print(f"   ✓ Modèle high chargé")
            
            return model_central, model_low, model_high, model_path, model_name
    
    # 3. Fallback vers les modèles par défaut
    if XGB_PATH_DEFAULT.exists():
        print(f"   ✓ Chargement des modèles par défaut")
        model_central = joblib.load(XGB_PATH_DEFAULT)
        model_low = joblib.load(XGB_LOW_PATH_DEFAULT) if XGB_LOW_PATH_DEFAULT.exists() else None
        model_high = joblib.load(XGB_HGH_PATH_DEFAULT) if XGB_HGH_PATH_DEFAULT.exists() else None
        model_path = str(XGB_PATH_DEFAULT)
        model_name = "default_xgb_global"
        return model_central, model_low, model_high, model_path, model_name
    
    raise FileNotFoundError("Aucun modèle global trouvé!")


def get_model_ensemble():
    """
    Optionnel : Récupère plusieurs modèles globaux pour faire un ensemble
    Retourne une liste de modèles
    """
    models = []
    
    if VERSIONS_DIR.exists():
        for model_file in VERSIONS_DIR.glob("xgboost_global_*.pkl"):
            if 'low' not in model_file.stem and 'high' not in model_file.stem:
                try:
                    model = joblib.load(model_file)
                    models.append({
                        'model': model,
                        'name': model_file.stem,
                        'mtime': model_file.stat().st_mtime
                    })
                except Exception as e:
                    print(f"   ⚠️ Erreur chargement {model_file.name}: {e}")
    
    # Trier par date (plus récent d'abord)
    models.sort(key=lambda x: x['mtime'], reverse=True)
    
    return models[:3]  # Garder les 3 plus récents


def predict_with_ensemble(models, X_next, use_ensemble=False):
    """
    Fait une prédiction avec un ensemble de modèles (moyenne pondérée)
    """
    if not use_ensemble or len(models) < 2:
        return None, None, None
    
    all_preds = []
    weights = []
    
    for m in models:
        try:
            pred = m['model'].predict(X_next)[0]
            all_preds.append(pred)
            # Poids basé sur la date (plus récent = plus de poids)
            weights.append(m['mtime'])
        except Exception as e:
            print(f"   ⚠️ Erreur prédiction avec {m['name']}: {e}")
    
    if not all_preds:
        return None, None, None
    
    # Normaliser les poids
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Moyenne pondérée
    pred_ensemble = np.average(all_preds, weights=weights)
    
    # Intervalle basé sur l'écart-type des prédictions
    pred_std = np.std(all_preds)
    pred_low = int(np.clip(pred_ensemble - pred_std, 0, None))
    pred_high = int(pred_ensemble + pred_std)
    
    return int(pred_ensemble), pred_low, pred_high


def run():
    print("=" * 55)
    print("  TIMESERIES GLOBAL — Démarrage")
    print("=" * 55)

    # ── 1. CHARGER LE MEILLEUR MODÈLE ─────────────────────────
    print("\n📦 Chargement des modèles...")
    try:
        xgb_model, xgb_low, xgb_high, model_path, model_name = get_best_global_model()
        print(f" ✅ Modèle central chargé: {Path(model_path).name}")
        
        # Optionnel : Charger des modèles supplémentaires pour ensemble
        ensemble_models = get_model_ensemble()
        use_ensemble = len(ensemble_models) >= 2
        if use_ensemble:
            print(f" 🎯 Mode ensemble activé avec {len(ensemble_models)} modèles")
    except Exception as e:
        print(f" ❌ Erreur chargement modèle: {e}")
        return

    # ── 2. CHARGER L'HISTORIQUE ───────────────────────────────
    if not HISTORY_CSV.exists():
        raise FileNotFoundError(
            f"weekly_history.csv introuvable : {HISTORY_CSV}\n"
            "   → Exporte-le depuis ton notebook : weekly_model.to_csv('data/weekly_history.csv', index=False)"
        )

    weekly = pd.read_csv(HISTORY_CSV, parse_dates=["date"])
    weekly = weekly.sort_values("date").reset_index(drop=True)
    print(f"\n📊 Historique chargé : {len(weekly)} semaines → jusqu'au {weekly['date'].max().date()}")

    # ── 3. TÉLÉCHARGEMENT USGS (7 derniers jours) ─────────────
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)

    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 2.0,
        "orderby": "time",
    }

    print(f"\n🌍 Fetch USGS semaine : {start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')}")
    resp = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params=params, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    print(f" 📊 {len(data['features'])} événements récupérés")

    # ── 4. PARSING → DataFrame ────────────────────────────────
    rows = []
    for feat in data["features"]:
        p = feat["properties"]
        g = feat["geometry"]["coordinates"]
        if p.get("type") != "earthquake":
            continue
        rows.append({
            "id": feat["id"],
            "time_utc": pd.to_datetime(p["time"], unit="ms", utc=True),
            "latitude": g[1],
            "longitude": g[0],
            "depth_km": g[2],
            "mag": p.get("mag"),
            "place": p.get("place"),
            "tsunami": p.get("tsunami", 0),
        })

    df_week = pd.DataFrame(rows)
    df_week["date"] = pd.to_datetime(df_week["time_utc"]).dt.date
    df_week["date"] = pd.to_datetime(df_week["date"])

    # ── 5. AGRÉGER EN UNE LIGNE HEBDOMADAIRE ──────────────────
    df_week["energy"] = 10 ** (1.5 * df_week["mag"].fillna(0))

    new_row = {
        "date": pd.Timestamp(end_time.date()),
        "nb_seismes": len(df_week),
        "mag_max": df_week["mag"].max(),
        "mag_mean": df_week["mag"].mean(),
        "mag_median": df_week["mag"].median(),
        "depth_mean": df_week["depth_km"].mean(),
        "tsunami_sum": df_week["tsunami"].sum(),
        "nb_zones": df_week["place"].nunique(),
        "energy_sum": df_week["energy"].sum(),
    }

    # ── 6. AJOUTER LA NOUVELLE SEMAINE À L'HISTORIQUE ─────────
    mask_existing = weekly["date"] == new_row["date"]
    if mask_existing.any():
        weekly = weekly[~mask_existing].copy()
        print(f"   ↻ Remplacement de la semaine {new_row['date'].date()} existante")

    weekly = pd.concat([weekly, pd.DataFrame([new_row])], ignore_index=True)
    weekly = weekly.sort_values("date").reset_index(drop=True)

    # ── 7. RECALCULER TOUTES LES FEATURES ─────────────────────
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

    # ── 8. SAUVEGARDER L'HISTORIQUE ───────────────────────────
    weekly.to_csv(HISTORY_CSV, index=False)
    print(f"\n💾 Historique mis à jour → {len(weekly)} semaines")

    # ── 9. CONSTRUIRE X_next ─────────────────────────────────
    last = weekly.iloc[-1]
    next_date = last["date"] + pd.Timedelta(weeks=1)
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

    # ── 10. PRÉDICTION ────────────────────────────────────────
    # Essayer avec ensemble si disponible
    pred_ensemble, pred_low_ensemble, pred_high_ensemble = predict_with_ensemble(
        ensemble_models, X_next, use_ensemble
    )
    
    if pred_ensemble is not None:
        pred_central = pred_ensemble
        pred_low = pred_low_ensemble
        pred_high = pred_high_ensemble
        print(f"\n🔮 Prédiction avec ensemble ({len(ensemble_models)} modèles)")
    else:
        pred_central = int(np.clip(xgb_model.predict(X_next)[0], 0, None))
        
        if xgb_low is not None:
            pred_low = int(np.clip(xgb_low.predict(X_next)[0], 0, None))
        else:
            pred_low = max(0, pred_central - 50)
        
        if xgb_high is not None:
            pred_high = int(np.clip(xgb_high.predict(X_next)[0], 0, None))
        else:
            pred_high = pred_central + 50
        
        print(f"\n🔮 Prédiction avec modèle unique: {model_name}")

    print(f"\n📅 Semaine du {next_date.date()}")
    print(f"   Centrale  : {pred_central} séismes")
    print(f"   Intervalle: [{pred_low} – {pred_high}]")
    print(f"   Semaine actuelle : {int(last['nb_seismes'])} séismes")

    # ── 11. HISTORIQUE 12 DERNIÈRES SEMAINES ──────────────────
    last_12 = weekly.tail(12)[["date", "nb_seismes"]].copy()
    last_12["date"] = last_12["date"].dt.strftime("%Y-%m-%d")
    history_list = last_12.to_dict(orient="records")

    # ── 12. SAUVEGARDE OUTPUT ─────────────────────────────────
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "next_week_start": str(next_date.date()),
        "model_used": model_name if not use_ensemble else f"ensemble_{len(ensemble_models)}_models",
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