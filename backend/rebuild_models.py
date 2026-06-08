"""
rebuild_models.py
Recrée les modèles de zones dans l'environnement Docker
Exécuter une seule fois après le premier démarrage
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime

print("=" * 60)
print("  🔨 RECONSTRUCTION DES MODÈLES POUR DOCKER")
print("=" * 60)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VERSIONS_DIR = MODELS_DIR / "versions"

# Créer les dossiers
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Vérifier que les données existent
zone_hist_path = DATA_DIR / "zone_history.csv"
if not zone_hist_path.exists():
    print("❌ zone_history.csv introuvable!")
    print("   Veuillez d'abord exécuter: python run_all.py")
    sys.exit(1)

# Charger les données
zone_hist = pd.read_csv(zone_hist_path, parse_dates=["date"])
print(f"📊 Données chargées: {len(zone_hist)} lignes")

# Nettoyer les données
zone_hist = zone_hist.dropna(subset=['nb_seismes'])
print(f"📊 Après nettoyage: {len(zone_hist)} lignes")

# Features pour les modèles de zones
FEATURES_ZONE = ["lag_1", "lag_2", "lag_3", "lag_4", "rolling_mean_4", "rolling_mean_8"]

# Reconstruire les features pour chaque zone
zones = zone_hist['zone'].unique()
print(f"📍 Zones trouvées: {sorted(zones)}")

models_created = 0

for zone_id in zones:
    zone_data = zone_hist[zone_hist['zone'] == zone_id].copy()
    zone_data = zone_data.sort_values('date')
    
    # Calculer les features
    for lag in [1, 2, 3, 4]:
        zone_data[f'lag_{lag}'] = zone_data['nb_seismes'].shift(lag)
    
    for w in [4, 8]:
        zone_data[f'rolling_mean_{w}'] = zone_data['nb_seismes'].shift(1).rolling(w).mean()
    
    zone_data = zone_data.dropna()
    
    if len(zone_data) >= 20:
        X = zone_data[FEATURES_ZONE].values
        y = zone_data['nb_seismes'].values
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        model.fit(X, y)
        
        # Sauvegarder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = VERSIONS_DIR / f"xgboost_zone_zone_{int(zone_id)}_docker_{timestamp}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Zone {int(zone_id)}: modèle sauvegardé ({len(zone_data)} échantillons)")
        models_created += 1
    else:
        print(f"⚠️ Zone {int(zone_id)}: données insuffisantes ({len(zone_data)} échantillons)")

print(f"\n✅ Reconstruction terminée! {models_created} modèles créés.")