"""
force_retrain_docker.py
Force le retraining de tous les modèles dans l'environnement Docker
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from training_pipeline import TrainingPipeline
import pandas as pd
from datetime import datetime

print("=" * 60)
print("  🔥 FORCAGE DU RETRAINING POUR DOCKER")
print("=" * 60)
print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Créer une instance du pipeline
pipeline = TrainingPipeline()

# Forcer le retraining
pipeline.should_retrain = lambda: True

# Exécuter le retraining
results = pipeline.run_weekly_training()

print("\n" + "=" * 60)
print("  📊 RÉSULTAT FINAL")
print("=" * 60)
print(f"  Modèles globaux: {len(results.get('models_trained', []))}")
print(f"  Zones entraînées: {len(results.get('zones_trained', []))}")

if results.get('zones_trained'):
    print(f"\n  ✅ Zones entraînées: {results['zones_trained']}")