"""
run_all.py
==========
Lance les 3 scripts de fetch+predict en séquence.
Version simplifiée : pas de MLOPS, pas de versioning, pas de retraining.
MAIS garde la mise à jour du tableau de bord (pipeline_status.json).

Usage : python run_all.py
"""

import os
import sys
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Configuration de l'encodage
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ajouter le backend au path pour les imports
sys.path.insert(0, str(Path(__file__).parent))


def run_step(name, fn):
    print("\n" + "-" * 55)
    print(f"  > {name}")
    print("-" * 55)
    try:
        fn()
        print(f"  ✅ OK - {name}")
        # Ajoutez l'heure de fin
        print(f"     Terminé à {datetime.now().strftime('%H:%M:%S')}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR - {name} : {e}")
        traceback.print_exc()
        return False


def update_dashboard_status(fetch_results):
    """
    Met à jour un fichier de statut pour le dashboard
    Version simplifiée (sans infos MLOPS)
    """
    status_file = Path(__file__).parent.parent / "data" / "pipeline_status.json"
    
    status = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "fetch_results": fetch_results,
        "models_available": {}
    }
    
    # Vérifier quels modèles sont disponibles (uniquement les modèles fixes)
    models_dir = Path(__file__).parent.parent / "models"
    if models_dir.exists():
        for model_file in models_dir.glob("*.pkl"):
            # Ignorer le dossier versions s'il existe
            if "versions" not in str(model_file):
                status["models_available"][model_file.stem] = {
                    "path": str(model_file),
                    "size_kb": round(model_file.stat().st_size / 1024, 1)
                }
    
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)
    
    print(f"\n  📊 Statut dashboard mis à jour: {status_file}")


def main():
    """Fonction principale du pipeline simplifié"""
    print("=" * 55)
    print(f"  🌍 SEISMIC INTELLIGENCE PIPELINE")
    print(f"  📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)
    
    # Importer les modules après les configurations
    from fetch_classification import run as run_classif
    from fetch_timeseries import run as run_ts
    from fetch_zones import run as run_zones
    
    # Exécuter les étapes de fetch/prediction
    fetch_results = {
        "classification": run_step("Classification (24h)", run_classif),
        "timeseries_global": run_step("TimeSeries Global (semaine)", run_ts),
        "zones_risque": run_step("Zones de risque (semaine)", run_zones),
    }
    
    # Mettre à jour le statut pour le dashboard
    update_dashboard_status(fetch_results)
    
    # Afficher résumé
    print("\n" + "=" * 55)
    print("  📥 RÉSUMÉ DU PIPELINE")
    print("=" * 55)
    for name, ok in fetch_results.items():
        status = "✅ OK" if ok else "❌ FAILED"
        print(f"  {status}  {name}")
    
    print("\n" + "=" * 55)
    print("  🎯 PIPELINE TERMINÉ")
    print("=" * 55 + "\n")
    
    # Exit code : 0 si tout OK, 1 si au moins un échec
    if not all(fetch_results.values()):
        sys.exit(1)
    else:
        sys.exit(0)


# Point d'entrée pour le scheduler
def run_pipeline_for_scheduler():
    """Version simplifiée pour appel par scheduler"""
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding="utf-8")
        main()
        return True
    except Exception as e:
        print(f"Scheduler pipeline error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()