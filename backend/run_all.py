"""
run_all.py
==========
Lance les 3 scripts de fetch+predict en séquence.
Ajoute la gestion MLOPS : versioning, comparaison des modèles, et retraining hebdomadaire.
À appeler au démarrage de l'app et en tâche planifiée.

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
    """Exécute une étape du pipeline avec gestion d'erreurs"""
    print("\n" + "-" * 55)
    print(f"  > {name}")
    print("-" * 55)
    try:
        fn()
        print(f"  ✅ OK - {name}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR - {name} : {e}")
        traceback.print_exc()
        return False

def should_run_weekly_training():
    """
    Détermine si on doit exécuter le retraining hebdomadaire
    Condition: c'est dimanche ET le dernier entraînement date de plus de 6 jours
    """
    today = datetime.now()
    
    # Vérifier si c'est dimanche (weekday = 6)
    is_sunday = today.weekday() == 6
    
    if not is_sunday:
        print("\n  ⏭️  Pas de retraining aujourd'hui (programmé pour dimanche)")
        return False
    
    # Vérifier le dernier entraînement
    training_log = Path(__file__).parent.parent / "data" / "training_history.csv"
    
    if not training_log.exists():
        print("\n  🆕 Premier entraînement détecté → lancement")
        return True
    
    try:
        import pandas as pd
        df = pd.read_csv(training_log)
        if len(df) > 0:
            last_training = pd.to_datetime(df['date'].max())
            days_since = (datetime.now() - last_training).days
            print(f"\n  📅 Dernier entraînement: {last_training.strftime('%Y-%m-%d')} ({days_since} jours)")
            
            if days_since >= 7:
                print("  🔄 Plus de 7 jours → lancement du retraining")
                return True
            else:
                print("  ⏭️  Retraining déjà effectué cette semaine")
                return False
    except Exception as e:
        print(f"  ⚠️ Erreur lecture historique: {e}")
    
    return True

def run_mlops_pipeline():
    """
    Exécute le pipeline MLOPS complet:
    - Comparaison des modèles existants
    - Sélection du meilleur
    - Retraining si nécessaire
    """
    print("\n" + "=" * 55)
    print("  🤖 PIPELINE MLOPS")
    print("=" * 55)
    
    results = {
        "model_comparison": False,
        "model_selection": False,
        "weekly_training": False
    }
    
    try:
        # 1. Comparaison des modèles
        print("\n  📊 Étape 1: Comparaison des modèles")
        from model_comparator import ModelComparator
        comparator = ModelComparator()
        
        comparison_results = comparator.test_all_global_models(weeks_for_test=4)
        results["model_comparison"] = True
        
        # 2. Sélection du meilleur modèle
        print("\n  🏆 Étape 2: Sélection du meilleur modèle")
        selection = comparator.compare_and_select_best(force_retrain=False)
        results["model_selection"] = True
        results["selection_result"] = selection
        
        # Afficher le résultat de sélection
        if selection.get('action') == 'switched':
            print(f"\n  🎉 NOUVEAU MODÈLE ADOPTÉ: {selection['selected_model']}")
            print(f"     Amélioration: {selection.get('improvement_percent', 0):+.1f}%")
        elif selection.get('action') == 'kept_current':
            print(f"\n  ✅ Maintien du modèle actuel")
        else:
            print(f"\n  🆕 Premier déploiement: {selection.get('selected_model')}")
        
        # 3. Génération du rapport de comparaison
        print("\n  📄 Étape 3: Génération du rapport")
        report_path = comparator.generate_comparison_report()
        print(f"     Rapport sauvegardé: {report_path}")
        
        # 4. Retraining hebdomadaire (si dimanche)
        if should_run_weekly_training():
            print("\n  🔄 Étape 4: Retraining hebdomadaire")
            from training_pipeline import TrainingPipeline
            trainer = TrainingPipeline()
            training_results = trainer.run_weekly_training()
            results["weekly_training"] = True
            results["training_results"] = training_results
        else:
            print("\n  ⏭️  Étape 4: Retraining sauté (pas programmé)")
            results["weekly_training"] = "skipped"
        
        # Sauvegarder les résultats MLOPS
        mlops_log = Path(__file__).parent.parent / "data" / "mlops_pipeline_log.json"
        with open(mlops_log, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": {
                    "comparison_done": results["model_comparison"],
                    "selection_done": results["model_selection"],
                    "training_done": results["weekly_training"] if results["weekly_training"] == True else False,
                    "selected_model": selection.get('selected_model'),
                    "improvement": selection.get('improvement_percent')
                }
            }, f, indent=2)
        
        print("\n  ✅ Pipeline MLOPS terminé avec succès")
        
    except Exception as e:
        print(f"\n  ❌ Erreur dans le pipeline MLOPS: {e}")
        traceback.print_exc()
        results["error"] = str(e)
    
    return results

def update_dashboard_status(results):
    """
    Met à jour un fichier de statut pour le dashboard
    """
    status_file = Path(__file__).parent.parent / "data" / "pipeline_status.json"
    
    status = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "pipeline_results": results,
        "models_available": {}
    }
    
    # Vérifier quels modèles sont disponibles
    models_dir = Path(__file__).parent.parent / "models"
    if models_dir.exists():
        for model_file in models_dir.glob("*.pkl"):
            status["models_available"][model_file.stem] = {
                "path": str(model_file),
                "size_kb": round(model_file.stat().st_size / 1024, 1)
            }
    
    # Vérifier les versions
    versions_dir = models_dir / "versions"
    if versions_dir.exists():
        status["versioned_models_count"] = len(list(versions_dir.glob("*.pkl")))
    
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)
    
    print(f"\n  📊 Statut dashboard mis à jour: {status_file}")

def main():
    """Fonction principale du pipeline"""
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
        "Classification (24h)": run_step("Classification (24h)", run_classif),
        "TimeSeries Global (semaine)": run_step("TimeSeries Global (semaine)", run_ts),
        "Zones de risque (semaine)": run_step("Zones de risque (semaine)", run_zones),
    }
    
    # Afficher résumé des fetches
    print("\n" + "=" * 55)
    print("  📥 RÉSUMÉ FETCH & PREDICTION")
    print("=" * 55)
    for name, ok in fetch_results.items():
        status = "✅ OK" if ok else "❌ FAILED"
        print(f"  {status}  {name}")
    
    # Si les fetches ont réussi, exécuter le pipeline MLOPS
    mlops_results = {}
    if all(fetch_results.values()):
        print("\n" + "=" * 55)
        print("  🚀 LANCEMENT PIPELINE MLOPS")
        print("=" * 55)
        mlops_results = run_mlops_pipeline()
    else:
        print("\n  ⚠️  Pipeline MLOPS sauté à cause d'échecs dans les fetches")
        mlops_results = {"error": "fetch_failed", "message": "Les étapes de fetch/prediction ont échoué"}
    
    # Mettre à jour le statut pour le dashboard
    update_dashboard_status({
        "fetch": fetch_results,
        "mlops": mlops_results
    })
    
    # Résumé final
    print("\n" + "=" * 55)
    print("  📋 RÉSUMÉ FINAL DU PIPELINE")
    print("=" * 55)
    
    # Statut des fetches
    fetch_status = "✅ SUCCÈS" if all(fetch_results.values()) else "⚠️ PARTIEL"
    print(f"\n  FETCH & PREDICTION: {fetch_status}")
    
    # Statut MLOPS
    if mlops_results.get("error"):
        print(f"  MLOPS PIPELINE: ❌ ÉCHEC - {mlops_results.get('error')}")
    elif mlops_results.get("model_selection"):
        print(f"  MLOPS PIPELINE: ✅ SUCCÈS")
        selection = mlops_results.get("selection_result", {})
        if selection.get('action') == 'switched':
            print(f"    • Modèle changé: {selection.get('selected_model')}")
            print(f"    • Amélioration: {selection.get('improvement_percent', 0):+.1f}%")
        elif selection.get('action') == 'kept_current':
            print(f"    • Modèle maintenu")
    else:
        print(f"  MLOPS PIPELINE: ⏭️ NON EXÉCUTÉ")
    
    # Versioning
    models_dir = Path(__file__).parent.parent / "models" / "versions"
    if models_dir.exists():
        n_versions = len(list(models_dir.glob("*.pkl")))
        print(f"  VERSIONS MODÈLES: {n_versions} sauvegardées")
    
    print("\n" + "=" * 55)
    print("  🎯 PIPELINE TERMINÉ")
    print("=" * 55 + "\n")
    
    # Exit code
    if not all(fetch_results.values()):
        sys.exit(1)
    else:
        sys.exit(0)


# Point d'entrée pour le scheduler (cron / APScheduler)
def run_pipeline_for_scheduler():
    """
    Version simplifiée pour appel par scheduler
    Retourne True si succès, False sinon
    """
    try:
        # Configuration minimale pour scheduler
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding="utf-8")
        
        # Exécuter le pipeline
        main()
        return True
    except Exception as e:
        print(f"Scheduler pipeline error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()