"""
scheduler_config.py
===================
Configuration du scheduler pour exécuter le pipeline à intervalles réguliers
Utilisable avec APScheduler ou cron
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Ajouter le backend au path
sys.path.insert(0, str(Path(__file__).parent))

def setup_apscheduler():
    """
    Configure APScheduler pour exécuter le pipeline automatiquement
    Installation: pip install apscheduler
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from run_all import run_pipeline_for_scheduler
        
        scheduler = BackgroundScheduler()
        
        # Tâche 1: Pipeline complet toutes les 6 heures
        scheduler.add_job(
            func=run_pipeline_for_scheduler,
            trigger='interval',
            hours=6,
            id='pipeline_6h',
            name='Pipeline sismique complet',
            replace_existing=True
        )
        
        # Tâche 2: Retraining chaque dimanche à 02:00
        scheduler.add_job(
            func=run_pipeline_for_scheduler,
            trigger=CronTrigger(day_of_week='sun', hour=2, minute=0),
            id='weekly_retraining',
            name='Retraining hebdomadaire',
            replace_existing=True
        )
        
        # Tâche 3: Nettoyage des vieux modèles chaque mois
        scheduler.add_job(
            func=cleanup_old_models,
            trigger=CronTrigger(day=1, hour=3, minute=0),
            id='monthly_cleanup',
            name='Nettoyage modèles',
            replace_existing=True
        )
        
        scheduler.start()
        print("✅ Scheduler démarré avec succès")
        return scheduler
        
    except ImportError:
        print("⚠️ APScheduler non installé. Installation: pip install apscheduler")
        return None

def cleanup_old_models():
    """Supprime les modèles de plus de 3 mois"""
    from pathlib import Path
    from datetime import datetime, timedelta
    
    models_dir = Path(__file__).parent.parent / "models" / "versions"
    if not models_dir.exists():
        return
    
    three_months_ago = datetime.now() - timedelta(days=90)
    deleted_count = 0
    
    for model_file in models_dir.glob("*.pkl"):
        # Extraire la date du nom du fichier (format: nom_2024W40_timestamp.pkl)
        try:
            mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
            if mtime < three_months_ago:
                model_file.unlink()
                deleted_count += 1
        except Exception:
            continue
    
    print(f"🧹 Nettoyage: {deleted_count} anciens modèles supprimés")

# Pour utilisation dans main.py
if __name__ == "__main__":
    scheduler = setup_apscheduler()
    
    if scheduler:
        print("Scheduler en cours d'exécution. Ctrl+C pour arrêter.")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            scheduler.shutdown()
            print("\nScheduler arrêté")