"""
main.py
=======
FastAPI backend — Seismic Dashboard (version simplifiée sans MLOPS)
Avec lifespan (plus de warning) et port configurable
"""

import json
import subprocess
import threading
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configuration de l'encodage
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── CHEMINS ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND = BASE_DIR / "frontend"
BACKEND_DIR = BASE_DIR / "backend"

# Fichiers JSON produits par les scripts fetch_*
CLASSIFICATION_JSON = DATA_DIR / "output_classification.json"
FORECAST_JSON = DATA_DIR / "output_forecast.json"
ZONES_JSON = DATA_DIR / "output_zones.json"
PIPELINE_STATUS_JSON = DATA_DIR / "pipeline_status.json"  # statut simplifié

# Variables globales pour le scheduler
scheduler_thread = None
pipeline_running = False


# ── LIFESPAN CONTEXT MANAGER ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le démarrage et l'arrêt de l'application"""
    global scheduler_thread, pipeline_running
    
    print("=" * 55)
    print("  🌍 SEISMIC INTELLIGENCE API (version allégée)")
    print("  📡 Données rafraîchies périodiquement")
    print("=" * 55)
    
    # === STARTUP (avant le yield) ===
    DATA_DIR.mkdir(exist_ok=True)
    
    data_exists = CLASSIFICATION_JSON.exists() and FORECAST_JSON.exists()
    
    if not data_exists:
        print("📡 Données manquantes → lancement pipeline initial...")
        threading.Thread(target=run_pipeline, daemon=True).start()
    else:
        print("✅ Données existantes trouvées")
    
    scheduler_thread = threading.Thread(
        target=background_scheduler,
        args=(6,),
        daemon=True
    )
    scheduler_thread.start()
    print("⏰ Scheduler démarré (refresh toutes les 6h)")
    print("=" * 55)
    
    # Yield pour que l'application tourne
    yield
    
    # === SHUTDOWN (après le yield) ===
    print("\n🛑 Arrêt de l'API Seismic Intelligence...")
    pipeline_running = False
    print("✅ Nettoyage terminé")


# ── APP ───────────────────────────────────────────────────────
app = FastAPI(
    title="Seismic Intelligence API",
    version="2.0.0",
    description="API de prédiction sismique (version simplifiée)",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Serve les fichiers statiques
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


# ── HELPER FUNCTIONS ─────────────────────────────────────────
def read_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Data not ready yet: {path.name}. Pipeline still running."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_pipeline_status(status: str, message: str = ""):
    """Met à jour le fichier de statut simplifié"""
    status_data = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "message": message,
    }
    try:
        with open(PIPELINE_STATUS_JSON, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        print(f"Erreur mise à jour statut: {e}")


# ── BACKGROUND REFRESH ────────────────────────────────────────
def run_pipeline():
    """Exécute le pipeline simplifié (run_all.py)"""
    global pipeline_running
    
    if pipeline_running:
        print("⚠️ Pipeline déjà en cours d'exécution")
        return
    
    pipeline_running = True
    
    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] Pipeline refresh...")
    
    try:
        result = subprocess.run(
            ["python", "run_all.py"],
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            print("✅ Pipeline terminé avec succès")
            update_pipeline_status("success", "Pipeline completed successfully")
        else:
            print(f"❌ Pipeline erreur:\n{result.stderr}")
            update_pipeline_status("error", result.stderr[:500])
            
    except subprocess.TimeoutExpired:
        print("⏰ Pipeline timeout (>10min)")
        update_pipeline_status("timeout", "Pipeline execution exceeded 10 minutes")
    except Exception as e:
        print(f"💥 Pipeline exception: {e}")
        update_pipeline_status("exception", str(e))
    finally:
        pipeline_running = False


def background_scheduler(interval_hours: int = 6):
    """Boucle infinie qui lance le pipeline toutes les X heures"""
    while True:
        time.sleep(interval_hours * 3600)
        print(f"\n🔄 Exécution planifiée du pipeline (interval {interval_hours}h)")
        run_pipeline()


# ── ENDPOINTS ─────────────────────────────────────────────────

@app.get("/")
def index():
    html_file = FRONTEND / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="index.html introuvable")
    return FileResponse(str(html_file))


@app.get("/api/status")
def status():
    def file_info(path: Path):
        if path.exists():
            mtime = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            size_kb = round(path.stat().st_size / 1024, 1)
            return {"exists": True, "updated_at": mtime, "size_kb": size_kb}
        return {"exists": False}

    return JSONResponse({
        "api_time": datetime.now(timezone.utc).isoformat(),
        "classification": file_info(CLASSIFICATION_JSON),
        "forecast": file_info(FORECAST_JSON),
        "zones": file_info(ZONES_JSON),
        "pipeline_status": read_json(PIPELINE_STATUS_JSON) if PIPELINE_STATUS_JSON.exists() else None
    })


@app.get("/api/classification")
def get_classification():
    return JSONResponse(read_json(CLASSIFICATION_JSON))


@app.get("/api/forecast")
def get_forecast():
    return JSONResponse(read_json(FORECAST_JSON))


@app.get("/api/zones")
def get_zones():
    return JSONResponse(read_json(ZONES_JSON))


@app.post("/api/refresh")
def trigger_refresh(background_tasks: BackgroundTasks):
    """Déclenche manuellement le pipeline (fetch + prediction)"""
    background_tasks.add_task(run_pipeline)
    return JSONResponse({
        "status": "started",
        "message": "Pipeline lancé en arrière-plan",
        "started_at": datetime.now(timezone.utc).isoformat()
    })


@app.get("/api/health")
def health_check():
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0"
    })


# Endpoints MLOPS désactivés (anciennement utilisés)
@app.get("/api/models/current")
def get_current_model():
    return JSONResponse({"status": "disabled", "message": "MLOPS non actif"})


@app.get("/api/models/compare")
def compare_models():
    return JSONResponse({"status": "disabled", "message": "MLOPS non actif"})


@app.get("/api/zones/models")
def get_zones_models():
    return JSONResponse({"status": "disabled", "message": "MLOPS non actif"})


@app.get("/api/zones/model/{zone_id}")
def get_zone_model(zone_id: int):
    return JSONResponse({"status": "disabled", "message": "MLOPS non actif"})


# ── RUN LOCAL ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port (défaut: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (défaut: 0.0.0.0)")
    args = parser.parse_args()
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║     SEISMIC INTELLIGENCE API (version simplifiée)        ║
    ║     🚀 http://{args.host}:{args.port}                            ║
    ║     📊 http://{args.host}:{args.port}/api/docs                  ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )