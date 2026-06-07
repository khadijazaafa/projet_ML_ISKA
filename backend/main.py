"""
main.py
=======
FastAPI backend — Seismic Dashboard avec MLOPS
Version avec lifespan (plus de warning) et port configurable
"""

import json
import subprocess
import threading
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
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
MODELS_DIR = BASE_DIR / "models"
VERSIONS_DIR = MODELS_DIR / "versions"

# Fichiers JSON existants
CLASSIFICATION_JSON = DATA_DIR / "output_classification.json"
FORECAST_JSON = DATA_DIR / "output_forecast.json"
ZONES_JSON = DATA_DIR / "output_zones.json"

# Nouveaux fichiers MLOPS
MODEL_METRICS_CSV = DATA_DIR / "model_metrics.csv"
CURRENT_BEST_JSON = DATA_DIR / "current_best_model.json"
PIPELINE_STATUS_JSON = DATA_DIR / "pipeline_status.json"
MLOPS_LOG_JSON = DATA_DIR / "mlops_pipeline_log.json"

# Variables globales pour le scheduler
scheduler_thread = None
pipeline_running = False


# ── LIFESPAN CONTEXT MANAGER (remplace on_event) ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le démarrage et l'arrêt de l'application"""
    global scheduler_thread, pipeline_running
    
    print("=" * 55)
    print("  🌍 SEISMIC INTELLIGENCE API v2.0")
    print("  🤖 MLOPS Enabled")
    print("=" * 55)
    
    # === STARTUP (avant le yield) ===
    DATA_DIR.mkdir(exist_ok=True)
    VERSIONS_DIR.mkdir(exist_ok=True, parents=True)
    
    data_exists = CLASSIFICATION_JSON.exists() and FORECAST_JSON.exists()
    
    if not data_exists:
        print("📡 Données manquantes → lancement pipeline initial...")
        threading.Thread(target=run_pipeline, daemon=True).start()
    else:
        print("✅ Données existantes trouvées")
        
        if CURRENT_BEST_JSON.exists():
            try:
                with open(CURRENT_BEST_JSON, 'r') as f:
                    current = json.load(f)
                    prod_model = current.get('production_model', {})
                    print(f"🎯 Modèle en production: {prod_model.get('model_name', 'unknown')}")
            except Exception:
                pass
        
        if VERSIONS_DIR.exists():
            n_versions = len(list(VERSIONS_DIR.glob("*.pkl")))
            print(f"📦 Modèles versionnés: {n_versions}")
    
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
    description="API de prédiction sismique avec MLOPS intégré",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan  # Plus de warning !
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


def read_csv_as_json(csv_path: Path, limit: int = None) -> list:
    if not csv_path.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if limit:
            df = df.tail(limit)
        return df.to_dict(orient='records')
    except Exception:
        return []


def get_model_performance_summary() -> Dict:
    summary = {
        "current_model": None,
        "best_model_history": [],
        "recent_performance": [],
        "improvement_trend": None
    }
    
    if CURRENT_BEST_JSON.exists():
        try:
            with open(CURRENT_BEST_JSON, 'r') as f:
                current = json.load(f)
                summary["current_model"] = current.get('production_model')
        except Exception:
            pass
    
    if MODEL_METRICS_CSV.exists():
        metrics_df = read_csv_as_json(MODEL_METRICS_CSV, limit=20)
        summary["recent_performance"] = metrics_df
        
        if len(metrics_df) >= 2:
            try:
                last_rmse = metrics_df[-1].get('rmse', 0)
                prev_rmse = metrics_df[-2].get('rmse', 0)
                if prev_rmse > 0:
                    improvement = ((prev_rmse - last_rmse) / prev_rmse) * 100
                    summary["improvement_trend"] = round(improvement, 1)
            except Exception:
                pass
    
    return summary


# ── BACKGROUND REFRESH ────────────────────────────────────────
def run_pipeline(run_mlops: bool = True):
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


def update_pipeline_status(status: str, message: str = ""):
    status_data = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "message": message,
        "next_scheduled": None
    }
    try:
        with open(PIPELINE_STATUS_JSON, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        print(f"Erreur mise à jour statut: {e}")


def background_scheduler(interval_hours: int = 6):
    while True:
        time.sleep(interval_hours * 3600)
        print(f"\n🔄 Exécution planifiée du pipeline (interval {interval_hours}h)")
        run_pipeline(run_mlops=True)


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

    mlops_info = {}
    if CURRENT_BEST_JSON.exists():
        try:
            with open(CURRENT_BEST_JSON, 'r') as f:
                mlops_info["current_best"] = json.load(f)
        except Exception:
            pass
    
    if MODEL_METRICS_CSV.exists():
        mlops_info["metrics_available"] = True
    
    return JSONResponse({
        "api_time": datetime.now(timezone.utc).isoformat(),
        "classification": file_info(CLASSIFICATION_JSON),
        "forecast": file_info(FORECAST_JSON),
        "zones": file_info(ZONES_JSON),
        "mlops": mlops_info
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
    background_tasks.add_task(run_pipeline, run_mlops=True)
    return JSONResponse({
        "status": "started",
        "message": "Pipeline lancé en background avec MLOPS",
        "started_at": datetime.now(timezone.utc).isoformat()
    })


@app.get("/api/models/current")
def get_current_model():
    if not CURRENT_BEST_JSON.exists():
        return JSONResponse({"status": "no_model"})
    try:
        with open(CURRENT_BEST_JSON, 'r') as f:
            return JSONResponse(json.load(f))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/compare")
def compare_models(weeks: int = 4):
    if not MODEL_METRICS_CSV.exists():
        return JSONResponse({"status": "no_data"})
    try:
        import pandas as pd
        df = pd.read_csv(MODEL_METRICS_CSV)
        df['week_start'] = pd.to_datetime(df['week_start'])
        last_weeks = df['week_start'].max() - pd.Timedelta(weeks=weeks)
        df_recent = df[df['week_start'] >= last_weeks]
        comparison = df_recent.groupby('model_name').agg({
            'rmse': 'mean', 'mae': 'mean', 'r2': 'mean'
        }).round(4).sort_values('rmse')
        return JSONResponse({
            "weeks_analyzed": weeks,
            "comparison": comparison.to_dict(orient='index')
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0"
    })
@app.get("/api/zones/models")
def get_zones_models():
    """Liste tous les modèles de zones disponibles"""
    metadata_path = DATA_DIR / "zones_models_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"status": "no_models"})

@app.get("/api/zones/model/{zone_id}")
def get_zone_model(zone_id: int):
    """Récupère le modèle pour une zone spécifique"""
    metadata_path = DATA_DIR / "zones_models_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return JSONResponse(metadata.get(str(zone_id), {"status": "not_found"}))
    return JSONResponse({"status": "no_models"})

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
    ║     SEISMIC INTELLIGENCE API v2.0 avec MLOPS           ║
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