"""
model_registry.py
=================
Gestionnaire de versions des modèles avec suivi des performances
Stocke tous les modèles entraînés avec leur métriques pour comparaison
"""

import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import shutil
from dataclasses import dataclass, asdict
from enum import Enum

class ModelType(Enum):
    GLOBAL = "global"
    ZONES = "zones"
    CLASSIFICATION = "classification"

@dataclass
class ModelMetrics:
    """Métriques d'un modèle"""
    model_name: str
    model_type: str
    week_start: str
    training_date: str
    rmse: float
    mae: float
    r2: float
    mape: float  # Mean Absolute Percentage Error
    prediction_coverage: float  # Intervalle de confiance
    training_time_sec: float
    is_current: bool = False
    model_path: str = ""

class ModelRegistry:
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = base_dir
        self.models_dir = base_dir / "models" / "versions"
        self.metrics_file = base_dir / "data" / "model_metrics.csv"
        self.current_best_file = base_dir / "data" / "current_best_model.json"
        self.config_file = base_dir / "data" / "model_registry_config.json"
        
        # Créer les dossiers nécessaires
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._init_metrics_file()
        self._init_config()
    
    def _init_metrics_file(self):
        """Initialiser fichier des métriques si inexistant"""
        if not self.metrics_file.exists():
            df = pd.DataFrame(columns=[
                'model_name', 'model_type', 'week_start', 'training_date',
                'rmse', 'mae', 'r2', 'mape', 'prediction_coverage',
                'training_time_sec', 'is_current', 'model_path'
            ])
            df.to_csv(self.metrics_file, index=False)
    
    def _init_config(self):
        """Initialiser fichier de configuration"""
        if not self.config_file.exists():
            config = {
                "production_model": None,
                "backtest_weeks": 4,
                "min_improvement_threshold": 0.05,  # 5% improvement needed
                "retrain_schedule": "weekly",
                "models_to_keep": 12  # Garder 12 versions
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
    
    def get_current_week(self) -> str:
        """Retourne la semaine actuelle au format YYYY-WXX"""
        now = datetime.now()
        week = now.strftime("%Y-W%W")
        return week
    
    def generate_model_name(self, base_name: str, week: str = None) -> str:
        """Générer nom unique pour le modèle"""
        if week is None:
            week = self.get_current_week()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{week}_{timestamp}"
    
    def save_model(self, model, model_name: str, model_type: ModelType, 
                   metrics: Dict[str, float], training_time: float) -> str:
        """
        Sauvegarde un modèle avec ses métriques
        
        Args:
            model: Le modèle sklearn/xgboost à sauvegarder
            model_name: Nom de base du modèle
            model_type: Type (GLOBAL/ZONES/CLASSIFICATION)
            metrics: Dictionnaire des métriques {rmse, mae, r2, mape, prediction_coverage}
            training_time: Temps d'entraînement en secondes
        
        Returns:
            Chemin du modèle sauvegardé
        """
        week = self.get_current_week()
        full_name = self.generate_model_name(model_name, week)
        model_path = self.models_dir / f"{full_name}.pkl"
        
        # Sauvegarder le modèle
        joblib.dump(model, model_path)
        
        # Créer l'entrée des métriques
        metrics_entry = ModelMetrics(
            model_name=full_name,
            model_type=model_type.value,
            week_start=week,
            training_date=datetime.now().isoformat(),
            rmse=metrics.get('rmse', 0),
            mae=metrics.get('mae', 0),
            r2=metrics.get('r2', 0),
            mape=metrics.get('mape', 0),
            prediction_coverage=metrics.get('prediction_coverage', 0),
            training_time_sec=training_time,
            is_current=False,
            model_path=str(model_path)
        )
        
        # Ajouter aux métriques
        self._add_metrics_entry(metrics_entry)
        
        # Nettoyer les vieux modèles
        self._cleanup_old_models(model_type)
        
        return str(model_path)
    
    def _add_metrics_entry(self, entry: ModelMetrics):
        """Ajouter une entrée dans le fichier CSV des métriques"""
        df = pd.read_csv(self.metrics_file)
        new_row = pd.DataFrame([asdict(entry)])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.metrics_file, index=False)
    
    def update_current_best(self, model_name: str, model_type: ModelType):
        """Marquer un modèle comme le meilleur actuel pour son type"""
        df = pd.read_csv(self.metrics_file)
        
        # Réinitialiser is_current pour ce type
        df.loc[df['model_type'] == model_type.value, 'is_current'] = False
        
        # Marquer le nouveau
        df.loc[df['model_name'] == model_name, 'is_current'] = True
        
        df.to_csv(self.metrics_file, index=False)
        
        # Mettre à jour config
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        config['production_model'] = {
            'model_name': model_name,
            'model_type': model_type.value,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Créer un lien symbolique vers le meilleur modèle
        best_model_path = self.base_dir / "models" / "current" / f"{model_type.value}_best.pkl"
        best_model_path.parent.mkdir(exist_ok=True)
        
        model_path = self.models_dir / f"{model_name}.pkl"
        if best_model_path.exists():
            best_model_path.unlink()
        
        # Copier au lieu de lien symbolique (pour Windows compatibilité)
        shutil.copy(model_path, best_model_path)
    
    def get_best_model(self, model_type: ModelType, metric: str = 'rmse') -> Tuple[Any, Dict]:
        """
        Récupère le meilleur modèle basé sur l'historique
        
        Args:
            model_type: Type de modèle
            metric: Métrique à utiliser pour la comparaison ('rmse', 'mae', 'r2')
        
        Returns:
            Tuple (modèle chargé, dict des métriques)
        """
        df = pd.read_csv(self.metrics_file)
        
        # Filtrer par type
        type_df = df[df['model_type'] == model_type.value].copy()
        
        if len(type_df) == 0:
            return None, None
        
        # Pour R2, plus haut est mieux; pour les autres, plus bas est mieux
        ascending = metric != 'r2'
        best_row = type_df.nsmallest(1, metric) if ascending else type_df.nlargest(1, metric)
        best_row = best_row.iloc[0]
        
        # Charger le modèle
        model_path = Path(best_row['model_path'])
        if not model_path.exists():
            # Fallback vers fichier original
            original_path = self.base_dir / "models" / f"model_xgb_{model_type.value}.pkl"
            if original_path.exists():
                model = joblib.load(original_path)
                return model, best_row.to_dict()
            return None, None
        
        model = joblib.load(model_path)
        
        metrics = {
            'rmse': best_row['rmse'],
            'mae': best_row['mae'],
            'r2': best_row['r2'],
            'mape': best_row['mape'],
            'week': best_row['week_start']
        }
        
        return model, metrics
    
    def get_current_production_model(self) -> Tuple[Any, Dict]:
        """Récupère le modèle actuellement en production"""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        prod_info = config.get('production_model')
        if not prod_info:
            return None, None
        
        model_path = self.models_dir / f"{prod_info['model_name']}.pkl"
        
        if not model_path.exists():
            # Fallback
            return self.get_best_model(ModelType(prod_info['model_type']))
        
        model = joblib.load(model_path)
        
        # Récupérer métriques
        df = pd.read_csv(self.metrics_file)
        metrics_row = df[df['model_name'] == prod_info['model_name']]
        
        if len(metrics_row) > 0:
            metrics = metrics_row.iloc[0].to_dict()
        else:
            metrics = {}
        
        return model, metrics
    
    def compare_models(self, model_type: ModelType, weeks: int = 4) -> pd.DataFrame:
        """
        Compare les performances des différents modèles sur les dernières semaines
        
        Args:
            model_type: Type de modèle à comparer
            weeks: Nombre de semaines à considérer
        
        Returns:
            DataFrame avec comparaison des modèles
        """
        df = pd.read_csv(self.metrics_file)
        
        # Filtrer par type
        type_df = df[df['model_type'] == model_type.value].copy()
        
        if len(type_df) == 0:
            return pd.DataFrame()
        
        # Prendre les dernières semaines
        last_weeks = sorted(type_df['week_start'].unique(), reverse=True)[:weeks]
        type_df = type_df[type_df['week_start'].isin(last_weeks)]
        
        # Grouper par modèle (sans la semaine)
        type_df['base_name'] = type_df['model_name'].apply(
            lambda x: '_'.join(x.split('_')[:-2])  # Enlever semaine et timestamp
        )
        
        # Calculer moyenne des métriques par modèle
        comparison = type_df.groupby('base_name').agg({
            'rmse': 'mean',
            'mae': 'mean',
            'r2': 'mean',
            'mape': 'mean',
            'training_time_sec': 'mean'
        }).round(4)
        
        comparison = comparison.sort_values('rmse')
        
        return comparison
    
    def _cleanup_old_models(self, model_type: ModelType):
        """Supprime les vieux modèles pour garder seulement les N derniers"""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        models_to_keep = config.get('models_to_keep', 12)
        
        df = pd.read_csv(self.metrics_file)
        type_df = df[df['model_type'] == model_type.value].copy()
        
        if len(type_df) <= models_to_keep:
            return
        
        # Trier par date et garder les plus récents
        type_df = type_df.sort_values('training_date', ascending=False)
        to_keep = type_df.head(models_to_keep)
        to_delete = type_df[~type_df['model_name'].isin(to_keep['model_name'])]
        
        # Supprimer les fichiers
        for _, row in to_delete.iterrows():
            model_path = Path(row['model_path'])
            if model_path.exists() and model_path.parent == self.models_dir:
                model_path.unlink()
        
        # Supprimer des métriques
        df = df[~df['model_name'].isin(to_delete['model_name'])]
        df.to_csv(self.metrics_file, index=False)
    
    def get_model_performance_trend(self, model_type: ModelType, 
                                     base_name: str = None) -> pd.DataFrame:
        """
        Récupère la tendance de performance d'un modèle sur les semaines
        
        Args:
            model_type: Type de modèle
            base_name: Nom de base du modèle (si None, prend le meilleur)
        
        Returns:
            DataFrame avec performance par semaine
        """
        df = pd.read_csv(self.metrics_file)
        
        if base_name:
            trend_df = df[df['model_name'].str.contains(base_name)].copy()
        else:
            # Prendre le meilleur modèle récent
            best_model, _ = self.get_best_model(model_type)
            if not best_model:
                return pd.DataFrame()
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            prod_name = config.get('production_model', {}).get('model_name', '')
            if prod_name:
                trend_df = df[df['model_name'] == prod_name].copy()
            else:
                trend_df = df[df['model_type'] == model_type.value].copy()
        
        trend_df = trend_df.sort_values('week_start')
        
        return trend_df[['week_start', 'rmse', 'mae', 'r2', 'mape']]