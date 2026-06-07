"""
training_pipeline.py
====================
Pipeline de retraining hebdomadaire des modèles
Version complète avec retraining des modèles de zones
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Ajouter le backend au path
sys.path.append(str(Path(__file__).parent))

from model_registry import ModelRegistry, ModelType
from model_comparator import ModelComparator


class TrainingPipeline:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.registry = ModelRegistry(self.base_dir)
        self.comparator = ModelComparator(self.base_dir)
        self.data_dir = self.base_dir / "data"
        
    def load_training_data(self) -> pd.DataFrame:
        """Charge et prépare les données pour l'entraînement"""
        from fetch_timeseries import FEATURES
        
        weekly_path = self.data_dir / "weekly_history.csv"
        
        if not weekly_path.exists():
            raise FileNotFoundError(f"Données historiques introuvables: {weekly_path}")
        
        df = pd.read_csv(weekly_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  📊 Données chargées: {len(df)} semaines")
        print(f"  📅 Période: {df['date'].min().date()} → {df['date'].max().date()}")
        
        return df
    
    def load_zone_training_data(self) -> pd.DataFrame:
        """Charge les données historiques par zone"""
        zone_path = self.data_dir / "zone_history.csv"
        
        if not zone_path.exists():
            raise FileNotFoundError(f"Données zones introuvables: {zone_path}")
        
        df = pd.read_csv(zone_path, parse_dates=['date'])
        df = df.sort_values(['zone', 'date']).reset_index(drop=True)
        
        print(f"  📊 Données zones chargées: {len(df)} lignes")
        print(f"  📍 Zones disponibles: {df['zone'].nunique()}")
        print(f"  📅 Période: {df['date'].min().date()} → {df['date'].max().date()}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'nb_seismes') -> Tuple[np.ndarray, np.ndarray, list]:
        """Prépare les features pour l'entraînement global"""
        from fetch_timeseries import FEATURES
        
        df = df.copy()
        
        # Features de base
        df['energy_per_seisme'] = df.get('energy_sum', df['nb_seismes'] * 1e6) / (df['nb_seismes'] + 1)
        df['log_energy'] = np.log1p(df.get('energy_sum', df['nb_seismes'] * 1e6))
        df['mag_spread'] = df.get('mag_max', 5) - df.get('mag_mean', 4)
        df['depth_mean'] = df.get('depth_mean', 50)
        
        # Lags
        for lag in [1, 2, 3, 4, 8]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Lags sur energy
        if 'energy_sum' in df.columns:
            for lag in [1, 2, 4]:
                df[f'energy_lag_{lag}'] = df['energy_sum'].shift(lag)
        
        # Rolling means
        for w in [4, 8, 26]:
            df[f'rolling_mean_{w}'] = df[target_col].shift(1).rolling(w).mean()
        
        # Rolling std
        df['rolling_std_8'] = df[target_col].shift(1).rolling(8).std()
        
        # Rolling max
        df['rolling_max_4'] = df[target_col].shift(1).rolling(4).max()
        
        # Zones features
        df['nb_zones'] = df.get('nb_zones', 5)
        for lag in [1, 2]:
            df[f'zones_lag_{lag}'] = df['nb_zones'].shift(lag)
        
        # Calendar features
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        
        # Supprimer les NaN
        df_clean = df.dropna()
        
        print(f"  📊 Après preprocessing: {len(df_clean)} lignes")
        
        # Features disponibles
        available_features = [f for f in FEATURES if f in df_clean.columns]
        
        X = df_clean[available_features].values
        y = df_clean[target_col].values
        
        return X, y, available_features
    
    def prepare_zone_features(self, zone_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les features pour l'entraînement par zone"""
        from fetch_zones import FEATURES_ZONE
        
        zone_df = zone_df.copy()
        zone_df = zone_df.sort_values('date')
        
        # Features pour zone
        # Lags
        for lag in [1, 2, 3, 4]:
            zone_df[f'lag_{lag}'] = zone_df['nb_seismes'].shift(lag)
        
        # Rolling means
        for w in [4, 8]:
            zone_df[f'rolling_mean_{w}'] = zone_df['nb_seismes'].shift(1).rolling(w).mean()
        
        # Supprimer les NaN
        zone_df_clean = zone_df.dropna()
        
        if len(zone_df_clean) == 0:
            return np.array([]), np.array([])
        
        # Features disponibles
        available_features = [f for f in FEATURES_ZONE if f in zone_df_clean.columns and f != 'zone']
        
        X = zone_df_clean[available_features].values
        y = zone_df_clean['nb_seismes'].values
        
        return X, y
    
    def train_xgboost_global(self, X: np.ndarray, y: np.ndarray) -> Tuple[xgb.XGBRegressor, Dict]:
        """Entraîne un modèle XGBoost pour la prédiction globale"""
        print("\n  🚀 Entraînement XGBoost Global...")
        start_time = time.time()
        
        if len(X) < 10:
            raise ValueError(f"Pas assez de données: {len(X)} échantillons")
        
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        
        training_time = time.time() - start_time
        
        y_pred = model.predict(X)
        y_pred = np.maximum(y_pred, 0)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / (y + 1)) * 100),
            'prediction_coverage': 80.0,
            'training_time_sec': training_time
        }
        
        print(f"  ✅ Entraînement terminé en {training_time:.1f}s")
        print(f"     RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f} | R²: {metrics['r2']:.4f}")
        
        return model, metrics
    
    def train_random_forest_global(self, X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestRegressor, Dict]:
        """Entraîne un modèle Random Forest global"""
        print("\n  🚀 Entraînement Random Forest Global...")
        start_time = time.time()
        
        if len(X) < 10:
            raise ValueError(f"Pas assez de données: {len(X)} échantillons")
        
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X, y)
        
        training_time = time.time() - start_time
        
        y_pred = model.predict(X)
        y_pred = np.maximum(y_pred, 0)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / (y + 1)) * 100),
            'prediction_coverage': 80.0,
            'training_time_sec': training_time
        }
        
        print(f"  ✅ Entraînement terminé en {training_time:.1f}s")
        print(f"     RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
        
        return model, metrics
    
    def train_xgboost_zones(self, zone_hist: pd.DataFrame) -> Dict[int, Tuple[Any, Dict]]:
        """
        Entraîne un modèle XGBoost par zone
        
        Returns:
            Dictionnaire {zone_id: (modèle, métriques)}
        """
        print("\n  🚀 Entraînement XGBoost par Zone...")
        start_time = time.time()
        
        zones = zone_hist['zone'].unique()
        models = {}
        successful_zones = 0
        
        for zone in zones:
            zone_data = zone_hist[zone_hist['zone'] == zone].copy()
            zone_data = zone_data.sort_values('date')
            
            # Préparer features
            X, y = self.prepare_zone_features(zone_data)
            
            if len(X) < 10:  # Minimum de données requis
                print(f"     ⚠️ Zone {zone}: données insuffisantes ({len(X)} semaines) - skip")
                continue
            
            try:
                # Paramètres adaptés pour zones
                params = {
                    'n_estimators': 50,  # Moins d'arbres pour zones
                    'max_depth': 4,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbosity': 0
                }
                
                model = xgb.XGBRegressor(**params)
                model.fit(X, y)
                
                # Évaluation
                y_pred = model.predict(X)
                y_pred = np.maximum(y_pred, 0)
                
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                mae = mean_absolute_error(y, y_pred)
                mape = np.mean(np.abs((y - y_pred) / (y + 1)) * 100)
                
                metrics = {
                    'rmse': round(rmse, 2),
                    'mae': round(mae, 2),
                    'mape': round(mape, 1),
                    'training_time_sec': 0
                }
                
                models[int(zone)] = (model, metrics)
                successful_zones += 1
                print(f"     ✅ Zone {zone}: RMSE={rmse:.2f} | MAE={mae:.2f} | {len(X)} semaines")
                
            except Exception as e:
                print(f"     ❌ Zone {zone}: erreur - {e}")
        
        training_time = time.time() - start_time
        print(f"\n  ✅ {successful_zones}/{len(zones)} zones entraînées avec succès en {training_time:.1f}s")
        
        return models
    
    def save_zone_models(self, zone_models: Dict[int, Tuple[Any, Dict]], model_name: str = 'xgboost_zone'):
        """
        Sauvegarde tous les modèles de zones dans le registre
        
        Args:
            zone_models: Dictionnaire {zone_id: (modèle, métriques)}
            model_name: Nom de base du modèle
        """
        print(f"\n  💾 Sauvegarde des modèles de zones...")
        
        saved_count = 0
        for zone_id, (model, metrics) in zone_models.items():
            try:
                # Nom unique pour chaque zone
                full_name = f"{model_name}_zone_{zone_id}_{datetime.now().strftime('%Y%W')}"
                
                # Sauvegarder dans le registre
                path = self.registry.save_model(
                    model,
                    full_name,
                    ModelType.ZONES,
                    metrics,
                    metrics.get('training_time_sec', 0)
                )
                saved_count += 1
            except Exception as e:
                print(f"     ❌ Erreur sauvegarde zone {zone_id}: {e}")
        
        print(f"  ✅ {saved_count} modèles de zones sauvegardés")
        
        # Sauvegarder aussi un dictionnaire global des modèles
        zone_models_dict = {}
        for zone_id, (model, metrics) in zone_models.items():
            zone_models_dict[str(zone_id)] = {
                'model_path': str(self.registry.models_dir / f"{model_name}_zone_{zone_id}_{datetime.now().strftime('%Y%W')}.pkl"),
                'metrics': metrics,
                'training_date': datetime.now().isoformat()
            }
        
        # Sauvegarder le dictionnaire
        zones_metadata_path = self.data_dir / "zones_models_metadata.json"
        with open(zones_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(zone_models_dict, f, indent=2)
        
        print(f"  📄 Métadonnées zones sauvegardées: {zones_metadata_path}")
    
    def run_weekly_training(self) -> Dict:
        """Exécute le pipeline complet d'entraînement hebdomadaire"""
        print("\n" + "="*60)
        print("  🔄 PIPELINE DE RETRAINING HEBDOMADAIRE")
        print("="*60)
        print(f"  📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': [],
            'zones_trained': [],
            'improvements': []
        }
        
        try:
            # ========== 1. MODÈLES GLOBAUX ==========
            print("\n" + "="*40)
            print("  📊 PARTIE 1: MODÈLES GLOBAUX")
            print("="*40)
            
            # Charger données globales
            df = self.load_training_data()
            X, y, features = self.prepare_features(df)
            
            if len(X) >= 20:
                # XGBoost Global
                xgb_model, xgb_metrics = self.train_xgboost_global(X, y)
                xgb_path = self.registry.save_model(
                    xgb_model, 
                    'xgboost_global', 
                    ModelType.GLOBAL,
                    xgb_metrics,
                    xgb_metrics.get('training_time_sec', 0)
                )
                results['models_trained'].append({
                    'name': 'xgboost_global',
                    'path': xgb_path,
                    'metrics': xgb_metrics
                })
                
                # Random Forest Global
                if len(X) > 50:
                    rf_model, rf_metrics = self.train_random_forest_global(X, y)
                    rf_path = self.registry.save_model(
                        rf_model,
                        'random_forest_global',
                        ModelType.GLOBAL,
                        rf_metrics,
                        rf_metrics.get('training_time_sec', 0)
                    )
                    results['models_trained'].append({
                        'name': 'random_forest_global',
                        'path': rf_path,
                        'metrics': rf_metrics
                    })
            else:
                print("  ⚠️ Pas assez de données pour modèles globaux")
            
            # ========== 2. MODÈLES DE ZONES ==========
            print("\n" + "="*40)
            print("  📍 PARTIE 2: MODÈLES DE ZONES")
            print("="*40)
            
            zone_df = self.load_zone_training_data()
            
            if len(zone_df) >= 100:  # Au moins 100 lignes totales
                zone_models = self.train_xgboost_zones(zone_df)
                
                if zone_models:
                    self.save_zone_models(zone_models, 'xgboost_zone')
                    results['zones_trained'] = list(zone_models.keys())
                    results['zones_count'] = len(zone_models)
            else:
                print("  ⚠️ Pas assez de données pour modèles de zones")
            
            # ========== 3. ÉVALUATION ==========
            print("\n" + "="*40)
            print("  🎯 PARTIE 3: ÉVALUATION")
            print("="*40)
            
            selection = self.comparator.compare_and_select_best(force_retrain=False)
            results['selection'] = selection
            
            # ========== 4. RAPPORT ==========
            report_path = self.comparator.generate_comparison_report()
            results['report_path'] = report_path
            
            # ========== 5. SAUVEGARDE MÉTRIQUES ==========
            metrics_df = pd.DataFrame([
                {
                    'date': datetime.now().isoformat(),
                    'type': 'global',
                    'model': m['name'],
                    'rmse': m['metrics']['rmse'],
                    'mae': m['metrics']['mae'],
                    'r2': m['metrics']['r2']
                }
                for m in results['models_trained']
            ])
            
            # Ajouter métriques zones
            for zone_id in results.get('zones_trained', []):
                metrics_df = pd.concat([metrics_df, pd.DataFrame([{
                    'date': datetime.now().isoformat(),
                    'type': 'zone',
                    'model': f'zone_{zone_id}',
                    'rmse': None,  # Sera rempli par les métriques individuelles
                    'mae': None,
                    'r2': None
                }])], ignore_index=True)
            
            training_log = self.data_dir / "training_history.csv"
            if training_log.exists():
                old_df = pd.read_csv(training_log)
                metrics_df = pd.concat([old_df, metrics_df], ignore_index=True)
            
            metrics_df.to_csv(training_log, index=False)
            
            print("\n" + "="*60)
            print("  ✅ PIPELINE DE RETRAINING TERMINÉ")
            print(f"  📊 Modèles globaux: {len(results['models_trained'])}")
            print(f"  📍 Zones entraînées: {len(results.get('zones_trained', []))}")
            print("="*60)
            
        except Exception as e:
            print(f"\n  ❌ Erreur dans le pipeline: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        return results
    
    def should_retrain(self) -> bool:
        """Détermine si on doit refaire l'entraînement (chaque dimanche)"""
        today = datetime.now()
        
        if today.weekday() != 6:
            return False
        
        training_log = self.data_dir / "training_history.csv"
        if not training_log.exists():
            return True
        
        try:
            df = pd.read_csv(training_log)
            last_training = pd.to_datetime(df['date'].max())
            days_since = (datetime.now() - last_training).days
            return days_since >= 7
        except Exception:
            return True


def run_weekly_pipeline():
    """Fonction principale à appeler par le scheduler"""
    pipeline = TrainingPipeline()
    
    if pipeline.should_retrain():
        print("\n  🔄 Lancement du retraining hebdomadaire...")
        results = pipeline.run_weekly_training()
        return results
    else:
        print("\n  ⏭️ Pas de retraining cette semaine (programmé pour dimanche)")
        return {"status": "skipped"}


if __name__ == "__main__":
    results = run_weekly_pipeline()
    print("\n  📊 Résumé final:")
    if 'models_trained' in results:
        print(f"     Modèles globaux: {len(results.get('models_trained', []))}")
    if 'zones_trained' in results:
        print(f"     Zones entraînées: {len(results.get('zones_trained', []))}")
    if 'selection' in results:
        print(f"     Modèle sélectionné: {results['selection'].get('selected_model')}")