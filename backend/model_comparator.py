"""
model_comparator.py
===================
Système de comparaison et d'évaluation des modèles
Version corrigée avec gestion des cas sans données
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Ajouter le backend au path
sys.path.append(str(Path(__file__).parent))

from model_registry import ModelRegistry, ModelType

class ModelComparator:
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = base_dir
        self.registry = ModelRegistry(self.base_dir)
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        
    def load_historical_data(self) -> pd.DataFrame:
        """Charge les données historiques pour évaluation"""
        weekly_path = self.data_dir / "weekly_history.csv"
        
        if not weekly_path.exists():
            raise FileNotFoundError(f"Fichier historique introuvable: {weekly_path}")
        
        df = pd.read_csv(weekly_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def prepare_features_for_backtest(self, df: pd.DataFrame, 
                                       target_col: str = 'nb_seismes') -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """Prépare les features pour le backtesting"""
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
        
        # Features disponibles
        available_features = [f for f in FEATURES if f in df_clean.columns]
        
        if len(df_clean) == 0:
            return np.array([]), np.array([]), pd.Index([])
        
        X = df_clean[available_features].values
        y = df_clean[target_col].values
        
        return X, y, df_clean.index
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Évalue un modèle sur les données de test"""
        if len(X_test) == 0:
            return {
                'rmse': 0,
                'mae': 0,
                'r2': 0,
                'mape': 0,
                'prediction_coverage': 0
            }
        
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
        
        # Prediction interval coverage
        residuals = y_test - y_pred
        std_residual = np.std(residuals) if len(residuals) > 1 else 0
        lower_bound = y_pred - 1.28 * std_residual
        upper_bound = y_pred + 1.28 * std_residual
        coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound)) * 100 if len(y_test) > 0 else 0
        
        return {
            'rmse': round(rmse, 2),
            'mae': round(mae, 2),
            'r2': round(r2, 4),
            'mape': round(mape, 1),
            'prediction_coverage': round(coverage, 1)
        }
    
    def test_all_global_models(self, weeks_for_test: int = 4) -> Dict[str, Dict]:
        """Teste tous les modèles globaux disponibles"""
        print("\n" + "="*60)
        print("  📊 COMPARAISON DES MODÈLES GLOBAUX")
        print("="*60)
        
        # Charger données
        df = self.load_historical_data()
        
        # Vérifier qu'on a assez de données
        if len(df) < weeks_for_test + 10:
            print(f"  ⚠️ Pas assez de données historiques ({len(df)} semaines)")
            print(f"  Minimum requis: {weeks_for_test + 10} semaines")
            return {}
        
        # Séparer train/test
        test_size = min(weeks_for_test, len(df) // 4)
        if test_size == 0:
            print("  ⚠️ Pas assez de données pour le test")
            return {}
        
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]
        
        # Préparer features
        X_train, y_train, _ = self.prepare_features_for_backtest(train_df)
        X_test, y_test, _ = self.prepare_features_for_backtest(test_df)
        
        if len(X_test) == 0:
            print("  ⚠️ Pas assez de données pour préparer les features de test")
            return {}
        
        print(f"  📅 Période d'entraînement: {train_df['date'].min().date()} → {train_df['date'].max().date()}")
        print(f"  📅 Période de test: {test_df['date'].min().date()} → {test_df['date'].max().date()}")
        print(f"  📊 Taille train: {len(X_train)}, test: {len(X_test)}\n")
        
        # Liste des modèles à tester
        models_to_test = {}
        
        # Modèles existants
        for model_name in ['model_xgb_global', 'model_Random_Forest']:
            model_path = self.models_dir / f"{model_name}.pkl"
            if model_path.exists():
                models_to_test[model_name.replace('model_', '')] = model_path
        
        # Modèles versionnés
        versions_dir = self.models_dir / "versions"
        if versions_dir.exists():
            for model_file in versions_dir.glob("*global*.pkl"):
                model_name = model_file.stem
                models_to_test[model_name] = model_file
        
        if not models_to_test:
            print("  ⚠️ Aucun modèle trouvé")
            return {}
        
        results = {}
        
        for model_name, model_path in models_to_test.items():
            try:
                print(f"  🔍 Test de: {model_name}")
                model = joblib.load(model_path)
                metrics = self.evaluate_model(model, X_test, y_test)
                results[model_name] = metrics
                print(f"     ✓ RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | R²: {metrics['r2']}")
            except Exception as e:
                print(f"  ❌ Erreur avec {model_name}: {e}")
        
        return results
    
    def compare_and_select_best(self, force_retrain: bool = False) -> Dict:
        """Compare tous les modèles et sélectionne le meilleur"""
        print("\n" + "="*60)
        print("  🏆 SÉLECTION DU MEILLEUR MODÈLE")
        print("="*60)
        
        # Tester tous les modèles
        results = self.test_all_global_models(weeks_for_test=4)
        
        if not results:
            print("  ⚠️ Aucun modèle testé avec succès")
            return {
                "selected_model": None,
                "metrics": None,
                "improvement_percent": 0,
                "action": "no_models_tested"
            }
        
        # Trouver le meilleur modèle (basé sur RMSE)
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        best_name = best_model[0]
        best_metrics = best_model[1]
        
        print(f"\n  🥇 MEILLEUR MODÈLE: {best_name}")
        print(f"     RMSE: {best_metrics['rmse']}")
        print(f"     MAE: {best_metrics['mae']}")
        print(f"     R²: {best_metrics['r2']}")
        
        # Récupérer modèle actuel
        current_model, current_metrics = self.registry.get_current_production_model()
        
        improvement = None
        should_switch = True
        
        if current_metrics and current_metrics.get('rmse'):
            current_rmse = current_metrics.get('rmse', float('inf'))
            if current_rmse > 0:
                improvement = (current_rmse - best_metrics['rmse']) / current_rmse * 100
            else:
                improvement = 100 if best_metrics['rmse'] == 0 else 0
            
            print(f"\n  📈 Comparaison avec modèle actuel:")
            print(f"     Actuel RMSE: {current_rmse:.2f}")
            print(f"     Nouveau RMSE: {best_metrics['rmse']:.2f}")
            print(f"     Amélioration: {improvement:+.1f}%")
            
            # Décider si on change
            if improvement > 5:
                print(f"  ✅ AMÉLIORATION SIGNIFICATIVE → Adoption du nouveau modèle")
            elif improvement > 0:
                print(f"  ⚠️ Légère amélioration ({improvement:.1f}%) → Adoption (force={force_retrain})")
                should_switch = force_retrain
            else:
                print(f"  ❌ Pas d'amélioration → Maintien du modèle actuel")
                should_switch = False
        
        if should_switch and best_name != "current":
            # Déterminer le type de modèle
            if 'zone' in best_name.lower():
                model_type = ModelType.ZONES
            else:
                model_type = ModelType.GLOBAL
            
            # Mettre à jour le best model
            self.registry.update_current_best(best_name, model_type)
            
            return {
                "selected_model": best_name,
                "metrics": best_metrics,
                "improvement_percent": improvement,
                "action": "switched" if improvement else "initial_deployment"
            }
        else:
            return {
                "selected_model": "current",
                "metrics": current_metrics,
                "improvement_percent": improvement,
                "action": "kept_current"
            }
    
    def generate_comparison_report(self) -> str:
        """Génère un rapport HTML de comparaison des modèles"""
        results = self.test_all_global_models()
        
        if not results:
            # Générer un rapport minimal si pas de résultats
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Rapport de Comparaison des Modèles</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .info {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>📊 Rapport de Comparaison des Modèles</h1>
                <div class="info">
                    <p>⚠️ Pas assez de données pour comparer les modèles actuellement.</p>
                    <p>📅 Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
                    <p>💡 Le système collecte actuellement les données historiques. Reviens dans quelques semaines.</p>
                </div>
            </body>
            </html>
            """
        else:
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Rapport de Comparaison des Modèles</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                    h1 { color: #333; }
                    table { border-collapse: collapse; width: 100%; background-color: white; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                    th { background-color: #3b82f6; color: white; }
                    .best { background-color: #22c55e20; font-weight: bold; }
                    .info { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <h1>📊 Rapport de Comparaison des Modèles</h1>
                <div class="info">
                    <p>📅 Généré le: {date}</p>
                    <p>📊 Nombre de modèles comparés: {n_models}</p>
                </div>
                <table>
                    <tr>
                        <th>Modèle</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>R²</th>
                        <th>MAPE (%)</th>
                        <th>Coverage (%)</th>
                    </tr>
            """.format(
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                n_models=len(results)
            )
            
            best_rmse = min([r['rmse'] for r in results.values()]) if results else 0
            
            for name, metrics in sorted(results.items(), key=lambda x: x[1]['rmse']):
                is_best = metrics['rmse'] == best_rmse and best_rmse > 0
                row_class = 'class="best"' if is_best else ''
                html += f"""
                    <tr {row_class}>
                        <td>{name}</td>
                        <td>{metrics['rmse']}</td>
                        <td>{metrics['mae']}</td>
                        <td>{metrics['r2']}</td>
                        <td>{metrics['mape']}</td>
                        <td>{metrics['prediction_coverage']}</td>
                    </tr>
                """
            
            html += """
                </table>
            </body>
            </html>
            """
        
        # Sauvegarder le rapport
        report_path = self.data_dir / "model_comparison_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\n  📄 Rapport sauvegardé: {report_path}")
        
        return str(report_path)


if __name__ == "__main__":
    comparator = ModelComparator()
    results = comparator.test_all_global_models()
    selection = comparator.compare_and_select_best()
    print(f"\n  Résultat sélection: {selection}")
    comparator.generate_comparison_report()