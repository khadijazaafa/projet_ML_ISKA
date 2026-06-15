# 🌍 Plateforme Intelligente de Surveillance et de Prévision Sismique

## Présentation

Ce projet a pour objectif de développer une plateforme intelligente capable d'analyser l'activité sismique mondiale à partir des données ouvertes de l'USGS (United States Geological Survey).

Le système combine plusieurs techniques de Machine Learning afin de :

* Classifier les séismes récents selon leur niveau de magnitude.
* Prévoir le nombre de séismes attendus durant la semaine suivante.
* Identifier les zones géographiques présentant un risque sismique élevé.
* Visualiser les résultats dans un tableau de bord interactif.

## Source des données

Les données utilisées proviennent de l'API officielle de l'USGS.

Chaque événement sismique contient notamment :

* Magnitude
* Profondeur
* Latitude
* Longitude
* Date et heure
* Informations sur les tsunamis
* Indicateurs de qualité des mesures

## Fonctionnalités principales

### 1. Classification des séismes

Le premier modèle prédit la classe de magnitude d'un séisme à partir de ses caractéristiques.

Classes utilisées :

* Faible
* Modéré
* Fort

Meilleur modèle : **Random Forest**

### 2. Prévision temporelle

Le deuxième modèle estime le nombre de séismes attendus pour la semaine suivante.

Meilleur modèle : **XGBoost**

### 3. Prévision spatiale

Le troisième modèle identifie les zones géographiques les plus exposées aux risques sismiques.

Méthodes utilisées :

* K-Means Clustering
* XGBoost

### 4. Dashboard interactif

Le tableau de bord permet de visualiser :

* Les séismes récents
* Les prédictions de magnitude
* Les prévisions hebdomadaires
* Les zones à risque
* Les indicateurs d'activité sismique

## Structure du projet

```text
PROJET_VF
│
├── backend
│   ├── main.py
│   ├── run_all.py
│   ├── fetch_classification.py
│   ├── fetch_timeseries.py
│   ├── fetch_zones.py
│   └── check_weekly_diff.py
│
├── choix_des_models
│   ├── modele2+3.ipynb
│   └── traitements-modèle1.ipynb
│
├── data
│   ├── output_classification.json
│   ├── output_forecast.json
│   ├── output_zones.json
│   ├── pipeline_status.json
│   ├── training_history.csv
│   ├── weekly_history.csv
│   ├── zone_history.csv
│   └── zones_models_metadata.json
│
├── frontend
│
├── models
│
├── rapport_ML_vf.docx
│
├── README.md
│
└── requirements.txt
```

## Modèles utilisés

### Classification

* Random Forest
* XGBoost
* Gradient Boosting
* Decision Tree
* KNN
* Logistic Regression
* SVM

### Prévision temporelle

* XGBoost
* Random Forest
* LightGBM
* ARIMA
* SARIMA
* Holt-Winters

### Prévision spatiale

* K-Means
* XGBoost

## Résultats obtenus

### Classification

* Accuracy : 99.7 %
* F1-Macro : 0.989

### Prévision temporelle

* MAE : 28.6
* MAPE : 11.36 %
* R² : 0.78

## Auteur

Projet de Fin du module

Earthquake Monitoring and Forecasting using Machine Learning.


# Installation et lancement

## 1. Cloner le projet

```bash
git clone https://github.com/khadijazaafa/projet_ML_ISKA.git
cd PROJET_VF
```

## 2. Créer un environnement virtuel

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 4. Lancer l'API Backend

```bash
cd backend
python main.py
```

Le serveur démarre sur :

```text
http://localhost:8000
```

## 5. Générer les prédictions

Depuis le dossier backend :

```bash
python run_all.py
```

Cette commande exécute automatiquement :

* La classification des séismes récents
* La prévision du nombre de séismes
* La prédiction des zones à risque

Les résultats sont enregistrés dans :

```text
data/output_classification.json
data/output_forecast.json
data/output_zones.json
```
