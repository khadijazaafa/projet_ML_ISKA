FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python (versions compatibles)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir -r requirements.txt

# Copier le projet
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/ ./data/
COPY frontend/ ./frontend/

WORKDIR /app/backend

# Commande: d'abord retraîner les modèles, puis lancer le pipeline
CMD python force_retrain_docker.py && python run_all.py && uvicorn main:app --host 0.0.0.0 --port 7860