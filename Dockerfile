FROM python:3.9-slim

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
    pip install --no-cache-dir pandas==1.5.3 && \
    pip install --no-cache-dir scikit-learn==1.2.2 && \
    pip install --no-cache-dir xgboost==1.7.6 && \
    pip install --no-cache-dir joblib==1.2.0 && \
    pip install --no-cache-dir requests==2.31.0 && \
    pip install --no-cache-dir fastapi==0.104.1 && \
    pip install --no-cache-dir uvicorn[standard]==0.24.0

# Copier tout le projet
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/ ./data/
COPY frontend/ ./frontend/

WORKDIR /app/backend

EXPOSE 7860

# Lancer l'API directement
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]