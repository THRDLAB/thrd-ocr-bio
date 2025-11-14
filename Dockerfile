# Dockerfile

FROM python:3.11-slim

# Empêche Python de bufferiser la sortie (logs en direct)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Mise à jour + dépendances système minimales + tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    libtesseract-dev \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Création dossier app
WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt /app/requirements.txt

# Installation des dépendances Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copie du reste du code
COPY . /app

# Variable d'env pour la langue OCR (fra+eng => TSH et contextes en français/anglais)
ENV OCR_LANG=fra+eng

# Exposition du port (optionnel mais pratique)
EXPOSE 8000

# Commande de lancement (FastAPI / Uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
