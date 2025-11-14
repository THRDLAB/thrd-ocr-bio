FROM python:3.11-slim

# ===== Install system dependencies =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===== Requirements =====
COPY requirements.txt .

# Must install numpy BEFORE opencv
RUN pip install --no-cache-dir numpy==1.26.4

RUN pip install --no-cache-dir -r requirements.txt

# ===== App files =====
COPY . .

EXPOSE 8000

# ===== Start server =====
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
