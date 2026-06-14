FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from rembg import remove; from PIL import Image; import numpy as np; remove(Image.fromarray(np.zeros((10,10,3), dtype=np.uint8)))" \
    && echo "rembg model pre-downloaded OK"

COPY . .
CMD ["sh", "-c", "uvicorn fileFinal:app --host 0.0.0.0 --port $PORT"]