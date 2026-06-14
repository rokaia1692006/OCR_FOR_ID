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
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_textline_orientation=True, lang='ar')"
RUN python -c "import easyocr; easyocr.Reader(['ar', 'en'], gpu=False)"
RUN python -c "from rembg import remove; import numpy as np; from PIL import Image; remove(np.zeros((100,100,3), dtype=np.uint8))"

COPY . .
CMD ["sh", "-c", "uvicorn fileFinal:app --host 0.0.0.0 --port $PORT"]