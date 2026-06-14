# OCR FOR IDS

Extracts the name and national ID number from national ID card images using computer vision and OCR

---

## What it does

1. Takes a photo of an Egyptian ID card
2. detects and flattens the card
3. corrects its orientation
4. runs Arabic OCR
5. returns the holder's name + 14-digit national ID number as JSON

---

## How to run

### 1. Clone the repo
```bash
git clone https://github.com/rokaia1692006/OCR_FOR_ID.git
cd OCR_FOR_ID
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Folder structure
```
jobtrailOCR/
├── fileFinal.py
├── requirements.txt
├── Dockerfile
├── static/
│   └── frontEnd.html
└── imagesToTest/
    └── testEgID.png
```

### 5. Run the server
```bash
python fileFinal.py
```
Then open localhost in your browser

### 6. Run inference directly (no server)
use ocrNoteBook.ipynb
---

## API

### POST  + extract

Upload an ID image -> receive extracted data as JSON

Request: `multipart/form-data` with field `file` containing the image.

Response:
```json
{
  "success": true,
  "nationalId": "xxxxxxxxxxxxxxxx",
  "name": "محمد عبد المنعم محمد السيد",
  "processing_time_seconds": 4.2
}
```

---

## Pipeline

### 1. Perspective Transformation
1. `findBESTCardContour` removes the background using `rembg`
2. tries 5 thresholding strategies (Otsu, inverted Otsu, adaptive Gaussian, Canny, foreground mask) to find the card's 4 corners `PerspectiveTransform` 
3. computes a homography matrix -> warps the card to a flat top-down view

### 2. Preprocessing
`makeImageBetter` converts the image to grayscale and upscales it to at least 1000px if needed. After testing CLAHE, denoising, and adaptive thresholding, PaddleOCR performs better on the minimally processed image.

### 3. Orientation Correction
1. `textDetection` runs PaddleOCR to get bounding boxes. 
2. `getOrientationFromText` fits a minimum area rectangle to each box and takes the median angle across all boxes. 
3. `rotatImage` then rotates the image without clipping any content by expanding the canvas to fit result.

### 4. OCR
`getText` runs PaddleOCR on the corrected image and returns detected text strings alongside their bounding box 

### 5. Post Processing
- `CleanText` strips all non-Arabic + non-digit characters from each OCR line
- `groupTextByLine` groups boxes sharing the same vertical position into lines + sorted right-to-left for Arabic reading order
- `extractIDFromText` finds the 14-digit national ID number + with a fallback that concatenates all digit strings in case the number was split across boxes
- `getName` uses raw OCR lines to find the first name and grouped lines to get the rest of the name
- `nameValidation` verifies the extracted name contains only Arabic characters

---

## Post Processing and Validation

### ID Number
- Arabic-Indic digits converted to Western digits
- All non-numeric characters stripped
- Validated against `\d{14}` regex
- If not found in a single box, all digit strings are concatenated and re-validated

### Name
- Located by finding lines after the `بطاقة تحقيق الشخصية` header
- Stops at the first digit line
- stripping tashkil marks
- Validated to contain Arabic letters and spaces


## Data Privacy

national IDs are sensitive PII. The following measures are applied:
- tmp files are deleted after processing 
- No image or extracted data is saved
- all traffic should go over HTTPS
- CORS should be locked to known frontend origins rather than `allow_origins=["*"]`
- Rate limiting and API key authentication should be added 

---



## Dependencies
- PaddleOCR + PaddlePaddle 
- OpenCV
- rembg 
- FastAPI + uvicorn
- Pillow,NumPy