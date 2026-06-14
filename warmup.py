print("Warming up models...")

from paddleocr import PaddleOCR
import easyocr
from rembg import remove
import numpy as np

PaddleOCR(
    use_textline_orientation=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
)

easyocr.Reader(['ar', 'en'], gpu=False)
remove(np.zeros((10, 10, 3), dtype=np.uint8))

print("Models ready.")