import pytesseract
from pytesseract import Output
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# -----------------------------------------------------------------------------
# Dataclass structurée : OCR brut premium
# -----------------------------------------------------------------------------

@dataclass
class OCRResult:
    raw_text: str
    boxes: List[dict]   # chaque box = {"text": str, "left": int, "top": int, "width": int, "height": int}


# -----------------------------------------------------------------------------
# Pré-traitements premium
# -----------------------------------------------------------------------------

def _load_image(path: str) -> Image.Image:
    """Charge en tant qu'image PIL en RGB."""
    return Image.open(path).convert("RGB")


def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convertit PIL -> OpenCV (BGR)."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _upscale(img: Image.Image, factor: float = 2.0) -> Image.Image:
    w, h = img.size
    return img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)


def _threshold(img: Image.Image) -> Image.Image:
    """Threshold brutal en noir/blanc."""
    gray = ImageOps.grayscale(img)
    np_gray = np.array(gray)
    _, threshed = cv2.threshold(np_gray, 150, 255, cv2.THRESH_BINARY)
    return Image.fromarray(threshed)


def _sharpen(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.SHARPEN)


# -----------------------------------------------------------------------------
# OCR multi-variant
# -----------------------------------------------------------------------------

def _run_tesseract(img: Image.Image):
    """
    Renvoie (texte_brut, boxes).
    boxes = liste de dicts contenant text + coordonnées + dimension.
    """
    # OCR texte brut
    raw_text = pytesseract.image_to_string(img, lang="fra+eng")

    # OCR détaillé pour récupérer les positions
    data = pytesseract.image_to_data(img, lang="fra+eng", output_type=Output.DICT)

    boxes = []
    for i in range(len(data["text"])):
        word = data["text"][i]
        if not word.strip():
            continue
        boxes.append({
            "text": word,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
        })

    return raw_text, boxes


def premium_extract_text(path: str) -> Optional[OCRResult]:
    """
    OCR premium :
    - Multi-pass (original / grayscale / autocontrast / upscale / sharpen / threshold)
    - Récupération des boxes (image_to_data)
    - Merge intelligent des textes
    """

    try:
        base = _load_image(path)
    except Exception:
        return None

    # Variantes
    variants = []

    # 1) Original
    variants.append(base)

    # 2) Grayscale
    variants.append(ImageOps.grayscale(base).convert("RGB"))

    # 3) Grayscale + autocontrast
    gray_ac = ImageOps.autocontrast(ImageOps.grayscale(base))
    variants.append(gray_ac.convert("RGB"))

    # 4) Upscale ×2
    variants.append(_upscale(base, 2.0))

    # 5) Sharpen
    variants.append(_sharpen(base))

    # 6) Threshold
    variants.append(_threshold(base).convert("RGB"))

    # Fusion des résultats
    all_texts = []
    all_boxes = []

    for img in variants:
        try:
            text, boxes = _run_tesseract(img)
            if text and text.strip():
                all_texts.append(text)
            if boxes:
                all_boxes.extend(boxes)
        except Exception:
            continue

    if not all_texts:
        return OCRResult(raw_text="", boxes=[])

    # Texte final = concat intelligemment les variantes
    merged_text = "\n".join(all_texts)

    return OCRResult(
        raw_text=merged_text,
        boxes=all_boxes,
    )
