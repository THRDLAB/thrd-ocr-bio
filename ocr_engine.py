import traceback
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageOps, ImageFilter
import pytesseract
from pytesseract import Output

TESS_LANG = "fra+eng"
TESS_BASE_CONFIG = "-c preserve_interword_spaces=1 tessedit_do_invert=0"


@dataclass
class OCRResult:
    raw_text: str
    boxes: List[dict]


def _load_image(path: str) -> Image.Image:
    """
    Chargement robuste de l'image source.
    """
    im = Image.open(path)
    return im.convert("RGB")


def preprocess_for_bio(im: Image.Image) -> Image.Image:
    """
    Pré-traitement spécifique bilans bio :

      - on garde surtout le bas du document (là où sont les valeurs),
      - niveaux de gris + autocontraste,
      - léger sharpen,
      - redimensionnement (uniquement si très grand) pour accélérer l'OCR.
    """
    w, h = im.size

    # Conserver le bas de la page (0.35 = partie haute coupée)
    cropped = im.crop((0, int(h * 0.35), w, h))

    # Gris + autocontraste
    gray = ImageOps.grayscale(cropped)
    gray = ImageOps.autocontrast(gray)

    # Sharpen léger
    gray = gray.filter(ImageFilter.SHARPEN)

    # Resize uniquement si l'image est très grande
    max_side = 1600
    w2, h2 = gray.size
    m = max(w2, h2)
    if m > max_side:
        r = max_side / m
        gray = gray.resize((int(w2 * r), int(h2 * r)), Image.LANCZOS)

    return gray


def _run_tesseract_string(img: Image.Image, psm: int = 6) -> str:
    """
    Exécution Tesseract pour récupérer le texte brut.
    """
    config = f"{TESS_BASE_CONFIG} --psm {psm} -l {TESS_LANG}"
    return pytesseract.image_to_string(img, config=config)


def _run_tesseract_data(img: Image.Image, psm: int = 6) -> List[dict]:
    """
    Exécution Tesseract pour récupérer les boxes (image_to_data).
    Retourne une liste de dicts {text,left,top,width,height,conf}.
    """
    config = f"{TESS_BASE_CONFIG} --psm {psm} -l {TESS_LANG}"
    data = pytesseract.image_to_data(
        img,
        config=config,
        output_type=Output.DICT,
    )

    boxes: List[dict] = []
    n = len(data.get("text", []))
    confs = data.get("conf", [0] * n)

    for i in range(n):
        txt = data["text"][i]
        if not txt or not txt.strip():
            continue
        try:
            boxes.append(
                {
                    "text": txt,
                    "left": int(data["left"][i]),
                    "top": int(data["top"][i]),
                    "width": int(data["width"][i]),
                    "height": int(data["height"][i]),
                    "conf": float(confs[i]),
                }
            )
        except Exception:
            # on ignore les boxes mal formées
            continue
    return boxes


# --------------------------------------------------------------------------
# NIVEAU 1 : OCR LIGHT (rapide, texte uniquement)
# --------------------------------------------------------------------------

def light_extract_text(path: str) -> Optional[OCRResult]:
    """
    Version light de l'OCR :

      - même pré-traitement de base que la version premium
      - un seul passage Tesseract pour le texte brut
      - pas de récupération des boxes

    Utilisée comme premier essai rapide ; si le parsing TSH échoue,
    on bascule sur premium puis optimum.
    """
    try:
        img = _load_image(path)
    except Exception as e:
        print("OCR-LIGHT ERROR: failed to load image:", e)
        print(traceback.format_exc())
        return None

    try:
        img = preprocess_for_bio(img)
    except Exception as e:
        print("OCR-LIGHT ERROR: preprocessing failed:", e)
        print(traceback.format_exc())
        # on tente quand même avec l'image brute
        pass

    try:
        raw_text = _run_tesseract_string(img, psm=6)
    except Exception as e:
        print("OCR-LIGHT ERROR: image_to_string failed:", e)
        print(traceback.format_exc())
        return None

    raw_text = raw_text or ""
    if not raw_text.strip():
        print("OCR-LIGHT ERROR: empty text")
        return None

    return OCRResult(raw_text=raw_text, boxes=[])


# --------------------------------------------------------------------------
# NIVEAU 2 : OCR PREMIUM (qualité standard, texte + boxes)
# --------------------------------------------------------------------------

def premium_extract_text(path: str) -> Optional[OCRResult]:
    """
    Pipeline OCR optimisé :

      1. Chargement robuste de l'image.
      2. Pré-traitement spécifique bilans bio.
      3. Un passage Tesseract pour le texte.
      4. Un passage Tesseract pour les boxes.
    """
    # 1) Load
    try:
        img = _load_image(path)
    except Exception as e:
        print("OCR ERROR: failed to load image:", e)
        print(traceback.format_exc())
        return None

    # 2) Preprocess
    try:
        img = preprocess_for_bio(img)
    except Exception as e:
        print("OCR ERROR: preprocessing failed:", e)
        print(traceback.format_exc())
        # On continue quand même avec l'image brute
        pass

    # 3) OCR texte brut
    try:
        raw_text = _run_tesseract_string(img, psm=6)
    except Exception as e:
        print("OCR ERROR: image_to_string failed:", e)
        print(traceback.format_exc())
        raw_text = ""

    raw_text = raw_text or ""

    # 4) OCR boxes
    try:
        boxes = _run_tesseract_data(img, psm=6)
    except Exception as e:
        print("OCR ERROR: image_to_data failed:", e)
        print(traceback.format_exc())
        boxes = []

    if (not raw_text.strip()) and not boxes:
        print("OCR ERROR: empty result (no text, no boxes)")
        return None

    return OCRResult(raw_text=raw_text, boxes=boxes)


# --------------------------------------------------------------------------
# NIVEAU 3 : OCR OPTIMUM (images difficiles, upscaling + binarisation)
# --------------------------------------------------------------------------

def optimum_extract_text(path: str) -> Optional[OCRResult]:
    """
    Version 'optimum' pour les bilans compliqués :

      - upscale de l'image
      - binarisation plus agressive
      - récupération du texte et des boxes

    Plus lente, mais utile en dernier recours.
    """
    try:
        img = _load_image(path)
    except Exception as e:
        print("OCR-OPTIMUM ERROR: failed to load image:", e)
        print(traceback.format_exc())
        return None

    # 1) Upscale x1.5 pour les petites polices
    try:
        w, h = img.size
        scale = 1.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    except Exception as e:
        print("OCR-OPTIMUM ERROR: upscale failed:", e)
        print(traceback.format_exc())

    # 2) Prétraitement + binarisation agressive
    try:
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray)

        def _threshold(p):
            return 255 if p > 160 else 0

        bin_img = gray.point(_threshold)
    except Exception as e:
        print("OCR-OPTIMUM ERROR: binarization failed:", e)
        print(traceback.format_exc())
        bin_img = img

    # 3) OCR texte brut
    try:
        raw_text = _run_tesseract_string(bin_img, psm=6)
    except Exception as e:
        print("OCR-OPTIMUM ERROR: image_to_string failed:", e)
        print(traceback.format_exc())
        raw_text = ""

    raw_text = raw_text or ""

    # 4) OCR boxes (on réutilise _run_tesseract_data qui renvoie une liste de dicts)
    try:
        boxes = _run_tesseract_data(bin_img, psm=6)
    except Exception as e:
        print("OCR-OPTIMUM ERROR: image_to_data failed:", e)
        print(traceback.format_exc())
        boxes = []

    if (not raw_text.strip()) and not boxes:
        print("OCR-OPTIMUM ERROR: empty result (no text, no boxes)")
        return None

    return OCRResult(raw_text=raw_text, boxes=boxes)
