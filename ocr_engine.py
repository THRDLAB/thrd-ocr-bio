import traceback
from dataclasses import dataclass
from typing import List, Optional

import pytesseract
from pytesseract import Output
from PIL import Image, ImageOps, ImageFilter


def preprocess_for_bio(im: Image.Image) -> Image.Image:
    """
    Pré-traitement spécifique pour les comptes-rendus biologiques.

    - On conserve surtout le bas de la page (zone examens / hormonologie).
    - Niveaux de gris + autocontraste.
    - Légère accentuation.
    - Redimensionnement pour avoir un grand côté ~2000 px max.
    """
    w, h = im.size

    # 1) Crop : on garde le bas de la page
    cropped = im.crop((0, int(h * 0.35), w, h))

    # 2) Gris + autocontraste
    gray = ImageOps.grayscale(cropped)
    gray = ImageOps.autocontrast(gray)

    # 3) Sharpen léger
    gray = gray.filter(ImageFilter.SHARPEN)

    # 4) Resize pour que le plus grand côté soit proche de 2000 px
    max_side = 2000
    w2, h2 = gray.size
    max_current_side = max(w2, h2)
    if max_current_side < max_side:
        resize_ratio = max_side / max_current_side
        gray = gray.resize(
            (int(w2 * resize_ratio), int(h2 * resize_ratio)),
            Image.LANCZOS,
        )

    return gray


@dataclass
class OCRResult:
    raw_text: str
    boxes: List[dict]


def _load_image(path: str) -> Image.Image:
    """
    Charge une image depuis un chemin disque, avec décodage forcé
    et conversion en RGB pour éviter les surprises (PNG, JPG, etc.).
    """
    img = Image.open(path)
    img.load()  # force le décodage ici
    return img.convert("RGB")


def _resize_if_needed(img: Image.Image, max_side: int = 1600) -> Image.Image:
    """
    Redimensionne l'image si le plus grand côté dépasse max_side.
    """
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _sharpen(img: Image.Image) -> Image.Image:
    """
    Accentuation légère pour améliorer la lisibilité du texte.
    """
    return img.filter(ImageFilter.SHARPEN)


def _run_tesseract_string(img: Image.Image, psm: int = 6) -> str:
    """
    Appel Tesseract pour extraire le texte brut.
    """
    return pytesseract.image_to_string(
        img,
        lang="fra+eng",
        config=f"--psm {psm}",
    )


def _run_tesseract_data(img: Image.Image, psm: int = 6):
    """
    Appel Tesseract pour récupérer les box (image_to_data).
    """
    return pytesseract.image_to_data(
        img,
        lang="fra+eng",
        config=f"--psm {psm}",
        output_type=Output.DICT,
    )


def premium_extract_text(path: str) -> Optional[OCRResult]:
    """
    Version light et optimisée :

    - Chargement image robuste.
    - Pré-traitement bio (crop bas de page + gris + contraste + sharpen).
    - Resize max 1600 px pour réduire le temps de calcul.
    - 1 seul image_to_string (texte brut).
    - 1 seul image_to_data (boxes).
    - Gestion d'erreurs verbeuse dans les logs, mais API propre côté app.py.

    Retourne :
        OCRResult(raw_text, boxes) ou None si l'OCR est totalement impossible.
    """
    # 1) Chargement image
    try:
        img = _load_image(path)
    except Exception as e:
        # Log clair dans les logs Northflank
        print("OCR ERROR: failed to load image:", e)
        print(traceback.format_exc())
        return None

    # 2) Pré-traitement spécifique bio + resize + sharpen
    try:
        img = preprocess_for_bio(img)
        img = _resize_if_needed(img, max_side=1600)
        img = _sharpen(img)
    except Exception as e:
        print("OCR ERROR: preprocessing failed:", e)
        print(traceback.format_exc())
        # On tente quand même de continuer avec l'image brute
        try:
            img = _resize_if_needed(img, max_side=1600)
        except Exception:
            return None

    # 3) OCR texte brut
    try:
        raw_text = _run_tesseract_string(img, psm=6)
    except Exception as e:
        print("OCR ERROR: image_to_string failed:", e)
        print(traceback.format_exc())
        raw_text = ""

    # 4) OCR data (boxes)
    boxes: List[dict] = []
    try:
        data = _run_tesseract_data(img, psm=6)
        texts = data.get("text", [])
        n = len(texts)
        for i in range(n):
            t = texts[i]
            if not t or not t.strip():
                continue
            boxes.append(
                {
                    "text": t,
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                }
            )
    except Exception as e:
        print("OCR ERROR: image_to_data failed:", e)
        print(traceback.format_exc())
        boxes = []

    # 5) Si vraiment rien n'est exploitable → None (app.py renverra "OCR failed")
    if (not raw_text or not raw_text.strip()) and not boxes:
        print("OCR ERROR: empty result (no text and no boxes)")
        return None

    # 6) Sinon on renvoie le résultat complet
    return OCRResult(raw_text=raw_text, boxes=boxes)
