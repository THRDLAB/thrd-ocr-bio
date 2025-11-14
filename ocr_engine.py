import os
import logging
from typing import Optional

import pytesseract
from PIL import Image, ImageOps
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Langue Tesseract : tu peux définir OCR_LANG=fra+eng dans les variables d'env
_TESS_LANG: Optional[str] = os.getenv("OCR_LANG")  # ex: "fra+eng"


def extract_text_from_file(path: str) -> str:
    """
    Point d'entrée principal pour l'OCR.
    - Si PDF : on tente d'extraire le texte avec pypdf.
    - Si image : OCR via Tesseract.
    Retourne une grosse string (peut être vide si rien trouvé).
    """
    if not path or not os.path.exists(path):
        logger.error(f"[OCR] File not found: {path}")
        return ""

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".pdf":
            logger.info(f"[OCR] Processing PDF: {path}")
            return _extract_text_from_pdf(path)
        else:
            logger.info(f"[OCR] Processing image: {path}")
            return _extract_text_from_image(path)
    except Exception as e:
        logger.exception(f"[OCR] Error while processing {path}: {e}")
        return ""


# --------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------

def _extract_text_from_pdf(path: str) -> str:
    """
    Extraction de texte depuis un PDF.
    - Pour l'instant : PDF texte uniquement (pypdf).
    - Si le PDF est un scan (images uniquement), pypdf retournera souvent ''
      → à ce moment-là, on pourrait plus tard ajouter une étape OCR page par page.
    """
    try:
        reader = PdfReader(path)
    except Exception as e:
        logger.exception(f"[OCR] Cannot read PDF {path}: {e}")
        return ""

    texts = []

    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
            logger.info(f"[OCR] PDF page {i}: extracted {len(page_text)} chars")
            texts.append(page_text)
        except Exception as e:
            logger.warning(f"[OCR] Error extracting text from page {i} of {path}: {e}")

    full_text = "\n\n".join(t for t in texts if t)

    # TODO plus tard : si full_text est très court / vide → fallback OCR sur les pages (pdf -> images -> pytesseract)
    if not full_text:
        logger.warning(f"[OCR] PDF {path} yielded empty text (possibly scanned image).")

    return _normalize_text(full_text)


# --------------------------------------------------------------------
# Images
# --------------------------------------------------------------------

def _extract_text_from_image(path: str) -> str:
    """
    Extraction de texte depuis une image avec Tesseract.
    On fait un pré-process minimal (grayscale, léger resize).
    """
    try:
        img = Image.open(path)
    except Exception as e:
        logger.exception(f"[OCR] Cannot open image {path}: {e}")
        return ""

    img = _preprocess_image_for_ocr(img)

    try:
        # Si OCR_LANG est défini (fra, fra+eng, etc.), on l'utilise. Sinon, Tesseract utilisera sa langue par défaut.
        if _TESS_LANG:
            text = pytesseract.image_to_string(img, lang=_TESS_LANG)
        else:
            text = pytesseract.image_to_string(img)

        logger.info(f"[OCR] Image {path}: extracted {len(text)} chars")
    except Exception as e:
        logger.exception(f"[OCR] Tesseract failed on image {path}: {e}")
        return ""

    return _normalize_text(text)


def _preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Préparation simple de l'image pour l'OCR :
    - conversion en niveaux de gris
    - légère amélioration du contraste via autocontrast
    - resize si l'image est immense pour éviter de plomber les perfs
    """
    # Grayscale
    img = ImageOps.grayscale(img)

    # Autocontraste pour améliorer la lisibilité
    img = ImageOps.autocontrast(img)

    # Resize si très grand (par ex. photo de smartphone 4000px)
    max_side = 2000
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        ratio = max_side / float(m)
        new_size = (int(w * ratio), int(h * ratio))
        logger.info(f"[OCR] Resizing image from {img.size} to {new_size}")
        img = img.resize(new_size)

    return img


# --------------------------------------------------------------------
# Normalisation du texte
# --------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Normalisation légère :
    - strip global
    - normalisation des sauts de ligne
    - garde la casse telle quelle (utile pour les abréviations labo)
    La logique plus fine de parsing (TSH, unités, etc.) sera dans parsers/tsh.py.
    """
    if text is None:
        return ""

    # Nettoyage basique : strips et normalisation des fins de lignes
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    cleaned = cleaned.strip()

    logger.debug(f"[OCR] Normalized text length: {len(cleaned)}")
    return cleaned
