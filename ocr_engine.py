import pytesseract
from pytesseract import Output
from PIL import Image, ImageOps, ImageFilter
from dataclasses import dataclass
from typing import List, Optional


def preprocess_for_bio(im: Image.Image) -> Image.Image:
    """Pré-traitement spécifique pour les comptes-rendus biologiques."""

    # On conserve uniquement le bas de la page où se situent les examens.
    w, h = im.size
    cropped = im.crop((0, int(h * 0.35), w, h))

    # Niveaux de gris + autocontraste pour améliorer la lisibilité.
    gray = ImageOps.grayscale(cropped)
    gray = ImageOps.autocontrast(gray)

    # Légère accentuation pour faire ressortir le texte.
    gray = gray.filter(ImageFilter.SHARPEN)

    # Redimensionnement pour avoir un plus grand côté proche de 2000 px.
    max_side = 2000
    w2, h2 = gray.size
    max_current_side = max(w2, h2)
    if max_current_side < max_side:
        resize_ratio = max_side / max_current_side
        gray = gray.resize((int(w2 * resize_ratio), int(h2 * resize_ratio)), Image.LANCZOS)

    return gray


@dataclass
class OCRResult:
    raw_text: str
    boxes: List[dict]


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _resize_if_needed(img: Image.Image, max_side: int = 1800) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _upscale(img: Image.Image, factor: float = 1.5) -> Image.Image:
    w, h = img.size
    return img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)


def _sharpen(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.SHARPEN)


def _run_tesseract_string(img: Image.Image, psm: int = 6) -> str:
    return pytesseract.image_to_string(img, lang="fra+eng", config=f"--psm {psm}")


def _run_tesseract_data(img: Image.Image, psm: int = 6):
    return pytesseract.image_to_data(img, lang="fra+eng", config=f"--psm {psm}", output_type=Output.DICT)


def premium_extract_text(path: str) -> Optional[OCRResult]:
    """
    Version light :
    - resize max 1400px
    - grayscale + autocontrast + sharpen
    - 1 seul image_to_string
    - 1 seul image_to_data
    """
    try:
        img = _load_image(path)
    except Exception:
        return None

    # Prétraitement spécifique bio
    img = preprocess_for_bio(img)

    # Prétraitement simple
    img = _resize_if_needed(img, max_side=1400)
    img = ImageOps.autocontrast(ImageOps.grayscale(img)).convert("RGB")
    img = _sharpen(img)

    # OCR texte
    text = pytesseract.image_to_string(img, lang="fra+eng", config="--psm 6")

    # OCR boxes
    try:
        data = pytesseract.image_to_data(img, lang="fra+eng", config="--psm 6", output_type=Output.DICT)
        boxes = []
        for i in range(len(data["text"])):
            t = data["text"][i]
            if not t.strip():
                continue
            boxes.append({
                "text": t,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            })
    except Exception:
        boxes = []
