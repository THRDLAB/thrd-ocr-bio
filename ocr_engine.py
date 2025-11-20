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
    Charge une image depuis le disque en forçant le décodage
    et en la convertissant en RGB.
    """
    img = Image.open(path)
    img.load()
    return img.convert("RGB")


def preprocess_for_bio(im: Image.Image) -> Image.Image:
    """
    Pré-traitement spécifique pour les comptes-rendus biologiques.

    - on conserve surtout le bas de la page (zone examens),
    - niveaux de gris + autocontraste,
    - léger sharpen,
    - redimensionnement (uniquement si très grand) pour accélérer l'OCR.
    """
    w, h = im.size

    # Conserver le bas de la page (ajuste 0.35 si besoin)
    cropped = im.crop((0, int(h * 0.35), w, h))

    # Gris + autocontraste
    gray = ImageOps.grayscale(cropped)
    gray = ImageOps.autocontrast(gray)

    # Sharpen léger
    gray = gray.filter(ImageFilter.SHARPEN)

    # Resize uniquement si l'image est très grande
    max_side = 1550
    w2, h2 = gray.size
    m = max(w2, h2)
    if m > max_side:
        r = max_side / m
        gray = gray.resize((int(w2 * r), int(h2 * r)), Image.LANCZOS)

    return gray


def _run_tesseract_string(img: Image.Image, psm: int = 6) -> str:
    """
    Exécution Tesseract pour récupérer le texte brut.
    On force :
      - OEM 1 (LSTM only)
      - PSM adapté (par défaut 6)
      - conservation des espaces
      - pas d'inversion automatique noir/blanc
    """
    config = f"--oem 1 --psm {psm} {TESS_BASE_CONFIG}"
    return pytesseract.image_to_string(img, lang=TESS_LANG, config=config)



def _run_tesseract_data(img: Image.Image, psm: int = 6) -> List[dict]:
    """
    Exécution Tesseract pour récupérer les boxes (image_to_data).
    Même config que pour le texte pour rester cohérent.
    """
    config = f"--oem 1 --psm {psm} {TESS_BASE_CONFIG}"
    data = pytesseract.image_to_data(
        img,
        lang=TESS_LANG,
        config=config,
        output_type=Output.DICT,
    )

    boxes: List[dict] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = data["text"][i]
        if not txt or not txt.strip():
            continue
        boxes.append(
            {
                "text": txt,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
        )
    return boxes

def light_extract_text(path: str) -> Optional[OCRResult]:
    """
    Version light de l'OCR :

      - même pré-traitement que la version premium
      - un seul passage Tesseract pour le texte brut
      - pas de récupération des boxes (image_to_data)

    Utilisée comme premier essai rapide ; si le parsing TSH échoue,
    on bascule sur `premium_extract_text`.
    """
    # 1) Load
    try:
        img = _load_image(path)
    except Exception as e:
        print("OCR-LIGHT ERROR: failed to load image:", e)
        print(traceback.format_exc())
        return None

    # 2) Preprocess (même logique que le premium)
    try:
        img = preprocess_for_bio(img)
    except Exception as e:
        print("OCR-LIGHT ERROR: preprocessing failed:", e)
        print(traceback.format_exc())
        # Fallback : on tente quand même avec l'image brute redimensionnée
        try:
            w, h = img.size
            max_side = 1400  # un poil plus agressif que le premium
            m = max(w, h)
            if m > max_side:
                r = max_side / m
                img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
        except Exception:
            return None

    # 3) OCR texte uniquement (PAS de boxes en mode light)
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

    # On renvoie des boxes vides : le parser peut vivre sans,
    # et si ça ne suffit pas on passera en premium.
    return OCRResult(raw_text=raw_text, boxes=[])


def premium_extract_text(path: str) -> Optional[OCRResult]:
    """
    Pipeline OCR optimisé :

      1. Chargement robuste de l'image.
      2. Pré-traitement spécifique bilans bio.
      3. Un seul passage Tesseract pour le texte + un pour les boxes.
      4. Gestion d'erreurs explicite avec logs.

    Retourne :
        OCRResult(raw_text, boxes) ou None si l'OCR est impossible.
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
        # On tente quand même avec l'image brute redimensionnée
        try:
            w, h = img.size
            max_side = 1600
            m = max(w, h)
            if m > max_side:
                r = max_side / m
                img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
        except Exception:
            return None

    # 3) OCR texte
    try:
        raw_text = _run_tesseract_string(img, psm=6)
    except Exception as e:
        print("OCR ERROR: image_to_string failed:", e)
        print(traceback.format_exc())
        raw_text = ""

    # 4) OCR boxes
    try:
        boxes = _run_tesseract_data(img, psm=6)
    except Exception as e:
        print("OCR ERROR: image_to_data failed:", e)
        print(traceback.format_exc())
        boxes = []

    if (not raw_text or not raw_text.strip()) and not boxes:
        print("OCR ERROR: empty result (no text, no boxes)")
        return None

    return OCRResult(raw_text=raw_text, boxes=boxes)

def optimum_extract_text(path: str) -> Optional[OCRResult]:
    """
    Version 'optimum' pour les bilans compliqués :
      - upscale de l'image
      - binarisation plus agressive
      - récupération du texte même avec des conf basses
    Plus lente, mais utile en dernier recours.
    """
    try:
        img = _load_image(path)
    except Exception as e:
        print("OCR-OPTIMUM ERROR: failed to load image:", e)
        print(traceback.format_exc())
        return None

    # 1) Upscale x1.5 pour les polices petites
    try:
        w, h = img.size
        scale = 1.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    except Exception as e:
        print("OCR-OPTIMUM ERROR: upscale failed:", e)
        print(traceback.format_exc())

    # 2) Prétraitement plus agressif : niveaux de gris + autocontrast + seuil
    try:
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray)

        # simple binarisation globale
        def _threshold(p):
            return 255 if p > 160 else 0
        bin_img = gray.point(_threshold)
    except Exception as e:
        print("OCR-OPTIMUM ERROR: binarization failed:", e)
        print(traceback.format_exc())
        bin_img = img  # fallback

    # 3) OCR texte brut sans filtrer par conf
    try:
        raw_text = _run_tesseract_string(bin_img, psm=6)
    except Exception as e:
        print("OCR-OPTIMUM ERROR: image_to_string failed:", e)
        print(traceback.format_exc())
        raw_text = ""

    raw_text = raw_text or ""

    # 4) OCR avec boxes mais conf très bas (ex. >= 20)
try:
    data = _run_tesseract_data(bin_img, psm=6)

    # Sécurité anti-crash : Tesseract peut retourner une liste ou un string
    if not isinstance(data, dict) or "text" not in data:
        print("OCR-OPTIMUM WARNING: malformed data from Tesseract")
        data = {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": []}

    boxes = []
    for i, txt in enumerate(data["text"]):
        if not txt or not txt.strip():
            continue

        # conf peut aussi être 'nan' -> try/except
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = 0.0

        if conf < 20:
            continue

        try:
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
        except Exception:
            continue

        boxes.append(OCRBox(text=txt, conf=conf, x=x, y=y, w=w, h=h))

except Exception as e:
    print(f"OCR-OPTIMUM ERROR: image_to_data failed: {e}")
    print(traceback.format_exc())
    boxes = []

    if not raw_text.strip() and not boxes:
        return None

    return OCRResult(raw_text=raw_text, boxes=boxes)

