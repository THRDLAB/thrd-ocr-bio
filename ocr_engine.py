
import pytesseract
from pytesseract import Output
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional


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
    try:
        base = _load_image(path)
    except Exception:
        return None

    base = _resize_if_needed(base)

    variants = [
        ImageOps.autocontrast(ImageOps.grayscale(base)).convert("RGB"),
        _upscale(base),
        _sharpen(base),
    ]

    all_text = []
    best_variant = None
    best_length = 0

    for img in variants:
        try:
            txt = _run_tesseract_string(img)
            if txt and len(txt) > best_length:
                best_length = len(txt)
                best_variant = img
            all_text.append(txt)
        except:
            continue

    if not all_text:
        return OCRResult(raw_text="", boxes=[])

    merged = "\n".join(all_text)

    if best_variant is None:
        return OCRResult(raw_text=merged, boxes=[])

    try:
        data = _run_tesseract_data(best_variant)
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
    except:
        boxes = []

    return OCRResult(raw_text=merged, boxes=boxes)
