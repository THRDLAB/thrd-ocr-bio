import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedTSH:
    ok: bool
    value: Optional[float]
    unit: Optional[str]
    ref_min: Optional[float]
    ref_max: Optional[float]
    confidence: Optional[float]   # ⚠️ float pour matcher TSHResponse
    error: Optional[str]


def _normalize(text: str) -> str:
    """Minuscule + suppression des accents + normalisation des espaces."""
    if not text:
        return ""
    # minuscules
    text = text.lower()
    # suppression accents
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    # espaces normalisés
    text = re.sub(r"\s+", " ", text)
    return text


def _to_float(s: str) -> Optional[float]:
    try:
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def _extract_ref_interval(context: str) -> tuple[Optional[float], Optional[float]]:
    """
    Cherche un intervalle de référence dans un petit bout de texte.
    Ex : "0.27 a 4.20", "0.27 - 4.20", "0.
