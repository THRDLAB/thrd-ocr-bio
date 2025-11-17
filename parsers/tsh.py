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
    confidence: str  # "high" | "medium" | "low"
    error: Optional[str]


def _normalize(text: str) -> str:
    """
    Minuscule + suppression des accents + normalisation des espaces.
    """
    if not text:
        return ""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    # normaliser les espaces
    text = re.sub(r"\s+", " ", text)
    return text


def _to_float(s: str) -> Optional[float]:
    try:
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def _extract_ref_interval(context: str) -> (Optional[float], Optional[float]):
    """
    Tente de trouver un intervalle de référence dans un bout de texte.
    On cherche deux nombres qui se suivent avec du texte entre les deux
    (ex: "0.27 a 4.20", "0.27 4 4.20", "0.27-4.20").
    """
    context = context.replace(",", ".")
    m = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\D+([0-9]+(?:\.[0-9]+)?)",
        context,
    )
    if not m:
        return None, None
    a = _to_float(m.group(1))
    b = _to_float(m.group(2))
    if a is None or b is None:
        return None, None
    if a > b:
        a, b = b, a
    return a, b


# Regex forte : TSH explicite
TSH_DIRECT_RE = re.compile(
    r"tsh[^0-9]{0,30}([0-9]+(?:[.,][0-9]+)?)\s*"
    r"(m[uµ]i/?l|mui/?l|µui/?l|ui/ml|ui/?l|m?ui/l)?",
    re.IGNORECASE,
)

# Regex générique valeur + unité (y compris OCR foireux type "mia")
VALUE_UNIT_RE = re.compile(
    r"([0-9]+(?:[.,][0-9]+)?)\s*"
    r"(m[uµ]i/?l|mui/?l|µui/?l|ui/ml|ui/?l|mia)",
    re.IGNORECASE,
)


def premium_parse_tsh(raw_text: str) -> ParsedTSH:
    """
    Parse la TSH dans un texte OCR de compte-rendu biologique.

    Stratégie :
      1. Chercher "tsh" + valeur (cas normaux) => high.
      2. Chercher "macro-tsh" et prendre la dernière valeur + unité avant => medium.
      3. Prendre la première valeur + unité plausible du document => low.

    Dans tous les cas, on vérifie que la valeur reste dans [0, 150] mUI/L.
    """
    if not raw_text or not raw_text.strip():
        return ParsedTSH(False, None, None, None, None, "low", "TSH_NOT_FOUND")

    norm = _normalize(raw_text)

    # 1) Cas direct "tsh ... valeur"
    m = TSH_DIRECT_RE.search(norm)
    if m:
        value = _to_float(m.group(1))
        if value is not None and 0 <= value <= 150:
            unit_raw = m.group(2) or "mui/l"
            unit = "mUI/L"  # on normalise
            # contexte local pour l'intervalle
            start = max(0, m.start())
            end = min(len(norm), m.end() + 80)
            ref_min, ref_max = _extract_ref_interval(norm[start:end])
            return ParsedTSH(True, value, unit, ref_min, ref_max, "high", None)

    # 2) Fallback via "macro-tsh" dans le texte
    idx_macro = norm.find("macro-tsh")
    if idx_macro != -1:
        before = norm[:idx_macro]
        candidates = list(VALUE_UNIT_RE.finditer(before))
        if candidates:
            last = candidates[-1]
            value = _to_float(last.group(1))
            if value is not None and 0 <= value <= 150:
                unit = "mUI/L"
                start = max(0, last.start())
                end = min(len(norm), last.end() + 80)
                ref_min, ref_max = _extract_ref_interval(norm[start:end])
                return ParsedTSH(True, value, unit, ref_min, ref_max, "medium", None)

    # 3) Fallback général : première valeur + unité plausible
    m2 = VALUE_UNIT_RE.search(norm)
    if m2:
        value = _to_float(m2.group(1))
        if value is not None and 0 <= value <= 150:
            unit = "mUI/L"
            start = max(0, m2.start())
            end = min(len(norm), m2.end() + 80)
            ref_min, ref_max = _extract_ref_interval(norm[start:end])
            return ParsedTSH(True, value, unit, ref_min, ref_max, "low", None)

    # Rien trouvé
    return ParsedTSH(False, None, None, None, None, "low", "TSH_NOT_FOUND")
