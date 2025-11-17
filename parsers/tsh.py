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
    Ex : "0.27 a 4.20", "0.27 - 4.20", "0.27 4 4.20".
    """
    if not context:
        return None, None

    ctx = context.replace(",", ".")
    m = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\D+([0-9]+(?:\.[0-9]+)?)",
        ctx,
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


# ---------- Regex principales ----------

# 1) Cas standard : "tsh ... 7,34 mUI/l"
TSH_DIRECT_RE = re.compile(
    r"tsh[^0-9]{0,30}([0-9]+(?:[.,][0-9]+)?)\s*"
    r"(m[uµ]i/?l|mui/?l|µui/?l|ui/ml|ui/?l|m?ui/l)?",
    re.IGNORECASE,
)

# 2) Valeur + unité générique (pour fallback et macro-tsh)
VALUE_UNIT_RE = re.compile(
    r"([0-9]+(?:[.,][0-9]+)?)\s*"
    r"(m[uµ]i/?l|mui/?l|µui/?l|ui/ml|ui/?l|mia)",
    re.IGNORECASE,
)


def _confidence(level: str) -> float:
    """Map 'high'/'medium'/'low' -> float."""
    if level == "high":
        return 0.9
    if level == "medium":
        return 0.6
    if level == "low":
        return 0.3
    return 0.0


def premium_parse_tsh(raw_text: str) -> ParsedTSH:
    """
    Parse la TSH dans un texte OCR de compte-rendu biologique.

    Stratégie :
      1. TSH direct (cas normal) -> confiance haute.
      2. Texte contenant 'macro-tsh' : on prend la dernière valeur + unité avant -> moyenne.
      3. Fallback : première valeur + unité plausible -> faible.

    Toujours :
      - on normalise le texte,
      - on filtre les valeurs hors [0, 150] mUI/L,
      - on essaie d'extraire un intervalle de référence local.
    """
    if not raw_text or not raw_text.strip():
        return ParsedTSH(False, None, None, None, None, 0.0, "TSH_NOT_FOUND")

    norm = _normalize(raw_text)

    # ---------- 1) Cas direct "tsh ... valeur" ----------
    m = TSH_DIRECT_RE.search(norm)
    if m:
        value = _to_float(m.group(1))
        if value is not None and 0 <= value <= 150:
            unit = "mUI/L"  # on normalise l'unité
            start = max(0, m.start())
            end = min(len(norm), m.end() + 80)
            ref_min, ref_max = _extract_ref_interval(norm[start:end])
            return ParsedTSH(
                ok=True,
                value=value,
                unit=unit,
                ref_min=ref_min,
                ref_max=ref_max,
                confidence=_confidence("high"),
                error=None,
            )

    # ---------- 2) Cas "macro-tsh" : on regarde avant ----------
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
                return ParsedTSH(
                    ok=True,
                    value=value,
                    unit=unit,
                    ref_min=ref_min,
                    ref_max=ref_max,
                    confidence=_confidence("medium"),
                    error=None,
                )

    # ---------- 3) Fallback général : 1ère valeur + unité plausible ----------
    m2 = VALUE_UNIT_RE.search(norm)
    if m2:
        value = _to_float(m2.group(1))
        if value is not None and 0 <= value <= 150:
            unit = "mUI/L"
            start = max(0, m2.start())
            end = min(len(norm), m2.end() + 80)
            ref_min, ref_max = _extract_ref_interval(norm[start:end])
            return ParsedTSH(
                ok=True,
                value=value,
                unit=unit,
                ref_min=ref_min,
                ref_max=ref_max,
                confidence=_confidence("low"),
                error=None,
            )

    # ---------- Rien trouvé ----------
    return ParsedTSH(False, None, None, None, None, 0.0, "TSH_NOT_FOUND")
