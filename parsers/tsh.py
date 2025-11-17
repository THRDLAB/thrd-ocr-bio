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
    confidence: Optional[str]   # "high" | "medium" | "low"
    error: Optional[str]


def _normalize(text: str) -> str:
    """Minuscules + accents retirés + espaces normalisés."""
    if not text:
        return ""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"\s+", " ", text)
    return text


def _to_float(s: str) -> Optional[float]:
    try:
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def _extract_ref_interval(context: str) -> tuple[Optional[float], Optional[float]]:
    """Cherche un intervalle type '0.27 ... 4.20' dans un bout de texte."""
    if not context:
        return None, None
    ctx = context.replace(",", ".")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\D+([0-9]+(?:\.[0-9]+)?)", ctx)
    if not m:
        return None, None
    a = _to_float(m.group(1))
    b = _to_float(m.group(2))
    if a is None or b is None:
        return None, None
    if a > b:
        a, b = b, a
    return a, b


# 1) Cas standard : "tsh ... 7,34 mUI/l"
TSH_DIRECT_RE = re.compile(
    r"tsh[^0-9]{0,30}([0-9]+(?:[.,][0-9]+)?)\s*"
    r"(m[uµ]i/?l|mui/?l|µui/?l|ui/ml|ui/?l|m?ui/l)?",
    re.IGNORECASE,
)

# 2) Valeur + unité générique (fallback + macro-TSH)
VALUE_UNIT_RE = re.compile(
    r"([0-9]+(?:[.,][0-9]+)?)\s*"
    r"(m[uµ]i/?l|mui/?l|µui/?l|ui/ml|ui/?l|mia)",
    re.IGNORECASE,
)


def _conf(level: str) -> str:
    if level in {"high", "medium", "low"}:
        return level
    return "low"


def premium_parse_tsh(raw_text: str, boxes=None) -> ParsedTSH:
    """
    Parser TSH tolérant les OCR foireux.

    Stratégie :
      1. TSH direct → "high"
      2. 'macro-tsh' → valeur + unité juste avant → "medium"
      3. Première valeur + unité plausible → "low"
    """
    if not raw_text or not raw_text.strip():
        return ParsedTSH(False, None, None, None, None, None, "TSH_NOT_FOUND")

    norm = _normalize(raw_text)

    # ---------- 1) Cas direct TSH ----------
    m = TSH_DIRECT_RE.search(norm)
    if m:
        value = _to_float(m.group(1))
        if value is not None and 0 <= value <= 150:
            unit = "mUI/L"
            start = max(0, m.start())
            end = min(len(norm), m.end() + 80)
            ref_min, ref_max = _extract_ref_interval(norm[start:end])
            return ParsedTSH(True, value, unit, ref_min, ref_max, _conf("high"), None)

    # ---------- 2) Cas "macro-tsh" ----------
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
                return ParsedTSH(True, value, unit, ref_min, ref_max, _conf("medium"), None)

    # ---------- 3) Fallback général ----------
    m2 = VALUE_UNIT_RE.search(norm)
    if m2:
        value = _to_float(m2.group(1))
        if value is not None and 0 <= value <= 150:
            unit = "mUI/L"
            start = max(0, m2.start())
            end = min(len(norm), m2.end() + 80)
            ref_min, ref_max = _extract_ref_interval(norm[start:end])
            return ParsedTSH(True, value, unit, ref_min, ref_max, _conf("low"), None)

    # ---------- Rien trouvé ----------
    return ParsedTSH(False, None, None, None, None, None, "TSH_NOT_FOUND")
