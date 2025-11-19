import re
from dataclasses import dataclass
from typing import Optional, List, Tuple


# --------------------------------------------------------------------
# Structures de données
# --------------------------------------------------------------------

@dataclass
class ParsedTSH:
    ok: bool
    value: Optional[float] = None
    unit: Optional[str] = None
    ref_min: Optional[float] = None
    ref_max: Optional[float] = None
    confidence: str = "low"
    error: Optional[str] = None


@dataclass
class TSHMatch:
    label: str
    value: float
    ref_min: Optional[float]
    ref_max: Optional[float]
    unit: Optional[str]
    span: Tuple[int, int]
    raw_segment: str


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text


def _to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------------------------
# REGEX solides TSH
# --------------------------------------------------------------------

TSH_LABEL = r"""
\b(?:
    (?<![a-z0-9])t\s*\.?\s*s\s*\.?\s*h(?![a-z0-9])   # formes décomposées exactes T S H
  | tsh(?:\s*us)?                                    # TSH, TSHus
  | tsh\s*ultra\s*sensible                           # TSH ultra sensible
  | tsh\s*3(?:e|ème)\s*gen                            # TSH 3e/3ème GEN
  | thyr[eé]ostimuline                                # thyréostimuline
  | thyrotrop(?:ine|e)                                # thyrotropine
)\b
"""

NUMBER = r"[+-]?\d+(?:[.,]\d+)?"

TSH_UNIT = r"(?:m ?UI/?L|µ ?UI/?L|u ?UI/?mL|mIU/?L|mU/?L|pUI/?mL|UI/?L|mUI|µUI|uUI)?"


TSH_BLOCK_REGEX = re.compile(
    rf"""
    (?P<label>{TSH_LABEL})

    [^0-9\n]{{0,60}}                      # bruit

    (?P<value>{NUMBER})

    \s*
    (?P<unit>{TSH_UNIT})

    (?:                                  # Plage optionnelle
        [^\d\n]{{0,20}}
        (?P<ref_min>{NUMBER})
        \s*
        (?:-|–|—|~|à|a|>|<|≤|>=|to|et|&)
        \s*
        (?P<ref_max>{NUMBER})
    )?
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


# --------------------------------------------------------------------
# Extraction candidats
# --------------------------------------------------------------------

def _find_tsh_candidates(raw_text: str) -> List[TSHMatch]:
    text = _normalize_text(raw_text)
    candidates: List[TSHMatch] = []

    for m in TSH_BLOCK_REGEX.finditer(text):
        label = m.group("label") or ""
        value = _to_float(m.group("value"))
        ref_min = _to_float(m.group("ref_min"))
        ref_max = _to_float(m.group("ref_max"))
        unit = (m.group("unit") or "").strip() or None

        if value is None:
            continue

        start, end = m.span()
        segment = text[max(0, start - 40): min(len(text), end + 40)]

        candidates.append(
            TSHMatch(
                label=label,
                value=value,
                ref_min=ref_min,
                ref_max=ref_max,
                unit=unit,
                span=(start, end),
                raw_segment=segment,
            )
        )

    return candidates


# --------------------------------------------------------------------
# Scoring candidats TSH : cœur du fix
# --------------------------------------------------------------------

def _score_candidate(c: TSHMatch) -> tuple:
    """
    Score triable :
    1. Valide la plage (ou rejette)
    2. Label solide (TSH > thyréostimuline)
    3. Cohérence valeur/plage
    4. Valeur physiologiquement correcte
    """

    # Sécurité physiologique TSH
    if c.value < 0.001 or c.value > 100:
        return (999, 999, 999, 999)

    # Vérification plage si existante
    if c.ref_min is not None and c.ref_max is not None:

        # plages impossibles → rejet
        if c.ref_min <= 0 or c.ref_max <= 0:
            return (999, 999, 999, 999)
        if c.ref_min > c.ref_max:
            return (999, 999, 999, 999)
        if c.ref_max < 0.05:  # trop petit = OCR raté
            return (999, 999, 999, 999)

        # cohérence valeur/plage
        if not (c.ref_min - 0.5 <= c.value <= c.ref_max + 0.5):
            range_penalty = 1
        else:
            range_penalty = 0
    else:
        # pas de plage → moins bon
        range_penalty = 2

    # Label strength
    label = c.label.lower()
    if "tsh" in label:
        label_penalty = 0
    elif "thyr" in label:
        label_penalty = 1
    else:
        label_penalty = 2

    # On trie selon :
    # (a une plage?, label, cohérence, position)
    has_range = 0 if (c.ref_min is not None and c.ref_max is not None) else 1

    return (has_range, label_penalty, range_penalty)


def pick_best_tsh_match(candidates: List[TSHMatch]) -> Optional[TSHMatch]:
    if not candidates:
        return None

    candidates_sorted = sorted(
        candidates,
        key=lambda c: (*_score_candidate(c), c.span[0])
    )
    return candidates_sorted[0]


# --------------------------------------------------------------------
# Parser final
# --------------------------------------------------------------------

def premium_parse_tsh(raw_text: str, boxes) -> ParsedTSH:
    text = raw_text or ""
    candidates = _find_tsh_candidates(text)

    if not candidates:
        return ParsedTSH(ok=False, error="TSH_NOT_FOUND")

    best = pick_best_tsh_match(candidates)
    if not best:
        return ParsedTSH(ok=False, error="TSH_NOT_FOUND")

    # Estimation de confiance
    if best.ref_min is not None and best.ref_max is not None:
        if best.ref_min - 0.5 <= best.value <= best.ref_max + 0.5:
            confidence = "high"
        else:
            confidence = "medium"
    else:
        confidence = "low"

    return ParsedTSH(
        ok=True,
        value=best.value,
        unit=best.unit,
        ref_min=best.ref_min,
        ref_max=best.ref_max,
        confidence=confidence,
        error=None,
    )
