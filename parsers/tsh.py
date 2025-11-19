import re
from dataclasses import dataclass
from typing import Optional, List, Tuple


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


def _normalize_text(text: str) -> str:
    """
    Normalisation légère du texte OCR :
    - homogénéise les retours à la ligne
    - réduit les espaces multiples
    """
    if not text:
        return ""
    text = text.replace("\r", "\n")
    # Réduire les suites d'espaces / tabs
    text = re.sub(r"[ \t\f\v]+", " ", text)
    # Réduire les multiples sauts de ligne
    text = re.sub(r"\n+", "\n", text)
    return text


def _to_float(s: Optional[str]) -> Optional[float]:
    """
    Convertit une chaîne avec virgule ou point en float Python.
    Renvoie None si non convertible.
    """
    if not s:
        return None
    s = s.replace(" ", "").replace("\u00a0", "")  # espaces normaux + insécables
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Motifs regex pour le bloc TSH complet
# ---------------------------------------------------------------------------

# Label TSH : gère plusieurs variantes
TSH_LABEL = r"""
\b(?:                               # frontière de mot
    t\s*\.?\s*s\s*\.?\s*h           # formes décomposées "t s h"
  | tsh(?:\s*us)?                   # TSH, TSHus
  | tsh\s*ultra\s*sensible          # TSH ultra sensible
  | tsh\s*3(?:e|ème)\s*gen          # TSH 3e/3ème GEN
  | thyr[eé]ostimuline              # thyréostimuline / thyreostimuline
  | thyrotrop(?:ine|e)              # thyrotropine / thyrotrope
)\b
"""

# Nombre avec virgule ou point
NUMBER = r"[+-]?\d+(?:[.,]\d+)?"

# Unités classiques de TSH (on reste permissif)
TSH_UNIT = r"(?:m ?UI/?L|µ ?UI/?L|u ?UI/?mL|mIU/?L|mU/?L|pUI/?mL|UI/?L|mUI|µUI|uUI)?"

# Regex principale pour un bloc TSH : label + valeur + (unit) + (plage ref)
TSH_BLOCK_REGEX = re.compile(
    rf"""
    (?P<label>{TSH_LABEL})               # 1. Label TSH

    [^0-9\n]{{0,60}}                      # un peu de bruit (points, parenthèses, etc.)

    (?P<value>{NUMBER})                  # 2. Valeur TSH

    \s*
    (?P<unit>{TSH_UNIT})                 # 3. Unité (optionnelle)

    # 4. Plage de référence (optionnelle)
    (?:                                  # groupe non capturant pour la plage
        [^\d\n]{{0,20}}                  # petit bruit (ex: ' ', ':', etc.)
        (?P<ref_min>{NUMBER})            # borne basse
        \s*
        (?:-|–|—|~|à|a|>|<|≤|>=|to|et|&) # séparateur très permissif
        \s*
        (?P<ref_max>{NUMBER})            # borne haute
    )?
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


def _find_tsh_candidates(raw_text: str) -> List[TSHMatch]:
    """
    Extrait tous les blocs candidats TSH dans le texte OCR.
    """
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
        # Petit contexte autour du match pour debug éventuel
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


def _score_candidate(c: TSHMatch) -> tuple:
    """
    Calcule un score triable pour ordonner les candidats TSH.
    Plus le score est petit, plus le candidat est "bon".
    Critères :
      1. Avoir une plage de référence complète (ref_min et ref_max)
      2. Label contenant explicitement 'TSH'
      3. Valeur cohérente avec la plage (si dispo)
    """
    # 1. Plage complète ou non
    has_range = 0 if (c.ref_min is not None and c.ref_max is not None) else 1

    # 2. Qualité du label
    label = c.label.lower()
    if "tsh" in label:
        label_penalty = 0
    elif "thyr" in label:
        label_penalty = 1
    else:
        label_penalty = 2

    # 3. Cohérence valeur/plage (si on a la plage)
    range_penalty = 0
    if c.ref_min is not None and c.ref_max is not None:
        # petite marge de tolérance à cause des approximations OCR
        low = c.ref_min - 0.5
        high = c.ref_max + 0.5
        if not (low <= c.value <= high):
            range_penalty = 1

    # On renvoie un tuple : Python triera dans cet ordre
    return has_range, label_penalty, range_penalty


def pick_best_tsh_match(candidates: List[TSHMatch]) -> Optional[TSHMatch]:
    """
    Sélectionne le "meilleur" bloc TSH parmi tous les candidats.
    """
    if not candidates:
        return None

    candidates_sorted = sorted(
        candidates,
        key=lambda c: (*_score_candidate(c), c.span[0]),
    )
    return candidates_sorted[0]


def premium_parse_tsh(raw_text: str, boxes) -> ParsedTSH:
    """
    Parser principal TSH.
    - Analyse le texte OCR brut avec une regex robuste
    - Sélectionne le meilleur candidat via des heuristiques simples
    - Renvoie un objet ParsedTSH avec ok/value/unit/ref_min/ref_max/confidence/error

    Paramètres :
      raw_text : texte brut renvoyé par l'OCR
      boxes    : boxes OCR (non utilisées pour l'instant mais gardées pour compat)
    """
    text = raw_text or ""
    candidates = _find_tsh_candidates(text)

    if not candidates:
        return ParsedTSH(ok=False, error="TSH_NOT_FOUND")

    best = pick_best_tsh_match(candidates)
    if not best:
        return ParsedTSH(ok=False, error="TSH_NOT_FOUND")

    # Estimation très simple de la confiance
    confidence = "low"
    if best.ref_min is not None and best.ref_max is not None:
        # Valeur dans la plage => confiance élevée
        if best.ref_min - 0.5 <= best.value <= best.ref_max + 0.5:
            confidence = "high"
        else:
            confidence = "medium"
    elif best.ref_min is not None or best.ref_max is not None:
        confidence = "medium"

    return ParsedTSH(
        ok=True,
        value=best.value,
        unit=best.unit,
        ref_min=best.ref_min,
        ref_max=best.ref_max,
        confidence=confidence,
        error=None,
    )
