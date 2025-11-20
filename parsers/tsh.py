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
    unit: Optional[str]
    ref_min: Optional[float]
    ref_max: Optional[float]
    span: Tuple[int, int]
    raw_line: str
    raw_snippet: str


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Nettoyage léger du texte OCR."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text


def _to_float(s: str) -> Optional[float]:
    """Convertit une chaîne en float (gère la virgule)."""
    if not s:
        return None
    s = s.replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _adjust_ref_value(raw: str) -> Optional[float]:
    """
    Corrige des références typiques OCR :
    - "0,40" -> 0.40
    - "027"  -> 0.27
    - "494"  -> 4.94
    - "4000" -> 4.0
    """
    if not raw:
        return None

    # Si déjà un séparateur décimal → on garde tel quel
    if "," in raw or "." in raw:
        return _to_float(raw)

    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None

    try:
        val_int = int(digits)
    except ValueError:
        return None

    # Heuristique : selon le nombre de chiffres, on re-scale
    n = len(digits)

    # 4 chiffres ou plus, souvent "4000" = 4.000
    if n >= 4:
        return val_int / 1000.0

    # 3 chiffres : souvent "027" = 0.27, "494" = 4.94
    if n == 3:
        return val_int / 100.0

    # 1 ou 2 chiffres, on laisse tel quel
    return float(val_int)


# --------------------------------------------------------------------
# Regex pour détecter la ligne TSH + nombres
# --------------------------------------------------------------------

# Label TSH : couvre les variantes vues dans tes docs
TSH_LABEL_PATTERN = (
    r"(?:"
    r"TSHus\s*3(?:e|ème)\s*GEN"                  # TSHus 3ème GEN
    r"|TSH\s*3(?:e|ème)\s*g[ée]n[ée]?ration?"    # TSH 3ème génération
    r"|TSH\s*ultra\s*sensible"                  # TSH ultra sensible
    r"|TSHus\b"                                 # TSHus
    r"|TSH\b"                                   # TSH simple
    r"|thyr[eé]ostimuline"                      # thyréostimuline
    r"|thyrotropine"                            # thyrotropine
    r")"
)

LABEL_RE = re.compile(TSH_LABEL_PATTERN, re.IGNORECASE)

# Nombre avec . ou , (style FR/US)
NUM_RE = re.compile(r"[+-]?\d+(?:[.,]\d+)?")

# Plage de références : x - y, x à y, x & y, etc.
RANGE_RE = re.compile(
    r"(?P<min>[+-]?\d+(?:[.,]\d+)?)\s*"
    r"(?:-|–|—|~|à|a|to|&)\s*"
    r"(?P<max>[+-]?\d+(?:[.,]\d+)?)"
)


# --------------------------------------------------------------------
# Extraction candidate sur une ligne
# --------------------------------------------------------------------

def _extract_tsh_from_line(line: str) -> Optional[TSHMatch]:
    """
    Cherche une ligne contenant un label TSH, puis :
      - première valeur numérique après le label = TSH
      - plage éventuelle x - y après cette valeur = ref_min/ref_max
    """
    m = LABEL_RE.search(line)
    if not m:
        return None

    label = m.group(0)
    snippet = line[m.end():]  # tout ce qu'il y a après le label

    # 1) TSH value : premier nombre après le label
    nums = list(NUM_RE.finditer(snippet))
    if not nums:
        return None

    tsh_num = nums[0]
    tsh_value = _to_float(tsh_num.group())
    if tsh_value is None:
        return None

    # 2) Unité éventuelle : très simple, on regarde juste autour
    unit = None
    unit_window = snippet[tsh_num.end():tsh_num.end() + 20]
    # On prend "UI/L", "mUI/L", "µUI/L", "uUI/mL", etc. si présents
    unit_match = re.search(
        r"(m ?UI/?L|µ ?UI/?L|u ?UI/?mL|mIU/?L|mU/?L|pUI/?mL|UI/?L|mUI|µUI|uUI)",
        unit_window,
        re.IGNORECASE,
    )
    if unit_match:
        unit = unit_match.group(0)

    # 3) Plage de référence : après la valeur TSH
    after_pos = tsh_num.end()
    range_match = RANGE_RE.search(snippet, after_pos)
    ref_min = ref_max = None
    if range_match:
        ref_min = _adjust_ref_value(range_match.group("min"))
        ref_max = _adjust_ref_value(range_match.group("max"))

    return TSHMatch(
        label=label,
        value=tsh_value,
        unit=unit,
        ref_min=ref_min,
        ref_max=ref_max,
        span=m.span(),
        raw_line=line,
        raw_snippet=snippet,
    )


def _find_tsh_candidates(raw_text: str) -> List[TSHMatch]:
    text = _normalize_text(raw_text)
    lines = text.split("\n")
    candidates: List[TSHMatch] = []

    for line in lines:
        if "TSH" not in line and "tsh" not in line and "Thyr" not in line and "thyr" not in line:
            # petit filtre rapide pour éviter des regex sur toutes les lignes
            continue

        cand = _extract_tsh_from_line(line)
        if cand:
            candidates.append(cand)

    return candidates


# --------------------------------------------------------------------
# Sélection du meilleur candidat
# --------------------------------------------------------------------

def _score_candidate(c: TSHMatch) -> tuple:
    """
    Score simple :
      1. A une plage de référence complète ou non
      2. Label "fort" (TSH, TSHus...) vs. plus générique
      3. Utilisé pour trier, pas pour rejeter.
    """
    has_range = 0 if (c.ref_min is not None and c.ref_max is not None) else 1

    label = c.label.lower()
    if "tsh" in label:
        label_penalty = 0
    elif "thyr" in label:
        label_penalty = 1
    else:
        label_penalty = 2

    return (has_range, label_penalty)


def _pick_best_candidate(candidates: List[TSHMatch]) -> Optional[TSHMatch]:
    if not candidates:
        return None
    # On trie par : a une plage ? / type de label / position dans le texte
    return sorted(
        candidates,
        key=lambda c: (*_score_candidate(c), c.span[0]),
    )[0]


# --------------------------------------------------------------------
# Parser principal
# --------------------------------------------------------------------

def premium_parse_tsh(raw_text: str, boxes) -> ParsedTSH:
    """
    Parser TSH utilisé partout :

      - prend raw_text (issu de n'importe quel niveau OCR : light, premium, optimum)
      - repère la ligne contenant TSH
      - extrait valeur + unité + plage si possible
    """
    text = raw_text or ""
    candidates = _find_tsh_candidates(text)

    if not candidates:
        return ParsedTSH(ok=False, error="TSH_NOT_FOUND")

    best = _pick_best_candidate(candidates)
    if not best:
        return ParsedTSH(ok=False, error="TSH_NOT_FOUND")

    # Estimation très simple de la confiance
    if best.ref_min is not None and best.ref_max is not None:
        confidence = "high"
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
