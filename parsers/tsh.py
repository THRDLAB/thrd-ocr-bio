import logging
import re
from typing import Optional, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TSHParseResult(BaseModel):
    tsh_value: float
    tsh_unit: Optional[str] = None
    ref_min: Optional[float] = None
    ref_max: Optional[float] = None
    confidence: Literal["low", "medium", "high"] = "low"


# Mots-clés autour de TSH
TSH_KEYWORDS = [
    "tsh",
    "tshus",
    "tsh ultra sensible",
    "t.s.h",
]

# Regex pour une valeur numérique (style labo)
FLOAT_RE = re.compile(r"(\d+[.,]\d+|\d+)")

# Regex pour une plage de référence "0,4 - 4,0" ou "0.4-4.0"
RANGE_RE = re.compile(
    r"(\d+[.,]\d+|\d+)\s*[-–]\s*(\d+[.,]\d+|\d+)"
)

# Quelques formes usuelles d’unités en minuscule
KNOWN_UNITS = [
    "mui/l",
    "µui/l",
    "µui/ml",
    "ui/l",
    "uiu/ml",
    "ui/ml",
    "uui/ml",
]


def parse_tsh(text: str) -> Optional[TSHParseResult]:
    """
    Parse le texte complet d'un compte-rendu pour en extraire la TSH.
    Stratégie simple :
    - On cherche des lignes contenant un mot-clé TSH.
    - On extrait la première valeur numérique sur ces lignes (ou ligne suivante si besoin).
    - On tente de trouver une unité proche.
    - On tente de trouver une plage de référence sur la même ligne ou voisine.

    Retourne TSHParseResult si trouvé, sinon None.
    """

    if not text:
        logger.info("[TSH] Empty text received, nothing to parse.")
        return None

    lines = text.splitlines()
    logger.info("[TSH] Parsing text with %d lines", len(lines))

    # On parcourt les lignes et on note l'index pour pouvoir regarder la suivante
    for idx, line in enumerate(lines):
        line_clean = line.strip()
        line_lower = line_clean.lower()

        if not line_clean:
            continue

        if not _line_contains_tsh_keyword(line_lower):
            continue

        logger.info("[TSH] Found TSH keyword on line %d: %s", idx, line_clean)

        # On construit un "bloc" : ligne courante + éventuellement la suivante
        block = line_clean
        if idx + 1 < len(lines):
            next_line = lines[idx + 1].strip()
            # Si la ligne actuelle est très courte, il est fréquent que la valeur soit sur la ligne suivante
            if len(line_clean) < 15 and next_line:
                block = f"{line_clean} {next_line}"

        result = _extract_tsh_from_block(block)

        if result is not None:
            logger.info("[TSH] TSH successfully parsed from block around line %d", idx)
            return result

    logger.info("[TSH] No TSH value found in text.")
    return None


def _normalize_for_keyword(s: str) -> str:
    """
    Supprime tout ce qui n'est pas une lettre et met en minuscule.
    Exemple :
    - "T.S.H"        -> "tsh"
    - "T S H"        -> "tsh"
    - "TSH 3ème gen" -> "tshèmegen"
    """
    return re.sub(r"[^a-z]", "", s.lower())


def _line_contains_tsh_keyword(line_lower: str) -> bool:
    """
    Détection robuste des mots-clés TSH en utilisant la version normalisée de la ligne.
    """
    norm_line = _normalize_for_keyword(line_lower)
    # On considère pour l'instant surtout "tsh" et "tshus"
    if "tshus" in norm_line:
        return True
    if "tsh" in norm_line:
        return True
    return False



def _extract_tsh_from_block(block: str) -> Optional[TSHParseResult]:
    """
    Extrait la TSH à partir d'un "bloc" de texte (ligne TSH + éventuelle ligne suivante).

    Stratégie :
    - on parcourt tous les nombres de la ligne ;
    - pour chaque nombre on regarde s'il y a une unité de TSH dans la zone qui suit,
      AVANT le prochain nombre ;
    - on choisit le premier nombre avec une unité valide ;
    - à défaut, on retombe sur le premier nombre plausible.
    """

    logger.debug("[TSH] Extracting from block: %s", block)

    matches = list(FLOAT_RE.finditer(block))
    if not matches:
        logger.debug("[TSH] No numeric value found in block.")
        return None

    candidate_result: Optional[TSHParseResult] = None

    for i, m in enumerate(matches):
        value_str = m.group(1).replace(",", ".")
        try:
            value = float(value_str)
        except ValueError:
            continue

        # Filtre grossier d’absurdités
        if value < 0 or value > 1000:
            continue

        # Position du prochain nombre (pour limiter la zone de recherche de l'unité)
        if i + 1 < len(matches):
            next_start = matches[i + 1].start()
        else:
            next_start = len(block)

        unit = _find_unit_near_value(block, m.start(), m.end(), next_start)
        ref_min, ref_max = _find_reference_range(block)
        confidence = _compute_confidence(value, unit, ref_min, ref_max)

        # Si on trouve une unité, on considère que c’est la vraie TSH
        if unit is not None:
            result = TSHParseResult(
                tsh_value=value,
                tsh_unit=unit,
                ref_min=ref_min,
                ref_max=ref_max,
                confidence=confidence,
            )
            logger.debug("[TSH] Parsed result with unit: %s", result)
            return result

        # Sinon on garde éventuellement comme candidat "au cas où"
        if candidate_result is None:
            candidate_result = TSHParseResult(
                tsh_value=value,
                tsh_unit=None,
                ref_min=ref_min,
                ref_max=ref_max,
                confidence=confidence,
            )

    if candidate_result is not None:
        logger.debug("[TSH] Parsed result without explicit unit: %s", candidate_result)
        return candidate_result

    logger.debug("[TSH] No plausible TSH value in block.")
    return None


def _find_unit_near_value(
    block: str,
    start_idx: int,
    end_idx: int,
    next_number_start: int,
) -> Optional[str]:
    """
    Cherche une unité dans la zone située après la valeur, mais avant le prochain nombre,
    avec une limite de longueur pour éviter de partir trop loin dans la ligne.
    Gère aussi les OCR foireux du type muI/I, mul/1, etc.
    """

    # Zone de recherche : entre la fin de la valeur et le prochain nombre
    max_span = end_idx + 25  # on ne va pas chercher à 3 km
    stop = min(next_number_start, len(block), max_span)
    search_zone = block[end_idx:stop]
    search_zone_lower = search_zone.lower()

    # 1. Unités écrites "proprement"
    for known in KNOWN_UNITS:
        if known in search_zone_lower:
            pattern = re.escape(known)
            m = re.search(pattern, search_zone, flags=re.IGNORECASE)
            if m:
                return m.group(0).strip()
            return known

    # 2. Regex permissive : accepte les erreurs OCR sur la dernière lettre (l <-> 1 <-> i)
    # ex : "mui/l", "mui/1", "mui/i", "mUl/l", etc.
    generic_unit_re = re.compile(
        r"[mµu]\s*u?\s*i?\s*/\s*[ml1i]",
        flags=re.IGNORECASE,
    )
    m2 = generic_unit_re.search(search_zone)
    if m2:
        return m2.group(0).strip()

    return None



def _find_reference_range(block: str) -> tuple[Optional[float], Optional[float]]:
    """
    Cherche une plage de référence dans le bloc, par ex.:
    - 0,4 - 4,0
    - N: 0.4–4.0
    """
    # On regarde d’abord la ligne brute
    m = RANGE_RE.search(block)
    if not m:
        # On peut tenter une recherche un peu plus permissive, par ex. après 'n', 'normes', 'ref'
        # mais pour le MVP on reste simple.
        return None, None

    min_str = m.group(1).replace(",", ".")
    max_str = m.group(2).replace(",", ".")

    try:
        ref_min = float(min_str)
        ref_max = float(max_str)
    except ValueError:
        return None, None

    if ref_min >= ref_max:
        return None, None

    return ref_min, ref_max


def _compute_confidence(
    value: float,
    unit: Optional[str],
    ref_min: Optional[float],
    ref_max: Optional[float],
) -> Literal["low", "medium", "high"]:
    """
    Logique simple de confiance :
    - high : valeur plausible + unité connue + plage de réf détectée.
    - medium : valeur plausible et (unité OU plage de réf).
    - low : juste une valeur plausible sans unité ni réf.
    """

    # Valeur plausible pour une TSH, on prend une large fourchette
    plausible = 0 <= value <= 100

    if plausible and unit is not None and ref_min is not None and ref_max is not None:
        return "high"

    if plausible and (unit is not None or (ref_min is not None and ref_max is not None)):
        return "medium"

    return "low"
