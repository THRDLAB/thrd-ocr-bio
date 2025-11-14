import re
from typing import Optional, List
from dataclasses import dataclass


# =============================================================================
# Dataclass structurée : Résultat TSH premium
# =============================================================================

@dataclass
class ParsedTSH:
    ok: bool
    value: Optional[float]
    unit: Optional[str]
    ref_min: Optional[float]
    ref_max: Optional[float]
    confidence: str        # "high", "medium", "low"
    error: Optional[str]   # "TSH_NOT_FOUND", "UNIT_NOT_FOUND", etc.


# =============================================================================
# Définition des unités reconnues
# =============================================================================

UNIT_PATTERNS = [
    r"µ?\s*u?i\s*/\s*l",     # µUI/L, uUI/L, UI/L, µUI / L
    r"µ?\s*u?i\s*/\s*ml",    # µUI/mL, mUI/mL
    r"mui\s*/\s*l",          # mUI/L
    r"mui\s*/\s*ml",         # mUI/mL
]

UNIT_CLEAN = {
    "mui/l": "mUI/L",
    "mui/ml": "mUI/mL",
    "ui/l": "UI/L",
    "uui/l": "uUI/L",
    "µui/l": "µUI/L",
    "µui/ml": "µUI/mL",
}


# =============================================================================
# Extraction générale
# =============================================================================

def _clean_text(s: str) -> str:
    return s.lower().replace(",", ".").strip()


def _extract_numbers(s: str) -> List[float]:
    nums = re.findall(r"\d+\.\d+|\d+,\d+|\d+", s)
    result = []
    for n in nums:
        try:
            n = n.replace(",", ".")
            result.append(float(n))
        except:
            continue
    return result


def _find_unit_in_line(line: str) -> Optional[str]:
    l = _clean_text(line)
    for pattern in UNIT_PATTERNS:
        if re.search(pattern, l):
            raw = re.search(pattern, l).group(0)
            cleaned = raw.replace(" ", "").replace("µ", "µ")
            cleaned = cleaned.lower()
            return UNIT_CLEAN.get(cleaned, cleaned)
    return None


# =============================================================================
# Organisation des lignes à partir des boxes
# =============================================================================

def _group_boxes_by_line(boxes: List[dict], tolerance: int = 12) -> List[List[dict]]:
    """
    Regroupe les boxes Tesseract en lignes horizontales.
    """
    lines = []
    for b in boxes:
        y = b["top"]
        placed = False

        for line in lines:
            # Si le mot est dans +/- tolerance px d'une ligne existante → même ligne
            if abs(line[0]["top"] - y) <= tolerance:
                line.append(b)
                placed = True
                break

        if not placed:
            lines.append([b])

    # Trie chaque ligne par position horizontale
    for line in lines:
        line.sort(key=lambda x: x["left"])

    # Tri vertical des lignes
    lines.sort(key=lambda x: x[0]["top"])
    return lines


def _merge_line_text(line: List[dict]) -> str:
    """Concatène les mots d'une même ligne."""
    return " ".join([w["text"] for w in line])


# =============================================================================
# Analyse des colonnes
# =============================================================================

def _compute_column_clusters(lines: List[List[dict]]) -> List[List[int]]:
    """
    Récupère les positions horizontales (left) de tous les nombres.
    On fait ensuite un clustering de colonnes (simplifié).
    """
    xs = []
    for line in lines:
        for w in line:
            txt = w["text"]
            if re.match(r"[0-9]", txt.replace(",", ".")):
                xs.append(w["left"])

    if not xs:
        return []

    xs = sorted(xs)

    clusters = [[xs[0]]]
    for x in xs[1:]:
        # si colonne proche → même cluster
        if abs(x - clusters[-1][-1]) < 40:  # 40px = largeur N° d'une même colonne
            clusters[-1].append(x)
        else:
            clusters.append([x])

    # Centre de chaque colonne
    col_centers = [sum(c) / len(c) for c in clusters]
    return sorted(col_centers)


def _extract_values_from_line(line: List[dict], col_centers: List[int]) -> List[float]:
    """
    Extrait les valeurs présentes dans chaque colonne.
    Renvoie un tableau par colonne.
    """
    text_line = _merge_line_text(line)
    raw_numbers = _extract_numbers(text_line)
    if not raw_numbers:
        return []

    # Trouver pour chaque nombre sa colonne
    values = []
    for w in line:
        try:
            val = float(w["text"].replace(",", "."))
        except:
            continue

        center = w["left"]
        distances = [abs(center - c) for c in col_centers]
        col_id = distances.index(min(distances))
        values.append((col_id, val))

    return values


# =============================================================================
# Fonction principale premium
# =============================================================================

def premium_parse_tsh(raw_text: str, ocr_boxes: List[dict]) -> Optional[ParsedTSH]:
    """
    Parsing premium :
    - Trouve la ligne contenant TSH
    - Analyse les colonnes (valeur actuelle / antériorité)
    - Trouve unité + valeurs de référence
    """
    text_lower = raw_text.lower()

    # 1) Vérifier que TSH apparaît
    if "tsh" not in text_lower:
        return ParsedTSH(
            ok=False,
            value=None,
            unit=None,
            ref_min=None,
            ref_max=None,
            confidence="low",
            error="TSH_NOT_FOUND",
        )

    # 2) Reconstituer les lignes avec positions
    lines = _group_boxes_by_line(ocr_boxes)

    # 3) Trouver la ligne où il y a le mot "TSH"
    tsh_lines = []
    for line in lines:
        merged = _merge_line_text(line).lower()
        if "tsh" in merged:
            tsh_lines.append(line)

    if not tsh_lines:
        return ParsedTSH(ok=False, value=None, unit=None, ref_min=None,
                         ref_max=None, confidence="low", error="TSH_NOT_FOUND")

    line = tsh_lines[0]  # principale

    # 4) Trouver toutes les colonnes (positions horizontales moyennes)
    col_centers = _compute_column_clusters(lines)

    # 5) Extraire les valeurs numériques de la ligne TSH
    values_by_col = _extract_values_from_line(line, col_centers)

    if not values_by_col:
        return ParsedTSH(ok=False, value=None, unit=None, ref_min=None,
                         ref_max=None, confidence="low", error="TSH_NOT_FOUND")

    # 6) Interprétation métier :
    # colonne la plus à gauche contenant un nombre = valeur actuelle
    col_ids = [col for (col, _) in values_by_col]
    first_col = min(col_ids)

    # valeurs de la première colonne
    main_values = [v for (col, v) in values_by_col if col == first_col]

    if not main_values:
        return ParsedTSH(ok=False, value=None, unit=None, ref_min=None,
                         ref_max=None, confidence="low", error="TSH_NOT_FOUND")

    # 7) Sélection de la valeur principale (souvent la première)
    tsh_value = main_values[0]

    # 8) Unité = chercher dans la ligne (ou ligne suivante)
    unit = _find_unit_in_line(_merge_line_text(line))
    if not unit:
        # chercher dans texte global
        unit = _find_unit_in_line(raw_text)

    # 9) Valeurs de référence
    # dans la même ligne
    refs = _extract_numbers(_merge_line_text(line))
    ref_min, ref_max = None, None
    if len(refs) >= 3:
        # exemple : [0.154, 0.4, 4.0] → TSH, min, max
        ref_min, ref_max = refs[-2], refs[-1]

    # 10) Déterminer la confiance
    confidence = "high"

    # si pas d'unité ou pas de bornes → medium
    if not unit:
        confidence = "medium"
    if ref_min is None or ref_max is None:
        confidence = "medium"

    # plausibilité médicale
    if tsh_value < 0 or tsh_value > 200:
        return ParsedTSH(ok=False, value=None, unit=None,
                         ref_min=None, ref_max=None, confidence="low",
                         error="TSH_NOT_FOUND")

    return ParsedTSH(
        ok=True,
        value=tsh_value,
        unit=unit,
        ref_min=ref_min,
        ref_max=ref_max,
        confidence=confidence,
        error=None,
    )
