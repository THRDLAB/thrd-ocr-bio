import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ParsedTSH:
    ok: bool
    value: Optional[float]
    unit: Optional[str]
    ref_min: Optional[float]
    ref_max: Optional[float]
    confidence: str
    error: Optional[str]


# -----------------------------------------------------------------------------
# Helpers généraux
# -----------------------------------------------------------------------------

def _clean(s: str) -> str:
    return s.lower().replace(",", ".").strip()


def _extract_numbers(s: str) -> List[float]:
    nums = re.findall(r"\d+\.\d+|\d+,\d+|\d+", s)
    out = []
    for n in nums:
        try:
            out.append(float(n.replace(",", ".")))
        except:
            pass
    return out


# -----------------------------------------------------------------------------
# Détection TSH (tolérante : TSH, T.S.H, T S H, thyreostimuline…)
# -----------------------------------------------------------------------------

# - t\s*\.?\s*s\s*\.?\s*h  => tsh / t.s.h / t s h
# - thyreo?stimuline      => thyreostimuline / thyreostimuline
TSH_REGEX = re.compile(r"(t\s*\.?\s*s\s*\.?\s*h)|thyreo?stimuline", re.IGNORECASE)


def _contains_tsh(text: str) -> bool:
    return bool(TSH_REGEX.search(text))


# -----------------------------------------------------------------------------
# Unités
# -----------------------------------------------------------------------------

UNIT_PATTERNS = [
    r"µ?\s*u?i\s*/\s*l",
    r"µ?\s*u?i\s*/\s*ml",
    r"mui\s*/\s*l",
    r"mui\s*/\s*ml",
]

UNIT_CLEAN = {
    "mui/l": "mUI/L",
    "mui/ml": "mUI/mL",
    "ui/l": "UI/L",
    "uui/l": "uUI/L",
    "µui/l": "µUI/L",
    "µui/ml": "µUI/mL",
}


def _find_unit(line: str) -> Optional[str]:
    line = _clean(line)
    for p in UNIT_PATTERNS:
        m = re.search(p, line)
        if m:
            raw = m.group(0).replace(" ", "").lower()
            return UNIT_CLEAN.get(raw, raw)
    return None


# -----------------------------------------------------------------------------
# Lignes à partir des boxes
# -----------------------------------------------------------------------------

def _group_lines(boxes: List[dict], tol: int = 12):
    lines = []
    for b in boxes:
        y = b["top"]
        placed = False
        for line in lines:
            if abs(line[0]["top"] - y) <= tol:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    for line in lines:
        line.sort(key=lambda w: w["left"])

    lines.sort(key=lambda w: w[0]["top"])
    return lines


def _merge_line(line: List[dict]):
    return " ".join([w["text"] for w in line])


def _find_tsh_line(lines):
    candidates = []
    for line in lines:
        txt = _merge_line(line)
        if _contains_tsh(txt):
            candidates.append(line)
    return candidates[0] if candidates else None


# -----------------------------------------------------------------------------
# Colonnes et valeurs
# -----------------------------------------------------------------------------

def _cluster_columns(lines):
    xs = []
    for line in lines:
        for w in line:
            t = w["text"]
            if re.match(r"[0-9]", t.replace(",", ".")):
                xs.append(w["left"])
    if not xs:
        return []

    xs = sorted(xs)
    clusters = [[xs[0]]]
    for x in xs[1:]:
        if abs(x - clusters[-1][-1]) < 40:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    centers = [sum(c) / len(c) for c in clusters]
    return sorted(centers)


def _extract_column_values(line, col_centers):
    out = []
    for w in line:
        txt = w["text"].replace(",", ".")
        try:
            val = float(txt)
        except:
            continue

        xc = w["left"]
        distances = [abs(xc - c) for c in col_centers]
        col_id = distances.index(min(distances))
        out.append((col_id, val))

    return out


# -----------------------------------------------------------------------------
# Intervalles de référence (min-max)
# -----------------------------------------------------------------------------

INTERVAL_REGEX = re.compile(
    r"(\d+[\.,]?\d*)\s*(?:-|à|;)\s*(\d+[\.,]?\d*)"
)


def _extract_reference_interval(text: str):
    matches = INTERVAL_REGEX.findall(text)
    if not matches:
        return None, None

    intervals = []
    for a, b in matches:
        try:
            a = float(a.replace(",", "."))
            b = float(b.replace(",", "."))
            # bornes plausibles pour TSH
            if 0 < a < b < 50:
                intervals.append((a, b))
        except:
            pass

    if not intervals:
        return None, None

    # On prend l'intervalle avec la plus petite borne (souvent la plage 0.4–4.0)
    intervals.sort(key=lambda x: x[0])
    return intervals[0]


# -----------------------------------------------------------------------------
# Fonction principale premium
# -----------------------------------------------------------------------------

def premium_parse_tsh(raw_text: str, ocr_boxes: List[dict]) -> Optional[ParsedTSH]:
    # 0. Sanity-check TSH dans le texte global (tolérant)
    if not _contains_tsh(raw_text):
        return ParsedTSH(False, None, None, None, None, "low", "TSH_NOT_FOUND")

    # 1. Reconstituer les lignes
    lines = _group_lines(ocr_boxes)
    tsh_line = _find_tsh_line(lines)

    if not tsh_line:
        return ParsedTSH(False, None, None, None, None, "low", "TSH_NOT_FOUND")

    # 2. Colonnes
    col_centers = _cluster_columns(lines)
    col_vals = _extract_column_values(tsh_line, col_centers)

    if not col_vals:
        return ParsedTSH(False, None, None, None, None, "low", "TSH_NOT_FOUND")

    # 3. Première colonne numérique = TSH actuelle
    first_col = min(col for col, _ in col_vals)
    main_vals = [v for col, v in col_vals if col == first_col]

    if not main_vals:
        return ParsedTSH(False, None, None, None, None, "low", "TSH_NOT_FOUND")

    tsh_value = main_vals[0]

    # 4. Unité : d'abord sur la ligne TSH, puis fallback texte global
    line_text = _merge_line(tsh_line)
    unit = _find_unit(line_text) or _find_unit(raw_text)

    # 5. Intervalles : d'abord dans la ligne TSH, sinon dans le texte
    ref_min, ref_max = _extract_reference_interval(line_text)
    if ref_min is None or ref_max is None:
        ref_min, ref_max = _extract_reference_interval(raw_text)

    # 6. Confiance
    confidence = "high"
    if not unit or ref_min is None or ref_max is None:
        confidence = "medium"

    # sécurité clinique
    if tsh_value < 0 or tsh_value > 150:
        return ParsedTSH(False, None, None, None, None, "low", "TSH_NOT_FOUND")

    return ParsedTSH(True, tsh_value, unit, ref_min, ref_max, confidence, None)
