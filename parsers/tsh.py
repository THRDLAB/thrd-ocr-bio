import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ParsedTSH:
    ok: bool
    value: Optional[float]
    unit: Optional[str]
    ref_min: Optional[float]
    ref_max: Optional[float]
    confidence: Optional[str]   # "high" | "medium" | "low"
    error: Optional[str]


# =========================================================
# Helpers
# =========================================================

NUM = r"\d+(?:[.,]\d+)?"
NUM_DEC = r"\d+[.,]\d+"


def _normalize(text: str) -> str:
    """
    Normalisation assez douce :
    - minuscule
    - suppression des accents
    - on garde chiffres / lettres / . , - / : >
    - espaces compactés
    """
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    # on garde uniquement les caractères utiles au parsing
    text = re.sub(r"[^a-z0-9.,<>:=/\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None


def _scale_ref(m1: float, m2: float) -> Tuple[float, float]:
    """
    Corrige grossièrement les ref type "0400 - 4,000" issues de l'OCR :
    tant que la valeur max est très grande, on divise par 10.
    On s'arrête quand max <= 50 ou après quelques itérations.
    """
    for _ in range(3):
        if max(m1, m2) <= 50:
            break
        m1 /= 10.0
        m2 /= 10.0
    if m1 > m2:
        m1, m2 = m2, m1
    return m1, m2


def _extract_ref_interval(snippet: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Cherche un intervalle de référence dans un bout de texte :
    - "0,4 - 4,0"
    - "0,4 4,0"
    - gère les cas OCR bizarres comme "0400 - 4,000"
    """
    norm = _normalize(snippet)

    # pattern classique "min - max"
    m = re.search(rf"({NUM})\s*[-–—aàto]+\s*({NUM})", norm)
    if not m:
        # fallback "min   max"
        m = re.search(rf"({NUM})\s+({NUM})", norm)
    if not m:
        return None, None

    m1 = _to_float(m.group(1))
    m2 = _to_float(m.group(2))
    if m1 is None or m2 is None:
        return None, None

    m1, m2 = _scale_ref(m1, m2)
    return m1, m2


def _conf(level: str) -> str:
    return level


def _first_value(ctx: str) -> Tuple[Optional[float], Optional[re.Match]]:
    """
    Renvoie la première valeur plausible dans un contexte :
    - on privilégie les nombres avec virgule/point (3,54 ; 0.40)
    - sinon on prend le premier entier.
    """
    m = re.search(NUM_DEC, ctx)
    if m:
        return _to_float(m.group(0)), m
    m = re.search(NUM, ctx)
    if m:
        return _to_float(m.group(0)), m
    return None, None


# =========================================================
# Public API
# =========================================================

def premium_parse_tsh(raw_text: str, boxes=None) -> ParsedTSH:
    """
    Parser TSH "premium" :
    - détecte TSH / T.S.H / T S H, "tsh ultra sensible", "3eme génération", etc.
    - récupère la première valeur plausible après le marqueur
    - essaie de trouver un intervalle de référence dans la foulée
    """
    if not raw_text:
        return ParsedTSH(False, None, None, None, None, None, "EMPTY_TEXT")

    norm = _normalize(raw_text)

    candidates: List[ParsedTSH] = []

    # 1) Recherche explicite autour de TSH
    #    on gère : "TSH", "T.S.H", "T S H", "thyreostimuline", etc.
    tsh_pattern = re.compile(
        r"(t\s*\.?\s*s\s*\.?\s*h|tsh|thyreostimuline|thyrotrop[eie]ne)",
        re.IGNORECASE,
    )

    for m in tsh_pattern.finditer(norm):
        # contexte après le mot TSH
        start = max(0, m.end())
        ctx = norm[start:start + 160]

        value, mv = _first_value(ctx)
        if mv is None or value is None:
            continue

        # bornes "plausibles" pour une TSH
        if not (0 <= value <= 150):
            continue

        # valeurs de référence dans ce qui suit
        ref_min, ref_max = _extract_ref_interval(ctx[mv.end():])

        candidates.append(
            ParsedTSH(
                ok=True,
                value=value,
                unit="mUI/L",  # on fixe l'unité, plus simple et plus robuste
                ref_min=ref_min,
                ref_max=ref_max,
                confidence=_conf("high" if ref_min is not None else "medium"),
                error=None,
            )
        )

    if candidates:
        # on garde le candidat avec la meilleure "confidence"
        order = {"high": 2, "medium": 1, "low": 0}
        candidates.sort(key=lambda c: order.get(c.confidence or "", 0), reverse=True)
        return candidates[0]

    # 2) Fallback très large : n'importe quelle valeur "raisonnable"
    #    dans tout le texte.
    any_value, m_any = _first_value(norm)
    if m_any is not None and any_value is not None and 0 <= any_value <= 150:
        ref_min, ref_max = _extract_ref_interval(norm[m_any.end():m_any.end() + 80])
        return ParsedTSH(
            ok=True,
            value=any_value,
            unit="mUI/L",
            ref_min=ref_min,
            ref_max=ref_max,
            confidence=_conf("low"),
            error=None,
        )

    # 3) Rien trouvé
    return ParsedTSH(False, None, None, None, None, None, "TSH_NOT_FOUND")
