import os
import shutil
import uuid
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ocr_engine import light_extract_text, premium_extract_text, optimum_extract_text
from parsers.tsh import premium_parse_tsh


# =========================================================
# MODELS
# =========================================================

class TSHResponse(BaseModel):
    ok: bool
    marker: str = "TSH"
    tsh_value: float | None = None
    tsh_unit: str | None = None
    ref_min: float | None = None
    ref_max: float | None = None
    confidence: str | None = None
    error: str | None = None
    raw_text: str | None = None


# =========================================================
# FASTAPI INIT
# =========================================================

app = FastAPI(title="THRD OCR Bio", version="3.0 Multi-Level")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# UTILS
# =========================================================

def save_temp_file(upload: UploadFile) -> str:
    """Save file to /tmp with a random name."""
    ext = os.path.splitext(upload.filename)[1].lower()
    fname = f"tmp_{uuid.uuid4().hex}{ext}"
    path = os.path.join("/tmp", fname)

    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)

    return path


# =========================================================
# HEALTHCHECK
# =========================================================

@app.get("/health")
async def health():
    return {"ok": True}


# =========================================================
# INTERNAL HELPER
# =========================================================

def _run_and_parse(path: str, level: Literal["light", "premium", "optimum"]):
    """Run OCR at the given level and parse TSH.

    Returns:
        (parsed, error, raw_text)
        - parsed: result object from premium_parse_tsh or None
        - error: error string or None
        - raw_text: raw OCR text or None
    """
    if level == "light":
        extract = light_extract_text
    elif level == "premium":
        extract = premium_extract_text
    else:
        extract = optimum_extract_text

    ocr = extract(path)
    if not ocr:
        return None, "OCR_FAILED", None

    parsed = premium_parse_tsh(ocr.raw_text, ocr.boxes)
    if not parsed.ok:
        return None, parsed.error, ocr.raw_text

    return parsed, None, ocr.raw_text


# =========================================================
# OCR TSH ENDPOINT (MULTI-LEVEL)
# =========================================================

@app.post("/ocr/tsh", response_model=TSHResponse)
async def ocr_tsh(
    file: UploadFile = File(...),
    mode: Literal["auto", "light", "premium", "optimum"] = "auto",
):
    """Full OCR pipeline for TSH.

    Steps:
    - D'abord -> Enregistrer le fichier uploadé dans /tmp
    - En fonction du mode, exécuter un ou plusieurs niveaux d’OCR (Les fameux light, premium, optimum Thks la SNCF)
    - Analyser le bloc TSH avec le parseur premium (Seulement un lvl pour l'instant)
    - Retourner une réponse JSON simplifiée pour Bubble

    Modes:
      - light:    1 seul passage OCR rapide
      - premium:  qualité standard (texte + boxes)
      - optimum:  mode renforcé pour images difficiles
      - auto:     light -> premium -> optimum (fallback progressif)
    """
    if not file:
        raise HTTPException(status_code=400, detail="Missing file")

    # 1. Save file
    tmp_path = save_temp_file(file)

    # Explicit modes: single pass
    if mode in ("light", "premium", "optimum"):
        parsed, error, raw_text = _run_and_parse(tmp_path, mode)
        if not parsed:
            return TSHResponse(
                ok=False,
                error=error,
                raw_text=raw_text,
            )

        return TSHResponse(
            ok=True,
            tsh_value=parsed.value,
            tsh_unit=parsed.unit,
            ref_min=parsed.ref_min,
            ref_max=parsed.ref_max,
            confidence=parsed.confidence,
            raw_text=raw_text,
        )

      # MODE AUTO : light -> premium -> optimum
    if mode == "auto":
        # 1) Light
        parsed, error, raw_text = _run_and_parse(tmp_path, "light")
        if parsed and not (parsed.ref_min is None and parsed.ref_max is None):
            return TSHResponse(
                ok=True,
                tsh_value=parsed.value,
                tsh_unit=parsed.unit,
                ref_min=parsed.ref_min,
                ref_max=parsed.ref_max,
                confidence=parsed.confidence,
                raw_text=raw_text,
            )

        # 2) Premium
        parsed, error, raw_text = _run_and_parse(tmp_path, "premium")
        if parsed and not (parsed.ref_min is None and parsed.ref_max is None):
            return TSHResponse(
                ok=True,
                tsh_value=parsed.value,
                tsh_unit=parsed.unit,
                ref_min=parsed.ref_min,
                ref_max=parsed.ref_max,
                confidence=parsed.confidence,
                raw_text=raw_text,
            )

        # 3) Optimum – dernier recours, on accepte même sans bornes
        parsed, error, raw_text = _run_and_parse(tmp_path, "optimum")
        if parsed:
            return TSHResponse(
                ok=True,
                tsh_value=parsed.value,
                tsh_unit=parsed.unit,
                ref_min=parsed.ref_min,
                ref_max=parsed.ref_max,
                confidence=parsed.confidence,
                raw_text=raw_text,
            )

        # Rien n'a marché
        return TSHResponse(
            ok=False,
            error=error or "TSH_NOT_FOUND",
            raw_text=raw_text,
        )

