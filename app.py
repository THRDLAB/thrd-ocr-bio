from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import tempfile
import shutil
import os
import logging

# --- Premium OCR & Parsing ---
from models import HealthResponse, TSHResponse
from ocr_engine import premium_extract_text  # ⬅️ nouveau
from parsers.tsh import premium_parse_tsh    # ⬅️ nouveau

# ---------------------------------------------------------------------
# Config & init
# ---------------------------------------------------------------------

app = FastAPI(
    title="THRD OCR Bio",
    description="Service OCR intelligent pour extraire la TSH depuis des bilans sanguins (PDF / images).",
    version="0.2.0-premium",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Utils internes
# ---------------------------------------------------------------------

def _save_upload_to_temp(upload_file: UploadFile) -> str:
    """Enregistre le fichier uploadé dans un tmp local"""
    suffix = ""
    if upload_file.filename:
        _, ext = os.path.splitext(upload_file.filename)
        suffix = ext

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with tmp as f:
            shutil.copyfileobj(upload_file.file, f)
    finally:
        upload_file.file.close()

    logger.info(f"[temp] saved upload to {tmp.name}")
    return tmp.name


def _is_supported_content_type(content_type: Optional[str]) -> bool:
    if not content_type:
        return False
    return content_type in {
        "image/jpeg",
        "image/png",
        "image/webp",
        "application/pdf",
    }

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ok=True, service="thrd-ocr-bio", version="0.2.0-premium")


@app.post("/ocr/tsh", response_model=TSHResponse)
async def ocr_tsh(file: UploadFile = File(...)) -> TSHResponse:
    """
    Endpoint premium :
    - OCR multi-pass
    - Détection automatique de la ligne TSH
    - Analyse par colonnes (valeur actuelle vs antériorités)
    - Extraction robustes des unités & valeurs de référence
    - Fail-safe intelligent (ne renvoie jamais une mauvaise TSH)
    """
    if not _is_supported_content_type(file.content_type):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type: {file.content_type}. "
                f"Supported: image/jpeg, image/png, image/webp, application/pdf."
            ),
        )

    temp_path = _save_upload_to_temp(file)

    try:
        # OCR premium multi-pass
        ocr_result = premium_extract_text(temp_path)

        if not ocr_result or not ocr_result.raw_text:
            return TSHResponse(
                ok=False,
                error="OCR_EMPTY_TEXT",
                raw_text=None,
                debug={
                    "engine": "tesseract-premium",
                    "path": temp_path,
                },
            )

        # Parsing premium de la TSH
        parsed = premium_parse_tsh(
            raw_text=ocr_result.raw_text,
            ocr_boxes=ocr_result.boxes,  # ⬅️ positions ligne/colonnes
        )

        if parsed is None or not parsed.ok:
            # Fail-safe = pas de TSH valide trouvée
            return TSHResponse(
                ok=False,
                error=parsed.error if parsed else "TSH_NOT_FOUND",
                raw_text=ocr_result.raw_text,
                debug={
                    "engine": "tesseract-premium",
                    "path": temp_path,
                },
            )

        # OK : on renvoie la détection complète et fiabilisée
        return TSHResponse(
            ok=True,
            marker="TSH",
            tsh_value=parsed.value,
            tsh_unit=parsed.unit,
            ref_min=parsed.ref_min,
            ref_max=parsed.ref_max,
            confidence=parsed.confidence,
            raw_text=ocr_result.raw_text,
            debug={
                "engine": "tesseract-premium",
                "path": temp_path,
            },
        )

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


# Dev local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
