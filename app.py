from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal
import tempfile
import shutil
import os
import logging

# ---------------------------------------------------------------------
# Config & init
# ---------------------------------------------------------------------

app = FastAPI(
    title="THRD OCR Bio",
    description="Service OCR pour extraire les résultats TSH depuis des bilans sanguins (PDF / images).",
    version="0.1.0",
)

# CORS (à adapter si besoin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Modèles Pydantic
# ---------------------------------------------------------------------

class TSHParseResult(BaseModel):
    tsh_value: float
    tsh_unit: Optional[str] = None
    ref_min: Optional[float] = None
    ref_max: Optional[float] = None
    confidence: Literal["low", "medium", "high"] = "low"


class TSHResponse(BaseModel):
    ok: bool
    marker: str = "TSH"
    tsh_value: Optional[float] = None
    tsh_unit: Optional[str] = None
    ref_min: Optional[float] = None
    ref_max: Optional[float] = None
    confidence: Optional[Literal["low", "medium", "high"]] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None
    debug: Optional[dict] = None


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str


# ---------------------------------------------------------------------
# Utils internes (stubs pour l’instant)
# ---------------------------------------------------------------------

def extract_text_from_file(path: str) -> str:
    
    from ocr_engine import extract_text_from_file
    logger.info(f"[extract_text_from_file] Called with path={path}")
    return ""


def parse_tsh(text: str) -> Optional[TSHParseResult]:
    """
    TODO: implémenter le parsing réel de la TSH dans le texte.
    Pour l’instant, renvoie None pour simuler un 'TSH_NOT_FOUND'.
    """
    logger.info("[parse_tsh] Called (stub), length of text=%d", len(text or ""))
    # Ici tu implémenteras plus tard la logique de:
    # - recherche de lignes contenant 'tsh', 'tshus', etc.
    # - extraction de la valeur, unité, plage de référence, etc.
    return None


def _save_upload_to_temp(upload_file: UploadFile) -> str:
    """
    Sauvegarde le fichier uploadé dans un fichier temporaire
    et renvoie le chemin. À supprimer ensuite manuellement.
    """
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

    logger.info(f"[tempfile] Saved upload to {tmp.name}")
    return tmp.name


def _is_supported_content_type(content_type: Optional[str]) -> bool:
    if not content_type:
        return False
    allowed = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "application/pdf",
    }
    return content_type in allowed


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Endpoint de healthcheck pour Northflank / monitoring.
    """
    return HealthResponse(ok=True, service="thrd-ocr-bio", version="0.1.0")


@app.post("/ocr/tsh", response_model=TSHResponse)
async def ocr_tsh(file: UploadFile = File(...)) -> TSHResponse:
    """
    Endpoint principal :
    - Reçoit un PDF ou une image (multipart/form-data)
    - Extrait le texte via OCR
    - Parse la TSH dans le texte
    - Retourne un JSON normalisé
    """
    if not _is_supported_content_type(file.content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. "
                   f"Supported: image/jpeg, image/png, image/webp, application/pdf.",
        )

    temp_path = _save_upload_to_temp(file)

    try:
        raw_text = extract_text_from_file(temp_path)

        if not raw_text:
            # OCR n'a rien sorti → on renvoie ok=false
            return TSHResponse(
                ok=False,
                error="OCR_EMPTY_TEXT",
                raw_text=None,  # tu peux décider de renvoyer '' si tu préfères
                debug={
                    "engine": "tesseract",  # prévisionnel
                    "path": temp_path,
                },
            )

        parsed = parse_tsh(raw_text)

        if parsed is None:
            # Pas de TSH trouvée
            return TSHResponse(
                ok=False,
                error="TSH_NOT_FOUND",
                raw_text=raw_text,
                debug={
                    "engine": "tesseract",  # prévisionnel
                    "path": temp_path,
                },
            )

        # TSH trouvée → on renvoie la structure complète
        return TSHResponse(
            ok=True,
            marker="TSH",
            tsh_value=parsed.tsh_value,
            tsh_unit=parsed.tsh_unit,
            ref_min=parsed.ref_min,
            ref_max=parsed.ref_max,
            confidence=parsed.confidence,
            raw_text=raw_text,
            debug={
                "engine": "tesseract",  # prévisionnel
                "path": temp_path,
            },
        )

    finally:
        # Nettoyage du fichier temporaire
        try:
            os.remove(temp_path)
            logger.info(f"[tempfile] Removed {temp_path}")
        except Exception as e:
            logger.warning(f"[tempfile] Could not remove temp file {temp_path}: {e}")


# ---------------------------------------------------------------------
# Développement local
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
