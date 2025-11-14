import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ocr_engine import premium_extract_text
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

app = FastAPI(title="THRD OCR Bio", version="2.0 Premium")

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
# OCR TSH ENDPOINT (PREMIUM)
# =========================================================

@app.post("/ocr/tsh", response_model=TSHResponse)
async def ocr_tsh(file: UploadFile = File(...)):
    """
    Full OCR pipeline:
    1. Save file
    2. Multi-variant OCR (premium)
    3. Premium TSH parser
    4. JSON response simplified for Bubble
    """
    if not file:
        raise HTTPException(status_code=400, detail="Missing file")

    # 1. Save file
    tmp_path = save_temp_file(file)

    # 2. Run OCR
    ocr = premium_extract_text(tmp_path)
    if not ocr:
        return TSHResponse(
            ok=False,
            error="OCR_FAILED",
            raw_text=None
        )

    # 3. Parse TSH
    parsed = premium_parse_tsh(ocr.raw_text, ocr.boxes)
    if not parsed.ok:
        return TSHResponse(
            ok=False,
            error=parsed.error,
            raw_text=ocr.raw_text
        )

    # 4. Build response
    return TSHResponse(
        ok=True,
        tsh_value=parsed.value,
        tsh_unit=parsed.unit,
        ref_min=parsed.ref_min,
        ref_max=parsed.ref_max,
        confidence=parsed.confidence,
        raw_text=ocr.raw_text
    )
