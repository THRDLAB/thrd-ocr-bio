# models.py

from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str


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
    debug: Optional[Dict[str, Any]] = None
