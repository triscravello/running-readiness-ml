from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt

try:
    from src.level7_runtime import ModelRuntime
except ModuleNotFoundError:
    from level7_runtime import ModelRuntime

BASE_DIR = Path(__file__).resolve().parent
runtime = ModelRuntime(BASE_DIR)
app = FastAPI(
    title="Running Readiness Level 7",
    version="4.0.0",
    description=(
        f"{runtime.manifest['model_notice']} "
        f"{runtime.manifest['inherited_v1_preprocessing_limitation']}"
    ),
)

@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Never echo client input: non-finite values (NaN) are not JSON serializable.
    detail = [
        {"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    weekly_km_to_date: Annotated[FiniteFloat, Field(ge=0.0, le=200.0)]
    sleep_hours_last_night: Annotated[FiniteFloat, Field(ge=0.0, le=14.0)]
    resting_hr_morning: Annotated[FiniteFloat, Field(ge=30.0, le=220.0)]
    soreness_score_evening: Annotated[FiniteFloat, Field(ge=0.0, le=10.0)]
    prior_injury: Annotated[StrictInt, Field(ge=0, le=1)]

class PredictionResponse(BaseModel):
    probability: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    decision: Annotated[StrictInt, Field(ge=0, le=1)]
    threshold: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    device: str
    selected_approach: str

class HealthResponse(BaseModel):
    status: str
    integrity_status: str
    device: str
    selected_approach: str
    evidence_scope: str
    model_notice: str
    inherited_v1_preprocessing_limitation: str

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        integrity_status=runtime.integrity_status,
        device=runtime.device,
        selected_approach=runtime.handoff["selected_approach"],
        evidence_scope=runtime.manifest["evidence_scope"],
        model_notice=runtime.manifest["model_notice"],
        inherited_v1_preprocessing_limitation=runtime.manifest[
            "inherited_v1_preprocessing_limitation"
        ],
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(**runtime.predict(request.model_dump()))