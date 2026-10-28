"""STUB of the Recommendation API for early integration (dashboard, CV team).

It checks input with the real contract models and answers with canned, clearly
marked placeholder data. It is NOT the recommendation logic. Replace the body of
recommend() with the baseline, then the ML models, without changing the contract.

Run from this folder:  uvicorn stub_app:app --port 8001 --reload
"""
import uuid
from copy import deepcopy
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import recommend_examples as ex
from recommend_schemas import (CANDIDATE_CROPS, ErrorEnvelope, HealthResponse, MetadataResponse,
                               RecommendRequest, RecommendResponse)

app = FastAPI(title="Recommendation API (STUB)", version="0.2.1")


def _rid() -> str:
    return uuid.uuid4().hex[:12]


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details, code = [], "VALIDATION_ERROR"
    for e in exc.errors():
        field = ".".join(str(p) for p in e["loc"] if p != "body")
        details.append({"field": field or None, "issue": e["msg"]})
        if e["type"] in ("greater_than_equal", "less_than_equal") and field in ("lat", "lon"):
            code = "OUT_OF_BOUNDS"
    body = ErrorEnvelope(error={"code": code, "message": "The request has invalid fields.",
                                "details": details, "request_id": _rid()})
    return JSONResponse(status_code=422, content=body.model_dump())


@app.get("/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="degraded", rules_loaded=False, model_loaded=False)


@app.get("/v1/metadata", response_model=MetadataResponse)
def metadata():
    return MetadataResponse(
        rules_version="stub", feature_schema_version="0.1",
        districts=[{"name": "Musanze", "province": "North", "in_training_data": True},
                   {"name": "Nyagatare", "province": "East", "in_training_data": True},
                   {"name": "Nyabihu", "province": "West", "in_training_data": True},
                   {"name": "Ruhango", "province": "South", "in_training_data": True}],
        crops=list(CANDIDATE_CROPS))


@app.post("/v1/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    out = deepcopy(ex.RESPONSE_OK)
    out["request_id"] = _rid()
    out["generated_at"] = datetime.now(timezone.utc)
    out["season"] = req.season
    out["warnings"].append({"code": "STUB_RESPONSE", "message": "Canned placeholder data. Not a real recommendation."})
    if "crops" not in req.include:
        out["crop_recommendations"] = []
    if "fertiliser" not in req.include:
        out["fertiliser"] = None
    out["crop_recommendations"] = out["crop_recommendations"][: req.top_k]
    return out
