"""Recommendation service: the real POST /v1/recommend, built on the B1 expert-rules baseline.

It follows the contract in docs/api_contracts.md (models in 03-shared/apis/recommend_schemas.py).
What it does for one request:
  1. Resolve the location to one of the four pilot districts (by name, or the nearest district centre).
  2. Build the inputs B1 needs: elevation, and the season climate normal moved to the point's height.
  3. Rank crops with B1. Each crop gets a score, reasons with sources, and a reliability level that also
     uses how B1 scored in that district in the official evaluation (reports/eval_v1).
  4. Ask the fertiliser engine for advice on the requested crop, or the rank 1 crop.
It never invents an answer: if no crop scores well enough it abstains, and if no fertiliser rule fits it says so.

Run from the repo root:
    python -m uvicorn service:app --app-dir 01-crop-recommendation/src/recommendation --port 8001
Then open http://127.0.0.1:8001/docs
"""
from __future__ import annotations

import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (HERE, HERE.parent / "models", REPO / "03-shared" / "apis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import expert_baseline as b1  # noqa: E402
import fertiliser_engine as fert  # noqa: E402
from recommend_schemas import (CONTRACT_VERSION, ErrorEnvelope, HealthResponse, MetadataResponse,  # noqa: E402
                               RecommendRequest, RecommendResponse)

PROFILES_PATH = HERE / "district_profiles.json"
FERTILISER_RULES_PATH = HERE / "rules" / "fertiliser_rules.yaml"

# Draft decision thresholds. They are documented in docs/api_contracts.md section 5 and can be tuned later.
MIN_SCORE = 0.5                 # abstain when the best crop scores below this
MAX_DISTRICT_DISTANCE_KM = 25   # a point further than this from every pilot district centre is "outside"
EVALUATED_SEASONS = {"B"}       # B1 was evaluated on season B data only (CIAT, 2025)
RULES_EVIDENCE_MIN = 0.5        # B1's tested macro-F1 in a district must reach this for "medium" (eval_v1)
KEY_SOIL_INPUTS = ("ph_water", "organic_carbon_pct", "total_nitrogen_pct")

CROP_RULES = b1.load_rules()
FERT_RULES = fert.load_rules(FERTILISER_RULES_PATH)
PROFILES = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
DISTRICTS = PROFILES["districts"]
SUPPORTED_CROPS = sorted(CROP_RULES["crops"])
RULES_VERSION = f"crops-{CROP_RULES['version']}+fertiliser-{FERT_RULES.version}"

app = FastAPI(title="Recommendation API (B1 expert-rules baseline)", version=CONTRACT_VERSION)


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int = 422, details: Optional[list] = None):
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details or []


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]


def _error(request: Request, status: int, code: str, message: str, details: Optional[list] = None) -> JSONResponse:
    body = ErrorEnvelope(error={"code": code, "message": message, "details": details or [], "request_id": _rid(request)})
    return JSONResponse(status_code=status, content=body.model_dump(), headers={"X-Request-ID": _rid(request)})


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details, code, status = [], "VALIDATION_ERROR", 422
    for e in exc.errors():
        field = ".".join(str(p) for p in e["loc"] if p != "body")
        details.append({"field": field or None, "issue": e["msg"]})
        if e["type"] == "json_invalid":
            code, status = "INVALID_JSON", 400
        elif e["type"] in ("greater_than_equal", "less_than_equal") and field in ("lat", "lon"):
            code = "OUT_OF_BOUNDS"
    message = "The request body is not valid JSON." if code == "INVALID_JSON" else "The request has invalid fields."
    return _error(request, status, code, message, details)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return _error(request, exc.status, exc.code, exc.message, exc.details)


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception):
    return _error(request, 500, "INTERNAL_ERROR", "Something went wrong on the server. Quote the request id.")


# ------------------------------------------------------------------ helpers
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _find_district(name: str) -> str:
    for d in DISTRICTS:
        if d.lower() == name.strip().lower():
            return d
    raise ApiError("UNKNOWN_DISTRICT", f"District '{name}' is not supported. See GET /v1/metadata.",
                   details=[{"field": "district", "issue": f"supported: {', '.join(sorted(DISTRICTS))}"}])


def resolve_location(req: RecommendRequest, warnings: list) -> tuple[Optional[str], dict]:
    """Return (district name or None, location dict for the response)."""
    named = _find_district(req.district) if req.district else None
    if req.lat is None:
        return named, {"district": named, "province": DISTRICTS[named]["province"], "lat": None, "lon": None,
                       "resolution": "district"}
    dist_km = {d: _haversine_km(req.lat, req.lon, p["centre"]["lat"], p["centre"]["lon"]) for d, p in DISTRICTS.items()}
    nearest = min(dist_km, key=dist_km.get)
    if dist_km[nearest] > MAX_DISTRICT_DISTANCE_KM:
        warnings.append({"code": "DISTRICT_NOT_IN_TRAINING_DATA",
                         "message": f"This point is {dist_km[nearest]:.0f} km from the nearest pilot district centre "
                                    f"({nearest}). Climate for this place is not available, so reliability is low."})
        district = None
    else:
        district = nearest
        warnings.append({"code": "LOCATION_MATCHED_TO_NEAREST_DISTRICT",
                         "message": f"Matched to {nearest}, whose centre is {dist_km[nearest]:.1f} km away. "
                                    "District boundaries are not loaded yet."})
    if named and district and named != district:
        warnings.append({"code": "DISTRICT_MISMATCH",
                         "message": f"The coordinates are closest to {district}, not {named}. The coordinates were used."})
    province = DISTRICTS[district]["province"] if district else None
    return district, {"district": district, "province": province, "lat": req.lat, "lon": req.lon, "resolution": "point"}


def site_inputs(req: RecommendRequest, district: Optional[str], warnings: list) -> tuple[dict, dict]:
    """Return (row of B1 inputs, site values with their sources for inputs_used)."""
    row: dict[str, Any] = {}
    site: dict[str, dict] = {}
    prof = DISTRICTS.get(district) if district else None
    if req.elevation_m is not None:
        row["elevation_m"] = float(req.elevation_m)
        site["elevation_m"] = {"value": row["elevation_m"], "source": "user"}
    elif prof:
        row["elevation_m"] = prof["elevation_m"]["median"]
        site["elevation_m"] = {"value": row["elevation_m"], "source": "district_median_ciat"}
        warnings.append({"code": "ELEVATION_FROM_DISTRICT_MEDIAN",
                         "message": f"Elevation was not given, so the median of the CIAT points in {district} "
                                    f"({row['elevation_m']:.0f} m) was used. Give elevation_m for a field-level answer."})
    if prof and req.season in prof["seasons"]:
        c = prof["seasons"][req.season]
        ref = prof["elevation_m"]["median"]
        elev = row.get("elevation_m", ref)
        tmean = (c["tmax_c"] + c["tmin_c"]) / 2
        row["precip_mm"] = c["precip_mm"]
        row["tmean_point_c"] = round(b1.adjust_to_height(tmean, elev, ref), 2)
        row["tmin_point_c"] = round(b1.adjust_to_height(c["tmin_c"], elev, ref), 2)
        src = f"nasa_power_normal_{c['years'].split(' (')[0].replace(' to ', '_')}"
        site["precip_mm"] = {"value": row["precip_mm"], "source": src}
        site["tmean_point_c"] = {"value": row["tmean_point_c"], "source": src + "_height_adjusted"}
        site["tmin_point_c"] = {"value": row["tmin_point_c"], "source": src + "_height_adjusted"}
    return row, site


def rules_tested(district: Optional[str]) -> Optional[dict]:
    """How B1 scored in this district in the official evaluation (main test, season B 2025), if known."""
    return DISTRICTS[district].get("b1_tested") if district else None


def rules_weak_here(district: Optional[str], season: str) -> bool:
    tested = rules_tested(district)
    return bool(tested) and season in EVALUATED_SEASONS and tested["macro_f1"] < RULES_EVIDENCE_MIN


def reliability_for(item: dict, district: Optional[str], season: str, row: dict) -> dict:
    """B1 reliability rules (contract section 5). B1 is never 'high', because no ML model is adopted.
    'medium' needs a pilot district where B1 scored at least RULES_EVIDENCE_MIN when tested, season B,
    a crop whose rules are not weak, and every rule input present. Anything else is 'low'."""
    codes = ["ML_NOT_ADOPTED"]
    low = False
    if district is None:
        codes.append("DISTRICT_NOT_IN_TRAINING")
        low = True
    if season not in EVALUATED_SEASONS:
        codes.append("SEASON_NOT_EVALUATED")
        low = True
    if rules_weak_here(district, season):
        codes.append("RULES_WEAK_IN_DISTRICT")
        low = True
    if item.get("reliability_cap") == "low":
        codes.append("CROP_RULES_WEAK")
        low = True
    for r in item["reasons"]:
        if r["value"] is None:
            codes.append(f"INPUT_MISSING_{r['factor'].upper()}")
            low = True
    return {"level": "low" if low else "medium", "reason_codes": codes}


def soil_inputs(req: RecommendRequest) -> tuple[dict, dict]:
    given = req.soil.model_dump(exclude_none=True) if req.soil else {}
    used = {"values": {k: {"value": v, "source": "user"} for k, v in given.items()},
            "missing": [k for k in KEY_SOIL_INPUTS if k not in given]}
    return given, used


# ---------------------------------------------------------------- endpoints
@app.get("/v1/health", response_model=HealthResponse)
def health():
    # "degraded" in the contract means: rules answer, but no trained ML model is loaded.
    return HealthResponse(status="degraded", rules_loaded=True, model_loaded=False)


@app.get("/v1/metadata", response_model=MetadataResponse)
def metadata():
    return MetadataResponse(
        rules_version=RULES_VERSION, feature_schema_version=PROFILES.get("source_table"),
        districts=[{"name": n, "province": p["province"], "in_training_data": p["in_training_data"]}
                   for n, p in sorted(DISTRICTS.items())],
        crops=SUPPORTED_CROPS)


@app.post("/v1/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, request: Request):
    if req.crop and req.crop not in SUPPORTED_CROPS:
        raise ApiError("UNSUPPORTED_CROP", f"Crop '{req.crop}' is not supported. See GET /v1/metadata.",
                       details=[{"field": "crop", "issue": f"supported: {', '.join(SUPPORTED_CROPS)}"}])
    warnings: list[dict] = []
    district, location = resolve_location(req, warnings)
    row, site = site_inputs(req, district, warnings)
    soil_given, soil_used = soil_inputs(req)

    if req.season not in EVALUATED_SEASONS:
        warnings.append({"code": "SEASON_NOT_EVALUATED",
                         "message": f"The baseline was tested on season B data only. Treat season {req.season} "
                                    "results with extra care."})
    if rules_weak_here(district, req.season):
        tested = rules_tested(district)
        warnings.append({"code": "RULES_WEAK_IN_DISTRICT",
                         "message": f"When tested on {tested['n_fields']} CIAT fields in {district} (season B 2025), "
                                    f"the expert rules scored {tested['macro_f1']:.2f} out of 1 for matching the "
                                    f"crop farmers planted, below the {RULES_EVIDENCE_MIN} needed for medium "
                                    "reliability. Treat this ranking with care."})
    if soil_given:
        warnings.append({"code": "SOIL_NOT_USED_BY_METHOD",
                         "message": "Soil values are recorded and passed to the fertiliser rules, but the crop "
                                    "baseline (B1) does not use soil yet."})

    # Crops with no evaluable rule (score None) cannot be assessed here, so they are not listed.
    ranked = [i for i in b1.rank_crops(row, CROP_RULES, PROFILES["crop_prior"]) if i["score"] is not None]
    best = ranked[0]["score"] if ranked else None
    abstain = best is None or best < MIN_SCORE
    crops_out = []
    if abstain:
        warnings.append({"code": "NO_CROP_ABOVE_MIN_SCORE",
                         "message": f"No crop reached the minimum score of {MIN_SCORE} for these conditions."})
    elif "crops" in req.include:
        tied = [i["crop"] for i in ranked if abs(i["score"] - ranked[0]["score"]) < 1e-9]
        if len(tied) > 1:
            warnings.append({"code": "CROPS_TIED",
                             "message": f"{', '.join(tied)} fit the rules equally well (score "
                                        f"{ranked[0]['score']:.2f}). Their order comes from a tie-break (number of "
                                        "rules checked, then how common each crop is in the CIAT data), not from a "
                                        "real difference in fit."})
        for rank, item in enumerate(ranked[: req.top_k], start=1):
            crops_out.append({"rank": rank, "crop": item["crop"], "score": round(item["score"], 3),
                              "method": "expert_rules",
                              "reliability": reliability_for(item, district, req.season, row),
                              "reasons": item["reasons"]})

    fertiliser = None
    if "fertiliser" in req.include:
        target = req.crop or (None if abstain else ranked[0]["crop"])
        if target:
            fertiliser = fert.recommend(FERT_RULES, target, soil_given)
            if fertiliser["status"] == "no_matching_rule":
                warnings.append({"code": "NO_GUIDELINE_RULE_FOUND",
                                 "message": "No fertiliser guideline rule fits. The rule file is waiting for the "
                                            "RAB fertiliser guideline, so no fertiliser advice is given yet."})
        else:
            warnings.append({"code": "FERTILISER_SKIPPED",
                             "message": "No crop to give fertiliser advice for. Name a crop in the request."})

    return {
        "request_id": _rid(request),
        "model_version": None,
        "rules_version": RULES_VERSION,
        "generated_at": datetime.now(timezone.utc),
        "location": location,
        "season": req.season,
        "abstain": abstain,
        "abstain_message": ("Uncertain. No crop fits these conditions well enough. Please consult an extension "
                            "officer.") if abstain else None,
        "crop_recommendations": crops_out,
        "fertiliser": fertiliser,
        "inputs_used": {"soil": soil_used, "site": site},
        "warnings": warnings,
    }
