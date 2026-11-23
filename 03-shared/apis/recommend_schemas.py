"""Pydantic models for the Recommendation API contract (version 1.0, final for the practicum).

These models are the executable form of docs/api_contracts.md. If a field changes,
change this file and the document together, and bump CONTRACT_VERSION.
Implemented by 01-crop-recommendation/src/recommendation/service.py; 03-shared/apis/stub_app.py returns placeholders.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "1.0"

# Approximate bounding box of Rwanda (decimal degrees, WGS84).
LAT_MIN, LAT_MAX = -2.9, -1.0
LON_MIN, LON_MAX = 28.8, 31.0

# Rwanda's land elevation runs from about 950 m (Rusizi river) to 4,507 m (Mount Karisimbi), with a small margin.
ELEV_MIN, ELEV_MAX = 900.0, 4600.0

# The four crops of the CIAT season B 2025 data, the only labelled data (final list, version 1.0).
# The service still checks crop codes at run time against GET /v1/metadata.
CANDIDATE_CROPS = ("beans", "irish_potato", "maize", "rice")

USDA_TEXTURES = (
    "sand", "loamy_sand", "sandy_loam", "loam", "silt_loam", "silt",
    "sandy_clay_loam", "clay_loam", "silty_clay_loam", "sandy_clay", "silty_clay", "clay",
)

DISCLAIMER = (
    "Decision support for extension officers and agronomists. "
    "It does not replace local field advice."
)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- request
class SoilInput(Strict):
    """Optional soil test values. Any field may be left out."""
    ph_water: Optional[float] = Field(None, ge=3.0, le=10.0, description="pH in water")
    organic_carbon_pct: Optional[float] = Field(None, ge=0.0, le=20.0, description="percent")
    total_nitrogen_pct: Optional[float] = Field(None, ge=0.0, le=5.0, description="percent")
    phosphorus_mgkg: Optional[float] = Field(None, ge=0.0, le=200.0, description="available P, mg per kg")
    potassium_cmolkg: Optional[float] = Field(None, ge=0.0, le=10.0, description="exchangeable K, cmol(+) per kg")
    texture: Optional[Literal[USDA_TEXTURES]] = Field(None, description="USDA texture class")  # type: ignore[valid-type]


class RecommendRequest(Strict):
    # The example the /docs page fills in, so "Try it out" works with one click on Execute.
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"district": "Musanze", "season": "B"}]})

    district: Optional[str] = Field(None, min_length=2, max_length=40)
    lat: Optional[float] = Field(None, ge=LAT_MIN, le=LAT_MAX)
    lon: Optional[float] = Field(None, ge=LON_MIN, le=LON_MAX)
    elevation_m: Optional[float] = Field(None, ge=ELEV_MIN, le=ELEV_MAX,
                                         description="Field elevation in metres, for example from a phone GPS. "
                                                     "If omitted, the district median is used.")
    season: Literal["A", "B", "C"]
    year: Optional[int] = Field(None, ge=2000, le=2100)
    crop: Optional[str] = Field(None, pattern=r"^[a-z_]{2,30}$",
                                description="Crop to give fertiliser advice for. If omitted, advice is for rank 1.")
    soil: Optional[SoilInput] = None
    top_k: int = Field(3, ge=1, le=5)
    include: list[Literal["crops", "fertiliser"]] = Field(default_factory=lambda: ["crops", "fertiliser"], min_length=1)

    @model_validator(mode="after")
    def location_is_given(self) -> "RecommendRequest":
        has_point = self.lat is not None and self.lon is not None
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be given together")
        if not has_point and not self.district:
            raise ValueError("give a district, or lat and lon")
        return self


# --------------------------------------------------------------- response
class Level(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class ResolvedLocation(Strict):
    district: Optional[str]
    province: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    resolution: Literal["point", "district"]


class Reliability(Strict):
    level: Level
    reason_codes: list[str] = Field(default_factory=list)


class Reason(Strict):
    factor: str
    direction: Literal["supports", "against", "neutral"]
    value: Optional[Union[float, str]] = None
    expected_range: Optional[str] = None
    text: str


class CropRecommendation(Strict):
    rank: int = Field(ge=1)
    crop: str
    score: float = Field(ge=0.0, le=1.0, description="Suitability score. Not a probability unless calibrated.")
    method: Literal["expert_rules", "random_forest", "xgboost"]
    reliability: Reliability
    reasons: list[Reason] = Field(min_length=1)


class SourceRef(Strict):
    document: str
    section: Optional[str] = None
    page: Optional[str] = None
    url: Optional[str] = None


class FertiliserItem(Strict):
    input_type: str = Field(description="e.g. lime, nitrogen, phosphorus, potassium, manure, compound_fertiliser")
    product: Optional[str] = None
    rate: Optional[float] = Field(None, ge=0.0)
    rate_unit: Optional[Literal["kg/ha", "t/ha"]] = None
    timing: Optional[str] = None
    condition_matched: str
    source: SourceRef
    explanation: Optional[str] = None
    explanation_method: Literal["rule_text", "rag"] = "rule_text"


class FertiliserAdvice(Strict):
    crop: str
    status: Literal["ok", "no_matching_rule"]
    engine_version: str
    items: list[FertiliserItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def ok_needs_items(self) -> "FertiliserAdvice":
        if self.status == "ok" and not self.items:
            raise ValueError("status ok needs at least one item")
        if self.status == "no_matching_rule" and self.items:
            raise ValueError("no_matching_rule must have no items")
        return self


class SoilValue(Strict):
    value: Union[float, str]
    source: Literal["user", "isdasoil", "district_default"]


class SoilUsed(Strict):
    values: dict[str, SoilValue] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list, description="Key soil inputs that were not available.")


class SiteValue(Strict):
    value: float
    source: str = Field(description="Where the value came from, for example user, district_median_ciat, "
                                    "or a NASA POWER season normal.")


class InputsUsed(Strict):
    soil: SoilUsed
    site: dict[str, SiteValue] = Field(default_factory=dict,
                                       description="Elevation and climate values the method used, with sources.")


class Warning_(Strict):
    code: str
    message: str


class RecommendResponse(Strict):
    request_id: str
    api_version: str = CONTRACT_VERSION
    model_version: Optional[str] = None
    rules_version: str
    generated_at: datetime
    location: ResolvedLocation
    season: Literal["A", "B", "C"]
    abstain: bool = False
    abstain_message: Optional[str] = None
    crop_recommendations: list[CropRecommendation] = Field(default_factory=list)
    fertiliser: Optional[FertiliserAdvice] = None
    inputs_used: InputsUsed
    warnings: list[Warning_] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def abstain_rules(self) -> "RecommendResponse":
        if self.abstain and not self.abstain_message:
            raise ValueError("abstain true needs abstain_message")
        if self.abstain and self.crop_recommendations:
            raise ValueError("abstain true means no crop_recommendations")
        ranks = [c.rank for c in self.crop_recommendations]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("ranks must be 1..n in order")
        return self


# ------------------------------------------------- one answer for a point (POST /v1/recommend/point)
class PointRequest(Strict):
    # The example the /docs page fills in: a point in Nyagatare, season B.
    model_config = ConfigDict(extra="forbid",
                              json_schema_extra={"examples": [{"lat": -1.35, "lon": 30.30, "season": "B"}]})

    lat: float = Field(..., ge=LAT_MIN, le=LAT_MAX)
    lon: float = Field(..., ge=LON_MIN, le=LON_MAX)
    season: Literal["A", "B", "C"]
    elevation_m: Optional[float] = Field(None, ge=ELEV_MIN, le=ELEV_MAX,
                                         description="Field elevation in metres. If omitted, the median GPS "
                                                     "elevation of the CIAT fields in the nearest 2 km block.")
    include_evaluation: bool = Field(False, description="Also return both methods' rankings and tested scores, "
                                                        "for evaluators. Users only need the recommendation.")


class Confidence(Strict):
    level: Level
    tested_hit_rate: Optional[float] = Field(None, ge=0, le=1,
                                             description="Share of new fields in this district where the answering "
                                                         "method's first crop was right in testing (eval_v1).")
    lowered_by: list[str] = Field(default_factory=list, description="Why the level is lower than the hit rate "
                                                                    "alone would give, in plain words.")


class PointAnswer(Strict):
    crop: str
    method: Literal["xgboost", "expert_rules"]
    confidence: Confidence
    close_second: Optional[str] = Field(None, description="A crop that fits almost as well, if any.")
    alternatives: list[str] = Field(default_factory=list, description="The method's next crops, in order.")
    why: list[str] = Field(..., min_length=1)


class SeasonUsed(Strict):
    asked: Literal["A", "B", "C"]
    used: Literal["B"] = "B"


class FieldDataFrom(Strict):
    n_ciat_fields: int = Field(..., ge=1, description="CIAT fields in the 2 km block the field inputs come from.")
    distance_km: float = Field(..., ge=0, description="Distance from the point to that block.")


class PointInputs(Strict):
    elevation_m: float
    elevation_source: str
    soil_texture: Optional[str] = None
    soil_0_20cm: dict[str, float]
    field_data_from: FieldDataFrom
    climate_season_b_2025: dict[str, float]
    nisr_crop_shares: dict[str, float]


class RankedCrop(Strict):
    crop: str
    score: float = Field(..., ge=0, le=1)


class PointEvaluation(Strict):
    xgboost_ranking: list[RankedCrop]
    rules_ranking: list[RankedCrop]
    tested_main_score_new_fields: dict[Literal["xgboost", "expert_rules"], Optional[float]]


class PointResponse(Strict):
    request_id: str
    api_version: str = CONTRACT_VERSION
    model_version: str
    rules_version: str
    generated_at: datetime
    location: ResolvedLocation
    season: SeasonUsed
    abstain: bool = False
    abstain_message: Optional[str] = None
    recommendation: Optional[PointAnswer] = None
    fertiliser: Optional[FertiliserAdvice] = None
    inputs_used: Optional[PointInputs] = None
    evaluation: Optional[PointEvaluation] = None
    warnings: list[Warning_] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def one_answer_or_abstain(self) -> "PointResponse":
        if self.abstain and (not self.abstain_message or self.recommendation is not None):
            raise ValueError("abstain true needs abstain_message and no recommendation")
        if not self.abstain and self.recommendation is None:
            raise ValueError("abstain false needs a recommendation")
        if self.recommendation and self.fertiliser and self.fertiliser.crop != self.recommendation.crop:
            raise ValueError("fertiliser advice must be for the recommended crop")
        return self


# ----------------------------------------------------------------- errors
class ErrorDetail(Strict):
    field: Optional[str] = None
    issue: str


class ErrorBody(Strict):
    code: Literal["INVALID_JSON", "VALIDATION_ERROR", "OUT_OF_BOUNDS", "UNKNOWN_DISTRICT",
                  "UNSUPPORTED_CROP", "MODEL_UNAVAILABLE", "INTERNAL_ERROR"]
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)
    request_id: str


class ErrorEnvelope(Strict):
    error: ErrorBody


# --------------------------------------------------------------- metadata
class DistrictInfo(Strict):
    name: str
    province: str
    in_training_data: bool


class MetadataResponse(Strict):
    api_version: str = CONTRACT_VERSION
    model_version: Optional[str] = None
    rules_version: str
    feature_schema_version: Optional[str] = None
    districts: list[DistrictInfo]
    crops: list[str]
    seasons: list[Literal["A", "B", "C"]] = ["A", "B", "C"]


class HealthResponse(Strict):
    status: Literal["ok", "degraded", "down"]
    rules_loaded: bool
    model_loaded: bool
    api_version: str = CONTRACT_VERSION
