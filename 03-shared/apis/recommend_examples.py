"""ILLUSTRATIVE payloads for docs/api_contracts.md and the contract tests.

All values are placeholders to show the shape of the contract. They are not
agronomic advice and not real model output. Document and section names are
deliberately left as placeholders until the real guideline source is chosen.
"""

REQUEST_DISTRICT = {
    "district": "Musanze",
    "season": "B",
    "soil": {"ph_water": 5.1, "organic_carbon_pct": 2.0},
    "top_k": 3,
}

REQUEST_POINT_FERTILISER_ONLY = {
    "lat": -1.5,
    "lon": 29.6,
    "season": "A",
    "crop": "maize",
    "include": ["fertiliser"],
}

RESPONSE_OK = {
    "request_id": "example-0001",
    "api_version": "0.1.0",
    "model_version": None,
    "rules_version": "rules-0.1-example",
    "generated_at": "2026-10-21T08:00:00Z",
    "location": {"district": "Musanze", "province": "North", "lat": None, "lon": None, "resolution": "district"},
    "season": "B",
    "abstain": False,
    "abstain_message": None,
    "crop_recommendations": [
        {
            "rank": 1, "crop": "maize", "score": 0.7, "method": "expert_rules",
            "reliability": {"level": "medium", "reason_codes": ["ML_NOT_ADOPTED", "INPUT_MISSING_TOTAL_NITROGEN"]},
            "reasons": [
                {"factor": "soil_ph", "direction": "supports", "value": 5.1,
                 "expected_range": "<range from the cited agronomy source>",
                 "text": "ILLUSTRATIVE TEXT. Soil pH is inside the range used for this crop."},
                {"factor": "elevation_m", "direction": "neutral", "value": None,
                 "expected_range": None, "text": "ILLUSTRATIVE TEXT. Elevation was not used in this example."},
            ],
        },
        {
            "rank": 2, "crop": "beans", "score": 0.6, "method": "expert_rules",
            "reliability": {"level": "medium", "reason_codes": ["ML_NOT_ADOPTED", "INPUT_MISSING_TOTAL_NITROGEN"]},
            "reasons": [{"factor": "soil_ph", "direction": "supports", "value": 5.1,
                         "expected_range": None, "text": "ILLUSTRATIVE TEXT."}],
        },
    ],
    "fertiliser": {
        "crop": "maize",
        "status": "ok",
        "engine_version": "rules-0.1-example",
        "items": [
            {
                "input_type": "lime", "product": None, "rate": None, "rate_unit": None, "timing": None,
                "condition_matched": "ILLUSTRATIVE. Soil pH is below the threshold given in the cited guideline.",
                "source": {"document": "<guideline title, filled from the real source>",
                           "section": "<section>", "page": None, "url": None},
                "explanation": "ILLUSTRATIVE TEXT ONLY. Rate and timing come from the cited guideline, not from this example.",
                "explanation_method": "rule_text",
            }
        ],
    },
    "inputs_used": {
        "soil": {
            "values": {
                "ph_water": {"value": 5.1, "source": "user"},
                "organic_carbon_pct": {"value": 2.0, "source": "user"},
            },
            "missing": ["total_nitrogen_pct"],
        }
    },
    "warnings": [{"code": "SOIL_VALUE_MISSING", "message": "Total nitrogen was not given, so reliability is capped at medium."}],
    "disclaimer": "Decision support for extension officers and agronomists. It does not replace local field advice.",
}

RESPONSE_ABSTAIN = {
    "request_id": "example-0002",
    "api_version": "0.1.0",
    "model_version": None,
    "rules_version": "rules-0.1-example",
    "generated_at": "2026-10-21T08:05:00Z",
    "location": {"district": None, "province": None, "lat": -1.5, "lon": 29.6, "resolution": "point"},
    "season": "A",
    "abstain": True,
    "abstain_message": "Uncertain. Please consult an extension officer.",
    "crop_recommendations": [],
    "fertiliser": None,
    "inputs_used": {"soil": {"values": {}, "missing": ["ph_water", "organic_carbon_pct", "total_nitrogen_pct"]}},
    "warnings": [{"code": "NO_CROP_ABOVE_MIN_SCORE", "message": "No crop reached the minimum score."}],
    "disclaimer": "Decision support for extension officers and agronomists. It does not replace local field advice.",
}

ERROR_VALIDATION = {
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "The request has invalid fields.",
        "details": [{"field": "soil.ph_water", "issue": "must be between 3.0 and 10.0"}],
        "request_id": "example-0003",
    }
}
