"""Payloads for docs/api_contracts.md and the contract tests.

The POST /v1/recommend examples are ILLUSTRATIVE placeholders that show the shape of the contract: they are not
agronomic advice and not real output. The POST /v1/recommend/point examples (added in 1.0, updated in 1.1 for the tuned
Random Forest on the map and in 1.2 for the reasons text) are real answers of the service on 25 Nov 2026, with only
request_id and generated_at fixed. Their fertiliser rates are the interim national rates, labelled as such.
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
    "api_version": "1.3",
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
    "api_version": "1.3",
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


# ------------------------------------------------------------ added in 1.0
REQUEST_POINT = {"lat": -1.35, "lon": 30.30, "season": "B"}

RESPONSE_POINT_OK = {
    "request_id": "example-0004",
    "api_version": "1.3",
    "model_version": "RF[all_sat] tuned, trained on the 1,904 CIAT fields",
    "rules_version": "crops-0.1-draft+fertiliser-0.1-interim-2026-11-18",
    "generated_at": "2026-11-25T09:30:00Z",
    "location": {
        "district": "Nyagatare",
        "province": "East",
        "lat": -1.35,
        "lon": 30.3,
        "resolution": "point",
    },
    "season": {
        "asked": "B",
        "used": "B",
    },
    "abstain": False,
    "abstain_message": None,
    "recommendation": {
        "crop": "maize",
        "method": "random_forest",
        "confidence": {
            "level": "high",
            "tested_hit_rate": 0.775946,
            "lowered_by": [],
            "basis": "first_crop",
            "tested_fields": 1031,
            "situation": "Nyagatare, field data within 5 km, one clear first crop",
        },
        "kind": "one_crop",
        "crops": ["maize"],
        "close_second": None,
        "alternatives": ["rice", "irish_potato", "beans"],
        "why": [
            ("In Nyagatare, Random Forest named the right crop first for 76% of the "
                "new fields in testing, against 22% for the expert rules, so the answer "
                "comes from Random Forest."),
            ("In testing, answers like this one, with one clear first crop, were right "
                "for 78% of the fields (1031 fields in Nyagatare, field data within 5 km)."),
        ],
    },
    "fertiliser": {
        "crop": "maize",
        "status": "ok",
        "engine_version": "0.2.0+rules-0.1-interim-2026-11-18",
        "items": [
            {
                "input_type": "compound_fertiliser",
                "product": "DAP (18-46-0)",
                "rate": 100.0,
                "rate_unit": "kg/ha",
                "timing": "At planting",
                "condition_matched": "maize-national-interim-01: applies to this crop in general",
                "source": {
                    "document": ("B. T. Mudereri et al., Smart Fertiliser Recommendations for Maize in "
                        "Rwanda, CGIAR Sustainable Farming Science Program Report, Nov. 2025"),
                    "section": "Table 2, treatment BR (blanket recommendation)",
                    "page": "7",
                    "url": "https://hdl.handle.net/10568/178714",
                },
                "explanation": ("Interim: published national rate, not yet confirmed by RAB. National "
                    "blanket recommendation as used in the RAB and CGIAR maize trials: 100 "
                    "kg/ha of DAP at planting."),
                "explanation_method": "rule_text",
            },
            {
                "input_type": "nitrogen",
                "product": "Urea (46% N)",
                "rate": 100.0,
                "rate_unit": "kg/ha",
                "timing": "Top dressing at 6 to 8 leaves, about 4 to 6 weeks after emergence",
                "condition_matched": "maize-national-interim-01: applies to this crop in general",
                "source": {
                    "document": ("B. T. Mudereri et al., Smart Fertiliser Recommendations for Maize in "
                        "Rwanda, CGIAR Sustainable Farming Science Program Report, Nov. 2025"),
                    "section": "Table 2, treatment BR (blanket recommendation)",
                    "page": "7",
                    "url": "https://hdl.handle.net/10568/178714",
                },
                "explanation": ("Interim: published national rate, not yet confirmed by RAB. National "
                    "blanket recommendation as used in the RAB and CGIAR maize trials: 100 "
                    "kg/ha of urea as top dressing. An older report lists 50 kg/ha of urea "
                    "for the government recommendation (IFDC, 2018, Table 2), so this rate "
                    "needs RAB's confirmation."),
                "explanation_method": "rule_text",
            },
        ],
    },
    "inputs_used": {
        "elevation_m": 1350.2,
        "elevation_source": "median GPS elevation of the CIAT fields in the block",
        "soil_texture": "sandy clay loam",
        "soil_0_20cm": {
            "nitrogen": 1.37,
            "phosphorous_extractable": 18.09,
            "potassium_extractable": 171.07,
            "ph": 5.84,
            "sand": 49.78,
            "clay": 32.78,
            "silt": 20.44,
            "carbon_organic": 14.57,
            "cation_exchange_capacity": 10.94,
        },
        "field_data_from": {
            "n_ciat_fields": 9,
            "distance_km": 0.0,
        },
        "climate_season_b_2025": {
            "rainfall_mm": 441.6,
            "tmax_c": 27.67,
            "tmin_c": 16.42,
            "mean_temp_at_field_c": 22.42,
            "solar_mj_m2_day": 17.65,
        },
        "nisr_crop_shares": {
            "Beans": 0.531,
            "Irish potato": 0.01,
            "Maize": 0.421,
            "Rice": 0.038,
        },
        "satellite_at_field": {
            "srtm_elevation_m": 1362.0,
            "chirps_precip_mm": 399.8,
            "ndvi_annual": 0.465,
            "swir_ratio": 1.116,
        },
    },
    "evaluation": None,
    "warnings": [
        {
            "code": "FERTILISER_INTERIM",
            "message": ("Fertiliser rates are interim: national blanket rates from published "
                "sources, not yet confirmed by RAB, and the same for every field."),
        },
    ],
    "disclaimer": ("Decision support for extension officers and agronomists. It does not "
        "replace local field advice."),
}

# ------------------------------------------------------------ added in 1.3
REQUEST_POINT_TWO_CROPS = {"lat": -2.196, "lon": 29.592, "season": "B"}

# A point in Ruhango where beans and maize fit about equally: both are the answer, each with its fertiliser.
# A real answer of the service on 25 Nov 2026, with request_id and generated_at fixed and inputs_used left out.
RESPONSE_POINT_TWO_CROPS = {
    "request_id": "example-0006",
    "api_version": "1.3",
    "model_version": "RF[all_sat] tuned, trained on the 1,904 CIAT fields",
    "rules_version": "crops-0.1-draft+fertiliser-0.1-interim-2026-11-18",
    "generated_at": "2026-11-25T09:30:00Z",
    "location": {
        "district": "Ruhango",
        "province": "South",
        "lat": -2.196,
        "lon": 29.592,
        "resolution": "point"
    },
    "season": {
        "asked": "B",
        "used": "B"
    },
    "abstain": False,
    "abstain_message": None,
    "recommendation": {
        "crop": "beans",
        "method": "random_forest",
        "confidence": {
            "level": "high",
            "tested_hit_rate": 0.786885,
            "lowered_by": [
                "two crops fit almost as well"
            ],
            "basis": "first_two_crops",
            "tested_fields": 61,
            "situation": "Ruhango, field data within 5 km, first two crops close"
        },
        "kind": "two_crops",
        "crops": [
            "beans",
            "maize"
        ],
        "close_second": "maize",
        "alternatives": [
            "irish_potato",
            "rice"
        ],
        "why": [
            "In Ruhango, Random Forest named the right crop first for 63% of the new fields in testing, against 59% for the expert rules, so the answer comes from Random Forest.",
            "Maize fits almost as well (0.51 against 0.48), so both are given.",
            "In testing, when the first two crops were this close, the crop planted was one of the two for 79% of the fields (61 fields in Ruhango, field data within 5 km)."
        ]
    },
    "fertiliser": {
        "crop": "beans",
        "status": "ok",
        "engine_version": "0.2.0+rules-0.1-interim-2026-11-18",
        "items": [
            {
                "input_type": "compound_fertiliser",
                "product": "DAP (18-46-0)",
                "rate": 100.0,
                "rate_unit": "kg/ha",
                "timing": None,
                "condition_matched": "beans-national-interim-01: applies to this crop in general",
                "source": {
                    "document": "A. R. Cyamweshi, J. Kayumba and N. L. Nabahungu, Optimizing fertilizer use within the context of integrated soil fertility management in Rwanda, in Fertilizer Use Optimization in Sub-Saharan Africa, C. S. Wortmann and K. Sones, Eds. CABI, 2017, pp. 164-175",
                    "section": None,
                    "page": "165",
                    "url": "https://agronomy.unl.edu/sites/unl.edu.ianr.agronomy-horticulture/files/media/file/OFRA%20Chapter%2013_Rwanda_HR.pdf"
                },
                "explanation": "Interim: published national rate, not yet confirmed by RAB. Recommended rate for bean in Rwanda: 18 kg N and 46 kg P2O5 per ha. 100 kg/ha of DAP (18-46-0) gives exactly these amounts. The source gives the nutrients only, not the product or when to apply.",
                "explanation_method": "rule_text"
            }
        ]
    },
    "fertiliser_second": {
        "crop": "maize",
        "status": "ok",
        "engine_version": "0.2.0+rules-0.1-interim-2026-11-18",
        "items": [
            {
                "input_type": "compound_fertiliser",
                "product": "DAP (18-46-0)",
                "rate": 100.0,
                "rate_unit": "kg/ha",
                "timing": "At planting",
                "condition_matched": "maize-national-interim-01: applies to this crop in general",
                "source": {
                    "document": "B. T. Mudereri et al., Smart Fertiliser Recommendations for Maize in Rwanda, CGIAR Sustainable Farming Science Program Report, Nov. 2025",
                    "section": "Table 2, treatment BR (blanket recommendation)",
                    "page": "7",
                    "url": "https://hdl.handle.net/10568/178714"
                },
                "explanation": "Interim: published national rate, not yet confirmed by RAB. National blanket recommendation as used in the RAB and CGIAR maize trials: 100 kg/ha of DAP at planting.",
                "explanation_method": "rule_text"
            },
            {
                "input_type": "nitrogen",
                "product": "Urea (46% N)",
                "rate": 100.0,
                "rate_unit": "kg/ha",
                "timing": "Top dressing at 6 to 8 leaves, about 4 to 6 weeks after emergence",
                "condition_matched": "maize-national-interim-01: applies to this crop in general",
                "source": {
                    "document": "B. T. Mudereri et al., Smart Fertiliser Recommendations for Maize in Rwanda, CGIAR Sustainable Farming Science Program Report, Nov. 2025",
                    "section": "Table 2, treatment BR (blanket recommendation)",
                    "page": "7",
                    "url": "https://hdl.handle.net/10568/178714"
                },
                "explanation": "Interim: published national rate, not yet confirmed by RAB. National blanket recommendation as used in the RAB and CGIAR maize trials: 100 kg/ha of urea as top dressing. An older report lists 50 kg/ha of urea for the government recommendation (IFDC, 2018, Table 2), so this rate needs RAB's confirmation.",
                "explanation_method": "rule_text"
            }
        ]
    },
    "warnings": [
        {
            "code": "FERTILISER_INTERIM",
            "message": "Fertiliser rates are interim: national blanket rates from published sources, not yet confirmed by RAB, and the same for every field."
        }
    ],
    "disclaimer": "Decision support for extension officers and agronomists. It does not replace local field advice."
}

RESPONSE_POINT_OUTSIDE = {
    "request_id": "example-0005",
    "api_version": "1.3",
    "model_version": "RF[all_sat] tuned, trained on the 1,904 CIAT fields",
    "rules_version": "crops-0.1-draft+fertiliser-0.1-interim-2026-11-18",
    "generated_at": "2026-11-23T12:31:00Z",
    "location": {
        "district": "Nyarugenge",
        "province": None,
        "lat": -1.95,
        "lon": 30.06,
        "resolution": "point",
    },
    "season": {
        "asked": "B",
        "used": "B",
    },
    "abstain": True,
    "abstain_message": ("Nyarugenge is not one of the four pilot districts (Musanze, Nyabihu, "
        "Nyagatare, Ruhango). There is no CIAT data here, so no recommendation "
        "is given. Please consult an extension officer."),
    "recommendation": None,
    "fertiliser": None,
    "inputs_used": None,
    "evaluation": None,
    "warnings": [
        {
            "code": "OUTSIDE_PILOT_DISTRICTS",
            "message": ("Nyarugenge is not one of the four pilot districts (Musanze, Nyabihu, "
                "Nyagatare, Ruhango). There is no CIAT data here, so no recommendation "
                "is given."),
        },
    ],
    "disclaimer": ("Decision support for extension officers and agronomists. It does not "
        "replace local field advice."),
}
