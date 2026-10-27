"""Thin client for the Recommendation API used by the dashboard.

Two modes:
  mock  : calls the in-process stub (03-shared/apis/stub_app.py). No server needed. Placeholder data.
  live  : calls a running service over HTTP, for example http://127.0.0.1:8001
The dashboard never invents answers. If the service fails, the error is shown to the user.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx

APIS_DIR = Path(__file__).resolve().parents[1] / "apis"
if str(APIS_DIR) not in sys.path:
    sys.path.insert(0, str(APIS_DIR))

DEFAULT_BASE_URL = "http://127.0.0.1:8001"
TIMEOUT_S = 10.0


class ApiError(Exception):
    """The service answered with the contract error envelope, or could not be reached."""

    def __init__(self, message: str, code: str = "CLIENT_ERROR", details: list | None = None):
        super().__init__(message)
        self.code, self.details = code, details or []


def _mock_client():
    from fastapi.testclient import TestClient
    from stub_app import app  # noqa: WPS433 (imported late on purpose)
    return TestClient(app)


def _request(mode: str, method: str, path: str, base_url: str, json: dict | None = None) -> Any:
    try:
        if mode == "mock":
            resp = getattr(_mock_client(), method)(path, **({"json": json} if json is not None else {}))
        else:
            resp = httpx.request(method.upper(), base_url.rstrip("/") + path, json=json, timeout=TIMEOUT_S)
    except httpx.HTTPError as exc:
        raise ApiError(f"Could not reach the service at {base_url}. Is it running? ({type(exc).__name__})") from exc
    if resp.status_code >= 400:
        try:
            err = resp.json()["error"]
            raise ApiError(err.get("message", "Request failed."), err.get("code", "ERROR"), err.get("details"))
        except (KeyError, ValueError):
            raise ApiError(f"Service returned HTTP {resp.status_code}.") from None
    return resp.json()


def get_metadata(mode: str = "mock", base_url: str = DEFAULT_BASE_URL) -> dict:
    return _request(mode, "get", "/v1/metadata", base_url)


def get_health(mode: str = "mock", base_url: str = DEFAULT_BASE_URL) -> dict:
    return _request(mode, "get", "/v1/health", base_url)


def recommend(payload: dict, mode: str = "mock", base_url: str = DEFAULT_BASE_URL) -> dict:
    return _request(mode, "post", "/v1/recommend", base_url, json=payload)


def build_payload(district: str, season: str, top_k: int, include: list[str], crop: str | None,
                  soil: dict[str, Any]) -> dict:
    """Build a request body. Empty soil fields are left out so the service can fill or flag them."""
    body: dict[str, Any] = {"district": district, "season": season, "top_k": top_k, "include": include}
    if crop:
        body["crop"] = crop
    clean = {k: v for k, v in soil.items() if v not in (None, "")}
    if clean:
        body["soil"] = clean
    return body
