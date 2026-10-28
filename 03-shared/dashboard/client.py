"""Thin client for the Recommendation API used by the dashboard.

Three modes:
  baseline : calls the real recommendation service in-process (B1 expert rules). No server needed.
  mock     : calls the in-process stub (03-shared/apis/stub_app.py). No server needed. Placeholder data.
  live     : calls a running service over HTTP, for example http://127.0.0.1:8001
The dashboard never invents answers. If the service fails, the error is shown to the user.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx

APIS_DIR = Path(__file__).resolve().parents[1] / "apis"
SERVICE_DIR = Path(__file__).resolve().parents[2] / "01-crop-recommendation" / "src" / "recommendation"
for _p in (APIS_DIR, SERVICE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_BASE_URL = "http://127.0.0.1:8001"
TIMEOUT_S = 10.0


class ApiError(Exception):
    """The service answered with the contract error envelope, or could not be reached."""

    def __init__(self, message: str, code: str = "CLIENT_ERROR", details: list | None = None):
        super().__init__(message)
        self.code, self.details = code, details or []


_CLIENTS: dict = {}


def _in_process_client(mode: str):
    """Build (once) a client that calls the app inside this process, without a server."""
    if mode not in _CLIENTS:
        from fastapi.testclient import TestClient
        if mode == "baseline":
            from service import app  # the real service on B1 rules
        else:
            from stub_app import app  # placeholder stub
        _CLIENTS[mode] = TestClient(app)
    return _CLIENTS[mode]


def _request(mode: str, method: str, path: str, base_url: str, json: dict | None = None) -> Any:
    try:
        if mode in ("mock", "baseline"):
            resp = getattr(_in_process_client(mode), method)(path, **({"json": json} if json is not None else {}))
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


def get_metadata(mode: str = "baseline", base_url: str = DEFAULT_BASE_URL) -> dict:
    return _request(mode, "get", "/v1/metadata", base_url)


def get_health(mode: str = "baseline", base_url: str = DEFAULT_BASE_URL) -> dict:
    return _request(mode, "get", "/v1/health", base_url)


def recommend(payload: dict, mode: str = "baseline", base_url: str = DEFAULT_BASE_URL) -> dict:
    return _request(mode, "post", "/v1/recommend", base_url, json=payload)


def build_payload(district: str, season: str, top_k: int, include: list[str], crop: str | None,
                  soil: dict[str, Any], elevation_m: float | None = None) -> dict:
    """Build a request body. Empty soil fields are left out so the service can fill or flag them."""
    body: dict[str, Any] = {"district": district, "season": season, "top_k": top_k, "include": include}
    if crop:
        body["crop"] = crop
    if elevation_m is not None:
        body["elevation_m"] = elevation_m
    clean = {k: v for k, v in soil.items() if v not in (None, "")}
    if clean:
        body["soil"] = clean
    return body
