"""
nasa_power_fetcher.py
---------------------
Fetches daily NASA POWER climate data (2017-2025) for four pilot districts,
aggregates by Rwanda agricultural season (A/B/C), and writes a long-format CSV.

District centroids from rwa_adm2_2006_NISR_WGS1984_20181002 shapefile (OCHA).

Output: data/raw/nasa_power/nasa_power_seasonal.csv
Schema: district, season, year, variable, value, unit, lat, lon
"""

import csv, json, time, urllib.request, urllib.error
from pathlib import Path
from datetime import date, timedelta

DISTRICTS = {
    "Nyagatare": {"lat": -1.338251, "lon": 30.379895},
    "Musanze":   {"lat": -1.498533, "lon": 29.606584},
    "Nyabihu":   {"lat": -1.647387, "lon": 29.510505},
    "Ruhango":   {"lat": -2.193599, "lon": 29.771734},
}

PARAMETERS = ["PRECTOTCORR", "T2M_MAX", "T2M_MIN", "ALLSKY_SFC_SW_DWN"]

AGG_METHOD = {
    "PRECTOTCORR":        "sum",
    "T2M_MAX":            "mean",
    "T2M_MIN":            "mean",
    "ALLSKY_SFC_SW_DWN":  "mean",
}

UNITS = {
    "PRECTOTCORR":        "mm",
    "T2M_MAX":            "degC",
    "T2M_MIN":            "degC",
    "ALLSKY_SFC_SW_DWN":  "MJ/m2/day",
}

# Season A: Sep Y -> Jan Y+1  (labelled by start year Y)
# Season B: Mar Y+1 -> Jun Y+1
# Season C: Jun Y+1 -> Aug Y+1
def season_dates(year):
    return [
        ("A", date(year,     9,  1), date(year + 1, 1, 31)),
        ("B", date(year + 1, 2,  1), date(year + 1, 6, 30)),
        ("C", date(year + 1, 6,  1), date(year + 1, 8, 31)),
    ]

START_YEARS = list(range(2017, 2025))   # seasons through Aug 2025
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "nasa_power"


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_nasa_power(lat, lon, start, end, parameters):
    url = (
        f"{NASA_POWER_BASE}?parameters={','.join(parameters)}"
        f"&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}&format=JSON"
    )
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read())
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                raise RuntimeError(f"NASA POWER failed after 3 attempts: {e}") from e
            print(f"    retry {attempt}/3 ..."); time.sleep(5)
    daily = data["properties"]["parameter"]
    return {
        p: {k: (None if v in (-999.0, -99.0, -999, -99) else v)
            for k, v in vals.items()}
        for p, vals in daily.items()
    }


def aggregate_season(daily, start, end, param, method):
    vals = [daily.get(param, {}).get(d.strftime("%Y%m%d"))
            for d in daterange(start, end)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) if method == "sum" else sum(vals) / len(vals)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "nasa_power_seasonal.csv"
    rows = []

    for district, coords in DISTRICTS.items():
        lat, lon = coords["lat"], coords["lon"]
        print(f"\n{district}  ({lat}, {lon})")
        for start_year in START_YEARS:
            fetch_start = date(start_year, 9, 1)
            fetch_end   = date(start_year + 1, 8, 31)
            print(f"  {fetch_start} to {fetch_end} ...", end=" ", flush=True)
            try:
                daily = fetch_nasa_power(lat, lon, fetch_start, fetch_end, PARAMETERS)
                print("OK")
            except RuntimeError as e:
                print(f"FAILED: {e}"); continue
            for season_label, s_start, s_end in season_dates(start_year):
                for param in PARAMETERS:
                    val = aggregate_season(daily, s_start, s_end, param, AGG_METHOD[param])
                    rows.append({
                        "district": district, "season": season_label,
                        "year": start_year, "variable": param,
                        "value": round(val, 4) if val is not None else "",
                        "unit": UNITS[param], "lat": lat, "lon": lon,
                    })
            time.sleep(1)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["district","season","year","variable","value","unit","lat","lon"])
        writer.writeheader(); writer.writerows(rows)

    expected = len(DISTRICTS) * len(START_YEARS) * 3 * len(PARAMETERS)
    print(f"\nWrote {len(rows)} rows to {out_path}  (expected {expected})")
    if len(rows) != expected:
        print("WARNING: row count mismatch, check for fetch errors above")

if __name__ == "__main__":
    main()
