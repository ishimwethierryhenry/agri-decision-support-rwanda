"""
chirps_fetcher.py
Agri-DSS Practicum, CMU Africa Fall 2026
Author: Arsene (Part A, data ingestion)

Fetches Season B 2025 total rainfall (mm) from CHIRPS via the ClimateSERV API
for each CIAT fieldwork point.

Season B 2025 window: 2025-03-01 to 2025-06-30 (inclusive)
  - CHIRPS resolution: ~5 km
  - ClimateSERV API: https://climateserv.servirglobal.net/api  (no auth required)
  - Each request submits a job for one point, then polls until complete.
  - Runtime estimate: ~8 s per point => ~6 hours for 2,739 points.
  - Use --test-n 10 for a quick sanity check before the full run.

Output: data/raw/chirps/chirps_seasonal_b2025.csv
  Columns: point_id, latitude, longitude, chirps_precip_mm

Usage (run from repo root: cd agri-dss/):
  # Quick test (10 points)
  python3 src/data_processing/chirps_fetcher.py \\
      --points data/raw/ciat/ciat_points_clean.csv \\
      --test-n 10

  # Full run (resumes from checkpoint automatically)
  python3 src/data_processing/chirps_fetcher.py \\
      --points data/raw/ciat/ciat_points_clean.csv

NOTES:
  - GPS coordinates are read from the input file and are NEVER written to the
    repo (ciat_points_clean.csv is in .gitignore).
  - The output file contains lat/lon for the join but should not be committed
    directly; commit only chirps_precip_mm joined into feature_table_v2.csv.
  - Checkpoint/resume: if interrupted, re-run the same command. Points already
    written to the output file are skipped.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

# ---------------------------------------------------------------------------
# ClimateSERV API config
# ---------------------------------------------------------------------------

BASE_URL     = 'https://climateserv.servirglobal.net/chirps'
SUBMIT_URL   = BASE_URL + '/submitDataRequest/'
PROGRESS_URL = BASE_URL + '/getDataRequestProgress/'
DATA_URL     = BASE_URL + '/getDataFromRequest/'

# CHIRPS daily precipitation, dataset type 26 (CHIRPS_DAILY)
# ClimateSERV dataset type list: https://climateserv.servirglobal.net/api/getDatasetTypes/
DATASET_TYPE  = 26          # CHIRPS daily
OPERATION     = 5           # Seasonal total (sum)

SEASON_START  = '03/01/2025'   # MM/DD/YYYY as required by ClimateSERV
SEASON_END    = '06/30/2025'

POLL_INTERVAL = 5           # seconds between progress polls
POLL_TIMEOUT  = 600         # give up on a single point after this many seconds
MAX_RETRIES   = 3           # retries on submit/retrieve failures
RETRY_WAIT    = 5

SLEEP_BETWEEN = 1.0         # be polite: 1 second between points


# ---------------------------------------------------------------------------
# ClimateSERV API calls
# ---------------------------------------------------------------------------

def submit_request(lat, lon):
    """
    Submit a data request for a 0.05-degree bounding box around the point.
    ClimateSERV needs a geometry; a tiny box around the point gives one pixel.
    Returns the job_id string, or raises on error.
    """
    delta = 0.025   # half-width of bounding box in degrees (~2.5 km)
    geom = {
        'type': 'Polygon',
        'coordinates': [[
            [lon - delta, lat - delta],
            [lon + delta, lat - delta],
            [lon + delta, lat + delta],
            [lon - delta, lat + delta],
            [lon - delta, lat - delta],
        ]]
    }

    params = {
        'datatype':         DATASET_TYPE,
        'begintime':        SEASON_START,
        'endtime':          SEASON_END,
        'intervaltype':     0,          # 0 = daily
        'operationtype':    OPERATION,
        'geometry':         json.dumps(geom),
        'isZip_CurrentDataType': False,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            url = SUBMIT_URL + '?' + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=30) as r:
                body = json.loads(r.read())
            # Response is a list; first element is the job_id
            if isinstance(body, list) and len(body) > 0:
                return str(body[0])
            raise ValueError(f'Unexpected submit response: {body}')
        except (urllib.error.URLError, ValueError) as e:
            if attempt == MAX_RETRIES:
                raise
            print(f'  submit attempt {attempt} failed: {e}, retrying...')
            time.sleep(RETRY_WAIT)


def poll_progress(job_id, debug=False):
    """
    Poll until job is complete (progress == 100) or timeout.
    Returns True if complete, False if timed out.
    """
    url = PROGRESS_URL + '?' + urllib.parse.urlencode({'id': job_id})
    deadline = time.time() + POLL_TIMEOUT
    poll_n = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
                body = json.loads(raw)
            if debug:
                print(f'    poll {poll_n}: raw={raw[:200]}')
            # Response: list with one item, e.g. [{"progress": 100}]
            if isinstance(body, list) and body:
                item = body[0]
                # API returns plain float: [0.0] or [100.0]
                if isinstance(item, (int, float)):
                    progress = float(item)
                else:
                    progress = float(item.get('progress', 0))
                if debug:
                    print(f'    poll {poll_n}: progress={progress}')
                if progress >= 100:
                    return True
                if progress < 0:
                    print(f'  Job {job_id} returned error progress: {progress}')
                    return False
            poll_n += 1
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            if debug:
                print(f'    poll {poll_n} exception: {e}')
            poll_n += 1
            time.sleep(POLL_INTERVAL)
    return False


def retrieve_data(job_id):
    """
    Retrieve the computed result. Returns the seasonal total (mm) as float,
    or None if the result is empty or malformed.
    """
    url = DATA_URL + '?' + urllib.parse.urlencode({'id': job_id})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                body = json.loads(r.read())
            # Response schema:
            # [{"data": [{"date": "...", "value": [{"avg": ..., "max": ..., "min": ..., "sum": ...}]}]}]
            if not isinstance(body, list) or len(body) == 0:
                return None
            data_entries = body[0].get('data', [])
            if not data_entries:
                return None
            # Operation 5 = sum: use 'sum' field; fall back to 'avg' if missing
            value_obj = data_entries[0].get('value', [{}])[0]
            total = value_obj.get('sum') if value_obj.get('sum') is not None else value_obj.get('avg')
            return float(total) if total is not None else None
        except (urllib.error.URLError, ValueError, KeyError, IndexError) as e:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_WAIT)
    return None


def fetch_chirps_for_point(lat, lon, debug=False):
    """
    Full pipeline for one point: submit -> poll -> retrieve.
    Returns seasonal total rainfall in mm, or None on failure.
    """
    job_id = submit_request(lat, lon)
    completed = poll_progress(job_id, debug=debug)
    if not completed:
        print(f'  TIMEOUT waiting for job {job_id}')
        return None
    return retrieve_data(job_id)


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def load_checkpoint(out_path):
    """Return set of point_id strings already written to output."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and 'point_id' in reader.fieldnames:
                for row in reader:
                    done.add(row['point_id'])
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Fetch CHIRPS Season B 2025 rainfall for CIAT points via ClimateSERV.'
    )
    parser.add_argument(
        '--points', required=True,
        help='Path to CIAT points CSV (must have columns: F_index, latitude, longitude)'
    )
    parser.add_argument(
        '--out', default='data/raw/chirps/chirps_seasonal_b2025.csv',
        help='Output CSV path (default: data/raw/chirps/chirps_seasonal_b2025.csv)'
    )
    parser.add_argument(
        '--test-n', type=int, default=None,
        help='Limit to first N points (for testing before full run)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.points):
        print(f'ERROR: points file not found: {args.points}')
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load CIAT points
    import pandas as pd
    ciat = pd.read_csv(args.points)

    # Validate required columns
    required = ['F_index', 'latitude', 'longitude']
    missing = [c for c in required if c not in ciat.columns]
    if missing:
        # Try fallback column names
        col_map_try = {}
        for need in missing:
            if need == 'F_index':
                for alt in ['F_id', 'fid', 'FID', 'id', 'ID', 'point_id']:
                    if alt in ciat.columns:
                        col_map_try[alt] = 'F_index'
                        break
            elif need == 'latitude':
                for alt in ['lat', 'y', 'Y', 'Latitude']:
                    if alt in ciat.columns:
                        col_map_try[alt] = 'latitude'
                        break
            elif need == 'longitude':
                for alt in ['lon', 'lng', 'x', 'X', 'Longitude']:
                    if alt in ciat.columns:
                        col_map_try[alt] = 'longitude'
                        break
        if col_map_try:
            ciat = ciat.rename(columns=col_map_try)
            still_missing = [c for c in required if c not in ciat.columns]
            if still_missing:
                print(f'ERROR: cannot find columns {still_missing} in {args.points}')
                print(f'  Available: {list(ciat.columns)}')
                sys.exit(1)
        else:
            print(f'ERROR: missing required columns {missing} in {args.points}')
            print(f'  Available: {list(ciat.columns)}')
            sys.exit(1)

    total = len(ciat)
    if args.test_n:
        ciat = ciat.head(args.test_n)
        print(f'TEST MODE: running on first {args.test_n} of {total} points')
        total = len(ciat)
    else:
        print(f'Total CIAT points: {total}')

    # Resume from checkpoint
    done = load_checkpoint(args.out)
    remaining = total - len(done)
    print(f'Already done: {len(done)}, remaining: {remaining}')
    if remaining == 0:
        print('All points already fetched. Nothing to do.')
        sys.exit(0)

    est_hours = remaining * (POLL_INTERVAL * 5 + SLEEP_BETWEEN) / 3600
    print(f'Estimated runtime: {est_hours:.1f} hours (at ~{POLL_INTERVAL * 5 + SLEEP_BETWEEN:.0f} s/point)')
    print(f'Season: {SEASON_START} to {SEASON_END}')
    print(f'Output: {args.out}')
    print()

    out_cols = ['point_id', 'latitude', 'longitude', 'chirps_precip_mm']
    write_header = not os.path.exists(args.out) or len(done) == 0
    mode = 'w' if write_header else 'a'

    success_count = 0
    fail_count = 0

    with open(args.out, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        if write_header:
            writer.writeheader()

        for i, row in ciat.iterrows():
            point_id = str(row['F_index'])
            if point_id in done:
                continue

            lat = float(row['latitude'])
            lon = float(row['longitude'])

            # First point runs in debug mode so we can see the raw API response
            debug_mode = (len(done) == 0 and success_count == 0 and fail_count == 0)
            try:
                val = fetch_chirps_for_point(lat, lon, debug=debug_mode)
            except Exception as e:
                print(f'  [{point_id}] FAILED: {e}')
                val = None
                fail_count += 1

            if val is not None:
                success_count += 1

            writer.writerow({
                'point_id':        point_id,
                'latitude':        lat,
                'longitude':       lon,
                'chirps_precip_mm': round(val, 2) if val is not None else '',
            })
            f.flush()

            done.add(point_id)
            done_total = len(done)

            if done_total % 50 == 0 or done_total == 1 or done_total == total:
                null_rate = fail_count / done_total * 100
                print(
                    f'[{datetime.now():%H:%M:%S}] {done_total}/{total} done '
                    f'| ok={success_count} null={fail_count} ({null_rate:.1f}%)'
                )

            time.sleep(SLEEP_BETWEEN)

    print(f'\nDone. {success_count} values written, {fail_count} nulls.')
    print(f'Output: {args.out}')

    # Quick summary stats
    if success_count > 0:
        try:
            import pandas as pd
            out_df = pd.read_csv(args.out)
            vals = out_df['chirps_precip_mm'].dropna()
            print(f'\nchirps_precip_mm summary (n={len(vals)}):')
            print(f'  mean  = {vals.mean():.1f} mm')
            print(f'  min   = {vals.min():.1f} mm')
            print(f'  max   = {vals.max():.1f} mm')
            print(f'  nulls = {out_df["chirps_precip_mm"].isna().sum()}')
        except Exception:
            pass


if __name__ == '__main__':
    main()
