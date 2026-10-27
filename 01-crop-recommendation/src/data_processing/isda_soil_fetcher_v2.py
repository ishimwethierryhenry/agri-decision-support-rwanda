"""
isda_soil_fetcher_v2.py
Agri-DSS Practicum, CMU Africa Fall 2026
Author: Arsene (Part A, data ingestion)

Fetches soil properties from iSDAsoil REST API v2 for each CIAT point.
- Authenticates with JWT token (expires 1 hour, refreshes every 55 minutes)
- Resumes from checkpoint if interrupted
- Output: data/raw/isda/isda_soil_points_v2.csv

Usage:
  python3 src/ingestion/isda_soil_fetcher_v2.py \
    --points data/raw/ciat/ciat_points_clean.csv \
    --email amugisha@andrew.cmu.edu \
    --password 'Umutekano@123'
"""

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime

# API endpoints
LOGIN_URL   = 'https://api.isda-africa.com/login'
SOIL_URL    = 'https://api.isda-africa.com/isdasoil/v2/soilproperty'

# Properties and depths for v2 API
PROPERTIES = [
    'nitrogen_total',
    'phosphorous_extractable',
    'potassium_extractable',
    'ph',
    'sand_content',
    'clay_content',
    'silt_content',
    'carbon_organic',
    'cation_exchange_capacity',
]
DEPTHS = ['0-20', '20-50']

# Column name mapping for output CSV
COL_MAP = {
    'nitrogen_total':          'nitrogen',
    'phosphorous_extractable': 'phosphorous_extractable',
    'potassium_extractable':   'potassium_extractable',
    'ph':                      'ph',
    'sand_content':            'sand',
    'clay_content':            'clay',
    'silt_content':            'silt',
    'carbon_organic':          'carbon_organic',
    'cation_exchange_capacity':'cation_exchange_capacity',
}

TOKEN_REFRESH_SECS = 55 * 60  # refresh 5 minutes before expiry
RETRY_WAIT  = 3
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.2


def get_token(email, password):
    data = urllib.parse.urlencode({'username': email, 'password': password}).encode()
    req = urllib.request.Request(
        LOGIN_URL, data=data,
        headers={'Content-Type': 'application/x-www-form-urlencoded', 'accept': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read())
    token = result.get('access_token')
    if not token:
        raise ValueError(f'Login failed: {result}')
    print(f'[{datetime.now():%H:%M:%S}] Token refreshed.')
    return token, time.time()


def fetch_property(lat, lon, prop, depth, token):
    url = f'{SOIL_URL}?lat={lat}&lon={lon}&property={prop}&depth={depth}'
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}', 'accept': 'application/json'})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            entries = data.get('property', {}).get(prop, [])
            if entries:
                return entries[0]['value']['value']
            return None
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise  # token expired, let caller handle
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_WAIT)
        except Exception:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_WAIT)


def build_output_columns():
    cols = ['point_id', 'latitude', 'longitude']
    for prop in PROPERTIES:
        base = COL_MAP[prop]
        for depth in DEPTHS:
            suffix = depth.replace('-', '_')
            cols.append(f'{base}_{suffix}cm')
    return cols


def load_checkpoint(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row['point_id'])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points',   required=True, help='CIAT points CSV')
    parser.add_argument('--email',    required=True, help='iSDA account email')
    parser.add_argument('--password', required=True, help='iSDA account password')
    parser.add_argument('--out', default='data/raw/isda/isda_soil_points_v2.csv')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = build_output_columns()

    # Load CIAT points
    import pandas as pd
    ciat = pd.read_csv(args.points)
    total = len(ciat)
    print(f'Total CIAT points: {total}')

    # Resume from checkpoint
    done = load_checkpoint(args.out)
    print(f'Already done: {len(done)}, remaining: {total - len(done)}')

    # Get initial token
    token, token_time = get_token(args.email, args.password)

    write_header = not os.path.exists(args.out) or len(done) == 0
    mode = 'w' if write_header else 'a'

    with open(args.out, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()

        for i, row in ciat.iterrows():
            point_id = str(row['F_index'])
            if point_id in done:
                continue

            lat = row['latitude']
            lon = row['longitude']

            # Refresh token if needed
            if time.time() - token_time > TOKEN_REFRESH_SECS:
                token, token_time = get_token(args.email, args.password)

            out_row = {'point_id': point_id, 'latitude': lat, 'longitude': lon}

            for prop in PROPERTIES:
                base = COL_MAP[prop]
                for depth in DEPTHS:
                    suffix = depth.replace('-', '_')
                    col = f'{base}_{suffix}cm'
                    try:
                        val = fetch_property(lat, lon, prop, depth, token)
                    except urllib.error.HTTPError as e:
                        if e.code == 401:
                            print(f'Token expired mid-run, refreshing...')
                            token, token_time = get_token(args.email, args.password)
                            val = fetch_property(lat, lon, prop, depth, token)
                        else:
                            val = None
                    out_row[col] = val

                time.sleep(SLEEP_BETWEEN)

            writer.writerow(out_row)
            f.flush()

            done_count = len(done) + 1
            done.add(point_id)
            if done_count % 50 == 0 or done_count == 1:
                print(f'[{datetime.now():%H:%M:%S}] {done_count}/{total} points done')

    print(f'\nDone. Output: {args.out}')


if __name__ == '__main__':
    main()
