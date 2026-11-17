"""
build_feature_table.py
Agri-DSS Practicum, CMU Africa Fall 2026
Author: Arsene (Part A, data ingestion)

Joins all four data sources into one feature table:
  1. CIAT fieldwork points (base, 2,739 rows)
  2. NASA POWER seasonal climate (district + season + year)
  3. NISR crop statistics (district + season + year + crop)
  4. iSDAsoil soil properties (point level, if available)

Output: data/processed/feature_table.csv

Crop mapping decision (Option 1 - primary crop):
  Intercrop plots are mapped to the dominant/higher-value crop so every
  labelled point gets NISR statistics. Rationale: preserves all 1,567
  labelled rows for training; the intercrop label is retained in the
  'landcover' column so Thierry can group or split as needed.
  Decision documented for Prof. Okeyo review.
"""

import os
import sys
import pandas as pd

# Paths (relative to repo root agri-dss/)
CIAT_POINTS  = 'data/raw/ciat/ciat_points_clean.csv'
NASA_POWER   = 'data/raw/nasa_power/nasa_power_seasonal.csv'
NISR_YIELD   = 'data/raw/nisr/nisr_yield_kg_ha.csv'
NISR_PROD    = 'data/raw/nisr/nisr_production_mt.csv'
NISR_HARV    = 'data/raw/nisr/nisr_harvested_area_ha.csv'
NISR_CULT    = 'data/raw/nisr/nisr_cultivated_area_ha.csv'
ISDA_POINTS  = 'data/raw/isda/isda_soil_points.csv'
OUT_PATH     = 'data/processed/feature_table.csv'

# CIAT collection: May-June 2025 = Season B
# NASA POWER start-year convention: Season B year=2024 covers Mar-Jun 2025
CIAT_SEASON = 'B'
CIAT_YEAR   = 2024

# Intercrop mapped to primary/dominant crop (Option 1)
# Pure crops mapped to their NISR name
CROP_MAP = {
    'Maize':                  'Maize',
    'Bean':                   'Beans',
    'Rice':                   'Paddy rice',
    'Irish potato':           'Irish potato',
    'Irish Potato':           'Irish potato',
    # Intercrop: mapped to dominant crop
    'Bean and Maize':         'Maize',
    'Maize and Bean':         'Maize',
    'Irish Potato and Maize': 'Irish potato',
    'Irish potato and Maize': 'Irish potato',
    'Bean and Irish potato':  'Beans',
    'Cassava and Maize':      'Maize',
}


def load_ciat():
    print('Loading CIAT points...')
    ciat = pd.read_csv(CIAT_POINTS)
    ciat['season'] = CIAT_SEASON
    ciat['year'] = CIAT_YEAR
    ciat['nisr_crop'] = ciat['landcover'].map(CROP_MAP)
    print(f'  Rows: {len(ciat):,}')
    print(f'  Labelled: {ciat["landcover"].notna().sum():,}')
    print(f'  Unlabelled (no crop label): {ciat["landcover"].isna().sum():,}')
    return ciat


def load_climate():
    print('Loading NASA POWER climate...')
    climate = pd.read_csv(NASA_POWER)
    climate_wide = climate.pivot_table(
        index=['district', 'season', 'year'],
        columns='variable',
        values='value'
    ).reset_index()
    climate_wide.columns.name = None
    climate_wide = climate_wide.rename(columns={
        'PRECTOTCORR':       'precip_mm',
        'T2M_MAX':           'tmax_c',
        'T2M_MIN':           'tmin_c',
        'ALLSKY_SFC_SW_DWN': 'solar_mj_m2',
    })
    print(f'  District-season-year combinations: {len(climate_wide):,}')
    return climate_wide


def load_nisr():
    print('Loading NISR crop statistics...')
    nisr = pd.read_csv(NISR_YIELD).rename(columns={'value': 'nisr_yield_kg_ha'})
    nisr = nisr.merge(
        pd.read_csv(NISR_PROD).rename(columns={'value': 'nisr_production_mt'}),
        on=['year', 'season', 'district', 'crop'], how='outer'
    )
    nisr = nisr.merge(
        pd.read_csv(NISR_HARV).rename(columns={'value': 'nisr_harvested_area_ha'}),
        on=['year', 'season', 'district', 'crop'], how='outer'
    )
    nisr = nisr.merge(
        pd.read_csv(NISR_CULT).rename(columns={'value': 'nisr_cultivated_area_ha'}),
        on=['year', 'season', 'district', 'crop'], how='outer'
    )
    print(f'  Rows: {len(nisr):,}')
    return nisr


def load_isda():
    if not os.path.exists(ISDA_POINTS):
        print('  iSDAsoil file not found, skipping soil join.')
        print('  Run isda_soil_fetcher.py overnight and re-run this script.')
        return None
    print('Loading iSDAsoil point data...')
    isda = pd.read_csv(ISDA_POINTS)
    print(f'  Rows: {len(isda):,}  Columns: {len(isda.columns)}')
    return isda


def build(ciat, climate, nisr, isda):
    # Join 1: climate on district + season + year
    print('\nJoin 1: CIAT + climate...')
    merged = ciat.merge(
        climate,
        left_on=['District', 'season', 'year'],
        right_on=['district', 'season', 'year'],
        how='left'
    ).drop(columns=['district'])
    climate_cols = ['precip_mm', 'tmax_c', 'tmin_c', 'solar_mj_m2']
    nulls = merged[climate_cols].isnull().sum().sum()
    print(f'  Shape: {merged.shape}  Climate nulls: {nulls}')
    if nulls > 0:
        print('  WARNING: unexpected nulls in climate columns, check season/year alignment')

    # Join 2: NISR on district + season + year + crop
    print('Join 2: + NISR statistics...')
    merged = merged.merge(
        nisr,
        left_on=['District', 'season', 'year', 'nisr_crop'],
        right_on=['district', 'season', 'year', 'crop'],
        how='left'
    ).drop(columns=['district', 'crop'])
    nisr_cols = ['nisr_yield_kg_ha', 'nisr_production_mt', 'nisr_harvested_area_ha', 'nisr_cultivated_area_ha']
    nisr_nulls = merged[nisr_cols].isnull().sum()
    print(f'  Shape: {merged.shape}')
    print(f'  NISR nulls (should equal unlabelled rows): {nisr_nulls["nisr_yield_kg_ha"]:,}')

    # Join 3: iSDAsoil at point level
    if isda is not None:
        print('Join 3: + iSDAsoil soil properties...')
        merged = merged.merge(
            isda,
            left_on='F_index',
            right_on='point_id',
            how='left'
        ).drop(columns=['point_id'])
        soil_cols = [c for c in merged.columns if any(
            c.startswith(p) for p in ['nitrogen', 'phosphorous', 'potassium', 'ph_',
                                       'sand_', 'clay_', 'silt_', 'carbon_organic', 'cation_']
        )]
        soil_nulls = merged[soil_cols].isnull().sum().sum()
        print(f'  Shape: {merged.shape}  Soil nulls: {soil_nulls}')
    else:
        print('Join 3: iSDAsoil skipped (file not available)')

    return merged


def main():
    print('=' * 55)
    print('Building feature table')
    print('=' * 55)

    ciat    = load_ciat()
    climate = load_climate()
    nisr    = load_nisr()
    isda    = load_isda()

    merged = build(ciat, climate, nisr, isda)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print(f'\nSaved: {OUT_PATH}')
    print(f'Final shape: {merged.shape}')
    print(f'Labelled rows (has landcover): {merged["landcover"].notna().sum():,}')
    print(f'Columns: {len(merged.columns)}')
    print('\nDone.')


if __name__ == '__main__':
    main()
