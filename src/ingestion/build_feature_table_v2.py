"""
build_feature_table_v2.py
Agri-DSS Practicum, CMU Africa Fall 2026
Author: Arsene (Part A, data ingestion)

Joins all four data sources into one feature table (v2).

Changes from v1:
  - NISR crop-specific statistics replaced with district-level crop share.
    v1 leaked the label (a Maize point got Maize yield stats). v2 gives every
    point in a district the same five crop-share columns, which are knowable
    before the crop is identified.
  - Crop_plant retained as an alternative target column (1,913 pure-crop labels
    vs 1,567 from landcover). Thierry to decide which to use for training.

Output: data/processed/feature_table_v2.csv
"""

import os
import sys
import pandas as pd

# Paths (relative to repo root agri-dss/)
CIAT_POINTS  = 'data/raw/ciat/ciat_points_clean.csv'
NASA_POWER   = 'data/raw/nasa_power/nasa_power_seasonal.csv'
NISR_CULT    = 'data/raw/nisr/nisr_cultivated_area_ha.csv'
ISDA_POINTS  = 'data/raw/isda/isda_soil_all.csv'
OUT_PATH     = 'data/processed/feature_table_v2.csv'

# CIAT collection: May-June 2025 = Season B
# NASA POWER start-year convention: Season B year=2024 covers Mar 2025 to Jun 2025
CIAT_SEASON = 'B'
CIAT_YEAR   = 2024

# The four crops in our model scope, mapped to NISR crop names
MODEL_CROPS = {
    'Maize':       'Maize',
    'Beans':       'Beans',
    'Irish potato':'Irish potato',
    'Paddy rice':  'Paddy rice',
}


def load_ciat():
    print('Loading CIAT points...')
    ciat = pd.read_csv(CIAT_POINTS)
    ciat['season'] = CIAT_SEASON
    ciat['year']   = CIAT_YEAR
    print(f'  Rows: {len(ciat):,}')
    print(f'  Labelled (landcover): {ciat["landcover"].notna().sum():,}')
    print(f'  Labelled (Crop_plant pure): {ciat["Crop_plant"].isin(["Maize","Beans","Irish potatoes","Rice"]).sum():,}')
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


def load_nisr_crop_share():
    """
    Compute district-level crop share from NISR cultivated area.
    For each district and season, calculate what fraction of the four
    model crops' total cultivated area each crop occupies.
    This is the same value for every point in a district, so it does
    not leak the label into the features.
    """
    print('Loading NISR cultivated area for crop share...')
    nisr = pd.read_csv(NISR_CULT)

    # Filter to model crops only
    nisr_model = nisr[nisr['crop'].isin(MODEL_CROPS.values())].copy()

    # Compute total area per district-season-year across the four model crops
    totals = nisr_model.groupby(['district', 'season', 'year'])['value'].sum().reset_index()
    totals = totals.rename(columns={'value': 'total_model_crop_area_ha'})

    # Pivot to wide: one column per crop share
    nisr_wide = nisr_model.merge(totals, on=['district', 'season', 'year'])
    nisr_wide['share'] = nisr_wide['value'] / nisr_wide['total_model_crop_area_ha']

    pivot = nisr_wide.pivot_table(
        index=['district', 'season', 'year'],
        columns='crop',
        values='share'
    ).reset_index()
    pivot.columns.name = None

    # Rename to clear column names
    rename = {
        'Maize':        'nisr_share_maize',
        'Beans':        'nisr_share_beans',
        'Irish potato': 'nisr_share_irish_potato',
        'Paddy rice':   'nisr_share_paddy_rice',
    }
    pivot = pivot.rename(columns=rename)
    pivot['nisr_total_model_crop_area_ha'] = pivot.merge(
        totals, on=['district', 'season', 'year']
    )['total_model_crop_area_ha']

    print(f'  District-season-year combinations: {len(pivot):,}')
    print(f'  Columns: {[c for c in pivot.columns if c not in ["district","season","year"]]}')
    return pivot


def load_isda():
    if not os.path.exists(ISDA_POINTS):
        print('  iSDAsoil file not found, skipping soil join.')
        print('  Run isda_soil_fetcher_v2.py and re-run this script.')
        return None
    print('Loading iSDAsoil point data...')
    isda = pd.read_csv(ISDA_POINTS)
    print(f'  Rows: {len(isda):,}  Columns: {len(isda.columns)}')
    return isda


def build(ciat, climate, nisr_share, isda):
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

    # Join 2: NISR crop share on district + season + year
    print('Join 2: + NISR crop share...')
    merged = merged.merge(
        nisr_share,
        left_on=['District', 'season', 'year'],
        right_on=['district', 'season', 'year'],
        how='left'
    ).drop(columns=['district'])
    share_cols = ['nisr_share_maize', 'nisr_share_beans',
                  'nisr_share_irish_potato', 'nisr_share_paddy_rice']
    share_nulls = merged[share_cols].isnull().sum().sum()
    print(f'  Shape: {merged.shape}  Crop share nulls: {share_nulls}')
    if share_nulls > 0:
        print('  WARNING: some district-season-year combinations missing from NISR')

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
    print('Building feature table v2')
    print('=' * 55)

    ciat       = load_ciat()
    climate    = load_climate()
    nisr_share = load_nisr_crop_share()
    isda       = load_isda()

    merged = build(ciat, climate, nisr_share, isda)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print(f'\nSaved: {OUT_PATH}')
    print(f'Final shape: {merged.shape}')
    print(f'Labelled rows (landcover): {merged["landcover"].notna().sum():,}')
    print(f'Labelled rows (Crop_plant pure crops): {merged["Crop_plant"].isin(["Maize","Beans","Irish potatoes","Rice"]).sum():,}')
    print(f'Columns: {len(merged.columns)}')
    print('\nDone.')


if __name__ == '__main__':
    main()
