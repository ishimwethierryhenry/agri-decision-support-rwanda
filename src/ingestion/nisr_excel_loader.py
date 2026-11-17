"""
NISR SAS Excel batch loader (2017-2025).

Processes all NISR SAS annual Excel/XLS files in data/raw/nisr/ and appends
records into the four long-format CSVs produced by nisr_parser_v2.py:

  data/raw/nisr/nisr_cultivated_area_ha.csv
  data/raw/nisr/nisr_harvested_area_ha.csv
  data/raw/nisr/nisr_yield_kg_ha.csv
  data/raw/nisr/nisr_production_mt.csv

Each CSV uses columns: year, season, district, crop, value

Layout notes discovered by inspection:

  2018 (Appendix 1-11): rows = crops, cols = districts (TRANSPOSED)
    Appendix 1/2/3   cultivated area  A/B/C
    Appendix 4/5     harvested area   A/B
    Appendix 6/7/8   production       A/B/C
    Appendix 9/10/11 yield            A/B/C

  2023 (Table 10-24): rows = districts, cols = crops
    Table 10-12   cultivated area  A/B/C
    Table 13-15   harvested area   A/B/C
    Table 16-18   yield            A/B/C
    Table 19-21   SKIP (LSF breakdown)
    Table 22-24   production       A/B/C

  2024 (Table 10-23): rows = districts, cols = crops
    Table 10-12   cultivated area  A/B/C
    Table 13-15   harvested area   A/B/C
    Table 16-18   yield            A/B/C
    Table 19-20   SKIP (LSF breakdown)
    Table 21-23   production       A/B/C

  2025 (Table 13-26): rows = districts, cols = crops (numbering shifted +3)
    Table 13-15   cultivated area  A/B/C
    Table 16-18   harvested area   A/B/C
    Table 19-21   yield            A/B/C
    Table 22-23   SKIP (LSF breakdown)
    Table 24-26   production       A/B/C

  2019 (Table 7-20): rows = districts, cols = crops
    Table 7-9    cultivated area  A/B/C
    Table 10-12  harvested area   A/B/C
    Table 13-15  yield            A/B/C
    Table 16-17  SKIP (LSF yield A/B)
    Table 18-20  production       A/B/C

  2020 (Table 8-22): rows = districts, cols = crops; some sheet names have trailing spaces
    Table 8/9 /10   cultivated area  A/B/C
    Table 11/12 /13 harvested area   A/B/C
    Table 14/15 /16 yield            A/B/C
    Table 17/18 /19 SKIP (LSF yield)
    Table 20/21/22  production       A/B/C

  2021 (Table 11-26): rows = districts, cols = crops; some sheet names have trailing spaces
    Table 11/12 /13 cultivated area  A/B/C
    Table 14/15 /16 harvested area   A/B/C
    Table 17/18     yield            A/B
    Table 19        SKIP (detailed banana/beans)
    Table 20        yield            C
    Table 21/22 /23 SKIP (LSF yield)
    Table 24/25/26  production       A/B/C

  2022 (Table 11-25): rows = districts, cols = crops; some sheet names have trailing spaces
    Table 11/12 /13 cultivated area  A/B/C
    Table 14/15 /16 harvested area   A/B/C
    Table 17/18 /19 yield            A/B/C
    Table 20/21 /22 SKIP (LSF yield)
    Table 23/24/25  production       A/B/C

  2017: only Season C has district-level crop tables (A/B are province-level only)
    Table 96  cultivated area  C
    Table 97  harvested area   C  (inferred from pattern)
    Table 99  yield            C
    Table 100 production       C

Usage (run from project root):
  python -I src/ingestion/nisr_excel_loader.py [options]

Options:
  --dry-run      Parse and print counts; do not write CSVs
  --year YYYY    Process only this year
  --inspect      Print sheet names and first rows; stop before writing
  --no-pdf-year  Skip years whose data already exist in the CSVs
"""

import re
import sys
import argparse
import pathlib
import warnings
import pandas as pd
import openpyxl

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
NISR_DIR = PROJECT_ROOT / "data" / "raw" / "nisr"

OUT_FILES = {
    "cultivated_area_ha": NISR_DIR / "nisr_cultivated_area_ha.csv",
    "harvested_area_ha":  NISR_DIR / "nisr_harvested_area_ha.csv",
    "yield_kg_ha":        NISR_DIR / "nisr_yield_kg_ha.csv",
    "production_mt":      NISR_DIR / "nisr_production_mt.csv",
}

# ---------------------------------------------------------------------------
# Year detection from filename
# ---------------------------------------------------------------------------
YEAR_PATTERNS = [
    (re.compile(r"USAS.?2017", re.I),  2017),
    (re.compile(r"SAS.?2018",  re.I),  2018),
    (re.compile(r"SAS.?2019",  re.I),  2019),
    (re.compile(r"USAS.?2020", re.I),  2020),
    (re.compile(r"SAS.?2021",  re.I),  2021),
    (re.compile(r"SAS.?2022",  re.I),  2022),
    (re.compile(r"SAS.?2023",  re.I),  2023),
    (re.compile(r"SAS.?2024",  re.I),  2024),
    (re.compile(r"SAS.?2025",  re.I),  2025),
]


def get_year(filepath):
    for pattern, year in YEAR_PATTERNS:
        if pattern.search(filepath.name):
            return year
    return None


# ---------------------------------------------------------------------------
# District and crop name normalisation
# ---------------------------------------------------------------------------
RWANDA_DISTRICTS = {
    "nyarugenge", "gasabo", "kicukiro",
    "nyanza", "gisagara", "nyaruguru", "huye", "nyamagabe",
    "ruhango", "muhanga", "kamonyi",
    "karongi", "rutsiro", "rubavu", "nyabihu", "ngororero",
    "rusizi", "nyamasheke",
    "rulindo", "gakenke", "musanze", "burera", "gicumbi",
    "rwamagana", "nyagatare", "gatsibo", "kayonza", "kirehe",
    "ngoma", "bugesera",
}

# Canonical crop names matching nisr_parser_v2.py output
# Value = None means skip this column/row (aggregate or total)
CROP_ALIASES = {
    "maize":              "Maize",
    "sorghum":            "Sorghum",
    "paddy rice":         "Paddy rice",
    "paddy":              "Paddy rice",
    "rice":               "Paddy rice",
    "wheat":              "Wheat",
    "other cereals":      "Other cereals",
    "cassava":            "Cassava",
    "sweet potato":       "Sweet potato",
    "sweet potatoes":     "Sweet potato",
    "irish potato":       "Irish potato",
    "irish potatoes":     "Irish potato",
    "yams & taro":        "Yams & Taro",
    "taro &yams":         "Yams & Taro",
    "taro & yams":        "Yams & Taro",
    "yams":               "Yams & Taro",
    "taro":               "Yams & Taro",
    "banana":             "Banana",
    "bananas":            "Banana",
    "cooking banana":     "Cooking banana",
    "cooking bananas":    "Cooking banana",
    "dessert banana":     "Dessert banana",
    "dessert bananas":    "Dessert banana",
    "banana for beer":    "Banana for beer",
    "bananas for beer":   "Banana for beer",
    "beans":              "Beans",
    "bean":               "Beans",
    "beans ":             "Beans",
    "bush bean":          "Bush bean",
    "bush beans":         "Bush bean",
    "climbing bean":      "Climbing bean",
    "climbing beans":     "Climbing bean",
    "pea":                "Pea",
    "peas":               "Pea",
    "groundnut":          "Groundnut",
    "groundnuts":         "Groundnut",
    "soybean":            "Soybean",
    "soybeans":           "Soybean",
    "soya":               "Soybean",
    "soya bean":          "Soybean",
    "soya beans":         "Soybean",
    "vegetables":         "Vegetables",
    "fruits":             "Fruits",
    "fruit":              "Fruits",
    "fodder crops":       "Fodder Crops",
    "fodder":             "Fodder Crops",
    "other crops":        "Other crops",
    # Aggregates to skip
    "cereals":            None,
    "cereals ":           None,
    "tubers and roots":   None,
    "tubers & roots":     None,
    "legumes and pulses": None,
    "legumes & pulses":   None,
    "pulses":             None,
    "banana (total)":     None,
    "developed land":     None,
    "total developed land": None,
    "total":              None,
    "kigali city":        None,  # province aggregate in some tables
    "kigali city ":       None,
}


def normalise_crop(val):
    """Return canonical crop name, or None to skip this column/row."""
    if val is None:
        return None
    s = str(val).strip().lower()
    s = " ".join(s.split())   # collapse internal whitespace / newlines
    return CROP_ALIASES.get(s, None)


def is_district(val):
    if val is None:
        return False
    return str(val).strip().lower() in RWANDA_DISTRICTS


def normalise_district(val):
    return str(val).strip().title()


def clean_num(val):
    """Convert a cell value to float or None."""
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        return None if v != v else v   # NaN check
    s = str(val).strip()
    if s.startswith("="):
        return None   # formula string not recalculated
    s = s.replace(",", "").replace("\xa0", "")
    s = re.sub(r"[–—\-]+$", "", s).strip()
    if not s or s in ("-", "–", "—", "n/a", "na"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Sheet content classifier
# Reads the title cell (row 1 or 2) to extract metric and season.
# ---------------------------------------------------------------------------
METRIC_RE = [
    (re.compile(r"cultivat", re.I),   "cultivated_area_ha"),
    (re.compile(r"harvest",  re.I),   "harvested_area_ha"),
    (re.compile(r"\byield\b|productivity", re.I), "yield_kg_ha"),
    (re.compile(r"production|output", re.I),      "production_mt"),
]

SEASON_RE = [
    (re.compile(r"season[_\s]*a\b|[_\s]sa[_\s]|[_\s]a_", re.I), "A"),
    (re.compile(r"season[_\s]*b\b|[_\s]sb[_\s]|[_\s]b_", re.I), "B"),
    (re.compile(r"season[_\s]*c\b|[_\s]sc[_\s]|[_\s]c_", re.I), "C"),
]


def classify_title(title_text):
    """Return (metric, season) or (None, None)."""
    metric = None
    for pat, m in METRIC_RE:
        if pat.search(title_text):
            metric = m
            break
    season = None
    for pat, s in SEASON_RE:
        if pat.search(title_text):
            season = s
            break
    return metric, season


def get_title(ws):
    """Return the first non-empty cell text from the first 4 rows."""
    for row in ws.iter_rows(min_row=1, max_row=4, values_only=True):
        for cell in row:
            if cell is not None:
                s = str(cell).strip()
                if len(s) > 10:
                    return s
    return ""


# ---------------------------------------------------------------------------
# Layout detection
# A sheet is "district-as-rows" if the first data column contains district names.
# It is "district-as-cols" if the first data row contains district names (transposed).
# ---------------------------------------------------------------------------

def get_header_and_data(ws, data_only=False):
    """
    Read all rows from ws into a list of lists.
    If data_only=True, the workbook should already have been opened with data_only=True.
    """
    rows = []
    for row in ws.iter_rows(values_only=True):
        rows.append(list(row))
    return rows


def detect_orientation(rows):
    """
    Detects layout and leading-column offset.

    Two layouts:
      rows_are_districts: districts in the label column, crops in the header row.
        Label column may be 0 or 1 (if col 0 is blank, as in 2024 files).
      rows_are_crops: crops in label col 0, districts as column headers in col 1+.

    Returns (orientation_string, label_col_offset) or (None, 0).
    """
    for r_idx, row in enumerate(rows[:30]):
        if not row:
            continue
        # Try each of the first 3 columns as a candidate label column
        for col_offset in range(min(3, len(row))):
            val = row[col_offset]
            if val is None:
                continue
            # Case 1: label col contains a district -> rows_are_districts
            if is_district(val):
                return "rows_are_districts", col_offset
            # Case 2: row has >= 4 districts as column headers starting after col_offset
            #         -> rows_are_crops, label col = col_offset
            district_count = sum(1 for v in row[col_offset + 1:] if is_district(v))
            if district_count >= 4:
                return "rows_are_crops", col_offset
    return None, 0


# ---------------------------------------------------------------------------
# Parser: rows = districts, cols = crops  (2024-style)
# ---------------------------------------------------------------------------

def parse_district_rows(rows, title, offset=0):
    """
    Expect layout:
      row 0..k:  title / group-header rows
      row k:     crop header  (col offset = 'District', col offset+1+ = crop names)
      row k+1..: data rows    (col offset = district name, col offset+1+ = values)

    offset: number of leading blank columns before the actual data (0 for most years,
            1 for 2024-style files where col A is empty).

    Returns list of {district, crop, value}.
    """
    records = []

    # Find header row: the row just before the first district row
    header_row_idx = None
    first_district_idx = None
    for r_idx, row in enumerate(rows):
        if row and len(row) > offset and is_district(row[offset]):
            first_district_idx = r_idx
            header_row_idx = r_idx - 1
            break

    if first_district_idx is None or header_row_idx is None or header_row_idx < 0:
        return records

    header = rows[header_row_idx]
    # Map col index -> canonical crop name (only cols after the district col)
    crop_map = {}
    for c_idx, val in enumerate(header):
        if c_idx <= offset:
            continue
        canon = normalise_crop(val)
        if canon is not None:
            crop_map[c_idx] = canon

    # If the immediate row above the first district row has no crop names,
    # try one row higher. This handles layouts where a "District" label row
    # sits between the crop header row and the data rows (e.g. 2023 production tables).
    if not crop_map and header_row_idx >= 1:
        header = rows[header_row_idx - 1]
        for c_idx, val in enumerate(header):
            if c_idx <= offset:
                continue
            canon = normalise_crop(val)
            if canon is not None:
                crop_map[c_idx] = canon

    if not crop_map:
        return records

    for row in rows[first_district_idx:]:
        if not row or len(row) <= offset or not is_district(row[offset]):
            continue
        district = normalise_district(row[offset])
        for c_idx, crop in crop_map.items():
            if c_idx >= len(row):
                continue
            val = clean_num(row[c_idx])
            if val is not None:
                records.append({"district": district, "crop": crop, "value": val})

    return records


# ---------------------------------------------------------------------------
# Parser: rows = crops, cols = districts  (2018-style, transposed)
# ---------------------------------------------------------------------------

def parse_crop_rows(rows, title, offset=0):
    """
    Expect layout (transposed: rows=crops, cols=districts):
      row 0..k:  title / group-header rows
      row k:     district header  (col offset = 'Crop', col offset+1+ = district names)
      row k+1..: data rows        (col offset = crop name, col offset+1+ = values)

    Returns list of {district, crop, value}.
    """
    records = []

    # Find district header row: the row whose cols after offset contain district names
    header_row_idx = None
    for r_idx, row in enumerate(rows):
        if row and len(row) > offset and row[offset] is not None:
            district_count = sum(1 for v in row[offset + 1:] if is_district(v))
            if district_count >= 4:
                header_row_idx = r_idx
                break

    if header_row_idx is None:
        return records

    header = rows[header_row_idx]
    # Map col index -> district name
    district_map = {}
    for c_idx, val in enumerate(header[offset + 1:], start=offset + 1):
        if is_district(val):
            district_map[c_idx] = normalise_district(val)

    if not district_map:
        return records

    for row in rows[header_row_idx + 1:]:
        if not row or len(row) <= offset or row[offset] is None:
            continue
        canon_crop = normalise_crop(row[offset])
        if canon_crop is None:
            continue
        for c_idx, district in district_map.items():
            if c_idx >= len(row):
                continue
            val = clean_num(row[c_idx])
            if val is not None:
                records.append({"district": district, "crop": canon_crop, "value": val})

    return records


# ---------------------------------------------------------------------------
# Per-file sheet mapping
# Explicit mappings discovered by inspection where the title text alone is
# ambiguous (e.g. Appendix 3 = Season C cultivated but title omits "C").
# Keyed by (year, sheet_name) -> (metric, season) or "skip".
# ---------------------------------------------------------------------------
EXPLICIT_SHEET_MAP = {
    # 2018
    (2018, "Appendix 1"):   ("cultivated_area_ha", "A"),
    (2018, "Appendix 2"):   ("cultivated_area_ha", "B"),
    (2018, "Appendix 3"):   ("cultivated_area_ha", "C"),
    (2018, "Appendix 4"):   ("harvested_area_ha",  "A"),
    (2018, "Appendix 5"):   ("harvested_area_ha",  "B"),
    (2018, "Appendix 6 "): ("production_mt",      "A"),
    (2018, "Appendix 7"):   ("production_mt",      "B"),
    (2018, "Appendix 8"):   ("production_mt",      "C"),
    (2018, "Appendix 9 "): ("yield_kg_ha",        "A"),
    (2018, "Appendix 10"):  ("yield_kg_ha",        "B"),
    (2018, "Appendix 11"):  ("yield_kg_ha",        "C"),
    # 2024
    (2024, "Table 10"):     ("cultivated_area_ha", "A"),
    (2024, "Table 11"):     ("cultivated_area_ha", "B"),
    (2024, "Table 12"):     ("cultivated_area_ha", "C"),
    (2024, "Table 13"):     ("harvested_area_ha",  "A"),
    (2024, "Table 14"):     ("harvested_area_ha",  "B"),
    (2024, "Table 15"):     ("harvested_area_ha",  "C"),
    (2024, "Table 16"):     ("yield_kg_ha",        "A"),
    (2024, "Table 17"):     ("yield_kg_ha",        "B"),
    (2024, "Table 18"):     ("yield_kg_ha",        "C"),
    # Tables 19-20 are large-scale farmer yield breakdowns, not the main survey
    (2024, "Table 19"):     "skip",
    (2024, "Table 20"):     "skip",
    (2024, "Table 21"):     ("production_mt",      "A"),
    (2024, "Table 22"):     ("production_mt",      "B"),
    (2024, "Table 23"):     ("production_mt",      "C"),
    # 2018: Tables 6.x/7.x/9.x are province-level summaries; Appendix sheets have district data
    (2018, "Table 6.1 "):   "skip",
    (2018, "Table 6.2 "):   "skip",
    (2018, "Table 6.3"):    "skip",
    (2018, "Table 7.1 "):   "skip",
    (2018, "Table 7.2"):    "skip",
    (2018, "Table 9.1"):    "skip",
    (2018, "Table 9.2"):    "skip",
    (2018, "Table 9.3"):    "skip",
    # 2023: same table numbering as 2024 for cultivated/harvested/yield,
    # but Tables 19/20/21 are LSF breakdowns and production shifts to 22/23/24
    (2023, "Table 10"):     ("cultivated_area_ha", "A"),
    (2023, "Table 11"):     ("cultivated_area_ha", "B"),
    (2023, "Table 12"):     ("cultivated_area_ha", "C"),
    (2023, "Table 13"):     ("harvested_area_ha",  "A"),
    (2023, "Table 14"):     ("harvested_area_ha",  "B"),
    (2023, "Table 15"):     ("harvested_area_ha",  "C"),
    (2023, "Table 16"):     ("yield_kg_ha",        "A"),
    (2023, "Table 17"):     ("yield_kg_ha",        "B"),
    (2023, "Table 18"):     ("yield_kg_ha",        "C"),
    (2023, "Table 19"):     "skip",
    (2023, "Table 20"):     "skip",
    (2023, "Table 21"):     "skip",
    (2023, "Table 22"):     ("production_mt",      "A"),
    (2023, "Table 23"):     ("production_mt",      "B"),
    (2023, "Table 24"):     ("production_mt",      "C"),
    # 2025: table numbering shifted by 3 relative to 2024; Tables 22/23 are LSF breakdowns
    (2025, "Table 13"):     ("cultivated_area_ha", "A"),
    (2025, "Table 14"):     ("cultivated_area_ha", "B"),
    (2025, "Table 15"):     ("cultivated_area_ha", "C"),
    (2025, "Table 16"):     ("harvested_area_ha",  "A"),
    (2025, "Table 17"):     ("harvested_area_ha",  "B"),
    (2025, "Table 18"):     ("harvested_area_ha",  "C"),
    (2025, "Table 19"):     ("yield_kg_ha",        "A"),
    (2025, "Table 20"):     ("yield_kg_ha",        "B"),
    (2025, "Table 21"):     ("yield_kg_ha",        "C"),
    (2025, "Table 22"):     "skip",
    (2025, "Table 23"):     "skip",
    (2025, "Table 24"):     ("production_mt",      "A"),
    (2025, "Table 25"):     ("production_mt",      "B"),
    (2025, "Table 26"):     ("production_mt",      "C"),
    # 2019
    (2019, "Table 7"):      ("cultivated_area_ha", "A"),
    (2019, "Table 8"):      ("cultivated_area_ha", "B"),
    (2019, "Table 9"):      ("cultivated_area_ha", "C"),
    (2019, "Table 10"):     ("harvested_area_ha",  "A"),
    (2019, "Table 11"):     ("harvested_area_ha",  "B"),
    (2019, "Table 12"):     ("harvested_area_ha",  "C"),
    (2019, "Table 13"):     ("yield_kg_ha",        "A"),
    (2019, "Table 14"):     ("yield_kg_ha",        "B"),
    (2019, "Table 15"):     ("yield_kg_ha",        "C"),
    (2019, "Table 16"):     "skip",   # LSF yield A
    (2019, "Table 17"):     "skip",   # LSF yield B
    (2019, "Table 18"):     ("production_mt",      "A"),
    (2019, "Table 19"):     ("production_mt",      "B"),
    (2019, "Table 20"):     ("production_mt",      "C"),
    # 2020: note trailing-space sheet names for B tables
    (2020, "Table 8"):      ("cultivated_area_ha", "A"),
    (2020, "Table 9 "):     ("cultivated_area_ha", "B"),
    (2020, "Table 10"):     ("cultivated_area_ha", "C"),
    (2020, "Table 11"):     ("harvested_area_ha",  "A"),
    (2020, "Table 12 "):    ("harvested_area_ha",  "B"),
    (2020, "Table 13"):     ("harvested_area_ha",  "C"),
    (2020, "Table 14"):     ("yield_kg_ha",        "A"),
    (2020, "Table 15 "):    ("yield_kg_ha",        "B"),
    (2020, "Table 16"):     ("yield_kg_ha",        "C"),
    (2020, "Table 17"):     "skip",   # LSF yield A
    (2020, "Table 18 "):    "skip",   # LSF yield B
    (2020, "Table 19"):     "skip",   # LSF yield C (site-based)
    (2020, "Table 20"):     ("production_mt",      "A"),
    (2020, "Table 21"):     ("production_mt",      "B"),
    (2020, "Table 22"):     ("production_mt",      "C"),
    # 2021: note trailing-space sheet names and Table 19 is detailed banana/beans (skip)
    (2021, "Table 11"):     ("cultivated_area_ha", "A"),
    (2021, "Table 12 "):    ("cultivated_area_ha", "B"),
    (2021, "Table 13"):     ("cultivated_area_ha", "C"),
    (2021, "Table 14"):     ("harvested_area_ha",  "A"),
    (2021, "Table 15 "):    ("harvested_area_ha",  "B"),
    (2021, "Table 16"):     ("harvested_area_ha",  "C"),
    (2021, "Table 17"):     ("yield_kg_ha",        "A"),
    (2021, "Table 18 "):    ("yield_kg_ha",        "B"),
    (2021, "Table 19"):     "skip",   # detailed banana/beans breakdown
    (2021, "Table 20"):     ("yield_kg_ha",        "C"),
    (2021, "Table 21"):     "skip",   # LSF yield A
    (2021, "Table 22 "):    "skip",   # LSF yield B
    (2021, "Table 23"):     "skip",   # LSF yield C (site-based)
    (2021, "Table 24"):     ("production_mt",      "A"),
    (2021, "Table 25"):     ("production_mt",      "B"),
    (2021, "Table 26"):     ("production_mt",      "C"),
    # 2022: note trailing-space sheet names
    (2022, "Table 11"):     ("cultivated_area_ha", "A"),
    (2022, "Table 12 "):    ("cultivated_area_ha", "B"),
    (2022, "Table 13"):     ("cultivated_area_ha", "C"),
    (2022, "Table 14"):     ("harvested_area_ha",  "A"),
    (2022, "Table 15 "):    ("harvested_area_ha",  "B"),
    (2022, "Table 16"):     ("harvested_area_ha",  "C"),
    (2022, "Table 17"):     ("yield_kg_ha",        "A"),
    (2022, "Table 18 "):    ("yield_kg_ha",        "B"),
    (2022, "Table 19"):     ("yield_kg_ha",        "C"),
    (2022, "Table 20"):     "skip",   # LSF yield A
    (2022, "Table 21 "):    "skip",   # LSF yield B
    (2022, "Table 22"):     "skip",   # LSF yield C (site-based)
    (2022, "Table 23"):     ("production_mt",      "A"),
    (2022, "Table 24"):     ("production_mt",      "B"),
    (2022, "Table 25"):     ("production_mt",      "C"),
    # 2023: production A/B; sheet names have no trailing space
    (2023, "Table 22"):     ("production_mt",      "A"),
    (2023, "Table 23"):     ("production_mt",      "B"),
    # 2023: province-level production tables that follow the district ones
    (2023, "Table 25"):     "skip",
    (2023, "Table 26"):     "skip",
    (2023, "Table 27"):     "skip",
    # 2024: province-level tables that follow district production A/B/C
    (2024, "Table 24"):     "skip",
    (2024, "Table 25"):     "skip",
    (2024, "Table 26"):     "skip",
    # 2025: province-level tables that follow district production A/B/C
    (2025, "Table 27"):     "skip",
    (2025, "Table 28"):     "skip",
    (2025, "Table 29"):     "skip",
    # 2020: province-level production C duplicate
    (2020, "Table 25"):     "skip",
    # 2017: all individual Table sheets are province-level; skip them.
    # District-level data lives entirely in the 'Appendix' sheet (handled separately).
    (2017, "Table 8"):      "skip",
    (2017, "Table 52"):     "skip",
    (2017, "Table 55"):     "skip",
    (2017, "Table 96"):     "skip",
    (2017, "Table 97"):     "skip",
    (2017, "Table 99"):     "skip",
    (2017, "Table 100"):    "skip",
}


# ---------------------------------------------------------------------------
# 2017 Appendix parser
# The Appendix sheet contains 9 sub-tables (cultivated/yield/production x A/B/C)
# laid out vertically, each with crops as rows and districts as columns.
# No harvested area data exists for 2017 at district level.
# ---------------------------------------------------------------------------

# Row offsets within the Appendix sheet where each sub-table header row sits
APPENDIX_2017_SECTIONS = [
    (63,  "cultivated_area_ha", "A"),
    (96,  "yield_kg_ha",        "A"),
    (118, "production_mt",      "A"),
    (140, "cultivated_area_ha", "B"),
    (174, "yield_kg_ha",        "B"),
    (197, "production_mt",      "B"),
    (220, "cultivated_area_ha", "C"),
    (239, "yield_kg_ha",        "C"),
    (253, "production_mt",      "C"),
]


def parse_appendix_2017(wb_xls):
    """
    Parse district-level data from the 2017 Appendix sheet.
    Returns {metric: {season: [records]}}.
    """
    ws = wb_xls.sheet_by_name("Appendix")
    # Read all rows
    rows = []
    for r in range(ws.nrows):
        rows.append([ws.cell_value(r, c) for c in range(ws.ncols)])

    results = {}

    for (header_row, metric, season) in APPENDIX_2017_SECTIONS:
        # The header row contains district names starting at col 1
        hrow = rows[header_row]
        district_map = {}
        for c_idx, val in enumerate(hrow[1:], start=1):
            if is_district(val):
                district_map[c_idx] = normalise_district(val)

        if not district_map:
            print(f"    Appendix 2017 {metric}/{season}: no districts in row {header_row}")
            continue

        # Find the end of this section: stop when we hit another section header
        # or a non-crop, non-numeric row that isn't blank
        next_section_rows = [r for (r, _, _) in APPENDIX_2017_SECTIONS if r > header_row]
        end_row = next_section_rows[0] - 2 if next_section_rows else ws.nrows

        records = []
        for row in rows[header_row + 1: end_row]:
            if not row or row[0] is None:
                continue
            canon_crop = normalise_crop(row[0])
            if canon_crop is None:
                continue
            for c_idx, district in district_map.items():
                if c_idx >= len(row):
                    continue
                val = clean_num(row[c_idx])
                if val is not None:
                    records.append({"district": district, "crop": canon_crop, "value": val})

        results.setdefault(metric, {}).setdefault(season, [])
        results[metric][season].extend(records)
        print(f"    Appendix (2017)      -> {metric} / Season {season}: {len(records)} district-crop pairs")

    return results


# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------

def load_file(filepath, year, inspect_mode=False):
    """
    Opens an xlsx/xls file, iterates sheets, classifies each and parses.
    Returns {metric: {season: [records]}}.
    """
    suffix = filepath.suffix.lower()
    engine = "xlrd" if suffix == ".xls" else "openpyxl"

    # Open twice when needed: once for formula text, once for cached values.
    # For openpyxl we use data_only=True to get values (required for yield tables
    # which contain =IFERROR() formulas referencing other files).
    if engine == "openpyxl":
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        sheet_names = wb.sheetnames
    else:
        import xlrd
        wb_xls = xlrd.open_workbook(str(filepath))
        sheet_names = wb_xls.sheet_names()
        wb = None   # handled below per sheet

    results = {}

    for sname in sheet_names:
        if inspect_mode:
            if engine == "openpyxl":
                ws = wb[sname]
                # Collect first non-empty row AND first row that looks like a
                # real table title (longer than 20 chars, not "List of Tables")
                first_nonempty = None
                title_row = None
                for i, row in enumerate(ws.iter_rows(values_only=True)):
                    r = [v for v in row if v is not None]
                    if not r:
                        continue
                    if first_nonempty is None:
                        first_nonempty = r
                    text = " ".join(str(v) for v in r)
                    if len(text) > 20 and "list of table" not in text.lower():
                        title_row = r
                        break
                    if i >= 15:
                        break
                display = title_row or first_nonempty or []
                print(f"  '{sname}': {display[:3]}")
            else:
                ws_xls = wb_xls.sheet_by_name(sname)
                title_row = None
                for r in range(min(15, ws_xls.nrows)):
                    row_vals = [ws_xls.cell_value(r, c) for c in range(min(6, ws_xls.ncols))]
                    text = " ".join(str(v) for v in row_vals if v)
                    if len(text) > 20 and "list of table" not in text.lower():
                        title_row = [v for v in row_vals if v]
                        break
                print(f"  '{sname}': {(title_row or [])[:3]}")
            continue

        # 2017: district-level data lives entirely in the Appendix sheet.
        # All individual Table sheets were confirmed province-level and are
        # already mapped to "skip" in EXPLICIT_SHEET_MAP.
        if year == 2017 and sname == "Appendix":
            appendix_results = parse_appendix_2017(wb_xls)
            for m, seasons in appendix_results.items():
                for s, recs in seasons.items():
                    results.setdefault(m, {}).setdefault(s, [])
                    results[m][s].extend(recs)
            continue

        # Determine metric + season
        explicit = EXPLICIT_SHEET_MAP.get((year, sname))
        if explicit == "skip":
            continue

        if explicit is not None:
            metric, season = explicit
        else:
            # Auto-classify from title text
            if engine == "openpyxl":
                ws = wb[sname]
                title = get_title(ws)
            else:
                ws_xls = wb_xls.sheet_by_name(sname)
                title = ""
                for r in range(min(8, ws_xls.nrows)):
                    for c in range(ws_xls.ncols):
                        v = ws_xls.cell_value(r, c)
                        s = str(v).strip() if v else ""
                        if len(s) > 10:
                            title = s
                            break
                    if title:
                        break
            metric, season = classify_title(title)

        if metric is None or season is None:
            continue

        # Parse rows
        if engine == "openpyxl":
            ws = wb[sname]
            rows = [list(row) for row in ws.iter_rows(values_only=True)]
        else:
            ws_xls = wb_xls.sheet_by_name(sname)
            rows = []
            for r in range(ws_xls.nrows):
                rows.append([ws_xls.cell_value(r, c) for c in range(ws_xls.ncols)])

        # Determine layout orientation and leading-column offset
        orientation, offset = detect_orientation(rows)
        if orientation == "rows_are_districts":
            records = parse_district_rows(rows, "", offset=offset)
        elif orientation == "rows_are_crops":
            records = parse_crop_rows(rows, "", offset=offset)
        else:
            # detect_orientation found no districts; try both parsers as fallback
            # (some sheets have subtly different formatting that defeats the heuristic)
            records = parse_district_rows(rows, "", offset=0)
            if not records:
                records = parse_district_rows(rows, "", offset=1)
            if not records:
                records = parse_crop_rows(rows, "", offset=0)
            if not records:
                print(f"    {sname!r:20s} -> SKIP (orientation unknown, {metric}/{season})")
                continue

        if not records:
            continue

        results.setdefault(metric, {}).setdefault(season, [])
        results[metric][season].extend(records)
        print(f"    {sname!r:20s} -> {metric} / Season {season}: {len(records)} district-crop pairs")

    if engine == "openpyxl" and wb:
        wb.close()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--year",     type=int, default=None)
    parser.add_argument("--inspect",  action="store_true")
    parser.add_argument("--no-pdf-year", action="store_true",
                        help="Skip year 2024 if already in CSVs from the PDF parser")
    args = parser.parse_args()

    files = sorted(NISR_DIR.glob("*.xls*"))
    if not files:
        print(f"No Excel files found in {NISR_DIR}")
        sys.exit(1)

    # Load existing CSVs
    existing = {}
    for metric, path in OUT_FILES.items():
        if path.exists():
            existing[metric] = pd.read_csv(path)
        else:
            existing[metric] = pd.DataFrame(
                columns=["year", "season", "district", "crop", "value"])

    new_records = {m: [] for m in OUT_FILES}

    for filepath in files:
        year = get_year(filepath)
        if year is None:
            print(f"SKIP (year unknown): {filepath.name}")
            continue
        if args.year and year != args.year:
            continue
        if args.no_pdf_year:
            # Check if this year already has data
            has_data = any(
                year in existing[m]["year"].values
                for m in OUT_FILES
                if not existing[m].empty
            )
            if has_data:
                print(f"SKIP (already loaded): {filepath.name}")
                continue

        print(f"\n=== {filepath.name}  (year={year}) ===")

        if args.inspect:
            # handled inside load_file via inspect_mode=True
            load_file(filepath, year, inspect_mode=True)
            continue

        file_results = load_file(filepath, year)

        for metric, seasons in file_results.items():
            for season, records in seasons.items():
                for rec in records:
                    new_records[metric].append({
                        "year":     year,
                        "season":   season,
                        "district": rec["district"],
                        "crop":     rec["crop"],
                        "value":    rec["value"],
                    })

    if args.inspect:
        return

    if args.dry_run:
        print("\n--- Dry run: no files written ---")
        for metric, recs in new_records.items():
            print(f"  {metric}: {len(recs)} new records")
        return

    # Merge and write
    pilot = {"Nyagatare", "Ruhango", "Musanze", "Nyabihu"}

    for metric, path in OUT_FILES.items():
        new_df = pd.DataFrame(new_records[metric])
        if new_df.empty:
            print(f"\nWARNING: no new records for {metric}")
            continue

        combined = pd.concat([existing[metric], new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["year", "season", "district", "crop"])
        combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
        combined = combined.dropna(subset=["value"])
        combined = combined.sort_values(
            ["year", "season", "district", "crop"]).reset_index(drop=True)

        combined.to_csv(path, index=False)

        years_present = sorted(combined["year"].unique())
        n_rows = len(combined)
        print(f"\n{metric}: {n_rows} rows | years {years_present}")
        for d in sorted(pilot):
            sub = combined[combined["district"] == d]
            year_season = sub.groupby(["year", "season"])["crop"].nunique()
            sample = year_season.reset_index().rename(columns={"crop": "n_crops"})
            print(f"  {d}: {len(sample)} year-season entries")
            print(f"  {sample.head(6).to_string(index=False)}")

        print(f"  -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
