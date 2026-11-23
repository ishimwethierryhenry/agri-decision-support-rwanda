"""Model scores page: how well every model did in the official evaluation (reports/eval_v2, protocol 1.1).

Asked by the client and the team on 29 Oct 2026: show the performance of every model, per district and per crop,
next to the recommendations. Updated on 18 Nov 2026 for protocol 1.1, which adds four satellite inputs measured at
each field (the 12 combinations of the midterm give the same numbers as in eval_v1). Everything here is read from the
committed result files. Logic: 01-crop-recommendation/src/models/model_scores.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parents[3] / "01-crop-recommendation" / "src" / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
import model_scores as ms  # noqa: E402

MAP_MODEL = "XGB[all]"                     # the model on the map page (map_model.MODEL_NAME)
BASELINES = ("B0_majority", ms.BASELINE)

@st.cache_resource(show_spinner=False)
def load(analysis: str = "main") -> ms.Scores:
    return ms.Scores(analysis=analysis)


def fmt(df: pd.DataFrame, skip=("Model", "Inputs", "Fields", "Beats B1 in", "Passes rule"), digits: int = 3):
    """Numbers as text with fixed decimals; missing values (crops with too few fields) read "not scored"."""
    out = df.copy()
    for c in out.columns:
        if c not in skip and pd.api.types.is_numeric_dtype(df[c]):
            out[c] = [("not scored" if pd.isna(v) else f"{v:.{digits}f}") for v in df[c]]
    return out


def rows_height(n: int) -> int:
    """Tall enough to show every row without an inner scroll bar."""
    return 35 * (n + 1) + 3


def best_trained(table: pd.DataFrame) -> str:
    """The trained model (not B0 or B1) with the highest main score, in a main_scores table indexed by model."""
    return table.drop(index=[m for m in BASELINES if m in table.index])["Main score"].idxmax()


def passing_models(table: pd.DataFrame) -> list:
    """Short names of the models that pass the adoption rule, in a main_scores table indexed by model."""
    return [ms.short_label(m) for m in table.index[table["Passes rule"] == "yes"]]


# Column names as shown (client feedback of 19 Nov: every column says what it measures; top-3 removed, because with
# four crops it is close to 1 for every method and only confuses).
DISPLAY = {"Main score": "Main score (macro-F1, 0 to 1)", "Worst district": "Worst district (main score)",
           "Accuracy (mean)": "Accuracy (average of the 4 districts)", "Beats B1 in": "Better than the rules in",
           "Passes rule": "Passes the adoption rule"}
SCORE_COLUMNS = {
    DISPLAY["Main score"]: st.column_config.ProgressColumn(
        DISPLAY["Main score"], min_value=0.0, max_value=1.0, format="%.3f",
        help="In each district: how well the method names the right crop first, checked crop by crop so that every "
             "crop counts the same (macro-F1). Then the average of the four districts. 1 is perfect, 0 always wrong."),
    DISPLAY["Worst district"]: st.column_config.NumberColumn(
        format="%.3f", help="The main score in the district where this method does worst."),
    DISPLAY["Accuracy (mean)"]: st.column_config.NumberColumn(
        format="%.3f", help="Share of fields where the first crop is the right one, in each district, then the "
                            "average of the four districts."),
    DISPLAY["Beats B1 in"]: st.column_config.TextColumn(
        help="Number of districts where this method's main score beats the expert rules by more than 0.01."),
    DISPLAY["Passes rule"]: st.column_config.TextColumn(
        help="Yes only if, on the main test, its average is higher than the rules' and it is better than the rules "
             "in at least 3 of the 4 districts."),
}


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    """A main_scores table as shown on the page: plain column names, no top-3."""
    return df.drop(columns=["model", "Top-3 (mean)"]).rename(columns=DISPLAY)


st.header("How well each model did")
st.caption("Official evaluation of 18 November 2026 (protocol 1.1): 1,904 CIAT fields of season B 2025 in Musanze, "
           "Nyabihu, Nyagatare and Ruhango, 18 model and input combinations, the same tests and the same scores "
           "for all. Version 1.1 adds four inputs measured at each field by satellite: SRTM elevation, CHIRPS "
           "rainfall of the season, and two Sentinel-2 indices of 2024, the year before the season. The 12 "
           "combinations of the midterm evaluation (28 October, protocol 1.0) give exactly the same numbers as "
           "before.")

scores = load()
main_test = scores.main_scores("new_district").set_index("model")
fields = scores.main_scores("new_fields").set_index("model")
rules_main, rules_fields = main_test.loc[ms.BASELINE, "Main score"], fields.loc[ms.BASELINE, "Main score"]
best_main, best_fields = best_trained(main_test), best_trained(fields)
passing = passing_models(main_test)
if not passing:
    st.info("No trained model passes the adoption rule on the main test (a district the model never saw), so the "
            f"expert rules (B1) stay the default answer of the API. The closest is {ms.short_label(best_main)}: "
            f"{main_test.loc[best_main, 'Main score']:.3f} against {rules_main:.3f} for the rules, ahead by more "
            f"than 0.01 in {main_test.loc[best_main, 'Beats B1 in']}. On new fields in the four known districts, "
            f"{ms.short_label(best_fields)} is best ({fields.loc[best_fields, 'Main score']:.3f} against "
            f"{rules_fields:.3f} for the rules).")
else:
    st.success(f"Passing the adoption rule on the main test: {', '.join(passing)}.")

test = st.radio("Test", list(ms.TESTS), format_func=ms.TEST_NAMES.get, horizontal=True, key="scores_test")

st.subheader("Main score of every model")
table = score_table(scores.main_scores(test))
st.dataframe(table, hide_index=True, column_config=SCORE_COLUMNS, height=rows_height(len(table)))
st.caption("How to read it. The main score checks, in each district, how well the method names the right crop "
           "first, crop by crop, so that rice counts as much as maize (macro-F1, from 0 to 1; crops with fewer than 5 "
           "fields in a district are left out). The table shows the average of the four districts, and the worst "
           "district shows the same score where the method does worst. Accuracy is the share of fields where the "
           "first crop is right, averaged over the four districts. A trained model replaces the rules only if, on the "
           "main test, its average is higher and it is better than the rules by more than 0.01 in at least 3 of the "
           "4 districts. Top-3 is not shown: with only four crops it is close to 1 for every method, so it tells "
           "nothing apart.")

st.subheader("What the satellite inputs changed")
effect = scores.satellite_effect(test)
changes = [c for c in effect.columns if c.startswith("Change")]
st.dataframe(effect.style.format("{:.3f}").format("{:+.3f}", subset=changes), height=rows_height(len(effect)))
if test == "new_fields":
    near = fields.loc["XGB[field_sat]", "Main score"]
    st.caption("Main score of each method on the same inputs, without and with the four satellite inputs. On new "
               "fields every method gains, most when it has field inputs only. XGBoost on field + satellite inputs "
               f"({near:.3f}) uses no input that is the same for a whole district, yet comes within "
               f"{round(fields.loc[best_fields, 'Main score'], 3) - round(near, 3):.3f} of the best model.")
else:
    st.caption("Main score of each method on the same inputs, without and with the four satellite inputs. In a "
               "district the model never saw, Random Forest gains the most and is now the closest to the rules, but "
               "it still loses to them in the highlands (Musanze, Nyabihu). XGBoost barely moves, and Logistic "
               "Regression drops with all inputs.")

st.subheader("Main score in each district (macro-F1, 0 to 1)")
per_district = scores.per_district(test)
st.dataframe(per_district.style.format("{:.3f}"), height=rows_height(len(per_district)))
st.caption("The trained models win in the lowlands (Nyagatare, Ruhango), where the rules are weak, and lose in the "
           "highlands (Musanze, Nyabihu), where the rules already work well.")

st.subheader("Overall scores, all test fields together")
overall = scores.overall(test).drop(columns="model")
st.dataframe(overall, hide_index=True, height=rows_height(len(overall)),
             column_config={c: st.column_config.NumberColumn(format="%.3f") for c in overall.columns
                            if c not in ("Model", "Inputs")})
st.caption("Accuracy: share of fields where the first crop is the true crop; it mostly reflects Nyagatare, which "
           "has 1,084 of the 1,904 fields. Balanced accuracy and macro-F1 give every crop the same weight. "
           "Cohen's kappa: 0 is chance agreement, 1 is perfect; 0.41 to 0.60 is usually read as moderate.")

st.subheader("Each crop in each district")
c1, c2 = st.columns(2)
district = c1.selectbox("District", ms.DISTRICTS, key="scores_district")
choices = [m for m in scores.models if m != "XGB[all]"]
other = c2.selectbox("Compare XGBoost (all inputs) with", choices, index=choices.index("B1_expert_rules"),
                     format_func=ms.short_label, key="scores_other")
crops = scores.crop_table(test, district, ("XGB[all]", other))
st.dataframe(fmt(crops, digits=2))
st.caption("Precision: when the model says this crop, how often it is right. Recall: of the fields with this crop, "
           "how many the model found. F1 combines the two. Crops with fewer than 5 fields in the district are not "
           "scored.")

def gain(table: pd.DataFrame, new: str, old: str) -> str:
    """Difference of two main scores as shown on the page (rounded to 3 decimals first), with its sign."""
    return f"{round(table.loc[new, 'Main score'], 3) - round(table.loc[old, 'Main score'], 3):+.3f}"


# Same choice as map_model.method_for: the model answers in a district only if it beats the rules on new fields there
# by more than 0.01 (the margin of the adoption rule).
nf = scores.results["new_fields"].pivot(index="model", columns="district", values="macro_f1")
answered_by = {d: "XGBoost" if nf.loc[MAP_MODEL, d] > nf.loc[ms.BASELINE, d] + 0.01 else "the rules"
               for d in ms.DISTRICTS}

st.subheader("Why XGBoost on all inputs is the trained model on the map")
st.markdown(
    "- The map shows users one answer per point. In each district it comes from the method that did better on new "
    "fields there in testing: " + ", ".join(f"{who} in {d}" for d, who in answered_by.items()) + ".\n"
    f"- At the midterm it had the best main score on new fields in the four districts: "
    f"{fields.loc[MAP_MODEL, 'Main score']:.3f}, ahead of Random Forest on all inputs "
    f"({fields.loc['RF[all]', 'Main score']:.3f}) and the rules ({rules_fields:.3f}).\n"
    f"- With the satellite inputs it reaches {fields.loc['XGB[all_sat]', 'Main score']:.3f} on new fields "
    f"({gain(fields, 'XGB[all_sat]', MAP_MODEL)}) and {main_test.loc['XGB[all_sat]', 'Main score']:.3f} in a "
    f"district never seen ({gain(main_test, 'XGB[all_sat]', MAP_MODEL)}). That is too small to change the model on "
    "the map, which would also have to take the satellite values from the nearest CIAT block, like the soil.\n"
    "- Shallow boosted trees capture how crops respond to combinations of rainfall, temperature, elevation and "
    "soil; balanced class weights keep rice and potato from being ignored; it retrains in seconds.\n"
    f"- Limit: in a district it has never seen, it scores {main_test.loc[MAP_MODEL, 'Main score']:.3f}, below the "
    f"rules ({rules_main:.3f}). The closest trained model there, {ms.short_label(best_main)} "
    f"({main_test.loc[best_main, 'Main score']:.3f}), is below the rules too.\n"
    "- Close competitor: over all fields pooled, Random Forest on all inputs is slightly ahead, because it does a "
    "little better in Nyagatare.")

with st.expander("Second check: the 650 fields CIAT marked \"Used\""):
    used_scores = load("used_only")
    used = score_table(used_scores.main_scores(test))
    st.dataframe(used, hide_index=True, column_config=SCORE_COLUMNS, height=rows_height(len(used)))
    used_main = used_scores.main_scores("new_district").set_index("model")
    used_fields = used_scores.main_scores("new_fields").set_index("model")
    best_used = best_trained(used_fields)
    on_main = passing_models(used_main)
    on_fields = passing_models(used_fields)
    st.caption("For information only: this check does not decide anything. On the main test "
               + ("no model passes. " if not on_main else f"these would pass: {', '.join(on_main)}. ")
               + (f"On new fields these would pass the same rule: {'; '.join(on_fields)}. " if on_fields else
                  "On new fields no model would pass the same rule. ")
               + f"The best there is {ms.short_label(best_used)} ({used_fields.loc[best_used, 'Main score']:.3f} "
               f"against {used_fields.loc[ms.BASELINE, 'Main score']:.3f} for the rules), which suggests the models "
               "gain from cleaner labels.")
