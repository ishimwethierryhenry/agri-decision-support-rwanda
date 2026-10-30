"""Model scores page: how well every model did in the official evaluation (reports/eval_v1, protocol 1.0).

Asked by the client and the team on 29 Oct 2026: show the performance of every model, per district and per crop,
next to the recommendations. Everything here is read from the committed result files. Logic:
01-crop-recommendation/src/models/model_scores.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parents[3] / "01-crop-recommendation" / "src" / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
import model_scores as ms  # noqa: E402


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


SCORE_COLUMNS = {
    "Main score": st.column_config.ProgressColumn("Main score", min_value=0.0, max_value=1.0, format="%.3f"),
    "Worst district": st.column_config.NumberColumn(format="%.3f"),
    "Accuracy (mean)": st.column_config.NumberColumn(format="%.3f"),
    "Top-3 (mean)": st.column_config.NumberColumn(format="%.3f"),
}


st.header("How well each model did")
st.caption("Official evaluation of 28 October 2026 (protocol 1.0): 1,904 CIAT fields of season B 2025 in Musanze, "
           "Nyabihu, Nyagatare and Ruhango, 12 model and input combinations, the same tests and the same scores "
           "for all. Rerun on Colab with identical results.")

scores = load()
main_test = scores.main_scores("new_district").set_index("model")
fields = scores.main_scores("new_fields").set_index("model")
passing = main_test[main_test["Passes rule"] == "yes"]
if passing.empty:
    st.info("No trained model passes the adoption rule on the main test (a district the model never saw), so the "
            "expert rules (B1) stay the default answer of the API. On new fields in the four known districts, "
            f"XGBoost on all inputs has the best main score ({fields.loc['XGB[all]', 'Main score']:.3f} against "
            f"{fields.loc['B1_expert_rules', 'Main score']:.3f} for the rules), so it is the model on the map, "
            "shown next to the rules.")
else:
    st.success(f"Passing the adoption rule on the main test: {', '.join(passing['Model'])}.")

test = st.radio("Test", list(ms.TESTS), format_func=ms.TEST_NAMES.get, horizontal=True, key="scores_test")

st.subheader("Main score of every model")
table = scores.main_scores(test).drop(columns="model")
st.dataframe(table, hide_index=True, column_config=SCORE_COLUMNS, height=rows_height(len(table)))
st.caption("Main score: macro-F1 in each test district over the crops with at least 5 fields there, then the mean "
           "over the four districts (0 to 1). Accuracy and top-3 are also means over the four districts. Adoption "
           "rule: a trained model replaces the rules only if, on the main test, its mean is higher and it beats the "
           "rules by more than 0.01 in at least 3 of the 4 districts. Top-3 is close to 1 for every method because "
           "there are only four crops, so it is never used to compare.")

st.subheader("Main score in each district")
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

st.subheader("Why XGBoost is on the map")
st.markdown(
    f"- Best main score on new fields in the four districts: {fields.loc['XGB[all]', 'Main score']:.3f}, ahead of "
    f"Random Forest on all inputs ({fields.loc['RF[all]', 'Main score']:.3f}) and the rules "
    f"({fields.loc['B1_expert_rules', 'Main score']:.3f}).\n"
    "- Shallow boosted trees capture how crops respond to combinations of rainfall, temperature, elevation and "
    "soil; balanced class weights keep rice and potato from being ignored; it retrains in seconds.\n"
    f"- Limit: in a district it has never seen, it scores {main_test.loc['XGB[all]', 'Main score']:.3f} on all "
    f"inputs, below the rules ({main_test.loc['B1_expert_rules', 'Main score']:.3f}).\n"
    "- Close competitor: over all fields pooled, Random Forest on all inputs is slightly ahead, because it does a "
    "little better in Nyagatare. Tuning after the midterm, inside the training districts only, will decide.")

with st.expander("Second check: the 650 fields CIAT marked \"Used\""):
    used = load("used_only").main_scores(test).drop(columns="model")
    st.dataframe(used, hide_index=True, column_config=SCORE_COLUMNS, height=rows_height(len(used)))
    st.caption("For information only: this check does not decide anything. On the main test no model passes; on new "
               "fields XGBoost beats the rules in 3 of the 4 districts, which suggests the models gain from cleaner "
               "labels.")
