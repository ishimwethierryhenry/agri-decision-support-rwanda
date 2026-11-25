"""Model scores page: how well every model did in the official evaluation (reports/eval_v2, protocol 1.1).

Asked by the client and the team on 29 Oct 2026: show the performance of every model, per district and per crop,
next to the recommendations. Updated on 18 Nov 2026 for protocol 1.1, which adds four satellite inputs measured at
each field (the 12 combinations of the midterm give the same numbers as in eval_v1), and on 23 Nov 2026 for protocol
1.2: the tuning run (reports/tuning_v1) and why its tuned Random Forest is now the trained model on the map. On
25 Nov 2026 the "why" section also gives the result of the second round of tuning (protocol 1.3, reports/tuning_v2).
Everything here is read from the committed result files. Logic: 01-crop-recommendation/src/models/model_scores.py
Since 24 Nov 2026 the page uses the shared Agri DSS look (ui_theme.py, rec_ui.py): a hero heading and one card per
section.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parents[3] / "01-crop-recommendation" / "src" / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_scores as ms  # noqa: E402
import rec_ui as ui  # noqa: E402
from ui_theme import hero  # noqa: E402

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


hero("Model scores", "How well does each <em>model</em> do?",
     "Every method was tested the same way, on the same fields, with the same scores. Everything here is read "
     "from the committed result files.")
ui.styles()
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

test = st.radio("Which test", list(ms.TESTS), format_func=ms.TEST_NAMES.get, horizontal=True,
                key="pills_scores_test")

with st.container(key="card_main"):
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

with st.container(key="card_satellite"):
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

with st.container(key="card_districts"):
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

with st.container(key="card_crops"):
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

@st.cache_resource(show_spinner=False)
def load_tuning() -> ms.Tuning:
    return ms.Tuning()


def pct(x: float) -> str:
    return f"{x:.0%}"


tuning = load_tuning()
gains = tuning.gains(test)
algo_gain = gains.groupby("Model", sort=False)["Change"].mean()
with st.container(key="card_tuning"):
    st.subheader("Tuning: the same models with better settings")
    st.dataframe(gains.drop(columns="model"), hide_index=True, height=rows_height(len(gains)),
                 column_config={"Protocol settings": st.column_config.NumberColumn(format="%.3f"),
                                "Tuned": st.column_config.NumberColumn(format="%.3f"),
                                "Change": st.column_config.NumberColumn(format="%+.3f"),
                                "First crop right, all fields (tuned)": st.column_config.NumberColumn(format="%.3f")})
    how = ("Main score (macro-F1, 0 to 1) of each method with the settings of the protocol and with tuned settings "
           "(tuning run of 23 November 2026, protocol 1.2). Each method tried 61 settings. In every fold the settings "
           "were chosen with the training fields only, then scored once on the test fields, so the test fields never "
           "helped choose them. ")
    if test == "new_fields":
        st.caption(how + f"On new fields every model gains, {algo_gain.idxmax()} the most on average "
                         f"({algo_gain.max():+.3f}).")
    else:
        lowered = int((gains["Change"] < 0).sum())
        st.caption(how + f"In a district the model never saw, settings chosen on three districts did not carry over to "
                         f"the fourth: tuning lowered {lowered} of the {len(gains)} models, and none passes the adoption "
                         "rule, so the rules stay the default answer of the API.")

# The model on the map since 23 Nov: decision 2 of the protocol, section 11 (written before the tuning run).
mp = tuning.decisions["map"]
on_map = tuning.map_model
algo = ms.model_label(on_map.replace(ms.TUNED, ""))[0]
tf, tm = tuning.summary["new_fields"], tuning.summary["new_district"]
who = tuning.answered_by()
sure = tuning.when_sure()
at_07 = sure[sure["Answers only from"] == 0.7].iloc[0]
alt = tuning.alternatives().set_index("Choice")
r2 = ms.second_round()                     # protocol 1.3 (reports/tuning_v2); None if its results are not there
district_rule = "Most common crop of the field's district"
mixed = tuning.mixed.loc["all"]
with st.container(key="card_why"):
    st.subheader(f"Why the tuned {algo} is the trained model on the map")
    st.markdown(
        f"- **The rule was written before the run.** The map changes only if a tuned model beats the model on the map on "
        f"new fields by more than {mp['margin']}. The tuned {ms.short_label(on_map.replace(ms.TUNED, ''))} scores "
        f"{tf.loc[on_map, 'macro_f1_mean']:.3f} against {mp['today_score']:.3f} for {ms.short_label(mp['today'])} with "
        "the protocol settings, so the map moved to it.\n"
        + (f"- **A second, bigger search found nothing better.** A second round of tuning, finished on 25 Nov, tried "
           f"{sum(r2['settings'].values())} settings of Random Forest, Extra Trees and XGBoost, a blend of the three, "
           "and inputs that say which crops the nearby surveyed fields grow. The best, the tuned "
           f"{ms.short_label(r2['best'].replace(ms.TUNED, ''))}, scored {r2['best_score']:.3f} on new fields, not "
           f"more than {r2['margin']} above this model, so the map kept it.\n" if r2 and not r2["changes"] else "")
        + "- **One answer per point.** In each district it comes from the method that did better on new fields there: "
        + ", ".join(f"{algo if w == 'model' else 'the rules'} in {d}" for d, w in who.items()) + ".\n"
        f"- **It knows when to be trusted.** When it gives its first crop a probability of 0.7 or more "
        f"({pct(at_07['Fields answered'])} of the fields), that crop is right {pct(at_07['First crop right'])} of the "
        "time. When it is less sure, it is right less often, as the table below shows.\n"
        f"- **It beats the simple choices.** Its first crop is right for {pct(tf.loc[on_map, 'top1_all_fields'])} of "
        f"the fields, against {pct(alt.loc['Expert rules (B1)', 'First crop right, all fields'])} for the rules and "
        f"{pct(alt.loc[district_rule, 'First crop right, all fields'])} for the most common crop of the district, which "
        f"never names another crop (main score {alt.loc[district_rule, 'Main score']:.3f}).\n"
        f"- **The data has a limit.** {pct(mixed['blocks_with_more_than_one_crop'])} of the 2 km blocks grow more than "
        "one crop. Even the most common crop of each field's own block, which no model can know in advance, would be "
        f"right for only {pct(mixed['block_most_common_crop_right'])} of the fields.\n"
        f"- **Limit.** In a district it has never seen, it scores {tm.loc[on_map, 'macro_f1_mean']:.3f}, below the "
        f"rules ({tm.loc[ms.BASELINE, 'macro_f1_mean']:.3f}), so the rules stay the default answer of the API.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**If it answered only when sure (new fields)**")
        st.dataframe(pd.DataFrame({"Answers only from": [f"{v:.1f}" for v in sure["Answers only from"]],
                                   "Fields answered": [pct(v) for v in sure["Fields answered"]],
                                   "First crop right": [pct(v) for v in sure["First crop right"]]}),
                     hide_index=True, height=rows_height(len(sure)))
    with c2:
        st.markdown("**Simple choices on the same fields**")
        st.dataframe(pd.DataFrame({"Choice": alt.index,
                                   "Main score": [f"{v:.3f}" for v in alt["Main score"]],
                                   "First crop right": [pct(v) for v in alt["First crop right, all fields"]]}),
                     hide_index=True, height=rows_height(len(alt)))
    st.caption("Out-of-fold predictions of the new-fields test: no field was predicted by a model that saw its 2 km "
               "block. 'Answers only from' is the model's probability for its first crop. Full report: "
               "01-crop-recommendation/reports/tuning_v1/report.md"
               + ("; second round: 01-crop-recommendation/reports/tuning_v2/report.md." if r2 else "."))

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
