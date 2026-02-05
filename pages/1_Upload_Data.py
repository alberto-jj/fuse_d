from __future__ import annotations

import pandas as pd
import streamlit as st

from src.io import (
    DatasetSpec,
    dataset_summary,
    read_csv_from_upload,
    validate_wide_dataset,
)
from src.state import (
    clear_dataset,
    init_session_state,
    recompute_from_raw_if_available,
)


st.set_page_config(
    page_title="Upload Data",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()


with st.sidebar:
    #st.image("fused.logo.png", use_container_width=True)
    #st.divider()
    st.subheader("Progress")

    if st.session_state.df_wide is not None:
        st.success("Dataset loaded")
        st.caption(
            f"Rows: {st.session_state.df_wide.shape[0]} | Cols: {st.session_state.df_wide.shape[1]}"
        )
        st.button(
            "Clear dataset",
            key="upload_clear_dataset_btn",
            on_click=clear_dataset,
            use_container_width=True,
        )
    else:
        st.info("No dataset loaded")

    if st.session_state.baseline_results is not None:
        st.success("Baseline model fitted")
    else:
        st.info("Baseline model fitted")

    if st.session_state.feature_cols_selected:
        st.success("Features selected")
        st.caption(
            f"Selected Features: {len(st.session_state.feature_cols_selected)} | "
            f"All features: {len(st.session_state.feature_cols or [])}"
        )
    else:
        st.info("Features selected")

    if st.session_state.baseline_results is not None and st.session_state.baseline_selected_results is not None:
        st.success("Compare your models")
    else:
        st.info("Compare your models")


st.header("Upload wide-format CSV")
st.write(
    "- Expected format: one row per subject, columns are features plus optional metadata.\n"
    "- ID and target columns are excluded from features automatically.\n"
    "- Manually exclude any additional columns in Dataset Settings."
)

def _apply_dataset_settings() -> None:
    st.session_state.spec = DatasetSpec(
        id_col=st.session_state.upload_id_col_input,
        target_col=st.session_state.upload_target_col_input,
        age_col=None,
        center_col=None,
    )
    st.session_state.exclude_text = st.session_state.upload_exclude_cols_input
    st.session_state.drop_missing_target = st.session_state.upload_drop_missing_target_chk
    recompute_from_raw_if_available()


st.subheader("Data loader")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="upload_csv_uploader")

colA, _ = st.columns([1, 1])
with colA:
    encoding = st.selectbox(
        "CSV encoding (usually auto)",
        options=["auto", "utf-8", "latin1"],
        index=0,
        key="upload_encoding_select",
    )

if uploaded is not None:
    try:
        df = read_csv_from_upload(uploaded, encoding=None if encoding == "auto" else encoding)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    ok, msgs = validate_wide_dataset(df, st.session_state.spec, require_id=True, require_target=True)
    if not ok:
        for m in msgs:
            st.error(m)
        st.stop()
    for m in msgs:
        st.warning(m)

    st.session_state.df_raw = df.copy()
    _apply_dataset_settings()

    st.session_state.baseline_results = None
    st.session_state.baseline_cv_scores = None
    st.session_state.baseline_diag = None
    st.session_state.baseline_summary = {}
    st.session_state.vim_results_df = None
    st.session_state.vim_top_df = None
    st.session_state.feature_cols_selected = None
    st.session_state.vim_run_meta = {}
    st.session_state.baseline_selected_results = None
    st.session_state.baseline_selected_cv_scores = None
    st.session_state.baseline_selected_diag = None
    st.session_state.baseline_selected_summary = {}

    with st.expander("Raw upload preview", expanded=False):
        st.dataframe(df.head(30), use_container_width=True)


st.subheader("Dataset settings")
prev_spec = st.session_state.get("spec", None)
id_default = getattr(prev_spec, "id_col", "participant_id")
tgt_default = getattr(prev_spec, "target_col", "ad_syndrome")

if "upload_id_col_input" not in st.session_state:
    st.session_state.upload_id_col_input = id_default
if "upload_target_col_input" not in st.session_state:
    st.session_state.upload_target_col_input = tgt_default
if "upload_exclude_cols_input" not in st.session_state:
    st.session_state.upload_exclude_cols_input = st.session_state.exclude_text
if "upload_drop_missing_target_chk" not in st.session_state:
    st.session_state.upload_drop_missing_target_chk = st.session_state.drop_missing_target

st.text_input(
    "Subject ID column",
    key="upload_id_col_input",
)
st.text_input(
    "Target column",
    key="upload_target_col_input",
)
st.text_input(
    "Exclude features (comma-separated)",
    placeholder="e.g. sex, age, center",
    help="These columns will not be used as features.",
    key="upload_exclude_cols_input",
)
st.checkbox(
    "Drop rows with missing target",
    key="upload_drop_missing_target_chk",
)
st.button(
    "Apply filters",
    type="primary",
    key="upload_apply_filters_btn",
    on_click=_apply_dataset_settings,
)

if st.session_state.df_wide is not None:
    st.caption(f"Selected features: {len(st.session_state.feature_cols or [])}")
    if st.session_state.excluded_cols:
        st.caption(
            "Excluded (effective): "
            + ", ".join(st.session_state.excluded_cols[:10])
            + (" ..." if len(st.session_state.excluded_cols) > 10 else "")
        )
    if st.session_state.missing_manual_excludes:
        st.caption(
            "Not found: "
            + ", ".join(st.session_state.missing_manual_excludes[:10])
            + (" ..." if len(st.session_state.missing_manual_excludes) > 10 else "")
        )

if st.session_state.df_wide is not None:
    st.subheader("Preview of final dataset")
    df_clean = st.session_state.df_wide
    st.caption(f"Loaded: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")

    if st.session_state.missing_manual_excludes:
        st.warning(
            "Excluded columns not found: " + ", ".join(st.session_state.missing_manual_excludes)
        )

    st.dataframe(df_clean.head(30), use_container_width=True)

    summ = dataset_summary(df_clean, st.session_state.spec)

    st.subheader("Final dataset details")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", summ["n_rows"])
    c2.metric("Columns", summ["n_cols"])
    c3.metric("Missing values", summ["n_missing_total"])
    c4.metric("Numeric features", summ["n_numeric_features"])
    c5.metric("Selected features", len(st.session_state.feature_cols or []))

    if st.session_state.excluded_cols:
        st.caption("Excluded (effective): " + ", ".join(st.session_state.excluded_cols))

    with st.expander("Included feature list"):
        st.write((st.session_state.feature_cols or [])[:250])
        if st.session_state.feature_cols and len(st.session_state.feature_cols) > 250:
            st.caption(f"... ({len(st.session_state.feature_cols)} total)")

    st.subheader("Class distribution")
    if "target_counts" in summ:
        counts = pd.Series(summ["target_counts"], name="count").sort_values(ascending=False)
        total = float(counts.sum()) if len(counts) else 0.0

        if total > 0:
            plot_df = counts.reset_index()
            plot_df.columns = ["class", "count"]
            plot_df["label"] = plot_df.apply(
                lambda r: f"{r['class']} ({int(r['count'])} | {r['count'] / total:.1%})",
                axis=1,
            )

            st.vega_lite_chart(
                plot_df,
                {
                    "width": 220,
                    "height": 180,
                    "mark": {"type": "arc", "outerRadius": 80},
                    "encoding": {
                        "theta": {"field": "count", "type": "quantitative"},
                        "color": {"field": "label", "type": "nominal", "legend": None},
                        "tooltip": [
                            {"field": "class", "type": "nominal"},
                            {"field": "count", "type": "quantitative"},
                            {"field": "label", "type": "nominal"},
                        ],
                    },
                },
                use_container_width=False,
            )
