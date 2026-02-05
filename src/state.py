from __future__ import annotations

import streamlit as st

from src.io import DatasetSpec, dataset_summary, prepare_wide_dataset


def init_session_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = None

    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "df_wide" not in st.session_state:
        st.session_state.df_wide = None

    if "spec" not in st.session_state:
        st.session_state.spec = DatasetSpec(
            id_col="participant_id",
            target_col="ad_syndrome",
            age_col=None,
            center_col=None,
        )

    if "exclude_text" not in st.session_state:
        st.session_state.exclude_text = ""
    if "drop_missing_target" not in st.session_state:
        st.session_state.drop_missing_target = True

    if "feature_cols" not in st.session_state:
        st.session_state.feature_cols = None
    if "excluded_cols" not in st.session_state:
        st.session_state.excluded_cols = None
    if "missing_manual_excludes" not in st.session_state:
        st.session_state.missing_manual_excludes = []

    if "baseline_results" not in st.session_state:
        st.session_state.baseline_results = None
    if "baseline_cv_scores" not in st.session_state:
        st.session_state.baseline_cv_scores = None
    if "baseline_diag" not in st.session_state:
        st.session_state.baseline_diag = None
    if "baseline_summary" not in st.session_state:
        st.session_state.baseline_summary = {}

    if "vim_results_df" not in st.session_state:
        st.session_state.vim_results_df = None
    if "vim_top_df" not in st.session_state:
        st.session_state.vim_top_df = None
    if "feature_cols_selected" not in st.session_state:
        st.session_state.feature_cols_selected = None
    if "vim_run_meta" not in st.session_state:
        st.session_state.vim_run_meta = {}

    if "baseline_selected_results" not in st.session_state:
        st.session_state.baseline_selected_results = None
    if "baseline_selected_cv_scores" not in st.session_state:
        st.session_state.baseline_selected_cv_scores = None
    if "baseline_selected_diag" not in st.session_state:
        st.session_state.baseline_selected_diag = None
    if "baseline_selected_summary" not in st.session_state:
        st.session_state.baseline_selected_summary = {}

    if "n_splits" not in st.session_state:
        st.session_state.n_splits = 10
    if "metric_choice" not in st.session_state:
        st.session_state.metric_choice = "balanced_accuracy"
    if "diag" not in st.session_state:
        st.session_state.diag = True
    if "baseline_random_state" not in st.session_state:
        st.session_state.baseline_random_state = 42


def clear_dataset() -> None:
    st.session_state.df_raw = None
    st.session_state.df_wide = None
    st.session_state.feature_cols = None
    st.session_state.excluded_cols = None
    st.session_state.missing_manual_excludes = []
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


def require_data_loaded() -> None:
    if st.session_state.df_wide is None or st.session_state.feature_cols is None:
        st.info("No dataset loaded. Go to the Upload Data page and load a CSV.")
        st.stop()


def parse_exclude_text(txt: str) -> list[str]:
    if txt is None:
        return []
    txt = str(txt).strip()
    if not txt:
        return []

    parts: list[str] = []
    for chunk in txt.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([p.strip() for p in chunk.split() if p.strip()])

    seen = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def recompute_from_raw_if_available() -> None:
    if st.session_state.df_raw is None:
        return

    spec = st.session_state.spec
    exclude_cols = parse_exclude_text(st.session_state.exclude_text)
    drop_missing_target = st.session_state.drop_missing_target

    df_clean = prepare_wide_dataset(
        st.session_state.df_raw,
        spec,
        drop_missing_target=drop_missing_target,
        coerce_age_numeric=False,
    )

    summ = dataset_summary(df_clean, spec)

    numeric_features = set(summ.get("numeric_feature_cols", []))
    meta_exclude = {spec.id_col, spec.target_col}
    meta_exclude = {c for c in meta_exclude if c}

    manual_exclude = {c for c in exclude_cols if c in df_clean.columns}
    missing_manual = [c for c in exclude_cols if c not in df_clean.columns]

    final_features = sorted(list(numeric_features - meta_exclude - manual_exclude))

    st.session_state.df_wide = df_clean
    st.session_state.feature_cols = final_features
    st.session_state.excluded_cols = sorted(list(meta_exclude | manual_exclude))
    st.session_state.missing_manual_excludes = missing_manual
