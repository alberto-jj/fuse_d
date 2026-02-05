from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from src.state import clear_dataset, init_session_state, require_data_loaded


st.set_page_config(
    page_title="Feature Importance",
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
            key="fi_clear_dataset_btn",
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


st.header("Feature importance")
require_data_loaded()

df_wide = st.session_state.df_wide
feature_cols = st.session_state.feature_cols
spec = st.session_state.spec

X = df_wide[feature_cols].to_numpy()
y = df_wide[spec.target_col].astype(str).to_numpy()
classes = sorted(np.unique(y))


def binary_log_loss(y_true, y_proba):
    return log_loss(y_true, y_proba, labels=[0, 1])


def parse_feature_list(txt: str) -> list[str]:
    if txt is None:
        return []
    raw = str(txt).strip()
    if not raw:
        return []
    parts = []
    for chunk in raw.replace("\n", ",").replace("\t", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([p.strip() for p in chunk.split() if p.strip()])
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def build_estimator(model_choice: str):
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="saga",
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    if model_choice == "LogReg L2 only":
        return logreg

    svm_rbf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    return VotingClassifier(
        estimators=[("logreg", logreg), ("svm_rbf", svm_rbf)],
        voting="soft",
        n_jobs=-1,
    )


st.subheader("Run settings")
n_splits = st.slider(
    "Stratified K-folds",
    min_value=3,
    max_value=15,
    value=int(st.session_state.n_splits),
    step=1,
    key="vim_n_splits_slider",
)
st.session_state.n_splits = n_splits

n_permutations = st.slider(
    "Permutations",
    min_value=10,
    max_value=200,
    value=50,
    step=5,
    key="vim_n_permutations_slider",
)

n_jobs = st.selectbox(
    "n_jobs",
    options=[-1, 1, 2, 4, 8],
    index=0,
    key="vim_n_jobs_select",
)

random_state = st.number_input(
    "Random state",
    value=42,
    step=1,
    key="vim_random_state_input",
)

methods = st.multiselect(
    "Methods",
    ["PFI", "CFI", "LOCO"],
    default=["PFI", "CFI", "LOCO"],
    key="vim_methods_select",
)

class_choices = st.multiselect(
    "Classes",
    options=classes,
    default=classes,
    key="vim_classes_select",
)

top_k = st.slider(
    "Top-k per (class, method)",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    key="vim_top_k_slider",
)

model_choice = st.selectbox(
    "Model",
    ["VotingClassifier (LogRegL2 + SVMRBF)", "LogReg L2 only"],
    index=0,
    key="vim_model_choice_select",
)

run = st.button("Run feature importance", type="primary", key="vim_run_btn")
progress = st.progress(0)
run_status = st.empty()

def _set_progress_color(color: str) -> None:
    st.markdown(
        f"""
        <style>
        div[data-testid="stProgress"] > div > div {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown("Apply significance threshold")
pval_threshold = st.number_input(
    "p-value threshold",
    value=0.05,
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="vim_pval_threshold_input",
)

correction = st.selectbox(
    "Multiple testing correction",
    ["none", "fdr_bh"],
    index=1,
    key="vim_pval_correction_select",
)

apply_filters_btn = st.button(
    "Apply significance threshold",
    key="vim_apply_filters_btn",
)


def extract_results_from_attrs(vim, fallback_features):
    importances = np.asarray(getattr(vim, "importances_", None))
    pvals = np.asarray(getattr(vim, "pvalues_", None))

    if importances is None or pvals is None:
        raise AttributeError("Could not find results on VIM object.")

    if importances.ndim == 2:
        imp_mean = importances.mean(axis=1)
        imp_sd = importances.std(axis=1, ddof=1)
    else:
        imp_mean = importances
        imp_sd = np.full_like(imp_mean, np.nan, dtype=float)

    df = pd.DataFrame(
        {
            "feature": fallback_features,
            "importance_mean": imp_mean,
            "importance_sd": imp_sd,
            "pval": np.asarray(pvals),
        }
    )
    return df


def apply_filters_to_results() -> None:
    res_df = st.session_state.get("vim_results_df")
    if res_df is None or res_df.empty:
        return

    try:
        from statsmodels.stats.multitest import multipletests
    except Exception:
        multipletests = None

    res_df = res_df.copy()
    res_df["pval_adj"] = res_df["pval"]
    if correction == "fdr_bh" and multipletests is not None:
        for (cls, method), grp in res_df.groupby(["class_pos", "method"], dropna=False):
            if grp["pval"].notna().any():
                pvals = grp["pval"].fillna(1.0).to_numpy()
                _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
                res_df.loc[grp.index, "pval_adj"] = p_adj

    res_df["rank"] = (
        res_df.groupby(["class_pos", "method"], dropna=False)
        .apply(lambda g: g["pval_adj"].rank(method="dense", ascending=True))
        .reset_index(level=[0, 1], drop=True)
    )

    valid = res_df["feature"].notna()
    top_rows = []
    for (cls, method), grp in res_df[valid].groupby(["class_pos", "method"]):
        grp_sorted = grp.sort_values(
            by=["pval_adj", "importance_mean"], ascending=[True, False]
        )
        top_rows.append(grp_sorted.head(top_k))
    top_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()

    if pval_threshold is not None and not top_df.empty:
        thresh_col = "pval_adj" if correction != "none" else "pval"
        if thresh_col in top_df.columns:
            top_df = top_df[top_df[thresh_col] <= pval_threshold]

    st.session_state.vim_results_df = res_df
    st.session_state.vim_top_df = top_df


if run:
    _set_progress_color("#7AA2F7")
    try:
        from hidimstat import PFICV, CFICV, LOCOCV
        from statsmodels.stats.multitest import multipletests
    except Exception as e:
        st.error(f"Required package missing or failed to import: {e}")
        st.stop()

    total_steps = max(1, len(class_choices) * max(1, len(methods)))
    step = 0

    features_groups = {fn: [i] for i, fn in enumerate(feature_cols)}
    res_rows = []

    for cls in class_choices:
        y_bin = (y == cls).astype(int)
        pos = int(y_bin.sum())
        neg = int(len(y_bin) - pos)
        if pos < n_splits or neg < n_splits:
            res_rows.append(
                {
                    "class_pos": cls,
                    "method": "SKIP",
                    "feature": None,
                    "importance_mean": None,
                    "importance_sd": None,
                    "pval": None,
                    "pval_adj": None,
                    "rank": None,
                    "note": f"Skipped: class has pos={pos}, neg={neg} < n_splits={n_splits}",
                }
            )
            continue

        clf = build_estimator(model_choice)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for method in methods:
            step += 1
            progress.progress(min(step / total_steps, 1.0))
            run_status.info(f"Estimating {method} for class '{cls}' (folds={n_splits})")
            try:
                if method == "PFI":
                    vim = PFICV(
                        estimators=clf,
                        cv=cv,
                        statistical_test="nb-ttest",
                        method="predict_proba",
                        loss=binary_log_loss,
                        n_permutations=n_permutations,
                        features_groups=features_groups,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                elif method == "CFI":
                    vim = CFICV(
                        estimators=clf,
                        cv=cv,
                        statistical_test="nb-ttest",
                        method="predict_proba",
                        loss=binary_log_loss,
                        n_permutations=n_permutations,
                        imputation_model_continuous=RidgeCV(alphas=np.logspace(-3, 3, 10)),
                        imputation_model_categorical=LogisticRegressionCV(
                            Cs=10,
                            penalty="l2",
                            solver="liblinear",
                            max_iter=2000,
                            class_weight="balanced",
                        ),
                        feature_types="continuous",
                        features_groups=features_groups,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                else:
                    vim = LOCOCV(
                        estimators=clf,
                        cv=cv,
                        statistical_test="nb-ttest",
                        method="predict_proba",
                        loss=binary_log_loss,
                        features_groups=features_groups,
                        n_jobs=n_jobs,
                    )

                vim.fit_importance(X, y_bin)
                method_df = extract_results_from_attrs(vim, feature_cols)
                method_df["class_pos"] = cls
                method_df["method"] = method
                res_rows.append(method_df)
            except Exception as e:
                res_rows.append(
                    {
                        "class_pos": cls,
                        "method": method,
                        "feature": None,
                        "importance_mean": None,
                        "importance_sd": None,
                        "pval": None,
                        "pval_adj": None,
                        "rank": None,
                        "note": f"Failed: {e}",
                    }
                )

    progress.progress(1.0)
    _set_progress_color("#00C853")
    run_status.success("Feature importance run complete.")

    if len(res_rows) == 0:
        st.warning("No results were produced.")
    else:
        res_df = pd.concat(
            [r if isinstance(r, pd.DataFrame) else pd.DataFrame([r]) for r in res_rows],
            ignore_index=True,
        )

        if "pval_adj" not in res_df.columns:
            res_df["pval_adj"] = res_df["pval"]

        if correction == "fdr_bh":
            for (cls, method), grp in res_df.groupby(["class_pos", "method"], dropna=False):
                if grp["pval"].notna().any():
                    pvals = grp["pval"].fillna(1.0).to_numpy()
                    _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
                    res_df.loc[grp.index, "pval_adj"] = p_adj

        res_df["rank"] = (
            res_df.groupby(["class_pos", "method"], dropna=False)
            .apply(
                lambda g: g.assign(
                    _rank=g["pval_adj"].rank(method="dense", ascending=True)
                )["_rank"]
            )
            .reset_index(level=[0, 1], drop=True)
        )

        valid = res_df["feature"].notna()
        top_rows = []
        for (cls, method), grp in res_df[valid].groupby(["class_pos", "method"]):
            grp_sorted = grp.sort_values(
                by=["pval_adj", "importance_mean"], ascending=[True, False]
            )
            top_rows.append(grp_sorted.head(top_k))
        top_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()

        if pval_threshold is not None and not top_df.empty:
            thresh_col = "pval_adj" if correction != "none" else "pval"
            if thresh_col in top_df.columns:
                top_df = top_df[top_df[thresh_col] <= pval_threshold]

        st.session_state.vim_results_df = res_df
        st.session_state.vim_top_df = top_df
        st.session_state.vim_run_meta = {
            "n_splits": n_splits,
            "n_permutations": n_permutations,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "methods": methods,
            "classes": class_choices,
            "top_k": top_k,
            "pval_threshold": pval_threshold,
            "correction": correction,
            "model_choice": model_choice,
        }
        st.session_state.vim_filter_active = False
else:
    progress.empty()
    run_status.empty()

if apply_filters_btn and st.session_state.get("vim_results_df") is not None:
    apply_filters_to_results()
    st.session_state.vim_filter_active = True

res_df = st.session_state.get("vim_results_df")
top_df = st.session_state.get("vim_top_df")

if res_df is not None:
    st.subheader("Results")
    tabs = st.tabs([m for m in ["PFI", "CFI", "LOCO"] if m in res_df["method"].unique()])

    for tab, method in zip(tabs, [m for m in ["PFI", "CFI", "LOCO"] if m in res_df["method"].unique()]):
        with tab:
            for cls in classes:
                use_filtered = bool(st.session_state.get("vim_filter_active")) and top_df is not None and not top_df.empty
                if use_filtered:
                    cls_df = top_df[(top_df["method"] == method) & (top_df["class_pos"] == cls)]
                    if cls_df.empty:
                        cls_df = res_df[(res_df["method"] == method) & (res_df["class_pos"] == cls)]
                else:
                    cls_df = res_df[(res_df["method"] == method) & (res_df["class_pos"] == cls)]
                if cls_df.empty:
                    continue
                with st.expander(f"{method} - {cls}"):
                    st.dataframe(cls_df, use_container_width=True)

    st.download_button(
        "Download full results (CSV)",
        data=res_df.to_csv(index=False),
        file_name="vim_results.csv",
        mime="text/csv",
        key="vim_download_full_btn",
    )

    if top_df is not None and not top_df.empty:
        st.download_button(
            "Download top results (CSV)",
            data=top_df.to_csv(index=False),
            file_name="vim_results_top.csv",
            mime="text/csv",
            key="vim_download_top_btn",
        )


st.subheader("Build selected feature set")

method_opts = ["PFI", "CFI", "LOCO"]
tab_from_results, tab_manual = st.tabs(["From results", "Manual list"])

with tab_from_results:
    if res_df is None or res_df.empty:
        st.info("Run feature importance to enable selection from results.")
    else:
        sel_classes = st.multiselect(
            "Classes to include",
            options=classes,
            default=class_choices if class_choices else classes,
            key="vim_select_classes",
        )
        sel_methods = st.multiselect(
            "Methods to include",
            options=method_opts,
            default=methods if methods else method_opts,
            key="vim_select_methods",
        )
        combine_rule = st.selectbox(
            "Combine rule",
            ["union", "intersection"],
            index=0,
            key="vim_combine_rule_select",
        )
        criteria = st.radio(
            "Selection criteria",
            ["Top K", "p-value threshold"],
            index=0,
            key="vim_select_criteria_radio",
        )

        thresh_col = "pval_adj" if correction != "none" else "pval"
        res_valid = res_df[
            res_df["feature"].notna()
            & res_df["method"].isin(sel_methods)
            & res_df["class_pos"].isin(sel_classes)
        ].copy()

        selected_sets = []
        if criteria == "Top K":
            for (cls, method), grp in res_valid.groupby(["class_pos", "method"]):
                grp_sorted = grp.sort_values(
                    by=["pval_adj", "importance_mean"], ascending=[True, False]
                )
                selected_sets.append(set(grp_sorted.head(top_k)["feature"].tolist()))
        else:
            filt = res_valid[res_valid[thresh_col] <= pval_threshold]
            for (cls, method), grp in filt.groupby(["class_pos", "method"]):
                selected_sets.append(set(grp["feature"].tolist()))

        if selected_sets:
            if combine_rule == "union":
                selected_features = sorted(set.union(*selected_sets))
            else:
                selected_features = sorted(set.intersection(*selected_sets))
        else:
            selected_features = []

        st.caption(f"Selected features: {len(selected_features)}")
        st.write(selected_features[:50])
        if len(selected_features) > 50:
            st.caption(f"... ({len(selected_features)} total)")

        if st.button("Apply selected features", key="vim_apply_from_results_btn"):
            st.session_state.feature_cols_selected = selected_features
            st.success(f"Saved {len(selected_features)} selected features.")

with tab_manual:
    manual_text = st.text_area(
        "Paste feature list (comma, space, or newline separated)",
        value="",
        height=140,
        key="vim_manual_textarea",
    )
    manual_features = parse_feature_list(manual_text)
    missing = [f for f in manual_features if f not in feature_cols]
    valid = [f for f in manual_features if f in feature_cols]

    if missing:
        st.warning(f"Missing features ({len(missing)}): " + ", ".join(missing[:10]))
        if len(missing) > 10:
            st.caption("... (more missing features)")

    st.caption(f"Valid features: {len(valid)}")

    if st.button("Apply selected features", key="vim_apply_manual_btn"):
        st.session_state.feature_cols_selected = valid
        st.success(f"Saved {len(valid)} selected features.")

if st.button("Clear selected features", key="vim_clear_selected_btn"):
    st.session_state.feature_cols_selected = None
    st.info("Cleared selected features.")

if st.session_state.feature_cols_selected:
    st.subheader("Current selected feature set")
    st.caption(f"Count: {len(st.session_state.feature_cols_selected)}")
    st.write(st.session_state.feature_cols_selected[:50])
    if len(st.session_state.feature_cols_selected) > 50:
        st.caption(f"... ({len(st.session_state.feature_cols_selected)} total)")
