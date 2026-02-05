from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.state import clear_dataset, init_session_state, require_data_loaded


st.set_page_config(
    page_title="Baseline Model",
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
            key="baseline_clear_dataset_btn",
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


st.header("Baseline classifier - Single soft-voting ensemble")

require_data_loaded()

df_wide = st.session_state.df_wide
feature_cols = st.session_state.feature_cols
spec = st.session_state.spec

st.subheader("CV options")
n_random_state = st.number_input(
    "Random state",
    value=int(st.session_state.baseline_random_state),
    step=1,
    key="baseline_random_state_input",
)
st.session_state.baseline_random_state = int(n_random_state)
n_splits = st.slider(
    "Stratified K-folds",
    min_value=3,
    max_value=15,
    value=int(st.session_state.n_splits),
    step=1,
    key="baseline_n_splits_slider",
)
st.session_state.n_splits = n_splits

metric_choice = st.selectbox(
    "Optimization metric",
    ["balanced_accuracy", "f1_macro", "accuracy"],
    index=["balanced_accuracy", "f1_macro", "accuracy"].index(st.session_state.metric_choice),
    help="Primary metric to focus on. All metrics are reported.",
    key="baseline_metric_choice_select",
)
st.session_state.metric_choice = metric_choice

diag = st.checkbox(
    "Compute confusion matrix + per-class report (slower)",
    value=st.session_state.diag,
    key="baseline_diag_chk",
)
st.session_state.diag = diag

X = df_wide[feature_cols].to_numpy()
y = df_wide[spec.target_col].to_numpy()

class_counts = pd.Series(y).value_counts()
st.caption(f"X shape: {X.shape} | Classes: {len(class_counts)}")

st.subheader("Class counts")
st.dataframe(class_counts.to_frame("n"), use_container_width=True)

min_class = int(class_counts.min())
if min_class < n_splits:
    st.warning(
        f"Smallest class has {min_class} samples, but you chose {n_splits} folds. "
        "StratifiedKFold may fail. Reduce folds or merge rare classes."
    )

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(n_random_state))
scoring = {
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
}

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
            random_state=int(n_random_state),
                multi_class="auto",
            ),
        ),
    ]
)

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
            random_state=int(n_random_state),
            ),
        ),
    ]
)

model = VotingClassifier(
    estimators=[("logreg", logreg), ("svm_rbf", svm_rbf)],
    voting="soft",
    n_jobs=-1,
)

st.caption("Model = Soft voting ensemble of LogReg(L2) + SVM(RBF).")

run = st.button("Run baseline CV", type="primary", key="baseline_run_btn")
run_status = st.empty()
def mean_std(arr):
    return float(np.mean(arr)), float(np.std(arr))


def render_results() -> None:
    results_df = st.session_state.get("baseline_results")
    summary = st.session_state.get("baseline_summary", {})

    if results_df is None:
        st.info("Set CV options above, then click Run baseline CV.")
        return

    st.subheader("Baseline CV performance (mean +/- std across folds)")
    st.dataframe(results_df, use_container_width=True)

    if summary:
        st.success(
            "Done. Primary metric: "
            f"{summary.get('metric_choice', metric_choice)} "
            f"(balanced_acc={summary.get('balanced_acc', 0.0):.3f}, "
            f"f1_macro={summary.get('f1_macro', 0.0):.3f}, "
            f"acc={summary.get('accuracy', 0.0):.3f})"
        )

    diag_payload = st.session_state.get("baseline_diag")
    if diag_payload:
        labels = diag_payload["labels"]
        cm = diag_payload["cm"]
        cm_norm = diag_payload["cm_norm"]
        y_true = diag_payload.get("y_true")
        y_proba = diag_payload.get("y_proba")

        def _cm_to_df(matrix, labels_list, fmt):
            rows = []
            for i, true_label in enumerate(labels_list):
                for j, pred_label in enumerate(labels_list):
                    val = matrix[i, j]
                    rows.append(
                        {
                            "true": str(true_label),
                            "pred": str(pred_label),
                            "value": float(val),
                            "text": fmt(val),
                        }
                    )
            return pd.DataFrame(rows)

        cm_df = _cm_to_df(cm, labels, lambda v: f"{int(v)}")
        cmn_df = _cm_to_df(cm_norm, labels, lambda v: f"{v:.2f}")

        st.subheader("Confusion matrices (cross-validated predictions)")
        c1, c2 = st.columns(2)

        with c1:
            st.vega_lite_chart(
                cm_df,
                {
                    "width": 500,
                    "height": 500,
                    "layer": [
                        {
                            "mark": "rect",
                            "encoding": {
                                "x": {"field": "pred", "type": "nominal", "title": None},
                                "y": {"field": "true", "type": "nominal", "title": None},
                                "color": {"field": "value", "type": "quantitative","scale": {"scheme": "viridis"}},
                            },
                        },
                        {
                            "mark": {"type": "text", "baseline": "middle", "fontSize": 16},
                            "encoding": {
                                "x": {"field": "pred", "type": "nominal"},
                                "y": {"field": "true", "type": "nominal"},
                                "text": {"field": "text", "type": "nominal"},
                                "color": {"value": "black"},
                            },
                        },
                    ],
                },
                use_container_width=False,
            )

        with c2:
            st.vega_lite_chart(
                cmn_df,
                {
                    "width": 500,
                    "height": 500,
                    "layer": [
                        {
                            "mark": "rect",
                            "encoding": {
                                "x": {"field": "pred", "type": "nominal", "title": None},
                                "y": {"field": "true", "type": "nominal", "title": None},
                                "color": {"field": "value", "type": "quantitative","scale": {"scheme": "viridis"}},
                            },
                        },
                        {
                            "mark": {"type": "text", "baseline": "middle", "fontSize": 16},
                            "encoding": {
                                "x": {"field": "pred", "type": "nominal"},
                                "y": {"field": "true", "type": "nominal"},
                                "text": {"field": "text", "type": "nominal"},
                                "color": {"value": "black"},
                            },
                        },
                    ],
                },
                use_container_width=False,
            )

        st.subheader("Per-class classification report (cross-validated predictions)")
        st.dataframe(diag_payload["report_df_view"], use_container_width=True)

        with st.expander("Report (text)"):
            st.text(diag_payload["report_text"])

        if y_true is not None and y_proba is not None:
            st.subheader("ROC and PR curves")
            curve_view = st.selectbox(
                "Curve view",
                ["Per class", "Overall"],
                index=0,
                key="baseline_curve_view_select",
            )

            y_true_arr = np.asarray(y_true)
            y_proba_arr = np.asarray(y_proba)
            supports = pd.Series(y_true_arr).value_counts().reindex(labels).fillna(0).to_numpy()

            def _curve_df(curve_type: str):
                rows = []
                if curve_view == "Per class":
                    for i, cls in enumerate(labels):
                        y_bin = (y_true_arr == cls).astype(int)
                        scores = y_proba_arr[:, i] if y_proba_arr.ndim > 1 else y_proba_arr
                        if curve_type == "roc":
                            x, yv, _ = roc_curve(y_bin, scores)
                        else:
                            yv, x, _ = precision_recall_curve(y_bin, scores)
                        for a, b in zip(x, yv):
                            rows.append({"x": float(a), "y": float(b), "class": str(cls)})
                else:
                    grid = np.linspace(0, 1, 101)
                    curves = []
                    for i, cls in enumerate(labels):
                        y_bin = (y_true_arr == cls).astype(int)
                        scores = y_proba_arr[:, i] if y_proba_arr.ndim > 1 else y_proba_arr
                        if curve_type == "roc":
                            x, yv, _ = roc_curve(y_bin, scores)
                            y_interp = np.interp(grid, x, yv)
                        else:
                            prec, rec, _ = precision_recall_curve(y_bin, scores)
                            y_interp = np.interp(grid, rec, prec)
                        curves.append(y_interp)
                    curves = np.vstack(curves) if curves else np.zeros((1, len(grid)))
                    weights = supports / supports.sum() if supports.sum() > 0 else None
                    if curve_type == "roc":
                        macro_avg = np.mean(curves, axis=0)
                        weighted_avg = np.average(curves, axis=0, weights=weights) if weights is not None else macro_avg
                        rows = [{"x": float(a), "y": float(b), "class": "macro avg"} for a, b in zip(grid, macro_avg)]
                        rows += [{"x": float(a), "y": float(b), "class": "weighted avg"} for a, b in zip(grid, weighted_avg)]
                    else:
                        y_true_bin = (y_true_arr[:, None] == np.array(labels)[None, :]).astype(int)
                        if y_proba_arr.ndim == 1:
                            y_score = np.tile(y_proba_arr[:, None], (1, len(labels)))
                        else:
                            y_score = y_proba_arr
                        prec, rec, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
                        rows = [{"x": float(a), "y": float(b), "class": "overall"} for a, b in zip(rec, prec)]
                return pd.DataFrame(rows)

            roc_df = _curve_df("roc")
            pr_df = _curve_df("pr")
            pos_rate = (1.0 / len(labels)) if len(labels) else 0.0
            c1, c2 = st.columns(2)
            with c1:
                st.vega_lite_chart(
                    roc_df,
                    {
                        "width": 500,
                        "height": 500,
                        "layer": [
                            {
                                "mark": {"type": "line"},
                                "encoding": {
                                    "x": {"field": "x", "type": "quantitative", "title": "False positive rate"},
                                    "y": {"field": "y", "type": "quantitative", "title": "True positive rate"},
                                    "color": {"field": "class", "type": "nominal"},
                                },
                            },
                            {
                                "data": {"values": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]},
                                "mark": {"type": "line", "strokeDash": [4, 4], "color": "gray"},
                                "encoding": {
                                    "x": {"field": "x", "type": "quantitative"},
                                    "y": {"field": "y", "type": "quantitative"},
                                },
                            },
                        ],
                    },
                    use_container_width=False,
                )
            with c2:
                st.vega_lite_chart(
                    pr_df,
                    {
                        "width": 500,
                        "height": 500,
                        "layer": [
                            {
                                "mark": {"type": "line"},
                                "encoding": {
                                    "x": {"field": "x", "type": "quantitative", "title": "Recall"},
                                    "y": {"field": "y", "type": "quantitative", "title": "Precision"},
                                    "color": {"field": "class", "type": "nominal"},
                                },
                            },
                            {
                                "data": {"values": [{"x": 0, "y": pos_rate}, {"x": 1, "y": pos_rate}]},
                                "mark": {"type": "line", "strokeDash": [4, 4], "color": "gray"},
                                "encoding": {
                                    "x": {"field": "x", "type": "quantitative"},
                                    "y": {"field": "y", "type": "quantitative"},
                                },
                            },
                        ],
                    },
                    use_container_width=False,
                )


if run:
    run_status.info("Running cross-validated evaluation...")
    with st.spinner("Running cross-validated evaluation..."):
        cv_res = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
        )

    bal_m, bal_s = mean_std(cv_res["test_balanced_accuracy"])
    f1_m, f1_s = mean_std(cv_res["test_f1_macro"])
    acc_m, acc_s = mean_std(cv_res["test_accuracy"])

    results_df = pd.DataFrame(
        [
            {
                "model": "soft_voting_logreg+svm_rbf",
                "balanced_accuracy_mean": bal_m,
                "balanced_accuracy_std": bal_s,
                "f1_macro_mean": f1_m,
                "f1_macro_std": f1_s,
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
            }
        ]
    )

    st.session_state.baseline_results = results_df
    st.session_state.baseline_cv_scores = cv_res
    st.session_state.baseline_summary = {
        "balanced_acc": bal_m,
        "f1_macro": f1_m,
        "accuracy": acc_m,
        "metric_choice": metric_choice,
    }

    if diag:
        run_status.info("Computing cross-validated predictions for confusion matrix...")
        with st.spinner("Computing cross-validated predictions for confusion matrix..."):
            y_pred = cross_val_predict(
                model,
                X,
                y,
                cv=cv,
                method="predict",
                n_jobs=-1,
            )
            y_proba = cross_val_predict(
                model,
                X,
                y,
                cv=cv,
                method="predict_proba",
                n_jobs=-1,
            )

        labels = list(pd.Series(y).value_counts().index)
        cm = confusion_matrix(y, y_pred, labels=labels)

        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

        report_dict = classification_report(
            y,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).T.rename_axis("label").reset_index()

        keep_rows = set(labels) | {"macro avg", "weighted avg", "accuracy"}
        report_df_view = report_df[report_df["label"].isin(keep_rows)].copy()

        for col in ["precision", "recall", "f1-score"]:
            if col in report_df_view.columns:
                report_df_view[col] = report_df_view[col].astype(float).round(3)
        if "support" in report_df_view.columns:
            report_df_view["support"] = report_df_view["support"].fillna(0).astype(int)

        report_text = classification_report(
            y,
            y_pred,
            labels=labels,
            zero_division=0,
        )

        st.session_state.baseline_diag = {
            "labels": labels,
            "cm": cm,
            "cm_norm": cm_norm,
            "report_df_view": report_df_view,
            "report_text": report_text,
            "y_true": y,
            "y_proba": y_proba,
        }
    else:
        st.session_state.baseline_diag = None
    run_status.success("Baseline CV complete.")

render_results()
