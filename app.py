from __future__ import annotations

import streamlit as st

from src.state import clear_dataset, init_session_state


st.set_page_config(
    page_title="FUSE-D - Feature Selection",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

with st.sidebar:
    st.image("fused.logo.png", use_container_width=True)


def landing_page() -> None:
    with st.sidebar:
        #st.divider()
        st.subheader("Progress")

        if st.session_state.df_wide is not None:
            st.success("Dataset loaded")
            st.caption(
                f"Rows: {st.session_state.df_wide.shape[0]} | Cols: {st.session_state.df_wide.shape[1]}"
            )
            st.button(
                "Clear dataset",
                key="app_clear_dataset_btn",
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

    st.image("fused.logo.png", width=300)
    st.header("FUSE-D")
    st.write(
        "Welcome to the FUSE-D multipage app. Use the navigation in the sidebar to switch pages."
    )

    st.subheader("How to use")
    st.write(
        "- Start with the Upload Data page to load a CSV and configure columns.\n"
        "- Then run the Baseline Model page to evaluate a soft-voting classifier.\n"
        "- Use Feature Importance to select a reduced feature set.\n"
        "- Re-run Baseline (Selected Features) to compare against the full set."
    )
    st.divider()
    st.caption(
        "Methods in Feature Importance are based on hidimstat (Mind-Inria). "
        "Reference: `https://github.com/mind-inria/hidimstat`"
    )




overview = st.Page(landing_page, title="Overview", icon=":material/home:")
upload = st.Page("pages/1_Upload_Data.py", title="Upload Data", icon=":material/upload:")
baseline = st.Page("pages/2_Baseline_Model.py", title="Baseline Model", icon=":material/analytics:")
feature_importance = st.Page(
    "pages/3_Feature_Importance.py",
    title="Feature Importance",
    icon=":material/insights:",
)
baseline_selected = st.Page(
    "pages/4_ReRun_Baseline_Selected.py",
    title="Re-run Baseline (with Selected Features)",
    icon=":material/analytics:",
)

pg = st.navigation(
    {
        "Start here": [overview, upload, baseline, feature_importance, baseline_selected],
    }
)
pg.run()
