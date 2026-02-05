# =========================
# io.py (FULL) — NO changes required for the sidebar behavior
# (keeping your simplified spec with optional age/center)
# =========================
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    id_col: Optional[str] = "participant_id"
    target_col: Optional[str] = "ad_syndrome"
    age_col: Optional[str] = None
    center_col: Optional[str] = None


def read_csv_from_upload(uploaded_file, *, encoding: Optional[str] = None) -> pd.DataFrame:
    if encoding is None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(uploaded_file, encoding=encoding)


def validate_wide_dataset(
    df: pd.DataFrame,
    spec: DatasetSpec,
    *,
    require_id: bool = True,
    require_target: bool = True,
) -> Tuple[bool, list[str]]:
    msgs: list[str] = []

    if df is None or df.empty:
        return False, ["Dataset is empty."]

    # Duplicate column names
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        msgs.append(f"Duplicate column names detected: {dupes[:10]}{'...' if len(dupes) > 10 else ''}")

    # ID checks
    if require_id and spec.id_col:
        if spec.id_col not in df.columns:
            return False, [f"Missing required ID column: '{spec.id_col}'"]
        if df[spec.id_col].isna().any():
            msgs.append(f"ID column '{spec.id_col}' has missing values.")
        n = df.shape[0]
        nunique = df[spec.id_col].nunique(dropna=True)
        if nunique != n:
            msgs.append(f"ID column '{spec.id_col}' is not unique: {nunique}/{n} unique rows.")

    # Target checks
    if require_target and spec.target_col:
        if spec.target_col not in df.columns:
            return False, [f"Missing required target column: '{spec.target_col}'"]
        if df[spec.target_col].isna().all():
            return False, [f"Target column '{spec.target_col}' is entirely missing."]
        if df[spec.target_col].isna().any():
            msgs.append(f"Target column '{spec.target_col}' has missing values (rows may be dropped).")

    # Optional age checks (only if user chooses to keep it; we don't enforce it)
    if spec.age_col and spec.age_col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[spec.age_col]):
            msgs.append(f"Age column '{spec.age_col}' is not numeric; can be coerced if desired.")

    return True, msgs


def dataset_summary(df: pd.DataFrame, spec: DatasetSpec) -> dict:
    out = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_missing_total": int(df.isna().sum().sum()),
    }

    if spec.id_col and spec.id_col in df.columns:
        out["n_subjects_unique"] = int(df[spec.id_col].nunique(dropna=True))

    if spec.target_col and spec.target_col in df.columns:
        out["target_counts"] = df[spec.target_col].value_counts(dropna=False).to_dict()

    # Numeric feature columns excluding *only* ID/target by default
    # (other columns like age/sex can be excluded via sidebar)
    exclude = {spec.id_col, spec.target_col}
    numeric_features = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_features.append(c)

    out["n_numeric_features"] = int(len(numeric_features))
    out["numeric_feature_cols"] = numeric_features
    return out


def prepare_wide_dataset(
    df: pd.DataFrame,
    spec: DatasetSpec,
    *,
    drop_missing_target: bool = True,
    coerce_age_numeric: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    if drop_missing_target and spec.target_col and spec.target_col in out.columns:
        out = out.dropna(subset=[spec.target_col])

    # Optional coercion if you ever turn it on
    if coerce_age_numeric and spec.age_col and spec.age_col in out.columns:
        out[spec.age_col] = pd.to_numeric(out[spec.age_col], errors="coerce")

    return out
