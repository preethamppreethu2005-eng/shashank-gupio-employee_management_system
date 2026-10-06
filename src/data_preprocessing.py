"""Dataset loading and reusable preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def load_dataset(path: str | Path, target: str = "y") -> tuple[pd.DataFrame, pd.Series]:
    """Read the UCI bank marketing CSV and return features and binary labels."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # The original UCI bank-full.csv uses semicolons; sep=None also handles CSV copies.
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = frame.columns.astype(str).str.strip()
    if target not in frame.columns:
        raise ValueError(
            f"Target column {target!r} is missing. Available columns: {list(frame.columns)}"
        )

    labels = frame.pop(target)
    labels = labels.astype("string").str.strip().str.lower()
    if set(labels.dropna().unique()) <= {"yes", "no"}:
        labels = labels.map({"no": 0, "yes": 1})
    elif labels.isna().any() or labels.nunique() != 2:
        raise ValueError("The target must contain exactly two non-null classes.")
    else:
        classes = sorted(labels.dropna().unique())
        labels = labels.map({classes[0]: 0, classes[1]: 1})

    if labels.isna().any():
        raise ValueError("The target contains missing or unrecognized labels.")
    return frame, labels.astype("int8").rename(target)


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing that can be fitted safely inside a model pipeline."""
    numeric = Pipeline(steps=[("scale", StandardScaler())])
    categorical = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_include="object")),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
